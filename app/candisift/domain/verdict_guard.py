"""Deterministic output guardrails — run AFTER the synthesizer, BEFORE persistence.

The persona agents are *told* the rules in prose ("cap recommendation when a must-have
is unmet", "never fabricate evidence"). Prose is not enforcement: the model can drift,
especially on an adversarial AI-written resume. This module is the programmatic net the
build doc calls for (§4.2) — it re-checks the model's verdict against the hard structured
facts and downgrades / flags for human review when the verdict and the facts disagree.

No LLM, no I/O. Pure functions over the structured evals. For an IT-staffing pipeline
this is the defensibility backstop: a bad auto-shortlist burns a client, so a verdict
that contradicts the facts must never reach the recruiter as a clean "shortlist".

Three checks, all from the doc:
  1. Knockout cap   — unmet must-have  => recommendation can't be "shortlist".
  2. Fraud cap      — overlapping full-time / date-fraud signal => can't be "shortlist".
  3. Evidence net   — every surfaced claim's evidence must trace to a profile field;
                      claims that don't are listed and force human review.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import CandidateProfile, Finding, Recommendation, RiskEval, Synthesis, TechEval

_WORD = re.compile(r"[a-z0-9]+")

# risk-flag phrasing that signals staffing fraud (vs. a benign, explained gap). A hit
# caps the verdict regardless of fit, same as concurrent_fulltime on the profile.
# Fraud-SPECIFIC phrases only (matched against the evaluator's own `claim`, not the
# verbatim resume evidence): bare stems like "overlap"/"concurrent"/"inconsistent" were
# tripping benign or NEGATED prose ("skills overlap with the role", "no inconsistencies
# found", "handled concurrent projects"), demoting legitimate candidates. Each pattern
# now requires the fraud context (full-time overlap, fabricated/falsified, date conflict).
_FRAUD_RE = re.compile(
    r"overlap\w*\s+(?:concurrent\s+)?full[\s-]?time"
    r"|concurrent\s+full[\s-]?time"
    r"|simultaneous\s+full[\s-]?time"
    r"|two\s+(?:concurrent\s+)?full[\s-]?time"
    r"|fabricat\w*|falsif\w*"
    r"|inconsistent\s+dates?|date\s+inconsistenc\w*|conflicting\s+dates?"
    r"|impossible\s+(?:date|timeline)|date\s+mismatch",
    re.I,
)

# A NEGATED fraud phrase is the opposite signal: "no fabrication found", "no date
# inconsistencies", "without falsification" mean the auditor CLEARED the candidate.
# Without this veto the bare `fabricat\w*`/`inconsistent dates` stems flip a clean
# verdict to "maybe" + human-review — exactly backwards. Veto a match preceded (within
# a few words) by a negator.
_FRAUD_NEGATION_RE = re.compile(
    r"\b(?:no|not|none|never|without|n't|free of|absent|zero)\b[\w\s,'-]{0,24}?"
    r"(?:overlap|concurrent|simultaneous|fabricat\w*|falsif\w*|inconsisten\w*|"
    r"conflicting|impossible|mismatch)",
    re.I,
)


def _flag_signals_fraud(claim: str) -> bool:
    return bool(_FRAUD_RE.search(claim)) and not _FRAUD_NEGATION_RE.search(claim)

# common filler stripped before grounding so a fabrication can't clear the threshold on
# shared connective words alone (e.g. "Used Python on AWS to run a <invented> platform").
_STOPWORDS = {
    "a", "an", "the", "of", "on", "in", "at", "to", "for", "and", "or", "with", "as",
    "by", "is", "was", "were", "be", "been", "are", "am", "this", "that", "it", "its",
    "from", "into", "using", "used", "use", "via", "per", "across", "over", "under",
    "did", "do", "done", "has", "have", "had", "led", "run", "ran", "built", "build",
}

# An evidence quote is "grounded" if at least this fraction of its content words appear
# in the candidate profile the agent was actually shown.
# ponytail: token-containment, not exact substring — tolerates the model paraphrasing
# its own quote. Upgrade path: exact-span match against the raw resume text if false
# negatives ever bite. 0.6 chosen so a fabricated claim (few/no matching tokens) fails
# while a lightly-reworded real quote passes.
_GROUND_THRESHOLD = 0.6
# minimum distinct CONTENT words for a quote to be judged (filler excluded). 2 lets a
# terse-but-real quote through ("AWS certified", "Staff Engineer") while a trivial quote
# ("did") still falls below the floor.
_MIN_EVIDENCE_WORDS = 2


def _words(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


def _content_words(text: str) -> list[str]:
    """Words with filler removed — what a claim actually asserts. Both the corpus and
    each evidence quote are reduced to these, so shared connective words can't inflate
    the grounding ratio."""
    return [w for w in _words(text) if w not in _STOPWORDS]


def profile_corpus(profile: CandidateProfile) -> set[str]:
    """Every content word the agent could legitimately have quoted: all string values
    in the (PII-stripped) profile — summary, titles, certs, gaps, and each skill's name
    and evidence snippet. Evidence not drawn from here was invented."""
    words: set[str] = set()

    def walk(v) -> None:
        if isinstance(v, str):
            words.update(_content_words(v))
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)

    walk(profile.model_dump())
    return words


def is_grounded(evidence: str, corpus: set[str]) -> bool:
    ev = set(_content_words(evidence))
    if len(ev) < _MIN_EVIDENCE_WORDS:
        return False
    return len(ev & corpus) / len(ev) >= _GROUND_THRESHOLD


def ungrounded_claims(findings: list[Finding], corpus: set[str]) -> list[str]:
    """Claims whose cited evidence does not trace back to the profile — the
    anti-hallucination list. Returns the claim text (truncated) for the audit trail."""
    return [f.claim[:160] for f in findings if not is_grounded(f.evidence, corpus)]


def _risk_has_fraud(risk: RiskEval | None) -> bool:
    # match the evaluator's own characterization (claim), not the verbatim resume
    # evidence — a benign resume line that happens to contain a stem must not trip this.
    if risk is None:
        return False
    return any(_flag_signals_fraud(f.claim) for f in risk.flags)


@dataclass(frozen=True)
class GuardOutcome:
    recommendation: Recommendation        # possibly downgraded from the model's
    requires_human_review: bool
    review_reasons: list[str] = field(default_factory=list)
    ungrounded: list[str] = field(default_factory=list)


def apply_guards(
    synthesis: Synthesis,
    tech: TechEval | None,
    risk: RiskEval | None,
    profile: CandidateProfile,
    *,
    bias_flagged: bool = False,
) -> GuardOutcome:
    """Reconcile the model's verdict with the hard facts. Never *upgrades* a verdict;
    only caps an over-confident "shortlist" and surfaces reasons a human must look."""
    reasons: list[str] = []
    rec = synthesis.recommendation

    # Knockout + fraud: record the reason whenever the condition holds (so the audit
    # trail is complete even on a model verdict that's already 'maybe'/'reject'), but
    # only ever CAP a shortlist down to 'maybe' — never upgrade.
    missing = list(tech.missing_must_haves) if tech else []
    if missing:
        reasons.append(f"unmet must-have(s): {missing}")
        if rec is Recommendation.shortlist:
            rec = Recommendation.maybe

    if profile.concurrent_fulltime or _risk_has_fraud(risk):
        reasons.append("fraud-class risk (overlapping full-time / date inconsistency)")
        if rec is Recommendation.shortlist:
            rec = Recommendation.maybe

    # Evidence grounding — include tech.transferable: it is where the model is invited to
    # credit adjacent experience for a must-have they lack BY NAME, i.e. the field most
    # prone to a fabricated bridge that silently clears a knockout.
    corpus = profile_corpus(profile)
    ungrounded = ungrounded_claims(synthesis.strengths, corpus)
    if tech:
        ungrounded += ungrounded_claims(tech.matched, corpus)
        ungrounded += ungrounded_claims(tech.transferable, corpus)
    if ungrounded:
        reasons.append(f"{len(ungrounded)} ungrounded claim(s): evidence not traceable to the profile")

    if bias_flagged:
        reasons.append("protected-class proxy in verdict text")

    return GuardOutcome(
        recommendation=rec,
        requires_human_review=bool(reasons),
        review_reasons=reasons,
        ungrounded=ungrounded,
    )


def demo() -> None:
    """Runnable self-check: the three caps fire on bad verdicts, pass clean ones."""
    prof = CandidateProfile(summary="Led migration of monolith to microservices on AWS using Python",
                            concurrent_fulltime=False)
    corpus = profile_corpus(prof)
    assert is_grounded("migration of monolith to microservices", corpus)
    assert not is_grounded("ran a kubernetes cluster for fintech payments", corpus)  # invented
    assert not is_grounded("did", corpus)  # too short

    # knockout cap
    s = Synthesis(overall_fit=90, recommendation=Recommendation.shortlist)
    t = TechEval(missing_must_haves=["Distributed systems"])
    out = apply_guards(s, t, None, prof)
    assert out.recommendation is Recommendation.maybe and out.requires_human_review

    # fraud cap
    prof2 = CandidateProfile(summary="x", concurrent_fulltime=True)
    out2 = apply_guards(Synthesis(overall_fit=88, recommendation=Recommendation.shortlist),
                        TechEval(), None, prof2)
    assert out2.recommendation is Recommendation.maybe

    # clean verdict survives
    clean = Synthesis(overall_fit=80, recommendation=Recommendation.shortlist,
                      strengths=[Finding(claim="strong AWS Python migration experience",
                                         evidence="Led migration of monolith to microservices on AWS using Python")])
    out3 = apply_guards(clean, TechEval(), RiskEval(), prof)
    assert out3.recommendation is Recommendation.shortlist and not out3.requires_human_review
    print("verdict_guard demo: OK")


if __name__ == "__main__":
    demo()

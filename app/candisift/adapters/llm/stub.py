"""Deterministic, offline stand-ins for the LLM persona ports.

Used when no ANTHROPIC_API_KEY is set, so the whole funnel — ingest, queue,
screen, synthesis, UI — runs end-to-end with zero network and zero cost. That
makes the app bootable for a demo and the pipeline testable in CI. Same ports as
the Agno adapters, so swapping them in is one line in the composition root (LSP).

Not a real evaluator — keyword heuristics. Real reasoning lives in agno_personas.
"""
from __future__ import annotations

import re

from app.candisift.domain.models import (
    AuditFinding, CandidateProfile, CoverageAudit, EducationEntry, Finding,
    HREval, JDSpec, Proficiency, Recommendation, SkillItem, Synthesis,
    TechEval, RiskEval, WorkEntry,
)
from app.candisift.domain.services import canon

_KNOWN = [
    "python", "java", "javascript", "typescript", "go", "rust", "c++",
    "react", "nodejs", "django", "fastapi", "flask", "spring",
    "kubernetes", "docker", "terraform", "aws", "gcp", "azure",
    "postgresql", "mysql", "redis", "kafka", "sql", "graphql",
]
_YEARS = re.compile(r"(\d+)\+?\s*(?:years|yrs)", re.I)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?:\+?\d[\d\s().-]{7,}\d)")
_AUTH = re.compile(r"(us citizen|green card|h1b|h-1b|requires sponsorship|authorized to work)", re.I)
_GITHUB_URL = re.compile(r"(?:https?://)?(?:www\.)?github\.com/([a-zA-Z0-9_-]+)", re.I)
_LINKEDIN_URL = re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/in/([a-zA-Z0-9_-]+)", re.I)


def _skills_in(text: str) -> list[str]:
    low = text.lower()
    return [s for s in _KNOWN if re.search(rf"\b{re.escape(s)}\b", low)]


def _line_with(text: str, token: str) -> str:
    for line in text.splitlines():
        if token.lower() in line.lower():
            return line.strip()[:160]
    return token


def _first(pat: re.Pattern, text: str) -> str:
    m = pat.search(text)
    return m.group() if m else ""


class StubProfileExtractor:
    def extract(self, resume_text: str) -> CandidateProfile:
        years = max((int(m) for m in _YEARS.findall(resume_text)), default=0)
        skills = [
            SkillItem(name=s, years=float(years), proficiency=Proficiency.working,
                      evidence=_line_with(resume_text, s))
            for s in _skills_in(resume_text)
        ]
        titles = [ln.strip() for ln in resume_text.splitlines()
                  if re.search(r"(engineer|developer|manager|architect)", ln, re.I)][:3]
        # extract GitHub/LinkedIn URLs
        gh = _GITHUB_URL.search(resume_text)
        github_url = gh.group(0) if gh else ""
        li = _LINKEDIN_URL.search(resume_text)
        linkedin_url = li.group(0) if li else ""
        return CandidateProfile(
            email=_first(_EMAIL, resume_text),
            phone=_first(_PHONE, resume_text),
            work_authorization=_first(_AUTH, resume_text),
            total_years=float(years), skills=skills, titles=titles,
            summary=resume_text.strip()[:200],
            github_url=github_url,
            linkedin_url=linkedin_url,
        )


class StubJobSpecExtractor:
    def extract(self, jd_text: str) -> JDSpec:
        years = max((int(m) for m in _YEARS.findall(jd_text)), default=0)
        skills = _skills_in(jd_text)
        title = next((ln.strip() for ln in jd_text.splitlines() if ln.strip()), "Role")
        return JDSpec(title=title[:120], must_have_skills=skills, min_years=float(years))


class StubTechnicalEvaluator:
    def evaluate(self, profile: CandidateProfile, jd: JDSpec, persona: str = "") -> TechEval:
        from app.candisift.domain.benchmarks import adjacent
        prof = {canon(s.name): s for s in profile.skills}
        must = [canon(s) for s in jd.must_have_skills]
        matched = [
            Finding(claim=f"Has must-have {m}", evidence=prof[m].evidence or m, weight=5)
            for m in must if m in prof
        ]
        # for each missing must-have, credit a transferable adjacent skill if present
        transferable: list[Finding] = []
        missing: list[str] = []
        for m in (x for x in must if x not in prof):
            adj = next((a for a in adjacent(m) if canon(a) in prof), None)
            if adj:
                transferable.append(Finding(
                    claim=f"Transferable to {m} via {adj}",
                    evidence=prof[canon(adj)].evidence or adj, weight=3))
            else:
                missing.append(m)
        # transferable counts as half a match toward depth
        depth = int(round(100 * (len(matched) + 0.5 * len(transferable)) / len(must))) if must else 50
        return TechEval(matched=matched, transferable=transferable, missing_must_haves=missing,
                        depth_score=min(100, depth),
                        summary=f"covered {len(matched)}/{len(must)} must-haves, "
                                f"{len(transferable)} transferable")


class StubRiskEvaluator:
    def evaluate(self, profile: CandidateProfile, persona: str = "") -> RiskEval:
        flags: list[Finding] = []
        score = 0
        if profile.concurrent_fulltime:
            flags.append(Finding(claim="Overlapping concurrent full-time roles",
                                  evidence="concurrent_fulltime=true", weight=5))
            score += 60
        if profile.employment_gaps:
            flags.append(Finding(claim="Unexplained employment gap(s)",
                                  evidence="; ".join(profile.employment_gaps)[:160], weight=3))
            score += 20
        return RiskEval(flags=flags, risk_score=min(score, 100))


class StubHREvaluator:
    def evaluate(self, profile: CandidateProfile, jd: JDSpec, persona: str = "") -> HREval:
        strengths: list[Finding] = []
        concerns: list[Finding] = []
        score = 55
        if profile.titles:
            strengths.append(Finding(claim="Relevant role history",
                                     evidence="; ".join(profile.titles)[:160], weight=3))
            score += 15
        if profile.total_years >= 5:
            strengths.append(Finding(claim="Seasoned (5y+) — likely autonomous",
                                     evidence=f"{profile.total_years:.0f} years total", weight=3))
            score += 10
        if profile.summary:
            strengths.append(Finding(claim="Communicates background clearly",
                                     evidence=profile.summary[:160], weight=2))
            score += 5
        if profile.employment_gaps:
            concerns.append(Finding(claim="Career gap(s) to discuss (context, not a flag)",
                                    evidence="; ".join(profile.employment_gaps)[:160], weight=2))
            score -= 10
        return HREval(strengths=strengths, concerns=concerns,
                      people_score=max(0, min(100, score)),
                      summary=f"people-fit {max(0, min(100, score))} from {len(profile.titles)} role(s), "
                              f"{profile.total_years:.0f}y")


class StubSynthesizer:
    def synthesize(self, jd: JDSpec, tech: TechEval, risk: RiskEval,
                   hr: HREval | None = None, persona: str = "") -> Synthesis:
        fit = max(0, min(100, tech.depth_score - risk.risk_score // 2))
        rec = (Recommendation.shortlist if fit >= 70
               else Recommendation.maybe if fit >= 45 else Recommendation.reject)
        weaknesses = [Finding(claim=f"Missing must-have {m}", evidence="not found in resume", weight=4)
                      for m in tech.missing_must_haves]
        return Synthesis(
            overall_fit=fit, recommendation=rec,
            strengths=tech.matched, weaknesses=weaknesses + risk.flags,
            rationale=f"depth {tech.depth_score}, risk {risk.risk_score} -> fit {fit}",
        )


class StubCoverageAuditor:
    """Deterministic stand-in for the §5 QA auditor. Reuses the domain grounding +
    bias checks so the offline pipeline exercises the same hold-for-review logic
    the real LLM auditor would, with no network."""

    def audit(self, jd: JDSpec, profile: CandidateProfile, tech: TechEval | None,
              risk: RiskEval | None, hr: HREval | None, synthesis: Synthesis) -> CoverageAudit:
        from app.candisift.domain.guardrails import scan_bias_proxies
        from app.candisift.domain.verdict_guard import profile_corpus, ungrounded_claims

        failures: list[AuditFinding] = []
        corpus = profile_corpus(profile)
        ungrounded = ungrounded_claims(synthesis.strengths, corpus) if synthesis else []
        if tech is not None:
            ungrounded = ungrounded + ungrounded_claims(tech.matched, corpus)
        if ungrounded:
            failures.append(AuditFinding(check="GROUNDING",
                                         detail=f"{len(ungrounded)} claim(s) lack profile evidence"))

        # coverage: a must-have is "addressed" if it shows up anywhere the tech eval
        # spoke to it (matched, transferable, or explicitly listed missing).
        skipped: list[str] = []
        if tech is not None:
            covered = " ".join(
                [f.claim + " " + f.evidence for f in (tech.matched + tech.transferable)]
                + tech.missing_must_haves
            ).lower()
            skipped = [m for m in jd.must_have_skills
                       if canon(m) not in covered and m.lower() not in covered]
            if skipped:
                failures.append(AuditFinding(check="COVERAGE",
                                             detail=f"requirements not addressed: {skipped}"))

        bias = scan_bias_proxies(synthesis.rationale if synthesis else "")
        if bias:
            failures.append(AuditFinding(check="BIAS", detail=f"proxy terms: {bias}"))

        safe = not failures
        return CoverageAudit(
            overall="pass" if safe else "fail", failures=failures,
            skipped_requirements=skipped, ungrounded_claims=ungrounded,
            bias_or_proxy_hits=bias, arithmetic_ok=True,
            safe_to_surface_to_recruiter=safe, notes="stub auditor (deterministic checks)",
        )


class StubResumeOptimizer:
    """Keyword-gap resume optimizer — no LLM, keyword heuristics only.
    Appends missing must-have keywords to a skills line so the pipeline works
    end-to-end with no API key. The real optimizer lives in agno_personas."""

    def optimize(self, resume_text: str, jd: "JDSpec", job_title: str = ""):
        from app.candisift.domain.models import KeywordGapItem, OptimizerLLMOut, ResumeChangeItem
        text_lower = resume_text.lower()
        gaps: list[KeywordGapItem] = []
        missing_kws: list[str] = []
        for kw in jd.must_have_skills:
            if re.search(rf"\b{re.escape(kw.lower())}\b", text_lower):
                gaps.append(KeywordGapItem(keyword=kw, status="present"))
            else:
                missing_kws.append(kw)
                gaps.append(KeywordGapItem(
                    keyword=kw, status="missing",
                    suggestion=f"Add '{kw}' to your skills or a relevant bullet point",
                ))
        optimized = resume_text
        changes: list[ResumeChangeItem] = []
        if missing_kws:
            kw_line = "Relevant skills: " + ", ".join(missing_kws)
            optimized = resume_text.rstrip() + f"\n\nSKILLS\n{kw_line}"
            changes.append(ResumeChangeItem(
                section="Skills",
                original="(end of resume)",
                improved=kw_line,
                reason=f"Added {len(missing_kws)} missing must-have keyword(s) from JD",
            ))
        return OptimizerLLMOut(
            optimized_resume=optimized, keyword_gaps=gaps, changes=changes,
        )


class StubCoverLetterWriter:
    def write(self, resume_text: str, jd: "JDSpec", job_title: str = "", tone: str = "professional") -> str:
        skills = ", ".join(jd.must_have_skills[:5])
        return (
            f"Dear Hiring Manager,\n\n"
            f"I am excited to apply for the {job_title or jd.title} role. "
            f"My experience aligns strongly with your requirements in {skills}.\n\n"
            f"[Stub output — configure an LLM provider for a real cover letter.]\n\n"
            f"Best regards,\n[Candidate Name]"
        )


class StubGitHubSelector:
    def select(self, projects_json: str) -> list[dict]:
        import json
        try:
            projects = json.loads(projects_json)
        except (json.JSONDecodeError, TypeError):
            return []
        # Return top 7 by sorting on author_commit_count
        projects.sort(key=lambda x: x.get("author_commit_count", 0), reverse=True)
        return projects[:7]


class StubLinkedInSelector:
    """Offline LinkedIn digest: regex titles/skills only. The real adapter's
    deterministic fallback (from the parsed profile) is the production path; this
    keeps the stub provider LSP-complete for no-key/test runs."""
    def select(self, resume_text: str) -> dict:
        titles = [ln.strip() for ln in resume_text.splitlines()
                  if re.search(r"(engineer|developer|manager|architect)", ln, re.I)][:3]
        skills = _skills_in(resume_text)
        if not titles and not skills:
            return {}
        return {
            "headline": titles[0] if titles else "",
            "positions": [{"title": t, "company": "", "duration": "", "highlights": []}
                          for t in titles],
            "skills": skills[:15],
        }


class StubLLMProvider:
    """Implements ports.LLMProvider with the deterministic stubs. Model id is
    ignored — there is no model — so every requested model maps to the same
    offline behavior (handy for tests and no-key demos)."""

    def __init__(self) -> None:
        self._profile = StubProfileExtractor()
        self._jd = StubJobSpecExtractor()
        self._tech = StubTechnicalEvaluator()
        self._risk = StubRiskEvaluator()
        self._hr = StubHREvaluator()
        self._synth = StubSynthesizer()
        self._coverage = StubCoverageAuditor()
        self._optimizer = StubResumeOptimizer()
        self._coverletter = StubCoverLetterWriter()
        self._github = StubGitHubSelector()
        self._linkedin = StubLinkedInSelector()

    def profile_extractor(self, model: str):
        return self._profile

    def jd_extractor(self, model: str):
        return self._jd

    def technical(self, model: str):
        return self._tech

    def risk(self, model: str):
        return self._risk

    def hr(self, model: str):
        return self._hr

    def synthesizer(self, model: str):
        return self._synth

    def coverage_auditor(self, model: str):
        return self._coverage

    def resume_optimizer(self, model: str):
        return self._optimizer

    def cover_letter_writer(self, model: str):
        return self._coverletter

    def github_selector(self, model: str):
        return self._github

    def linkedin_selector(self, model: str):
        return self._linkedin

"""Pure domain rules — deterministic, no I/O, no framework. Unit-testable in isolation.

These are business invariants, not infrastructure: who is hard-filtered out, what
counts as the same candidate, what identity data must never reach an evaluator.
The ranking *algorithm* is an adapter (swappable for embeddings); the ranking
*inputs/rules* that are business policy live here.
"""
from __future__ import annotations

import hashlib
import re

from .models import CandidateProfile, JDSpec


# ---- skills ontology (stub) ----------------------------------------------

# ponytail: hand-map, not a real taxonomy. The proper upgrade is skillNER (spaCy)
# for extraction + ESCO/O*NET as the canonical skill graph — drop them in behind
# this function (canon stays the call site). Canonicalization is what lets
# "ReactJS" match "react" so the ranker and keyword checks aren't fooled by spelling.
_CANON = {
    "reactjs": "react", "react.js": "react", "reactnative": "react native",
    "js": "javascript", "ts": "typescript", "typescript": "typescript",
    "node": "nodejs", "node.js": "nodejs", "nodejs": "nodejs",
    "k8s": "kubernetes", "kube": "kubernetes",
    "postgres": "postgresql", "psql": "postgresql",
    "py": "python", "golang": "go", "gcp": "google cloud", "aws": "aws",
    "tf": "terraform", "k8": "kubernetes", "ml": "machine learning",
    "nlp": "natural language processing", "ci/cd": "cicd", "ci": "cicd",
    "rest": "rest api", "restful": "rest api", "gql": "graphql",
}


def canon(skill: str) -> str:
    s = skill.strip().lower()
    return _CANON.get(s, s)


# ---- identity / dedup -----------------------------------------------------

def dedup_key(profile: CandidateProfile) -> str:
    """Same person re-applies with tweaked resumes. Key on normalized
    name+email+phone so we don't re-screen or double-count.

    Returns "" when the profile has NO identity at all (name+email+phone all blank
    — e.g. extraction returned nothing). An empty identity must NOT produce a shared
    hash: that would collapse every identity-less applicant onto the first one and
    silently discard the rest. The caller treats "" as non-dedupable (exact re-uploads
    are still caught upstream by content_sha256)."""
    parts = [
        re.sub(r"\s+", "", v.lower())
        for v in (profile.name, profile.email, re.sub(r"\D", "", profile.phone))
    ]
    if not any(parts):
        return ""
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


# ---- compliance: strip protected-class proxies before screening ----------

# free-text identity that survives the scalar blanking: an email, a personal URL
# (LinkedIn/GitHub/portfolio), or the candidate's own name quoted inside a summary
# or an evidence snippet. We do NOT strip bare digit runs/years here — those are the
# metrics ("reduced cost 1.5M", "12 years") the evaluators must see; blanking them
# would gut the signal. grad_year (the age proxy) is already zeroed as a scalar.
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
# URLs and personal profile links — including bare domains with a path
# ("linkedin.com/in/jane") that carry no http/www prefix. A domain followed by a
# "/path" is a link, not prose, so redacting it is safe; "node.js" (no slash) is left
# alone. Known social/profile hosts are caught even without a path.
_URL_RE = re.compile(
    r"\b(?:https?://|www\.)\S+"
    r"|\b(?:linkedin|github|gitlab|bitbucket|twitter|x|behance|dribbble|medium|gitlab)\.com\S*"
    r"|\b[\w-]+\.(?:com|io|dev|me|net|org|co|ai)/\S+",
    re.IGNORECASE,
)
_REDACTED = "[redacted]"


def _redact_free_text(text: str, identifiers: list[str]) -> str:
    """Redact the candidate's own name/email/phone (passed in `identifiers`) and any
    email/personal-URL pattern from one free-text field, case-insensitively."""
    if not text:
        return text
    out = text
    # longest first so "john smith" is redacted before "john" leaves a dangling "smith"
    for ident in sorted(identifiers, key=len, reverse=True):
        if len(ident) >= 3:                      # avoid redacting 1-2 char noise
            out = re.sub(re.escape(ident), _REDACTED, out, flags=re.IGNORECASE)
    out = _EMAIL_RE.sub(_REDACTED, out)
    out = _URL_RE.sub(_REDACTED, out)
    return out


def _scrub_linkedin(digest: dict, idents: list[str]) -> dict:
    """Redact the candidate's identity from the resume-derived LinkedIn digest's
    free-text fields before it reaches an evaluator — same boundary as work_entries.
    Skills are canonical tech names, not identity, so they pass through unchanged."""
    if not digest:
        return digest
    positions = [
        {
            **pos,
            "title": _redact_free_text(pos.get("title", ""), idents),
            "company": _redact_free_text(pos.get("company", ""), idents),
            "highlights": [_redact_free_text(h, idents) for h in pos.get("highlights", [])],
        }
        for pos in digest.get("positions", [])
    ]
    return {**digest,
            "headline": _redact_free_text(digest.get("headline", ""), idents),
            "positions": positions}


def strip_pii(profile: CandidateProfile) -> CandidateProfile:
    """Blank name/email/phone/location/grad-year (age proxy) AND redact the same
    identity from every free-text field an evaluator sees — summary, titles,
    employment-gap notes, and each skill's verbatim evidence quote — so the screen
    is on skills and experience, not identity. Without the free-text pass a quoted
    "Jane Doe led the team" snippet leaks the name straight to the persona LLMs.
    Original is retained for the recruiter UI shown only after a human decision."""
    # the candidate's own identifiers to scrub out of free text (name parts too, so
    # "Jane" and "Doe" are caught individually inside prose)
    idents = [profile.name, profile.email, profile.phone, *profile.name.split()]
    idents = [i.strip() for i in idents if i and i.strip()]

    scrubbed_skills = [
        s.model_copy(update={"evidence": _redact_free_text(s.evidence, idents)})
        for s in profile.skills
    ]
    scrubbed_work = [
        w.model_copy(update={
            "highlights": [_redact_free_text(h, idents) for h in w.highlights],
        })
        for w in profile.work_entries
    ]
    scrubbed_edu = [
        e.model_copy(update={
            "institution": _redact_free_text(e.institution, idents),
        })
        for e in profile.education
    ]
    return profile.model_copy(update={
        "name": "", "email": "", "phone": "", "location": "", "grad_year": 0,
        "linkedin_url": "", "portfolio_url": "",
        "linkedin_profile": _scrub_linkedin(profile.linkedin_profile, idents),
        "summary": _redact_free_text(profile.summary, idents),
        "titles": [_redact_free_text(t, idents) for t in profile.titles],
        "employment_gaps": [_redact_free_text(g, idents) for g in profile.employment_gaps],
        "skills": scrubbed_skills,
        "work_entries": scrubbed_work,
        "education": scrubbed_edu,
    })


# ---- deterministic experience validator ----------------------------------

_DATE_FORMATS = [
    r"(\d{4})-(\d{2})",            # 2020-01
    r"(\d{4})/(\d{2})",            # 2020/01
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+(\d{4})",  # Jan 2020
    r"(\d{4})",                     # bare year
]

_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _parse_date(date_str: str) -> tuple[int, int] | None:
    """Parse a date string into (year, month). Returns None if unparseable."""
    import datetime
    if not date_str:
        return None
    s = date_str.strip().lower()
    if s in ("present", "current", "now", ""):
        today = datetime.date.today()
        return (today.year, today.month)

    # try YYYY-MM or YYYY/MM
    m = re.match(r"(\d{4})[-/](\d{1,2})", s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    # try "Jan 2020" / "January 2020"
    m = re.match(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+(\d{4})", s)
    if m:
        return (int(m.group(2)), _MONTH_MAP[m.group(1)[:3]])
    # try bare year
    m = re.match(r"^(\d{4})$", s)
    if m:
        return (int(m.group(1)), 1)
    return None


def _months_between(start: tuple[int, int], end: tuple[int, int]) -> int:
    return (end[0] - start[0]) * 12 + (end[1] - start[1])


def validate_experience(profile: CandidateProfile) -> CandidateProfile:
    """Deterministic post-extraction validator. Cross-checks total_years against
    parseable work_entry dates and detects concurrent full-time roles. If work_entries
    have parseable dates, the computed total replaces the LLM-provided total_years
    (which may hallucinate). Also auto-detects concurrent_fulltime and gaps."""
    entries = profile.work_entries
    if not entries:
        return profile

    # parse all date ranges
    spans: list[tuple[int, int]] = []  # (start_month_idx, end_month_idx)
    for e in entries:
        start = _parse_date(e.start_date)
        end = _parse_date(e.end_date) if e.end_date else _parse_date("present")
        if start and end:
            s = start[0] * 12 + start[1]
            n = end[0] * 12 + end[1]
            if n >= s:
                spans.append((s, n))

    if not spans:
        return profile

    # sort and merge overlapping spans (union)
    spans.sort()
    merged: list[tuple[int, int]] = [spans[0]]
    for s, n in spans[1:]:
        if s <= merged[-1][1]:
            # overlap → extend
            merged[-1] = (merged[-1][0], max(merged[-1][1], n))
        else:
            merged.append((s, n))

    # total non-overlapping months → years (inclusive of both start and end months)
    total_months = sum(n - s + 1 for s, n in merged)
    computed_years = round(total_months / 12, 1)

    # detect concurrent full-time: any two original spans where one starts before the other ends
    concurrent = False
    sorted_spans = sorted(spans)
    for i in range(len(sorted_spans) - 1):
        if sorted_spans[i][1] > sorted_spans[i + 1][0] + 1:  # +1 for 1-month tolerance
            concurrent = True
            break

    # detect employment gaps (>6 months between merged spans)
    gaps = profile.employment_gaps[:]
    for i in range(len(merged) - 1):
        gap_months = merged[i + 1][0] - merged[i][1]
        if gap_months > 6:
            gap_years = round(gap_months / 12, 1)
            gaps.append(f"~{gap_years}y gap detected between employment entries")

    updates: dict = {"total_years": computed_years}
    if concurrent and not profile.concurrent_fulltime:
        updates["concurrent_fulltime"] = True
    if gaps and not profile.employment_gaps:
        updates["employment_gaps"] = gaps

    return profile.model_copy(update=updates)


# ---- hard filters (deterministic, ~free) ---------------------------------

def hard_filter(profile: CandidateProfile, jd: JDSpec) -> tuple[bool, list[str]]:
    """Stage that removes ~50-70% at zero cost. Returns (passed, rejection reasons).

    Required gates FAIL CLOSED: if the JD makes work-auth or an on-site location
    mandatory and the profile doesn't establish it (extraction empty), the candidate
    is rejected rather than silently waved through — you cannot assume an unstated
    legal work authorization. (Soft signals — years, certs — already failed closed.)"""
    reasons: list[str] = []

    if jd.required_work_auth:
        if not profile.work_authorization:
            reasons.append("work authorization not stated; required " + str(jd.required_work_auth))
        elif not any(a.lower() in profile.work_authorization.lower() for a in jd.required_work_auth):
            reasons.append(f"work auth '{profile.work_authorization}' not in {jd.required_work_auth}")

    if jd.locations and not (jd.remote_ok and profile.remote_ok):
        if not profile.location:
            reasons.append("location not stated; on-site role requires one of " + str(jd.locations))
        elif not any(loc.lower() in profile.location.lower() for loc in jd.locations):
            reasons.append(f"location '{profile.location}' not in {jd.locations}")

    if profile.total_years < jd.min_years:
        reasons.append(f"{profile.total_years}y experience < required {jd.min_years}y")

    have_certs = {c.lower() for c in profile.certifications}
    missing = [c for c in jd.required_certs if c.lower() not in have_certs]
    if missing:
        reasons.append(f"missing required certs: {missing}")

    # JD knockouts: explicit disqualifying keywords. The deterministic reading is
    # "auto-reject if this term is present in the candidate's skills/titles/summary"
    # (e.g. a contract-to-hire role knocking out "consultant only"). Matched here for
    # free rather than spending an LLM call to rediscover a stated hard rule.
    if jd.knockouts:
        haystack = " ".join(
            [profile.summary, *profile.titles, *(s.name for s in profile.skills)]
        ).lower()
        hit = [k for k in jd.knockouts if k.strip() and k.lower() in haystack]
        if hit:
            reasons.append(f"matched JD knockout(s): {hit}")

    return (not reasons), reasons

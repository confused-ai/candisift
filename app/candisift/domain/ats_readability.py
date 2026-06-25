"""ATS readability score — deterministic, no LLM. Inspired by open-resume's
"is this resume ATS-parseable?" check and ats-screener's per-section scoring.

Operates on the *parsed* profile: if the parser couldn't recover contact info,
skills, titles, or years, the resume is likely poorly formatted / scanned / image
based and a real ATS would mis-handle it too. Plus keyword coverage against the
role's must-haves. Cheap, resilient (no external deps), explainable.

ponytail: a fuller version scores the raw layout (columns, tables, fonts, sections)
like ats-screener does per platform — add a raw-text/layout signal alongside this
when you wire a richer parser.
"""
from __future__ import annotations

from .models import CandidateProfile, JDSpec
from .services import canon


def score(profile: CandidateProfile, jd: JDSpec) -> dict:
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str, weight: int) -> int:
        checks.append({"name": name, "ok": ok, "detail": detail})
        return weight if ok else 0

    earned = 0
    total = 0

    total += 15
    earned += add("contact info parsed", bool(profile.email or profile.phone),
                  profile.email or profile.phone or "none found", 15)
    total += 20
    earned += add("skills extracted", len(profile.skills) >= 3,
                  f"{len(profile.skills)} skills", 20)
    total += 15
    earned += add("job titles parsed", bool(profile.titles),
                  ", ".join(profile.titles) or "none", 15)
    total += 15
    earned += add("experience years stated", profile.total_years > 0,
                  f"{profile.total_years}y", 15)
    total += 10
    earned += add("summary present", len(profile.summary) >= 40,
                  f"{len(profile.summary)} chars", 10)

    # keyword coverage vs must-haves (25 pts)
    must = {canon(s) for s in jd.must_have_skills}
    have = {canon(s.name) for s in profile.skills}
    cov = len(must & have) / len(must) if must else 1.0
    total += 25
    earned += int(round(25 * cov))
    checks.append({"name": "must-have keyword coverage", "ok": cov >= 0.5,
                   "detail": f"{len(must & have)}/{len(must)} ({cov:.0%})"})

    return {"score": int(round(100 * earned / total)) if total else 0, "checks": checks}

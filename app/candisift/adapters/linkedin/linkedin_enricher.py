"""LinkedIn enricher — distils a LinkedIn-style professional digest from the resume.

Mirrors the GitHub enricher's shape (LLM selector + deterministic fallback) but
the data source is the resume text ONLY: LinkedIn exposes no free public API and
blocks scraping, so there is nothing to fetch. The selector refines what the
resume already states into a clean public-profile view; if the LLM path is
unavailable the deterministic fallback assembles the same view from the parsed
profile, so enrichment always returns useful, PII-free professional content.
"""
from __future__ import annotations

import logging

from app.candisift.domain import ports
from app.candisift.domain.models import CandidateProfile

log = logging.getLogger("candisift.linkedin")


def _duration(entry) -> str:
    start = (entry.start_date or "").strip()
    end = (entry.end_date or "").strip() or "Present"
    if not start:
        return end if end != "Present" else ""
    return f"{start} – {end}"


class LinkedInEnricherAdapter:
    def __init__(self, llm_provider: ports.LLMProvider, default_model: str) -> None:
        self.llm = llm_provider
        self.model = default_model

    def _fallback(self, profile: CandidateProfile) -> dict:
        """Deterministic digest straight from the parsed profile — no LLM, no PII."""
        positions = [
            {
                "title": w.title or "",
                "company": w.company or "",
                "duration": _duration(w),
                "highlights": (w.highlights or [])[:3],
            }
            for w in profile.work_entries[:7]
            if (w.title or w.company)
        ]
        if not positions and not profile.skills:
            return {}
        return {
            "headline": profile.titles[0] if profile.titles else "",
            "positions": positions,
            "skills": [s.name for s in profile.skills][:15],
        }

    def enrich(self, resume_text: str, profile: CandidateProfile) -> dict:
        # LLM selector path (parity with GitHub). Falls through to the deterministic
        # digest on any failure — including the wrapped-provider AttributeError, since
        # the tracing/resilient/throttle decorators don't forward *_selector (same as
        # GitHub). The fallback is the real workhorse and is always PII-free.
        try:
            selector = self.llm.linkedin_selector(self.model)
            digest = selector.select(resume_text)
            if digest and (digest.get("positions") or digest.get("skills")):
                return digest
        except Exception as e:  # noqa: BLE001
            log.warning(f"LinkedIn LLM selector unavailable, using deterministic digest: {e}")
        return self._fallback(profile)

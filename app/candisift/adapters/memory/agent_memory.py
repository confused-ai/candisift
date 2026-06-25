"""SqlAgentMemory — persistent memory the agents consult to ground judgments.

Stores prior agent evaluations and recruiter decisions in the `memory` table and
exposes two retrieval functions used as agent tools:

  - recall_similar_candidates(skills): token-overlap search over the candidate
    table — "have we seen people like this, and how did they screen?"
  - recall_recruiter_feedback(job): recent accept/reject decisions with rationale
    — lets the synthesizer learn the team's bar.

Retrieval is keyword/token-set (no embeddings/vector DB) — deterministic, no new
deps. Upgrade path: swap the scorer for an embedding index behind this same port.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from app.candisift.adapters.persistence.db import CandidateRow, MemoryRow, ResultRow
from app.candisift.domain.services import canon

log = logging.getLogger("candisift.memory")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _tokens(skills: list[str]) -> set[str]:
    out: set[str] = set()
    for s in skills or []:
        out.update(canon(str(s)).split())
    return {t for t in out if t}


def _overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)   # Jaccard


class SqlAgentMemory:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def remember(self, *, candidate_id: str, job_id: str, kind: str,
                 content: str, data: dict | None = None) -> None:
        try:
            with Session(self._engine) as s:
                s.add(MemoryRow(candidate_id=candidate_id, job_id=job_id, kind=kind,
                                content=content[:2000], data_json=data or {}, ts=_now()))
                s.commit()
        except Exception:
            log.exception("memory.remember failed")

    def recall_similar_candidates(self, skills: list[str], limit: int = 5) -> list[dict]:
        """Past candidates with overlapping skills + how they scored (from results)."""
        want = _tokens(skills)
        if not want:
            return []
        with Session(self._engine) as s:
            cands = s.exec(select(CandidateRow)).all()
            scored = []
            for c in cands:
                cskills = [sk.get("name", "") for sk in (c.profile_json.get("skills") or [])]
                sim = _overlap(want, _tokens(cskills))
                if sim > 0:
                    scored.append((sim, c))
            scored.sort(key=lambda t: t[0], reverse=True)
            out = []
            for sim, c in scored[:limit]:
                res = s.exec(select(ResultRow).where(ResultRow.candidate_id == c.id)).first()
                out.append({
                    "candidate_id": c.id,
                    "similarity": round(sim, 3),
                    "skills": [sk.get("name", "") for sk in (c.profile_json.get("skills") or [])][:12],
                    "prior_decision": res.decision if res else "none",
                    "prior_fit": (res.synthesis_json or {}).get("overall_fit") if res and res.synthesis_json else None,
                })
            return out

    def recall_recruiter_feedback(self, job_id: str = "", limit: int = 10) -> list[dict]:
        """Recent human accept/reject decisions (optionally for one job) + rationale."""
        with Session(self._engine) as s:
            stmt = select(ResultRow).where(ResultRow.decision.in_(("accepted", "rejected")))
            if job_id:
                stmt = stmt.where(ResultRow.job_id == job_id)
            rows = s.exec(stmt.order_by(ResultRow.created_at.desc()).limit(limit)).all()
            return [{
                "candidate_id": r.candidate_id,
                "job_id": r.job_id,
                "decision": r.decision,
                "fit": (r.synthesis_json or {}).get("overall_fit") if r.synthesis_json else None,
                "rationale": (r.synthesis_json or {}).get("rationale", "")[:240] if r.synthesis_json else "",
            } for r in rows]

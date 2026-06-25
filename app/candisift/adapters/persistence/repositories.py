"""Repository + audit adapters. Implement the persistence ports; map row <-> domain.

These are the only place that knows SQL exists. The application sees the port
interfaces only.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from app.candisift.domain.models import (
    Candidate, CandidateProfile, CoverageAudit, HREval, Job, JDSpec, RolePersonas,
    ScreeningResult, Synthesis, TechEval, RiskEval, Decision,
)
from .db import AuditRow, CandidateRow, JobRow, ResultRow, is_unique_violation

log = logging.getLogger("candisift.audit")


# ---- mappers --------------------------------------------------------------

def _to_candidate(row: CandidateRow) -> Candidate:
    return Candidate(
        id=row.id, dedup_key=row.dedup_key, content_sha256=row.content_sha256 or "",
        source_filename=row.source_filename,
        profile=CandidateProfile.model_validate(row.profile_json),
        near_duplicate_of=row.near_duplicate_of, duplicate_similarity=row.duplicate_similarity,
        created_at=row.created_at,
    )


def _to_job(row: JobRow) -> Job:
    return Job(id=row.id, title=row.title, raw_text=row.raw_text,
               spec=JDSpec.model_validate(row.spec_json),
               personas=RolePersonas.model_validate(row.personas_json) if row.personas_json else None,
               created_at=row.created_at)


def _to_result(row: ResultRow) -> ScreeningResult:
    return ScreeningResult(
        id=row.id, job_id=row.job_id, candidate_id=row.candidate_id,
        passed_hard_filters=row.passed_hard_filters, filter_reasons=row.filter_reasons,
        semantic_score=row.semantic_score,
        tech=TechEval.model_validate(row.tech_json) if row.tech_json else None,
        risk=RiskEval.model_validate(row.risk_json) if row.risk_json else None,
        hr=HREval.model_validate(row.hr_json) if getattr(row, "hr_json", None) else None,
        synthesis=Synthesis.model_validate(row.synthesis_json) if row.synthesis_json else None,
        bias_flags=getattr(row, "bias_flags", None) or [],
        requires_human_review=bool(getattr(row, "requires_human_review", False)),
        review_reasons=getattr(row, "review_reasons", None) or [],
        ungrounded_claims=getattr(row, "ungrounded_claims", None) or [],
        coverage=CoverageAudit.model_validate(row.coverage_json)
        if getattr(row, "coverage_json", None) else None,
        decision=Decision(row.decision), models_fingerprint=row.models_fingerprint or "",
        created_at=row.created_at,
    )


# ---- repositories ---------------------------------------------------------

class SqlCandidateRepository:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def add(self, candidate: Candidate) -> None:
        with Session(self._engine) as s:
            s.add(CandidateRow(
                id=candidate.id, dedup_key=candidate.dedup_key,
                content_sha256=candidate.content_sha256,
                source_filename=candidate.source_filename,
                profile_json=candidate.profile.model_dump(mode="json"),
                near_duplicate_of=candidate.near_duplicate_of,
                duplicate_similarity=candidate.duplicate_similarity,
                created_at=candidate.created_at,
            ))
            try:
                s.commit()
            except (IntegrityError, ValueError) as e:
                # a concurrent ingest already inserted this dedup_key/content hash
                # (the partial unique index fired). The other row wins; this is a
                # benign dedup race, not an error — keep going. (libSQL raises a raw
                # ValueError for the violation, pysqlite an IntegrityError.)
                s.rollback()
                if not is_unique_violation(e):
                    raise
                log.info("candidate add raced on a unique key; existing row kept (%s)",
                         candidate.dedup_key or candidate.content_sha256)

    def get(self, candidate_id: str) -> Candidate | None:
        with Session(self._engine) as s:
            row = s.get(CandidateRow, candidate_id)
            return _to_candidate(row) if row else None

    def find_by_dedup_key(self, dedup_key: str) -> Candidate | None:
        with Session(self._engine) as s:
            row = s.exec(select(CandidateRow).where(CandidateRow.dedup_key == dedup_key)).first()
            return _to_candidate(row) if row else None

    def find_by_content_hash(self, content_sha256: str) -> Candidate | None:
        if not content_sha256:
            return None
        with Session(self._engine) as s:
            row = s.exec(
                select(CandidateRow).where(CandidateRow.content_sha256 == content_sha256)
            ).first()
            return _to_candidate(row) if row else None

    def list_all(self) -> list[Candidate]:
        with Session(self._engine) as s:
            return [_to_candidate(r) for r in s.exec(select(CandidateRow)).all()]


class SqlJobRepository:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def add(self, job: Job) -> None:
        with Session(self._engine) as s:
            s.add(JobRow(id=job.id, title=job.title, raw_text=job.raw_text,
                         spec_json=job.spec.model_dump(mode="json"),
                         personas_json=job.personas.model_dump(mode="json") if job.personas else None,
                         created_at=job.created_at))
            s.commit()

    def get(self, job_id: str) -> Job | None:
        with Session(self._engine) as s:
            row = s.get(JobRow, job_id)
            return _to_job(row) if row else None

    def list_all(self) -> list[Job]:
        with Session(self._engine) as s:
            rows = s.exec(select(JobRow).order_by(JobRow.created_at.desc())).all()
            return [_to_job(r) for r in rows]


class SqlResultRepository:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def upsert(self, result: ScreeningResult) -> None:
        with Session(self._engine) as s:
            row = s.get(ResultRow, result.id)
            data = dict(
                job_id=result.job_id, candidate_id=result.candidate_id,
                passed_hard_filters=result.passed_hard_filters,
                filter_reasons=result.filter_reasons, semantic_score=result.semantic_score,
                tech_json=result.tech.model_dump(mode="json") if result.tech else None,
                risk_json=result.risk.model_dump(mode="json") if result.risk else None,
                hr_json=result.hr.model_dump(mode="json") if result.hr else None,
                synthesis_json=result.synthesis.model_dump(mode="json") if result.synthesis else None,
                bias_flags=result.bias_flags,
                requires_human_review=result.requires_human_review,
                review_reasons=result.review_reasons,
                ungrounded_claims=result.ungrounded_claims,
                coverage_json=result.coverage.model_dump(mode="json") if result.coverage else None,
                models_fingerprint=result.models_fingerprint,
                created_at=result.created_at,
            )
            if row is None:
                s.add(ResultRow(id=result.id, decision=result.decision.value, **data))
            else:
                for k, v in data.items():  # preserve a human decision across re-screens
                    setattr(row, k, v)
                s.add(row)
            s.commit()

    def get(self, result_id: str) -> ScreeningResult | None:
        with Session(self._engine) as s:
            row = s.get(ResultRow, result_id)
            return _to_result(row) if row else None

    def list_for_job(self, job_id: str) -> list[ScreeningResult]:
        with Session(self._engine) as s:
            rows = s.exec(select(ResultRow).where(ResultRow.job_id == job_id)).all()
        results = [_to_result(r) for r in rows]
        # latest screened first — recruiter sees newest completions on top (matches
        # the live job page). created_at is (re)stamped on every screen/upsert.
        results.sort(key=lambda r: r.created_at, reverse=True)
        return results

    def list_all(self, limit: int = 5000) -> list[ScreeningResult]:
        """Every result across jobs — powers the analytics dashboard funnel/decisions."""
        with Session(self._engine) as s:
            rows = s.exec(select(ResultRow).limit(limit)).all()
        return [_to_result(r) for r in rows]

    def set_decision(self, result_id: str, decision: str) -> None:
        with Session(self._engine) as s:
            row = s.get(ResultRow, result_id)
            if row is None:
                raise LookupError(f"result {result_id}")
            row.decision = Decision(decision).value
            s.add(row)
            s.commit()


class SqlAuditLog:
    """Append-only audit trail: every score, rationale, decision is reproducible."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def record(self, event: str, **fields) -> None:
        # Tag with job_id so the per-job activity log can filter. Screening events
        # carry result_id (= "<job_id>.<candidate_id>"), so job_id derives for free.
        job_id = fields.get("job_id") or str(fields.get("result_id", "")).split(".")[0]
        with Session(self._engine) as s:
            s.add(AuditRow(ts=datetime.now(timezone.utc), event=event,
                           job_id=job_id, data_json=fields))
            s.commit()
        log.info("audit %s %s", event, fields)

    def events_for_job(self, job_id: str, limit: int = 200) -> list[dict]:
        """Newest-first activity log for one job — powers the live log on the job page."""
        with Session(self._engine) as s:
            rows = s.exec(
                select(AuditRow).where(AuditRow.job_id == job_id)
                .order_by(AuditRow.ts.desc()).limit(limit)
            ).all()
        return [{"ts": r.ts, "event": r.event, "data": r.data_json or {}} for r in rows]

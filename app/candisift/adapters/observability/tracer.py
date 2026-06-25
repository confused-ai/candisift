"""SqlTracer — persists agent-run sessions (traces) and their spans.

A "session" is one screening (or ingest). `start_run` opens it and binds the
trace id to the current thread via a contextvar; `record_span` appends each
agent / tool / LLM call; `end_run` closes it and writes the rollups. The active
trace is per-thread, so the worker's parallel tasks each trace independently —
to keep spans from sub-threads (the parallel tech/risk personas) attached to the
parent run, the caller propagates the context (contextvars.copy_context).

Tracing must never break a screening: every method swallows its own errors.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy import func
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlmodel import Session, select

from app.candisift.domain.trace import RunTrace, Span
from app.candisift.adapters.persistence.db import SpanRow, TraceRow

log = logging.getLogger("candisift.tracer")


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class _RunState:
    trace_id: str
    t0: float
    ordinal: int = 0
    spans: int = 0
    cost: float = 0.0
    dropped: int = 0          # spans whose DB write failed -> NOT counted in rollups
    lock: threading.Lock = field(default_factory=threading.Lock)


# the active run for the current thread/context
_CURRENT: ContextVar[_RunState | None] = ContextVar("ats_trace_run", default=None)


class SqlTracer:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    # ---- lifecycle -------------------------------------------------------
    def start_run(self, kind: str, candidate_id: str = "", job_id: str = "") -> str:
        tid = uuid.uuid4().hex
        try:
            with Session(self._engine) as s:
                s.add(TraceRow(id=tid, kind=kind, candidate_id=candidate_id,
                               job_id=job_id, status="running", started_at=_now()))
                s.commit()
        except Exception:
            log.exception("tracer.start_run failed")
        _CURRENT.set(_RunState(trace_id=tid, t0=time.monotonic()))
        return tid

    def record_span(self, *, name: str, agent: str = "", model: str = "",
                    latency_ms: float = 0.0, cost_usd: float = 0.0,
                    cache_hit: bool = False, error: str = "") -> None:
        st = _CURRENT.get()
        if st is None:
            return  # no active run -> drop silently
        # allocate the ordinal under the lock, but DO NOT bump the span/cost rollups
        # until the row is actually persisted — otherwise a failed write leaves the
        # TraceRow claiming spans/cost that never landed in SpanRow (cost reporting lies).
        with st.lock:
            st.ordinal += 1
            ordinal = st.ordinal
        row = SpanRow(trace_id=st.trace_id, ordinal=ordinal, name=name,
                      agent=agent, model=model, latency_ms=latency_ms,
                      cost_usd=cost_usd, cache_hit=cache_hit, error=error, ts=_now())
        if self._write_span(row):
            with st.lock:
                st.spans += 1
                st.cost += cost_usd
        else:
            with st.lock:
                st.dropped += 1

    def _write_span(self, row) -> bool:
        """Persist one span, retrying briefly on SQLite 'database is locked' (the 3
        persona threads commit spans concurrently against one file). Returns False if
        the span was lost so the caller can avoid counting it in the rollups."""
        for attempt in range(3):
            try:
                with Session(self._engine) as s:
                    s.add(row)
                    s.commit()
                return True
            except OperationalError as e:
                if "locked" in str(e).lower() and attempt < 2:
                    time.sleep(0.1 * (attempt + 1))
                    continue
                log.warning("tracer span write failed (%s); span dropped", e.__class__.__name__)
                return False
            except Exception:
                log.exception("tracer.record_span failed")
                return False
        return False

    def end_run(self, status: str = "done", error: str = "", cache_hit: bool = False) -> None:
        st = _CURRENT.get()
        if st is None:
            return
        if st.dropped:
            log.warning("trace %s: %d span(s) dropped on write — rollup cost/count "
                        "reflect only persisted spans", st.trace_id, st.dropped)
        try:
            total_ms = (time.monotonic() - st.t0) * 1000.0
            with Session(self._engine) as s:
                row = s.get(TraceRow, st.trace_id)
                if row:
                    row.status = status
                    row.error = error[:2000]
                    row.cache_hit = cache_hit
                    row.span_count = st.spans
                    row.total_ms = round(total_ms, 1)
                    row.total_cost_usd = round(st.cost, 6)
                    row.ended_at = _now()
                    s.add(row)
                    s.commit()
        except Exception:
            log.exception("tracer.end_run failed")
        finally:
            _CURRENT.set(None)

    # ---- queries (UI / API) ---------------------------------------------
    def get_run(self, trace_id: str) -> RunTrace | None:
        with Session(self._engine) as s:
            row = s.get(TraceRow, trace_id)
            if row is None:
                return None
            spans = s.exec(
                select(SpanRow).where(SpanRow.trace_id == trace_id).order_by(SpanRow.ordinal)
            ).all()
        return _to_trace(row, spans)

    def list_runs(self, limit: int = 100, candidate_id: str = "",
                  job_id: str = "") -> list[RunTrace]:
        with Session(self._engine) as s:
            stmt = select(TraceRow)
            if candidate_id:
                stmt = stmt.where(TraceRow.candidate_id == candidate_id)
            if job_id:
                stmt = stmt.where(TraceRow.job_id == job_id)
            rows = s.exec(stmt.order_by(TraceRow.started_at.desc()).limit(limit)).all()
        return [_to_trace(r, []) for r in rows]

    def latest_run(self, candidate_id: str, job_id: str) -> RunTrace | None:
        """Most recent run for a (candidate, job) — used to attach spans to the
        candidate breakdown page. Loads spans for the matched run."""
        runs = self.list_runs(limit=1, candidate_id=candidate_id, job_id=job_id)
        return self.get_run(runs[0].id) if runs else None

    def agent_stats(self) -> list[dict]:
        """Per-agent rollup for the Agents UI: calls, cache-hit rate, avg latency, cost."""
        with Session(self._engine) as s:
            rows = s.exec(
                select(
                    SpanRow.agent,
                    func.count().label("calls"),
                    func.sum(SpanRow.cache_hit),
                    func.avg(SpanRow.latency_ms),
                    func.sum(SpanRow.cost_usd),
                ).group_by(SpanRow.agent)
            ).all()
        out = []
        for agent, calls, hits, avg_ms, cost in rows:
            calls = int(calls or 0)
            hits = int(hits or 0)
            out.append({
                "agent": agent or "?",
                "calls": calls,
                "cache_hits": hits,
                "cache_hit_rate": round((hits / calls), 3) if calls else 0.0,
                "avg_latency_ms": round(float(avg_ms or 0.0), 1),
                "total_cost_usd": round(float(cost or 0.0), 6),
            })
        out.sort(key=lambda d: d["calls"], reverse=True)
        return out

    def spend_by_model(self) -> list[dict]:
        """Cost + call count grouped by model — the spend breakdown on the dashboard."""
        with Session(self._engine) as s:
            rows = s.exec(
                select(
                    SpanRow.model,
                    func.count().label("calls"),
                    func.sum(SpanRow.cost_usd),
                ).where(SpanRow.model != "").group_by(SpanRow.model)
            ).all()
        out = [{"model": m or "?", "calls": int(c or 0), "cost_usd": round(float(cost or 0.0), 6)}
               for m, c, cost in rows]
        out.sort(key=lambda d: d["cost_usd"], reverse=True)
        return out


def _to_trace(row: TraceRow, span_rows) -> RunTrace:
    return RunTrace(
        id=row.id, kind=row.kind, candidate_id=row.candidate_id, job_id=row.job_id,
        status=row.status, cache_hit=row.cache_hit, span_count=row.span_count,
        total_ms=row.total_ms, total_cost_usd=row.total_cost_usd, error=row.error,
        started_at=row.started_at, ended_at=row.ended_at,
        spans=[Span(trace_id=sp.trace_id, ordinal=sp.ordinal, name=sp.name, agent=sp.agent,
                    model=sp.model, latency_ms=sp.latency_ms, cost_usd=sp.cost_usd,
                    cache_hit=sp.cache_hit, error=sp.error, ts=sp.ts) for sp in span_rows],
    )

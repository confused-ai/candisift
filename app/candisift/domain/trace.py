"""Agent-run trace value objects — the "sessions" surface.

A RunTrace is one screening (or ingest) session; Spans are the individual
agent / tool / LLM calls inside it, each with model, latency, cost and a
cache-hit flag. Pure domain types — the SqlTracer adapter persists them and the
UI renders them. No I/O here.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from .models import utcnow


class Span(BaseModel):
    trace_id: str
    ordinal: int = 0
    name: str = ""           # display label, e.g. "tech:claude-haiku-4-5"
    agent: str = ""          # role: profile | jd | tech | risk | synth | tool
    model: str = ""
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    cache_hit: bool = False
    error: str = ""
    ts: datetime = Field(default_factory=utcnow)


class RunTrace(BaseModel):
    id: str
    kind: str = "screen"     # screen | ingest
    candidate_id: str = ""
    job_id: str = ""
    status: str = "running"  # running | done | error
    cache_hit: bool = False  # whole run served from cache (no LLM spend)
    span_count: int = 0
    total_ms: float = 0.0
    total_cost_usd: float = 0.0
    error: str = ""
    started_at: datetime = Field(default_factory=utcnow)
    ended_at: datetime | None = None
    spans: list[Span] = []

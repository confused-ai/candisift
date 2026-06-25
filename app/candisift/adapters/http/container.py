"""Composition root — the ONE place adapters are bound to ports (manual DI).

Everything inward depends on abstractions; here at the edge we pick concretes.
Swap SQLite->Postgres, stub->Agno, token-cosine->embeddings by editing only this
file. The application and domain never change.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from app.candisift.config import Settings, load_settings
from app.candisift.application.screening_service import ScreeningService
from app.candisift.adapters.persistence.db import make_engine, init_db
from app.candisift.adapters.persistence.repositories import (
    SqlCandidateRepository, SqlJobRepository, SqlResultRepository, SqlAuditLog,
)
from app.candisift.adapters.persistence.queue import SqliteTaskQueue
from app.candisift.adapters.ranking.token_cosine import TokenCosineRanker
from app.candisift.adapters.ranking.embedding import EmbeddingRanker
from app.candisift.adapters.parsing.text_extractor import FileTextExtractor
from app.candisift.adapters.observability.tracer import SqlTracer
from app.candisift.adapters.memory.agent_memory import SqlAgentMemory

log = logging.getLogger("candisift.container")


@dataclass
class Container:
    settings: Settings
    service: ScreeningService
    queue: SqliteTaskQueue
    candidates: SqlCandidateRepository
    jobs: SqlJobRepository
    results: SqlResultRepository
    audit: SqlAuditLog
    tracer: SqlTracer
    memory: SqlAgentMemory
    parser: FileTextExtractor   # configured extractor (OCR caps) for ad-hoc UI uploads


def build_container(settings: Settings | None = None) -> Container:
    settings = settings or load_settings()
    engine = make_engine(settings.db_url)
    init_db(engine)

    candidates = SqlCandidateRepository(engine)
    jobs = SqlJobRepository(engine)
    results = SqlResultRepository(engine)
    audit = SqlAuditLog(engine)
    queue = SqliteTaskQueue(engine, max_attempts=settings.max_attempts,
                            retry_base_seconds=settings.worker_retry_base_seconds,
                            retry_max_seconds=settings.worker_retry_max_seconds)
    # semantic ranker (local embeddings) with the lexical ranker as offline fallback
    ranker = EmbeddingRanker(fallback=TokenCosineRanker())
    tracer = SqlTracer(engine)
    memory = SqlAgentMemory(engine)
    parser = FileTextExtractor(
        ocr=settings.ocr_enabled,
        ocr_lang=settings.ocr_lang,
        ocr_dpi=settings.ocr_dpi,
        max_ocr_pages=settings.ocr_max_pages,
        ocr_timeout_s=settings.ocr_timeout_s,
    )

    # LLM provider chain (each implements ports.LLMProvider, LSP-clean):
    #   Tracing( Resilient( Throttled( Agno ) ) )   [stub branch: Tracing( Stub )]
    # Throttle is INNERMOST so it is acquired/released PER ATTEMPT: a retry's backoff
    # sleep no longer holds a concurrency slot or rate token (the old outermost-throttle
    # order pinned both across every retry + timeout, collapsing throughput under
    # provider degradation). Resilient wraps the throttle (each attempt re-admits);
    # tracing outermost so a span's latency covers the whole admit+retry path.
    from app.candisift.adapters.llm.throttled import ThrottledLLMProvider
    from app.candisift.adapters.llm.traced import TracingLLMProvider
    from app.candisift.adapters.llm.persona_designer import TemplatePersonaDesigner
    if settings.has_llm:
        from app.candisift.adapters.llm.agno_personas import AgnoLLMProvider
        from app.candisift.adapters.llm.resilient import ResilientLLMProvider
        from app.candisift.adapters.llm.persona_designer import AgnoPersonaDesigner
        throttled = ThrottledLLMProvider(
            AgnoLLMProvider(memory=memory),
            max_concurrency=settings.llm_max_concurrency,
            rate_per_min=settings.llm_rate_per_min,
        )
        base = ResilientLLMProvider(throttled)
        # design personas with the strong tier; falls back to the template offline
        persona_designer = AgnoPersonaDesigner(settings.synth_model, TemplatePersonaDesigner())
        log.info("LLM: Agno multi-provider (throttled→resilient; default personas=%s synth=%s)",
                 settings.persona_model, settings.synth_model)
    else:
        from app.candisift.adapters.llm.stub import StubLLMProvider
        base = StubLLMProvider()        # instant + offline; no throttle/retry needed
        persona_designer = TemplatePersonaDesigner()
        log.warning("ANTHROPIC_API_KEY not set — using deterministic offline stub LLM + template personas")
    llm = TracingLLMProvider(base, tracer)

    service = ScreeningService(
        text_extractor=parser, llm=llm, ranker=ranker,
        candidates=candidates, jobs=jobs, results=results, audit=audit, queue=queue,
        default_persona_model=settings.persona_model,
        default_synth_model=settings.synth_model,
        top_n=settings.top_n,
        tracer=tracer, memory=memory, persona_designer=persona_designer,
        coverage_audit=settings.coverage_audit_enabled,
        hr_eval=settings.hr_eval_enabled,
    )
    return Container(settings=settings, service=service, queue=queue,
                     candidates=candidates, jobs=jobs, results=results, audit=audit,
                     tracer=tracer, memory=memory, parser=parser)

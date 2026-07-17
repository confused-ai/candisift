"""SQLModel engine + tables. Persistence detail — lives at the outermost ring.

Tables store domain value objects as JSON blobs (profile_json, spec_json, ...);
repositories map row <-> domain Pydantic. SQLite by default; point CANDISIFT_DB_URL at
Postgres and nothing else changes (the repos use SQLAlchemy, not sqlite specifics).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import JSON, Column, Index, event
from sqlalchemy.engine import Engine
from sqlmodel import Field, SQLModel, create_engine


class CandidateRow(SQLModel, table=True):
    __tablename__ = "candidate"
    id: str = Field(primary_key=True)
    dedup_key: str = Field(index=True)
    content_sha256: str = Field(default="", index=True)   # sha256 of raw upload bytes -> skip OCR+extract on exact repeat
    source_filename: str = ""
    profile_json: dict = Field(default_factory=dict, sa_column=Column(JSON))
    near_duplicate_of: str = ""
    duplicate_similarity: float = 0.0
    created_at: datetime


class JobRow(SQLModel, table=True):
    __tablename__ = "job"
    id: str = Field(primary_key=True)
    title: str = ""
    raw_text: str = ""
    spec_json: dict = Field(default_factory=dict, sa_column=Column(JSON))
    personas_json: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    # per-job model choice. Losing these on reload silently re-screened (and
    # cost-estimated) every job on the env-default models instead of the ones
    # the recruiter picked. "" (migrated legacy rows) resolves like "auto".
    persona_model: str = "auto"
    synth_model: str = "auto"
    created_at: datetime


class ResultRow(SQLModel, table=True):
    __tablename__ = "result"
    id: str = Field(primary_key=True)
    job_id: str = Field(index=True)
    candidate_id: str = Field(index=True)
    passed_hard_filters: bool = True
    filter_reasons: list = Field(default_factory=list, sa_column=Column(JSON))
    hard_filter_overridden: bool = False   # human overruled the gate; survives re-screens
    semantic_score: float = 0.0
    tech_json: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    risk_json: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    hr_json: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    synthesis_json: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    bias_flags: list = Field(default_factory=list, sa_column=Column(JSON))
    requires_human_review: bool = False
    review_reasons: list = Field(default_factory=list, sa_column=Column(JSON))
    ungrounded_claims: list = Field(default_factory=list, sa_column=Column(JSON))
    coverage_json: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    decision: str = "pending"
    models_fingerprint: str = ""   # sha256(persona|synth|spec) -> reuse verdict, skip LLM on exact repeat
    created_at: datetime


class AuditRow(SQLModel, table=True):
    __tablename__ = "audit"
    id: Optional[int] = Field(default=None, primary_key=True)
    ts: datetime
    event: str = Field(index=True)
    job_id: str = Field(default="", index=True)   # per-job activity log filter; derived from event fields
    data_json: dict = Field(default_factory=dict, sa_column=Column(JSON))


class TaskRow(SQLModel, table=True):
    __tablename__ = "task"
    id: str = Field(primary_key=True)
    type: str
    payload_json: dict = Field(default_factory=dict, sa_column=Column(JSON))
    status: str = Field(default="pending", index=True)
    attempts: int = 0
    max_attempts: int = 3
    last_error: str = ""
    lease_until: Optional[datetime] = None
    available_at: Optional[datetime] = None   # earliest claim time; future on retry (backoff)
    created_at: datetime
    updated_at: datetime


# composite index for the durable queue's "oldest pending" claim scan
Index("ix_task_status_created", TaskRow.status, TaskRow.created_at)
# the job page's activity log filters job_id + ORDER BY ts DESC on every render, and
# the audit table is append-only — the single-column indexes stop paying once it grows.
Index("ix_audit_job_ts", AuditRow.job_id, AuditRow.ts)


class TraceRow(SQLModel, table=True):
    """One agent-run session (a screening or ingest). Spans hang off it."""
    __tablename__ = "trace"
    id: str = Field(primary_key=True)
    kind: str = Field(default="screen", index=True)   # screen | ingest
    candidate_id: str = Field(default="", index=True)
    job_id: str = Field(default="", index=True)
    status: str = "running"                            # running | done | error
    cache_hit: bool = False                            # whole run served from cache
    span_count: int = 0
    total_ms: float = 0.0
    total_cost_usd: float = 0.0
    error: str = ""
    started_at: datetime
    ended_at: Optional[datetime] = None


class SpanRow(SQLModel, table=True):
    """One agent / tool / LLM call inside a trace."""
    __tablename__ = "span"
    id: Optional[int] = Field(default=None, primary_key=True)
    trace_id: str = Field(index=True)
    ordinal: int = 0
    name: str = ""                 # e.g. tech:claude-haiku-4-5
    agent: str = ""                # role: profile|jd|tech|risk|synth|tool
    model: str = ""
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    cache_hit: bool = False
    error: str = ""
    ts: datetime


class MemoryRow(SQLModel, table=True):
    """Persistent agent memory: prior evaluations + recruiter decisions, recalled
    by the retrieval tool to ground future judgments."""
    __tablename__ = "memory"
    id: Optional[int] = Field(default=None, primary_key=True)
    candidate_id: str = Field(default="", index=True)
    job_id: str = Field(default="", index=True)
    kind: str = Field(default="", index=True)   # tech_eval | risk_eval | synthesis | decision
    content: str = ""                           # short text summary, keyword-searchable
    data_json: dict = Field(default_factory=dict, sa_column=Column(JSON))
    ts: datetime


def _is_libsql(db_url: str) -> bool:
    return "+libsql" in db_url or db_url.startswith("libsql")


def is_unique_violation(exc: BaseException) -> bool:
    """True if `exc` is a UNIQUE/primary-key violation, across drivers. pysqlite
    surfaces it as SQLAlchemy IntegrityError; the libSQL (Turso) driver raises a raw
    ValueError('UNIQUE constraint failed: ...'); Postgres says 'duplicate key value
    violates unique constraint' (SQLSTATE 23505). All must be treated the same for
    idempotent inserts, or a duplicate enqueue/ingest would crash on that backend."""
    if getattr(getattr(exc, "orig", None), "pgcode", "") == "23505":
        return True
    msg = str(exc).lower()
    return "unique constraint failed" in msg or "duplicate key value" in msg


def make_engine(db_url: str) -> Engine:
    connect_args: dict[str, Any] = {}
    engine_kwargs: dict[str, Any] = {}
    if _is_libsql(db_url):
        # libSQL (Turso) driver. It is thread-safe per-connection on its own (no
        # pysqlite check_same_thread flag), and a remote Turso DB reads its auth token
        # from the TURSO_AUTH_TOKEN env var — passed through here so secrets never live
        # in the URL string / logs. Local-file libSQL ignores the token.
        import os
        token = os.getenv("TURSO_AUTH_TOKEN") or os.getenv("CANDISIFT_DB_AUTH_TOKEN")
        if token:
            connect_args["auth_token"] = token
    elif db_url.startswith("sqlite"):
        # plain pysqlite: threads (worker + web) share one file; WAL + busy_timeout
        # reduce "database is locked" errors.
        connect_args = {"check_same_thread": False, "timeout": 30}
    else:
        # server DB (Postgres/MySQL): validate pooled connections before handing
        # them out (survives DB restarts / idle timeouts) and recycle stale ones.
        # Pool sized for one web process + the background worker thread.
        engine_kwargs.update(pool_pre_ping=True, pool_size=5, max_overflow=10,
                             pool_recycle=1800)
    engine = create_engine(db_url, connect_args=connect_args, **engine_kwargs)
    if engine.url.get_backend_name() == "sqlite":
        # busy_timeout is PER-CONNECTION and resets on every new pooled connection.
        # Setting it once in init_db only covered one connection; the rest defaulted
        # to 0 => a concurrent write threw immediately ("database is locked") instead
        # of waiting. libSQL raises that as a bare ValueError, which killed the worker
        # thread. Set it on every connect. (pysqlite's timeout= arg already does this,
        # but the listener is harmless there and covers libSQL, which has no such arg.)
        @event.listens_for(engine, "connect")
        def _set_busy_timeout(dbapi_conn, _rec):  # noqa: ANN001
            try:
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA busy_timeout=30000")
                cur.close()
            except Exception:  # pragma: no cover - backend-dependent (remote Turso)
                pass
    return engine


def init_db(engine: Engine) -> None:
    SQLModel.metadata.create_all(engine)        # creates missing TABLES only
    # libSQL reports backend "sqlite" (it IS SQLite-compatible), so the same migration
    # + WAL + partial-index path applies to both pysqlite and libSQL/Turso.
    if engine.url.get_backend_name() == "sqlite":
        _migrate_sqlite(engine)                  # adds missing COLUMNS to existing tables
        with engine.connect() as conn:
            # PRAGMAs are a best-effort tune-up. A remote Turso DB manages journaling
            # itself and may reject/ignore these — never let that block boot.
            for pragma in ("PRAGMA journal_mode=WAL", "PRAGMA busy_timeout=30000"):
                try:
                    conn.exec_driver_sql(pragma)
                except Exception:  # pragma: no cover - backend-dependent
                    pass
    # partial unique indexes are valid on Postgres too — the dedup/cache
    # invariants must hold as DB constraints on every backend, not just SQLite.
    _ensure_unique_indexes(engine)


def _ensure_unique_indexes(engine: Engine) -> None:
    """PARTIAL unique indexes turn the candidate dedup/cache invariants into a DB
    constraint, so a check-then-insert race can't create two candidates for one person.
    Partial (WHERE <col> != '') because both keys legitimately default to '': an
    identity-less profile has dedup_key='' and must NOT collide with other empty ones.
    Best-effort: if a legacy DB already holds duplicates the CREATE fails — we log and
    move on rather than block boot (dedup still enforced at the app layer)."""
    import logging
    log = logging.getLogger("candisift.db")
    stmts = [
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_candidate_dedup_key "
        "ON candidate(dedup_key) WHERE dedup_key != ''",
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_candidate_content_sha "
        "ON candidate(content_sha256) WHERE content_sha256 != ''",
    ]
    with engine.connect() as conn:
        for sql in stmts:
            try:
                conn.exec_driver_sql(sql)
            except Exception as e:                # pre-existing duplicates -> keep booting
                log.warning("could not create unique index (%s): %s", sql.split()[5], e)
        conn.commit()


def _migrate_sqlite(engine: Engine) -> None:
    """Lightweight additive migration: ADD COLUMN for any model column missing from
    an existing table. SQLite create_all never alters existing tables, so a DB from
    an older schema would otherwise 500 on read. Idempotent; additive only (no drops,
    no type changes). Postgres users should use a real migration tool instead."""
    import logging
    from sqlalchemy import inspect
    log = logging.getLogger("candisift.db")
    insp = inspect(engine)
    tables = set(insp.get_table_names())
    with engine.begin() as conn:
        for table in SQLModel.metadata.sorted_tables:
            if table.name not in tables:
                continue                         # freshly created by create_all
            have = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name in have:
                    continue
                coltype = col.type.compile(dialect=engine.dialect)
                # ADD COLUMN can't be NOT NULL without a default -> give scalars a sane
                # default so existing rows stay valid for the Pydantic mappers. Derive it
                # from the compiled SQL type: SQLModel's str columns (AutoString/VARCHAR)
                # raise on .python_type, which previously left them with NO default -> the
                # column filled NULL and every legacy-row read 500'd in Pydantic.
                default = ""
                try:
                    pt = col.type.python_type
                    is_str = pt is str
                    is_int = pt in (int, bool)
                    is_float = pt is float
                except Exception:                # AutoString and friends land here
                    t = coltype.upper()
                    is_str = any(k in t for k in ("CHAR", "TEXT", "CLOB", "STRING"))
                    is_int = is_float = False
                default = (" DEFAULT ''" if is_str else
                           " DEFAULT 0" if is_int else
                           " DEFAULT 0.0" if is_float else "")   # JSON/datetime -> nullable
                conn.exec_driver_sql(
                    f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" {coltype}{default}')
                log.warning("migrated: added %s.%s", table.name, col.name)

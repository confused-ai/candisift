# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
./run.sh                              # venv + deps + launch (http://127.0.0.1:8000)
./run.sh --reload                     # dev autoreload (extra args pass to uvicorn)
PORT=9000 HOST=0.0.0.0 ./run.sh       # override port / bind network
uvicorn main:app --reload             # run without the install wrapper (deps already present)

python3 -m pytest -q                  # full suite, fully offline (no API key needed)
python3 -m pytest tests/test_domain.py::test_name   # single test
python3 -m pytest tests/test_hardening.py -q        # regression suite for the 53 hardening fixes

# the image sets CANDISIFT_ENV=prod (refuses default creds, wildcard CORS, HSTS-off).
# Local demo — override to dev so it boots on plain HTTP:
docker build -t candisift . && docker run -p 8000:8000 \
  -e CANDISIFT_ENV=dev -v candisift-data:/data candisift
# Real deploy (behind TLS) — set creds + HSTS + CORS instead; see README "Deploying".
```

No lint config in repo. Tests are the gate — run them after any change. Everything runs offline against the **stub LLM** unless `ANTHROPIC_API_KEY` (or another provider key) is set; tests never hit the network.

## Architecture

Hexagonal (ports & adapters). **Dependencies point inward only**: `domain/` has zero framework deps, `application/` depends only on `domain.ports.*`, `adapters/` implement those ports. The flow is a **cost-tiered funnel** — deterministic free filters drop most candidates; only survivors reach paid LLM calls:

```
ingest → strip PII → hard filter → rank → [survivors] persona agents (tech ‖ risk) → synthesis → persist+audit
        ╰──── deterministic, ~free, no key ────╯              ╰──── LLM, resilient ────╯
```

The **hard filter is the only cost gate** — rank scores every survivor but cuts nothing (`semantic_score` is advisory, shown to the recruiter). A real top-N cut needs a batch barrier the per-candidate durable worker doesn't have; the dead `top_n` setting that implied otherwise is gone.

That filter's rule: **reject on conflicting evidence, flag on missing evidence.** Extraction is imperfect and a resume is not a form, so "we couldn't parse it" (no location, unrecognized work-auth phrasing, unestablished years) passes with an advisory flag in `filter_reasons`, while a recognized conflict still rejects for free. Blank work auth is the one fail-closed case. Every auto-reject is overridable by a recruiter (`screen(..., override_hard_filter=True)`, audited) — an automated employment decision a human can't overrule is the thing regulators object to.

Orchestrator: `application/screening_service.py` — knows the funnel order and model-selection policy, nothing about Agno/SQL/HTTP.

### Two things to understand before editing

1. **`adapters/http/container.py` is the only place concretes are bound to ports.** Swapping SQLite↔Postgres, stub↔Agno, lexical↔embedding ranker, or adding a persona is a one-adapter change + one line here. The domain and application never change. Don't reach for a concrete adapter from inside `application/` or `domain/`.

2. **The LLM provider is a decorator chain, assembled in `container.py`:** `Tracing( Resilient( Throttled( Agno ) ) )` (or `Tracing( Stub )` offline). Order is load-bearing — **Throttle is innermost** so a retry's backoff sleep doesn't hold a concurrency slot/rate token (the old outermost-throttle order collapsed throughput under provider degradation). Each layer implements `ports.LLMProvider` (LSP-clean); preserve that when touching the chain.

### One process, durable background work

`uvicorn main:app` serves the JSON API, the recruiter UI, **and** the durable worker (started in the FastAPI `lifespan` in `app.py`). Tasks live in SQLite (`SqliteTaskQueue`), survive restarts; a dead worker's task is reclaimed via expired lease on next startup. At-least-once delivery, so **handlers must stay idempotent**:
- screen results use a deterministic id `f"{job_id}.{candidate_id}"` → retried screens upsert the same row.
- ingest dedups on email/phone hash (`services.dedup_key`).

### Invariants that bite if broken

- **PII is stripped (`services.strip_pii`) before any model sees a candidate** — compliance boundary, not cosmetic. Keep it ahead of every LLM call path (screening, resume optimize, cover letter).
- **Untrusted text passes through `domain/guardrails.py`** (prompt-injection fence + sanitizer) before reaching a prompt.
- **Cost estimate is an upper bound** (assumes every resume reaches the LLM stage). It must account for *all* per-screen LLM calls including the HR evaluator and coverage auditor when those flags are on — undercounting here was a prior bug.
- **Default DB is libSQL** (`sqlite+libsql:///candisift.db`), not pysqlite. The libsql driver raises `ValueError` (not `IntegrityError`) on a UNIQUE violation — cross-driver constraint handling for idempotent inserts lives in the persistence layer; don't assume `IntegrityError`.
- **`config.load_settings()` fails fast in prod** (`CANDISIFT_ENV=prod`) on default credentials, wildcard CORS, or HSTS-off. Don't weaken `validate_runtime`.

### Config

All settings use the `CANDISIFT_` prefix, read **only** in `candisift/config.py` (`.env` via `.env.example`). Provider keys (`ANTHROPIC_API_KEY`, …) are read directly from the process env by the Agno SDK, so `config.py` pushes `.env` into `os.environ` at import. `has_llm` is provider-aware (an OpenAI/Gemini/Ollama default with its own key enables real LLMs; Ollama needs no key).

### Pricing

Per-token prices come from pydantic `genai-prices` (embedded offline dataset). The static table in `pricing.py` is the UI label list + offline fallback.

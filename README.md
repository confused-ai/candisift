# CandiSift — AI resume screening & applicant tracking system (ATS)

> Open-source, production-grade **AI ATS**: LLM-powered resume screening, CV parsing, JD↔candidate matching, and bias-audited candidate ranking — self-hostable, privacy-first, offline-capable.

Agentic resume screening on a **cost-tiered funnel**: deterministic filters drop most
candidates for free, embeddings/keyword rank the rest, and LLM **persona agents** (Agno
on Claude) only ever see the survivors. Hexagonal architecture, durable background jobs,
estimate-before-you-spend, per-run model switching, and resilience as a first-class concern.

```
upload N resumes ─► parse ─► strip PII ─► HARD FILTERS ─► rank ─► [survivors] PERSONA AGENTS ─► synthesis ─► human review
                    stage1   stage2        stage3         stage4   tech ‖ risk (parallel)        lead recruiter
                    ╰──────────── deterministic, ~free, no API key ───────────╯   ╰──── Claude, resilient ────╯
```

One process (`uvicorn main:app`) serves the JSON API, the recruiter UI, and the durable
worker. No key? It runs on a deterministic **offline stub LLM** so you can demo and test
the whole pipeline at zero cost.

## Run

```bash
pip install -r requirements.txt
cp .env.example .env                 # edit auth + (optional) ANTHROPIC_API_KEY
python3 -m pytest -q                 # 18 tests, all offline
uvicorn main:app --reload            # http://localhost:8000  (login: recruiter / change-me)
# or: docker build -t candisift . && docker run -p 8000:8000 candisift
```

**Flow:** create a role (pick models or leave *Auto*) → upload resumes → **see the cost
estimate** → confirm → the worker screens in the background → ranked results table → drill
into any candidate's full evidence-cited breakdown → accept/reject.

## Architecture — hexagonal (ports & adapters), SOLID

Dependencies point **inward only**. The domain knows nothing about Agno, SQL, or HTTP.

```
candisift/
  domain/          ENTITIES + RULES + PORTS — zero framework deps
    models.py        entities & value objects (Candidate, Job, ScreeningResult, Task)
    ports.py         Protocol interfaces (TextExtractor, LLMProvider, Ranker, repos, TaskQueue…)
    services.py      pure rules: hard_filter, strip_pii (compliance), dedup_key, skill ontology
    guardrails.py    prompt-injection fence + sanitizer
    ats_readability.py, duplicate.py   deterministic ATS-readability + near-dup scorers
  application/
    screening_service.py   THE funnel orchestrator — depends only on ports (DIP)
  adapters/
    llm/             AgnoLLMProvider (Claude) · ResilientLLMProvider · StubLLMProvider (offline)
    persistence/     SQLModel tables · repositories · durable SqliteTaskQueue · audit log
    ranking/         TokenCosineRanker  (swap → embeddings)
    parsing/         FileTextExtractor  (swap → OCR / parser API)
    worker.py        durable background worker (lease, retry, reclaim)
    http/            FastAPI app · api.py (JSON) · ui.py (htmx-free HTML) · security · container (composition root)
  config.py · pricing.py
main.py
```

**SOLID in practice:** SRP — each adapter does one thing. OCP — a new persona or a new
ranker is a new adapter, zero edits to the orchestrator. LSP — stub/Agno/resilient
providers are interchangeable behind `LLMProvider`. ISP — narrow, single-purpose ports.
DIP — `ScreeningService` depends on `ports.*`, never on a concrete adapter; the composition
root (`http/container.py`) is the only place concretes are chosen.

## What's covered (end-to-end CandiSift)

| Stage | Capability | Status |
|------|------------|--------|
| 1 Ingest | PDF / DOCX / txt parse, multi-file upload | ✅ `parsing/` (OCR = documented swap) |
| 1 Ingest | Structured candidate profile (Agno or offline stub) | ✅ |
| 1 Ingest | Exact dedup (email/phone hash) | ✅ `services.dedup_key` |
| 1 Ingest | **Near-duplicate / resume-farming** detection (Jaccard) | ✅ `duplicate.py` |
| 1.5 | Skill **canonicalization / ontology** | ✅ `services._CANON` (skillNER/ESCO = swap) |
| 2 Rank | Semantic rank (token-cosine cosine; embeddings/FAISS = swap) | ✅ `ranking/` |
| 3 Filter | Deterministic **hard filters** (auth, location, years, certs) | ✅ free, pre-LLM |
| 4 Agents | **Persona agents**: Technical + Risk (parallel) + Synthesis | ✅ `llm/`, evidence-cited |
| 4 Agents | **Per-run model switching** + `auto` default + tiered pricing | ✅ `pricing.py` + UI picker |
| 5 Review | Recruiter UI: ranked table, full breakdown, accept/reject | ✅ `ui.py` |
| — | **Cost estimate before processing** (genai-prices) | ✅ staged-task confirm flow |
| — | **ATS-readability score** (parseability + keyword coverage) | ✅ `ats_readability.py` |
| Compliance | PII stripped before any model sees a candidate | ✅ `strip_pii` |
| Compliance | Append-only audit log (scores, rationale, decisions, versions) | ✅ |
| Compliance | Bias-audit endpoint (pass-through + recommendation mix) | ✅ `/api/jobs/{id}/bias-audit` |
| Security | Prompt-injection fence + flagging | ✅ `guardrails.py` |
| Security | HTTP Basic auth, security headers (CSP…), rate limit, body-size cap, upload validation, fail-fast secrets | ✅ `security.py` |

### Resilience (top priority)

- **Durable queue** — tasks live in SQLite, survive restarts. Worker death mid-screen →
  task reclaimed via expired lease on next startup. At-least-once; handlers idempotent
  (dedup on ingest, deterministic result id on screen).
- **Per-call LLM resilience** — every persona call gets a **timeout**, **retry with
  exponential backoff**, and a **circuit breaker** per (role, model) (`ResilientLLMProvider`).
  Exhaustion raises `LLMUnavailable`; the durable worker re-queues the task — defense in depth.
- **Dead-letter + manual requeue** — exhausted tasks land in `failed`; inspect via
  `GET /api/tasks?status=failed`, retry via `POST /api/tasks/{id}/requeue`.
- **Graceful degradation** — no API key → offline stub LLM; genai-prices miss → static
  price table; missing parser dep → falls back to plain-text read.
- **Safe errors** — unhandled exceptions return a generic 500 with a request id; internals
  never leak to clients.

## Pricing — pydantic `genai-prices`

Per-token prices come from [`genai-prices`](https://github.com/pydantic/genai-prices)
(embedded offline dataset, no network call), so the estimate tracks upstream Anthropic
pricing. Static table in `pricing.py` is the UI label list + offline fallback. The cost
shown is an **upper bound** (assumes every resume reaches the LLM stage); hard filters make
real spend much lower. Example: 100 resumes at Haiku-personas + Opus-synth ≈ **$4.28** vs.
all-Opus ≈ **$10.80** — the funnel's whole point.

## Inspiration → where it plugs in (port-ready swaps)

The named open-source projects map cleanly onto our ports — most are a one-adapter swap
because the architecture already isolates them:

| Project / library | Idea borrowed | Lands at | Status |
|---|---|---|---|
| **xitanggg/open-resume** | "is this resume ATS-parseable?" | `ats_readability.py` | ✅ implemented (deterministic) |
| **sunnypatell/ats-screener** | per-section format/keyword scoring | `ats_readability.py` | ✅ (raw-layout per-platform scoring = next swap) |
| **srbhr/Resume-Matcher** | JD↔resume embedding match | `ranking/` (`Ranker` port) | token-cosine now → swap `EmbeddingRanker` (sentence-transformers + FAISS) |
| **skillNER** + **ESCO/O\*NET** | skill extraction + canonical graph | `services._CANON` / profile extractor | hand-map now → drop-in skillNER adapter |
| PyMuPDF / Tesseract / PaddleOCR | scanned-PDF OCR | `parsing/` (`TextExtractor` port) | text now → add OCR fallback adapter |
| FastAPI + Celery + Docker | throughput skeleton | `worker.py` / `TaskQueue` port | durable SQLite worker now → swap Redis/Celery, same port |
| LangGraph / CrewAI | agent orchestration | `application/` + `llm/` | Agno personas now; orchestrator is swappable |

Each swap touches **one adapter + one line in `container.py`** — the domain and application
never change. That's the payoff of the hexagon.

> Honest caveats (per your landscape note): no shipped accuracy benchmark — build a
> ground-truth eval set before trusting scores. Prompt-injection, near-dup, and bias-audit
> are handled here (most repos skip them). Embeddings/FAISS/skillNER/OCR are deliberately
> left as documented swaps to keep the default install light and resilient.

## API (all under HTTP Basic; `/health`, `/ready` are open)

```
GET  /api/models                         model catalog for the picker
POST /api/jobs                           {jd_text, persona_model?, synth_model?}
GET  /api/jobs · GET /api/jobs/{id}
POST /api/jobs/{id}/upload   (multipart) stage batch + return cost estimate (no spend)
POST /api/jobs/{id}/confirm  {task_ids}  release staged batch → worker screens
GET  /api/jobs/{id}/results              ranked results
GET  /api/results/{id}                   full breakdown + ATS readability + dup flag
POST /api/results/{id}/decision {decision}
GET  /api/jobs/{id}/bias-audit
GET  /api/queue · GET /api/tasks?status=failed · POST /api/tasks/{id}/requeue
```

## Config

All settings use the `CANDISIFT_` prefix (`candisift/config.py`); see `.env.example`. Key ones:
`ANTHROPIC_API_KEY` (omit → offline stub), `CANDISIFT_PERSONA_MODEL` / `CANDISIFT_SYNTH_MODEL`,
`CANDISIFT_DB_URL` (SQLite → Postgres by URL), `CANDISIFT_BASIC_AUTH_PASS`, `CANDISIFT_ENV=prod`
(refuses to boot on the default password), upload/rate limits.

## Tests

`python3 -m pytest -q` — 18 tests, fully offline: domain rules, guardrails, durable-queue
staging/retry/requeue, resilience (retry + circuit breaker), ATS readability, near-dup, and
a full TestClient end-to-end (auth, security headers, estimate→confirm→screen→decision→bias).

## Topics

Open-source **AI ATS** / applicant tracking system · AI resume screening · resume parser ·
CV parsing · candidate screening & ranking · resume–job-description matching (resume matcher) ·
LLM hiring / AI recruiting · recruitment automation · talent acquisition · bias audit ·
ATS-readability score · PII-safe / privacy-first screening · prompt-injection guardrails ·
self-hosted · FastAPI · Python · hexagonal architecture · Agno · Claude.

*Keywords: ats, applicant-tracking-system, ai-ats, resume-screening, resume-parser,
cv-parser, ai-recruiting, candidate-screening, hiring-automation, resume-matcher,
recruitment, talent-acquisition, bias-audit, llm, claude, agno, fastapi, python,
hexagonal-architecture, open-source-ats.*

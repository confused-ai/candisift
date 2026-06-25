"""Typed configuration from environment / .env. The only place env vars are read."""
from __future__ import annotations

import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Push .env into os.environ at import. pydantic's own env_file only fills CANDISIFT_
# fields; provider keys (ANTHROPIC_API_KEY, ...) are read via os.getenv and by
# the Agno SDK, both of which need them in the real process environment.
load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CANDISIFT_", env_file=".env", extra="ignore")

    # persistence — libSQL (Turso's SQLite fork) as a drop-in for plain SQLite. Same
    # file format + SQL, accessed via the libsql driver. Point CANDISIFT_DB_URL at a Turso
    # cloud DB to go remote: "sqlite+libsql://<db>-<org>.turso.io?secure=true" with
    # TURSO_AUTH_TOKEN set. Plain "sqlite:///ats.db" still works (pysqlite path).
    db_url: str = "sqlite+libsql:///ats.db"

    # funnel
    top_n: int = 30
    max_attempts: int = 3
    worker_lease_seconds: int = 300
    # parallel worker threads (atomic claim is multi-safe). >1 overlaps the slow
    # synth call of one screen with cheap persona calls of another; the LLM
    # throttle (llm_max_concurrency) still bounds total provider load.
    worker_concurrency: int = 3
    # retry backoff: a failed task waits base*2^(attempts-1) seconds (capped) before it
    # can be re-claimed, so transient errors don't hot-loop through the attempt budget.
    worker_retry_base_seconds: float = 2.0
    worker_retry_max_seconds: float = 300.0
    # §5 QA auditor (LLM-as-judge) — a cheap second-opinion pass over each verdict.
    # Off => skip the extra call (cost). On => hold unsafe verdicts for human review.
    coverage_audit_enabled: bool = True
    # HR/people-fit evaluator — an extra persona LLM call per screen. It is advisory
    # only (people_score never gates the verdict; every downstream path is None-safe).
    # On => full signal. Off => drop the call to cut ~1/4 of per-screen persona cost,
    # trading away the soft people-fit lens. Default on (no quality change).
    hr_eval_enabled: bool = True

    # OCR (scanned PDFs / image resumes via Tesseract). Needs `tesseract` +
    # `poppler` system binaries; degrades to text-layer-only if absent.
    ocr_enabled: bool = True
    ocr_lang: str = "eng"
    ocr_dpi: int = 300
    ocr_max_pages: int = 50
    ocr_timeout_s: int = 30               # per-call cap on tesseract/poppler

    # model tiers (any provider Agno supports; see pricing.MODEL_CATALOG)
    persona_model: str = "claude-haiku-4-5"
    synth_model: str = "claude-opus-4-8"

    # LLM throttling (provider-side; distinct from the HTTP rate limiter)
    llm_max_concurrency: int = 4          # max simultaneous in-flight LLM calls
    llm_rate_per_min: int = 60            # token-bucket cap on LLM calls/minute

    # recruiter UI auth (HTTP Basic). Change before exposing publicly.
    basic_auth_user: str = "recruiter"
    basic_auth_pass: str = "change-me"

    # deployment env: "prod" enforces secret hardening (see validate_runtime)
    env: str = "dev"

    # ---- security limits / guardrails ----
    max_file_mb: int = 5                  # per-file upload cap
    max_files_per_batch: int = 200        # per-upload count cap
    max_request_mb: int = 64              # whole-request body cap
    rate_limit_per_min: int = 120         # per-client request budget
    cors_origins: str = ""                # comma-separated allowlist; empty = same-origin only
    hsts: bool = False                    # enable only behind HTTPS

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def has_key_for(self, model: str) -> bool:
        """True if the API key for this model's provider is configured. A local
        Ollama model needs no key (it talks to a local server), so it always
        counts as available — set CANDISIFT_*_MODEL=ollama/<name> to run fully offline."""
        from app.candisift import pricing
        if pricing.provider_for(model) == "ollama":
            return True
        return bool(os.getenv(pricing.provider_env_var(model)))

    @property
    def has_llm(self) -> bool:
        """True when the configured default models' provider keys are present;
        otherwise the deterministic offline stub adapters are used (boot, demo,
        and tests work with no key). Provider-aware so an OpenAI/Gemini/Groq
        default with its own key enables real LLMs without an Anthropic key."""
        return self.has_key_for(self.persona_model) and self.has_key_for(self.synth_model)

    def validate_runtime(self) -> None:
        """Fail fast in prod on insecure defaults — better to refuse boot than to
        serve an open door. Covers every credential + transport default, not just the
        password: a default username, a wildcard CORS origin on this authenticated
        admin surface, or HSTS-off all weaken a prod deployment."""
        if self.env.lower() == "prod":
            problems = []
            if self.basic_auth_pass in ("", "change-me"):
                problems.append("CANDISIFT_BASIC_AUTH_PASS is unset/default")
            if self.basic_auth_user in ("", "recruiter"):
                problems.append("CANDISIFT_BASIC_AUTH_USER is unset/default")
            if "*" in self.cors_list:
                problems.append("CANDISIFT_CORS_ORIGINS contains '*' (wildcard) — set explicit origins")
            if not self.hsts:
                problems.append("CANDISIFT_HSTS is off — enable behind HTTPS")
            if problems:
                raise RuntimeError("insecure production config: " + "; ".join(problems))


def load_settings() -> Settings:
    s = Settings()
    s.validate_runtime()
    return s

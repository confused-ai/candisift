"""Model catalog + cost estimation.

Per-token prices come from pydantic's `genai-prices` (embedded offline dataset,
no network call) so the numbers track upstream instead of a table we hand-maintain.
The static MODEL_CATALOG remains as the UI label list and an offline fallback if a
model id isn't in genai-prices' dataset.

The estimate is an UPPER BOUND: it assumes every uploaded resume survives the hard
filter and reaches the persona + synthesis stages. Real spend is lower.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger("candisift.pricing")

try:
    from genai_prices import Usage, calc_price
    _HAS_GENAI_PRICES = True
except Exception:  # pragma: no cover - library optional
    _HAS_GENAI_PRICES = False


@dataclass(frozen=True)
class ModelInfo:
    id: str
    label: str
    fallback_input_per_mtok: float
    fallback_output_per_mtok: float
    provider: str = "anthropic"


# UI dropdown + offline fallback prices (USD / 1M tokens). Multi-provider: Agno
# routes each id to the right SDK (see agno_personas._model).
MODEL_CATALOG: list[ModelInfo] = [
    ModelInfo("claude-haiku-4-5", "Haiku 4.5 (fast, cheap)", 1.0, 5.0, "anthropic"),
    ModelInfo("claude-sonnet-4-6", "Sonnet 4.6 (balanced)", 3.0, 15.0, "anthropic"),
    ModelInfo("claude-opus-4-8", "Opus 4.8 (most capable)", 5.0, 25.0, "anthropic"),
    ModelInfo("claude-opus-4-7", "Opus 4.7", 5.0, 25.0, "anthropic"),
    ModelInfo("claude-fable-5", "Fable 5 (frontier)", 10.0, 50.0, "anthropic"),
    ModelInfo("gpt-4o", "GPT-4o (OpenAI)", 2.5, 10.0, "openai"),
    ModelInfo("gpt-4o-mini", "GPT-4o mini (OpenAI, cheap)", 0.15, 0.6, "openai"),
    ModelInfo("gemini-1.5-pro", "Gemini 1.5 Pro (Google)", 1.25, 5.0, "google"),
    ModelInfo("gemini-1.5-flash", "Gemini 1.5 Flash (Google, cheap)", 0.075, 0.3, "google"),
    ModelInfo("llama-3.3-70b-versatile", "Llama 3.3 70B (Groq)", 0.59, 0.79, "groq"),
    # local models via Ollama — $0/token (your hardware). "ollama/" prefix routes
    # to the Ollama adapter and disambiguates from the groq llama-* ids above.
    # ponytail: local LLM support is scaffolded but not default; set CANDISIFT_PERSONA_MODEL
    # or CANDISIFT_SYNTH_MODEL=ollama/<name> to activate, or pick from the UI model dropdown.
    ModelInfo("ollama/llama3.1", "Llama 3.1 (local · Ollama)", 0.0, 0.0, "ollama"),
    ModelInfo("ollama/qwen2.5", "Qwen 2.5 (local · Ollama)", 0.0, 0.0, "ollama"),
    ModelInfo("ollama/qwen2.5:7b", "Qwen 2.5 7B (local · Ollama)", 0.0, 0.0, "ollama"),
    ModelInfo("ollama/qwen3:8b", "Qwen 3 8B (local · Ollama)", 0.0, 0.0, "ollama"),
    ModelInfo("ollama/mistral", "Mistral 7B (local · Ollama)", 0.0, 0.0, "ollama"),
]
_BY_ID = {m.id: m for m in MODEL_CATALOG}

# id prefix -> provider, for ids not in the static catalog
_PROVIDER_PREFIXES = (
    ("ollama/", "ollama"),  # explicit prefix, checked before bare "llama"->groq
    ("openai/", "openai"),  # any OpenAI-compatible endpoint (DeepInfra, Together,
                            # OpenRouter, Fireworks, vLLM, LM Studio, ...) via OPENAI_BASE_URL
    ("claude", "anthropic"), ("gpt", "openai"), ("o1", "openai"), ("o3", "openai"),
    ("gemini", "google"), ("llama", "groq"), ("mixtral", "groq"), ("mistral", "mistral"),
)


def provider_for(model: str) -> str:
    """Infer the LLM provider for a model id (catalog first, then prefix)."""
    info = _BY_ID.get(model)
    if info is not None:
        return info.provider
    low = (model or "").lower()
    for prefix, provider in _PROVIDER_PREFIXES:
        if low.startswith(prefix):
            return provider
    return "anthropic"


# provider -> the env var holding its API key
PROVIDER_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "ollama": "OLLAMA_API_KEY",  # only for Ollama Cloud; a local server needs no key
}


def provider_env_var(model: str) -> str:
    """Env var that must hold the API key for this model's provider."""
    return PROVIDER_ENV.get(provider_for(model), "ANTHROPIC_API_KEY")

# rough per-resume token usage by funnel stage (input, output). These persona-tier
# stages always run for a survivor.
_PERSONA_STAGES = {"profile_extract": (2500, 600), "technical": (2500, 800), "risk": (1800, 500)}
# optional persona-tier stages, gated by Settings flags (hr_eval / coverage_audit).
# Both default ON, so leaving them out of the estimate under-quotes real spend.
_HR_STAGE = (2500, 600)            # people-fit evaluator (persona model)
_COVERAGE_STAGE = (3000, 500)      # QA auditor sees spec+profile+evals (persona model)
_SYNTH_STAGE = (1800, 700)


def catalog() -> list[dict]:
    """For the UI model picker."""
    return [{"id": "auto", "label": "Auto (configured default)"}] + [
        {"id": m.id, "label": m.label} for m in MODEL_CATALOG
    ]


def is_known_model(model: str) -> bool:
    return model == "auto" or model in _BY_ID


def resolve_model(model: str, default: str) -> str:
    return default if model in ("", "auto") else model


def _stage_price(model: str, tin: int, tout: int) -> tuple[float, str]:
    """(cost_usd, source) for one stage. source ∈ free | genai-prices |
    static-fallback | unknown. A paid OpenAI-compatible endpoint ('openai/<id>') is
    'unknown' (NOT free) — its per-token price isn't in any dataset, so reporting $0
    would hide real spend; surfacing 'unknown' is honest. A model id we can't price at
    all is 'unknown' too, never a raise — the estimate must not 500 after files are
    already staged."""
    if provider_for(model) == "ollama":
        return 0.0, "free"                       # local inference, genuinely $0
    if model.startswith("openai/"):
        return 0.0, "unknown"                    # paid custom endpoint, price unknown
    if _HAS_GENAI_PRICES:
        try:
            p = calc_price(Usage(input_tokens=tin, output_tokens=tout),
                           model_ref=model, provider_id=provider_for(model))
            return float(p.total_price), "genai-prices"
        except Exception:                        # fall through to static table
            log.debug("genai-prices miss for %s; using fallback table", model)
    info = _BY_ID.get(model)
    if info is None:
        return 0.0, "unknown"                    # unpriceable -> resilient, not a crash
    return (tin / 1e6 * info.fallback_input_per_mtok
            + tout / 1e6 * info.fallback_output_per_mtok), "static-fallback"


def _stage_cost(model: str, tin: int, tout: int) -> float:
    return _stage_price(model, tin, tout)[0]


def call_cost(model: str, input_chars: int, output_chars: int) -> float:
    """Best-effort USD cost of a single LLM call, estimated from text length
    (~4 chars/token). Used by the tracer for per-span cost. Returns 0.0 for
    unknown/stub models rather than raising — tracing must never break a call."""
    try:
        tin, tout = input_chars // 4, output_chars // 4
        return round(_stage_cost(model, tin, tout), 6)
    except Exception:
        return 0.0


def _roll_source(sources: list[str]) -> str:
    """Collapse per-stage price sources into one label for the estimate. 'free' stages
    don't count against confidence; a mix of priced sources (or any 'unknown') is
    surfaced honestly rather than masked as a single clean source."""
    priced = {s for s in sources if s != "free"}
    if not priced:
        return "free"
    if priced == {"genai-prices"}:
        return "genai-prices"
    if priced == {"static-fallback"}:
        return "static-fallback"
    return "mixed" if "unknown" not in priced else "unknown"


def estimate_batch(n_resumes: int, persona_model: str, synth_model: str,
                   *, hr_eval: bool = True, coverage_audit: bool = True) -> dict:
    """Upper-bound cost for screening n resumes with the chosen models. Counts EVERY
    persona-tier call that actually runs for a survivor — profile, technical, risk,
    plus HR and the QA auditor when their flags are on — and the synthesis call. Omit
    a flag only if its Settings toggle is off, so the quote mirrors real spend and the
    `is_upper_bound` promise holds."""
    stages: list[tuple[float, str]] = [
        _stage_price(persona_model, tin, tout) for tin, tout in _PERSONA_STAGES.values()
    ]
    if hr_eval:
        stages.append(_stage_price(persona_model, *_HR_STAGE))
    if coverage_audit:
        stages.append(_stage_price(persona_model, *_COVERAGE_STAGE))
    stages.append(_stage_price(synth_model, *_SYNTH_STAGE))

    per_resume = round(sum(c for c, _ in stages), 4)
    return {
        "n_resumes": n_resumes,
        "persona_model": persona_model,
        "synth_model": synth_model,
        "hr_eval": hr_eval,
        "coverage_audit": coverage_audit,
        "per_resume_usd": per_resume,
        # base the total on the SAME rounded per-resume figure shown above so the two
        # numbers reconcile (displayed per_resume × n == displayed total).
        "estimated_total_usd": round(per_resume * n_resumes, 2),
        "price_source": _roll_source([s for _, s in stages]),
        "is_upper_bound": True,
        "note": "upper bound — assumes all resumes pass hard filters and reach the LLM stage",
    }

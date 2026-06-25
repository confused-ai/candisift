"""TracingLLMProvider — emits one span per persona/LLM call.

Wraps an inner LLMProvider; each method call is timed, its input/output sizes
measured, its cost estimated (pricing.call_cost), and a span recorded on the
active run via the Tracer. Composes with the resilient + throttled decorators —
order in the chain decides what the latency includes (we sit outside Resilient
so a span's latency covers retries).

Tracing failures never propagate: a bad span must not fail a screening.
"""
from __future__ import annotations

import logging
import time

from app.candisift import pricing

log = logging.getLogger("candisift.traced")

# inner factory name -> role label
_ROLES = {
    "profile_extractor": "profile",
    "jd_extractor": "jd",
    "technical": "tech",
    "risk": "risk",
    "hr": "hr",
    "synthesizer": "synth",
    "coverage_auditor": "coverage",
    "resume_optimizer": "optimize",
    "cover_letter_writer": "coverletter",
    "github_selector": "github",
    "linkedin_selector": "linkedin",
}


def _text_len(obj) -> int:
    try:
        if hasattr(obj, "model_dump_json"):
            return len(obj.model_dump_json())
        return len(str(obj))
    except Exception:
        return 0


class _TracingProxy:
    def __init__(self, inner, tracer, role: str, model: str) -> None:
        self._inner = inner
        self._tracer = tracer
        self._role = role
        self._model = model

    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr

        def wrapped(*args, **kwargs):
            in_chars = sum(_text_len(a) for a in args) + sum(_text_len(v) for v in kwargs.values())
            t0 = time.monotonic()
            err = ""
            result = None
            try:
                result = attr(*args, **kwargs)
                return result
            except Exception as e:  # noqa: BLE001 — record then re-raise
                err = f"{type(e).__name__}: {e}"[:500]
                raise
            finally:
                latency_ms = (time.monotonic() - t0) * 1000.0
                out_chars = _text_len(result) if result is not None else 0
                cost = pricing.call_cost(self._model, in_chars, out_chars) if not err else 0.0
                try:
                    self._tracer.record_span(
                        name=f"{self._role}:{self._model}", agent=self._role,
                        model=self._model, latency_ms=round(latency_ms, 1),
                        cost_usd=cost, cache_hit=False, error=err,
                    )
                except Exception:
                    log.debug("span record failed", exc_info=True)

        return wrapped


class TracingLLMProvider:
    """Implements ports.LLMProvider by wrapping an inner provider's adapters."""

    def __init__(self, inner, tracer) -> None:
        self._inner = inner
        self._tracer = tracer

    def _wrap(self, factory_name: str, model: str):
        adapter = getattr(self._inner, factory_name)(model)
        return _TracingProxy(adapter, self._tracer, _ROLES[factory_name], model)

    def profile_extractor(self, model: str):
        return self._wrap("profile_extractor", model)

    def jd_extractor(self, model: str):
        return self._wrap("jd_extractor", model)

    def technical(self, model: str):
        return self._wrap("technical", model)

    def risk(self, model: str):
        return self._wrap("risk", model)

    def hr(self, model: str):
        return self._wrap("hr", model)

    def synthesizer(self, model: str):
        return self._wrap("synthesizer", model)

    def coverage_auditor(self, model: str):
        return self._wrap("coverage_auditor", model)

    def resume_optimizer(self, model: str):
        return self._wrap("resume_optimizer", model)

    def cover_letter_writer(self, model: str):
        return self._wrap("cover_letter_writer", model)

    def github_selector(self, model: str):
        return self._wrap("github_selector", model)

    def linkedin_selector(self, model: str):
        return self._wrap("linkedin_selector", model)

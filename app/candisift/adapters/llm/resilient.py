"""Resilience decorator for any LLMProvider — top-priority hardening.

Wraps every persona call with three independent safety nets:
  1. timeout   — a hung HTTP call can't wedge the worker thread forever.
  2. retry     — transient errors (rate limit, 5xx, blip) get exponential backoff.
  3. circuit breaker — after N consecutive failures for a (role, model), the
     circuit opens and calls fast-fail for a cooldown, so we stop hammering a
     down model and let the durable queue retry the whole task later.

On exhaustion it raises LLMUnavailable; the durable worker catches it and re-queues
the task (at-least-once), so nothing is lost. This composes with the queue's own
retry/lease — defense in depth.

It implements ports.LLMProvider by returning guarded proxies, so the application
is unaware resilience exists (it just sees the port). LSP-clean drop-in.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

log = logging.getLogger("candisift.resilient")


class LLMUnavailable(RuntimeError):
    """All retries exhausted or circuit open — the task should be retried later."""


class _CircuitBreaker:
    """Per-(role,model) failure counter + cooldown gate. Shared across worker threads,
    so every read-modify-write of the counters is guarded by a lock — otherwise two
    concurrent failures race on `_fails[key] += 1`, undercount, and the breaker may
    never reach its threshold (it would never open, defeating the whole point)."""

    def __init__(self, threshold: int, cooldown_s: float) -> None:
        self._threshold = threshold
        self._cooldown = cooldown_s
        self._fails: dict[str, int] = {}
        self._open_until: dict[str, float] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        with self._lock:
            until = self._open_until.get(key, 0.0)
        if until and time.monotonic() < until:
            raise LLMUnavailable(f"circuit open for {key}")

    def record_success(self, key: str) -> None:
        with self._lock:
            self._fails.pop(key, None)
            self._open_until.pop(key, None)

    def record_failure(self, key: str) -> None:
        with self._lock:
            n = self._fails.get(key, 0) + 1
            self._fails[key] = n
            if n >= self._threshold:
                self._open_until[key] = time.monotonic() + self._cooldown
                log.warning("circuit OPEN for %s after %d failures", key, n)


class ResiliencePolicy:
    def __init__(self, *, timeout_s: float = 90.0, max_retries: int = 2,
                 backoff_base: float = 1.0, breaker_threshold: int = 5,
                 breaker_cooldown_s: float = 30.0, sleep=time.sleep) -> None:
        self._timeout = timeout_s
        self._max_retries = max_retries
        self._backoff = backoff_base
        self._breaker = _CircuitBreaker(breaker_threshold, breaker_cooldown_s)
        self._sleep = sleep
        self._pool = ThreadPoolExecutor(max_workers=16, thread_name_prefix="llm")

    def close(self) -> None:
        """Release the worker pool. Safe to call once at shutdown; the singleton
        provider lives for the whole process so this matters mainly for per-tenant /
        per-request instances that would otherwise leak 16 threads each."""
        self._pool.shutdown(wait=False, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def run(self, key: str, fn):
        self._breaker.check(key)
        last: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                # ponytail: futures timeout can't cancel the underlying sync HTTP call,
                # so a timed-out call leaves its pool thread running until it returns
                # (a slow zombie). The breaker bounds the blast radius — after
                # `breaker_threshold` consecutive timeouts the circuit opens and we
                # fast-fail without submitting, so zombies can't exhaust the pool.
                # Upgrade path if this bites: pass a hard request timeout to the SDK
                # client (anthropic/openai `timeout=`) so the call actually aborts.
                result = self._pool.submit(fn).result(timeout=self._timeout)
                self._breaker.record_success(key)
                return result
            except FutureTimeout:
                last = LLMUnavailable(f"{key} timed out after {self._timeout}s")
                self._breaker.record_failure(key)
            except Exception as e:  # noqa: BLE001
                last = e
                self._breaker.record_failure(key)
            if attempt < self._max_retries:
                self._sleep(self._backoff * (2 ** attempt))
        log.warning("%s failed after %d attempts: %s", key, self._max_retries + 1, last)
        raise LLMUnavailable(str(last)) from last


class _Guarded:
    """Proxies one persona adapter, applying the policy to each method call."""

    def __init__(self, inner, policy: ResiliencePolicy, label: str) -> None:
        self._inner = inner
        self._policy = policy
        self._label = label

    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr

        def wrapped(*args, **kwargs):
            return self._policy.run(self._label, lambda: attr(*args, **kwargs))

        return wrapped


class ResilientLLMProvider:
    """Implements ports.LLMProvider by wrapping an inner provider's adapters."""

    def __init__(self, inner, policy: ResiliencePolicy | None = None) -> None:
        self._inner = inner
        self._policy = policy or ResiliencePolicy()

    def profile_extractor(self, model: str):
        return _Guarded(self._inner.profile_extractor(model), self._policy, f"profile:{model}")

    def jd_extractor(self, model: str):
        return _Guarded(self._inner.jd_extractor(model), self._policy, f"jd:{model}")

    def technical(self, model: str):
        return _Guarded(self._inner.technical(model), self._policy, f"tech:{model}")

    def risk(self, model: str):
        return _Guarded(self._inner.risk(model), self._policy, f"risk:{model}")

    def hr(self, model: str):
        return _Guarded(self._inner.hr(model), self._policy, f"hr:{model}")

    def synthesizer(self, model: str):
        return _Guarded(self._inner.synthesizer(model), self._policy, f"synth:{model}")

    def coverage_auditor(self, model: str):
        return _Guarded(self._inner.coverage_auditor(model), self._policy, f"coverage:{model}")

    def resume_optimizer(self, model: str):
        return _Guarded(self._inner.resume_optimizer(model), self._policy, f"optimize:{model}")

    def cover_letter_writer(self, model: str):
        return _Guarded(self._inner.cover_letter_writer(model), self._policy, f"coverletter:{model}")

    def github_selector(self, model: str):
        return _Guarded(self._inner.github_selector(model), self._policy, f"github:{model}")

    def linkedin_selector(self, model: str):
        return _Guarded(self._inner.linkedin_selector(model), self._policy, f"linkedin:{model}")

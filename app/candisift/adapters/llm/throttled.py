"""ThrottledLLMProvider — caps LLM call concurrency and rate.

Distinct from the HTTP rate-limit middleware (which protects the web tier). This
protects the *provider*: a semaphore bounds simultaneous in-flight calls and a
token bucket bounds calls/minute, so a large confirmed batch can't burst past
the model's limits or blow the cost budget. Backpressure is a blocking wait on
the worker thread; the durable queue holds the rest.

Implements ports.LLMProvider by returning proxies that acquire the throttle
around every method call — composes with the resilient/tracing decorators.
"""
from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger("candisift.throttle")


class _TokenBucket:
    """Classic token bucket. capacity = burst, refills `rate_per_min` tokens/min."""

    def __init__(self, rate_per_min: int, sleep=time.sleep) -> None:
        self._rate_per_sec = max(rate_per_min, 1) / 60.0
        self._capacity = float(max(rate_per_min, 1))
        self._tokens = self._capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()
        self._sleep = sleep

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self._capacity,
                                   self._tokens + (now - self._last) * self._rate_per_sec)
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                deficit = 1.0 - self._tokens
                wait = deficit / self._rate_per_sec
            self._sleep(min(wait, 1.0))


class LLMThrottle:
    """Concurrency semaphore + rate bucket, usable as a context manager."""

    def __init__(self, max_concurrency: int = 4, rate_per_min: int = 60, sleep=time.sleep) -> None:
        self._sema = threading.BoundedSemaphore(max(1, int(max_concurrency)))
        self._bucket = _TokenBucket(int(rate_per_min), sleep=sleep)

    def __enter__(self):
        self._sema.acquire()
        try:
            self._bucket.acquire()
        except BaseException:
            self._sema.release()
            raise
        return self

    def __exit__(self, *exc) -> None:
        self._sema.release()


class _ThrottledProxy:
    def __init__(self, inner, throttle: LLMThrottle) -> None:
        self._inner = inner
        self._throttle = throttle

    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr

        def wrapped(*args, **kwargs):
            with self._throttle:
                return attr(*args, **kwargs)

        return wrapped


class ThrottledLLMProvider:
    """Implements ports.LLMProvider by wrapping an inner provider's adapters."""

    def __init__(self, inner, throttle: LLMThrottle | None = None,
                 *, max_concurrency: int = 4, rate_per_min: int = 60) -> None:
        self._inner = inner
        self._throttle = throttle or LLMThrottle(max_concurrency, rate_per_min)

    def _wrap(self, adapter):
        return _ThrottledProxy(adapter, self._throttle)

    def profile_extractor(self, model: str):
        return self._wrap(self._inner.profile_extractor(model))

    def jd_extractor(self, model: str):
        return self._wrap(self._inner.jd_extractor(model))

    def technical(self, model: str):
        return self._wrap(self._inner.technical(model))

    def risk(self, model: str):
        return self._wrap(self._inner.risk(model))

    def hr(self, model: str):
        return self._wrap(self._inner.hr(model))

    def synthesizer(self, model: str):
        return self._wrap(self._inner.synthesizer(model))

    def coverage_auditor(self, model: str):
        return self._wrap(self._inner.coverage_auditor(model))

    def resume_optimizer(self, model: str):
        return self._wrap(self._inner.resume_optimizer(model))

    def cover_letter_writer(self, model: str):
        return self._wrap(self._inner.cover_letter_writer(model))

    def github_selector(self, model: str):
        return self._wrap(self._inner.github_selector(model))

    def linkedin_selector(self, model: str):
        return self._wrap(self._inner.linkedin_selector(model))

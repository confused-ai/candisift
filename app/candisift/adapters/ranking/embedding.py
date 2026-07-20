"""Ranker adapter: semantic similarity via local sentence-transformer embeddings.

Catches transferable skills and terminology differences the lexical TokenCosineRanker
misses ("k8s" vs "Kubernetes", "built ML models" vs "machine learning"). Implements
the Ranker port, so it drops into the composition root in one line.

Hybrid by construction: the embedding model is heavy and OPTIONAL. If
sentence-transformers (or the model download) is unavailable, it falls back to the
injected lexical ranker — the app always boots, and tests/CI run with no extra deps.

ponytail: local MiniLM, offline, $0/call. Encoder is injectable (tests pass a fake);
swap it for a larger model or an embeddings API for higher recall — port + fallback
stay the same.
"""
from __future__ import annotations

import logging
import math
import os
import threading
import time

from app.candisift.domain.models import CandidateProfile, JDSpec
from app.candisift.adapters.ranking.token_cosine import TokenCosineRanker

log = logging.getLogger("candisift.ranker")

_MODEL = "all-MiniLM-L6-v2"   # ~80MB, strong quality/size tradeoff
# SentenceTransformer(...) phones HuggingFace Hub for metadata on construction; with
# no network (or a slow one) that BLOCKS the worker thread indefinitely on an SSL read.
# We load it in a side thread with a hard timeout and fall back to the lexical ranker
# if it doesn't finish — and retry later (the side thread keeps warming the on-disk
# cache, so a subsequent attempt loads instantly).
_LOAD_TIMEOUT_S = 20.0
_RETRY_COOLDOWN_S = 60.0


def _profile_text(p: CandidateProfile) -> str:
    return " ".join([s.name for s in p.skills] + p.titles + [p.summary]).strip()


def _jd_text(jd: JDSpec) -> str:
    return " ".join(jd.must_have_skills + jd.nice_to_have_skills + [jd.title]).strip()


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class EmbeddingRanker:
    """Implements ports.Ranker. `encoder` is any object with `.encode(str) -> vector`
    (a sentence_transformers.SentenceTransformer, or a fake in tests). Left None, the
    model is lazy-loaded on first score; if that fails, scoring uses the fallback."""

    def __init__(self, *, fallback=None, model_name: str = _MODEL, encoder=None,
                 load_timeout_s: float = _LOAD_TIMEOUT_S) -> None:
        self._fallback = fallback or TokenCosineRanker()
        self._model_name = model_name
        self._encoder = encoder
        self._load_timeout = load_timeout_s
        self._next_retry = 0.0            # monotonic time before which we don't retry load
        self._lock = threading.Lock()
        self._jd_cache: dict[str, list[float]] = {}   # JD text -> embedding (per-instance)

    def _ensure_encoder(self) -> None:
        if self._encoder is not None:
            return
        with self._lock:
            if self._encoder is not None:
                return
            if time.monotonic() < self._next_retry:
                return                    # in cooldown -> lexical fallback for now
            result: dict = {}

            def _load() -> None:
                try:
                    from sentence_transformers import SentenceTransformer
                    # device is pinned, NOT auto-selected: sentence-transformers picks
                    # MPS on Apple Silicon, and a torch MPS context built in this side
                    # thread then used from a uvicorn worker thread SIGBUSes on the first
                    # forward pass. MiniLM-L6 is ~2ms/text on CPU — MPS buys nothing at
                    # this size. Override with CANDISIFT_EMBED_DEVICE=mps|cuda.
                    device = os.environ.get("CANDISIFT_EMBED_DEVICE", "cpu")
                    result["enc"] = SentenceTransformer(self._model_name, device=device)
                except Exception as e:    # not installed / offline fetch failed
                    result["err"] = e

            t = threading.Thread(target=_load, name="st-load", daemon=True)
            t.start()
            t.join(self._load_timeout)
            if t.is_alive():
                # still downloading/blocked — don't wedge the screen; retry later
                # (this load thread keeps warming the on-disk cache meanwhile).
                self._next_retry = time.monotonic() + _RETRY_COOLDOWN_S
                log.warning("semantic ranker: load exceeded %ss — lexical fallback "
                            "(will retry); a model download may be in progress",
                            self._load_timeout)
                return
            if "enc" in result:
                self._encoder = result["enc"]
                log.info("semantic ranker: loaded %s", self._model_name)
            else:
                self._next_retry = time.monotonic() + _RETRY_COOLDOWN_S
                log.warning("embeddings unavailable (%s) — lexical fallback (will retry)",
                            type(result.get("err")).__name__)

    def _encode_jd(self, jd: JDSpec) -> list[float]:
        key = _jd_text(jd)
        v = self._jd_cache.get(key)
        if v is None:
            v = list(self._encoder.encode(key))     # re-encoding the JD per candidate
            if len(self._jd_cache) > 256:           # was the dominant cost at batch scale
                self._jd_cache.clear()
            self._jd_cache[key] = v
        return v

    def score(self, profile: CandidateProfile, jd: JDSpec) -> float:
        self._ensure_encoder()
        if self._encoder is None:
            return self._fallback.score(profile, jd)
        try:
            a = list(self._encoder.encode(_profile_text(profile)))
            b = self._encode_jd(jd)                  # cached across a batch's candidates
            # embeddings can go slightly negative -> clamp to [0,1]
            # float() coerces numpy float32 -> Python float so the score stays JSON
            # serializable downstream (audit log, result row).
            return float(round(max(0.0, min(1.0, _cosine(a, b))), 4))
        except Exception:
            log.warning("embedding score failed; lexical fallback", exc_info=True)
            return self._fallback.score(profile, jd)


def _demo() -> None:
    from app.candisift.domain.models import SkillItem

    class _Fake:                       # deterministic 3-dim "embedding" over keywords
        def encode(self, text: str):
            t = text.lower()
            return [float(t.count("python")), float(t.count("kubernetes")), float(t.count("java"))]

    r = EmbeddingRanker(encoder=_Fake())
    prof = CandidateProfile(skills=[SkillItem(name="python"), SkillItem(name="kubernetes")], titles=[], summary="")
    jd = JDSpec(title="Eng", must_have_skills=["python", "kubernetes"])
    assert r.score(prof, jd) > 0.9, r.score(prof, jd)
    jd2 = JDSpec(title="Eng", must_have_skills=["java"])
    assert r.score(prof, jd2) == 0.0, r.score(prof, jd2)        # unrelated -> low
    r2 = EmbeddingRanker(encoder=None); r2._loaded = True; r2._encoder = None
    assert isinstance(r2.score(prof, jd), float)               # no encoder -> lexical fallback
    print("embedding ranker demo ok")


if __name__ == "__main__":
    _demo()

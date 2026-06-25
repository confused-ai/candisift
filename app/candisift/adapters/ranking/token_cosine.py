"""Ranker adapter: bag-of-canonical-tokens cosine. Implements the Ranker port.

ponytail: real embeddings (sentence-transformers / an embeddings API) catch
synonyms this can't. Because ranking is a port, swapping to an EmbeddingRanker is
a new adapter + one line in the composition root — the application never changes.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from app.candisift.domain.models import CandidateProfile, JDSpec
from app.candisift.domain.services import canon

# \w is Unicode-aware in Python 3, so non-Latin skill names (e.g. accented or
# Cyrillic terms) are tokenized instead of silently dropped by an [a-z0-9] class.
# +#. are kept so "c++" / "c#" / ".net" survive as single tokens.
_WORD = re.compile(r"[\w+#.]+")


def _vec(text: str) -> Counter:
    return Counter(canon(w) for w in _WORD.findall(text.lower()))


def _cosine(a: Counter, b: Counter) -> float:
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


class TokenCosineRanker:
    def score(self, profile: CandidateProfile, jd: JDSpec) -> float:
        prof_text = " ".join(
            [s.name for s in profile.skills] + profile.titles + [profile.summary]
        )
        jd_text = " ".join(jd.must_have_skills + jd.nice_to_have_skills + [jd.title])
        return round(_cosine(_vec(prof_text), _vec(jd_text)), 4)

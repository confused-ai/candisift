"""Near-duplicate / resume-farming detection — beyond exact dedup.

Exact dedup (services.dedup_key) catches the same email/phone. This catches the
*same content* re-skinned: one person submitting tweaked resumes, or an agency
spraying near-identical CVs under different identities (a real IT-staffing fraud
signal). Token-set Jaccard over canonical skills + title words.

Pure + deterministic. ponytail: O(n) scan per ingest. For large corpora swap in
MinHash + LSH (datasketch) or a vector index — same fingerprint(), faster lookup.
"""
from __future__ import annotations

import re

from .models import CandidateProfile
from .services import canon

_WORD = re.compile(r"[a-z0-9+#.]+")
THRESHOLD = 0.9
# A fingerprint must carry at least this many distinct tokens to be comparable.
# Two near-empty profiles (failed extraction, one-skill stubs) would otherwise score
# Jaccard 1.0 and brand distinct, legitimate applicants as resume-farming fraud.
_MIN_FINGERPRINT = 5
# ...and they must actually SHARE this many tokens, so a high ratio over a tiny set
# (2-of-2 == 1.0) can't trip the flag on its own.
_MIN_SHARED = 5


def fingerprint(profile: CandidateProfile) -> set[str]:
    toks = {canon(s.name) for s in profile.skills}
    for t in profile.titles:
        toks |= set(_WORD.findall(t.lower()))
    toks |= set(_WORD.findall(profile.summary.lower()))
    return {t for t in toks if len(t) > 2}


def jaccard(a: set[str], b: set[str]) -> float:
    # both-empty is NOT "identical" for fraud purposes — two profiles with no
    # extractable content are unknown, not duplicates. Return 0.0, never 1.0.
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def find_near_duplicate(profile: CandidateProfile,
                        existing: list[tuple[str, CandidateProfile]],
                        threshold: float = THRESHOLD) -> tuple[str, float] | None:
    """Return (candidate_id, similarity) of the closest near-duplicate, or None.

    Sparse-profile guard: a fingerprint below _MIN_FINGERPRINT distinct tokens is too
    thin to judge (a one-skill stub matches every other stub at 1.0), so we skip it
    rather than emit a false fraud flag; and a real match must also share at least
    _MIN_SHARED tokens, not just clear the ratio over a tiny set."""
    fp = fingerprint(profile)
    if len(fp) < _MIN_FINGERPRINT:
        return None
    best: tuple[str, float] | None = None
    for cid, other in existing:
        ofp = fingerprint(other)
        if len(ofp) < _MIN_FINGERPRINT or len(fp & ofp) < _MIN_SHARED:
            continue
        sim = jaccard(fp, ofp)
        if sim >= threshold and (best is None or sim > best[1]):
            best = (cid, round(sim, 3))
    return best

"""Curated, offline industry benchmarks — the 'industry research' the evaluators
reference so judgments use market norms, not just the JD text.

Two uses:
  - adjacency: which skills commonly travel together, so a candidate gets credit
    for transferable/adjacent experience when the exact must-have term is absent
    (semantic, terminology-agnostic screening).
  - seniority bands: typical years for a level, to sanity-check claims.

ponytail: hand-maintained dict — deterministic, $0, no network. Coverage is
deliberately modest; swap benchmark_note() for a sourced dataset or a research
service if breadth matters. The evaluators degrade gracefully on a miss (empty
note → they just reason from the JD as before).
"""
from __future__ import annotations

from app.candisift.domain.services import canon

# canonical-skill -> commonly co-occurring / transferable skills (industry norm)
SKILL_ADJACENCY: dict[str, list[str]] = {
    "python": ["django", "fastapi", "flask", "pandas", "numpy"],
    "java": ["spring", "kotlin", "jvm", "hibernate"],
    "javascript": ["typescript", "nodejs", "react", "vue"],
    "typescript": ["javascript", "react", "nodejs", "nextjs"],
    "go": ["kubernetes", "docker", "grpc", "microservices"],
    "rust": ["c++", "systems programming", "wasm"],
    "react": ["typescript", "javascript", "nextjs", "redux"],
    "nodejs": ["javascript", "typescript", "express", "fastify"],
    "django": ["python", "postgresql", "celery", "drf"],
    "fastapi": ["python", "pydantic", "uvicorn", "async"],
    "spring": ["java", "hibernate", "kotlin", "microservices"],
    "kubernetes": ["docker", "terraform", "helm", "aws", "gcp"],
    "docker": ["kubernetes", "ci/cd", "linux", "containers"],
    "terraform": ["aws", "gcp", "azure", "kubernetes"],
    "aws": ["terraform", "kubernetes", "lambda", "s3", "ec2"],
    "gcp": ["kubernetes", "bigquery", "terraform"],
    "azure": ["terraform", "kubernetes", "dotnet"],
    "postgresql": ["sql", "mysql", "database design", "redis"],
    "mysql": ["sql", "postgresql", "database design"],
    "redis": ["caching", "postgresql", "kafka"],
    "kafka": ["event streaming", "redis", "rabbitmq", "spark"],
    "graphql": ["rest", "apollo", "typescript"],
    "pytorch": ["tensorflow", "numpy", "cuda", "machine learning"],
    "tensorflow": ["pytorch", "keras", "machine learning"],
    "machine learning": ["pytorch", "tensorflow", "pandas", "statistics"],
    "sql": ["postgresql", "mysql", "data modeling"],
}

# seniority label -> (min_years, max_years) typical industry band
SENIORITY_YEARS: dict[str, tuple[int, int]] = {
    "junior": (0, 2), "mid": (2, 5), "senior": (5, 8), "staff": (8, 99),
}


def adjacent(skill: str) -> list[str]:
    """Industry-adjacent / transferable skills for one skill (canonicalized)."""
    return SKILL_ADJACENCY.get(canon(skill), [])


def benchmark_note(spec) -> str:
    """A short industry-norms note for a JD's must-haves, injected into the
    evaluator prompt so it can credit transferable experience with grounding.
    Empty string when nothing in the curated set matches."""
    seen: list[str] = []
    for s in (spec.must_have_skills or [])[:6]:
        adj = adjacent(s)
        if adj:
            seen.append(f"{s} commonly pairs with {', '.join(adj[:4])}")
    if not seen:
        return ""
    return ("INDUSTRY NORMS (credit transferable/adjacent experience when an exact "
            "must-have term is absent): " + "; ".join(seen) + ".")


def _demo() -> None:
    from app.candisift.domain.models import JDSpec
    note = benchmark_note(JDSpec(title="x", must_have_skills=["kubernetes", "python"]))
    assert "kubernetes" in note and "docker" in note, note
    assert benchmark_note(JDSpec(title="x", must_have_skills=["basket weaving"])) == ""
    assert "docker" in adjacent("Kubernetes")
    print("benchmarks demo ok")


if __name__ == "__main__":
    _demo()

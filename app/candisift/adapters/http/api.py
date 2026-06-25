"""JSON API (driving adapter). Thin: validate, call the use-case, serialize.

No business logic — it all lives in application/domain. This router only
translates HTTP <-> use-case calls, and enforces the upload guardrails.
"""
from __future__ import annotations

from collections import Counter

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.candisift import pricing
from app.candisift.application.screening_service import NotFoundError
from app.candisift.domain import ats_readability
from app.candisift.domain.models import TaskStatus
from .container import Container
from .deps import get_container
from .security import check_size, validate_uploads

router = APIRouter()


class CreateJobIn(BaseModel):
    jd_text: str
    persona_model: str = "auto"
    synth_model: str = "auto"


class OptimizeIn(BaseModel):
    resume_text: str
    model: str = "auto"


class DecisionIn(BaseModel):
    decision: str  # accepted | rejected | pending


class ConfirmIn(BaseModel):
    task_ids: list[str]


def _validate_models(settings, *models: str) -> None:
    for m in models:
        if not pricing.is_known_model(m):
            raise HTTPException(400, f"unknown model: {m}")
        # in real-LLM mode, refuse an explicit model whose provider key is missing
        # rather than silently degrading to the offline stub.
        if m != "auto" and settings.has_llm and not settings.has_key_for(m):
            raise HTTPException(400, f"{m} requires {pricing.provider_env_var(m)} to be set")


# ---- catalog --------------------------------------------------------------

@router.get("/models")
def list_models():
    return pricing.catalog()


# ---- jobs -----------------------------------------------------------------

@router.post("/jobs")
def create_job(body: CreateJobIn, c: Container = Depends(get_container)):
    _validate_models(c.settings, body.persona_model, body.synth_model)
    job = c.service.create_job(body.jd_text, body.persona_model, body.synth_model)
    return {"id": job.id, "title": job.title, "spec": job.spec.model_dump(),
            "persona_model": job.persona_model, "synth_model": job.synth_model}


@router.get("/jobs")
def list_jobs(c: Container = Depends(get_container)):
    return [{"id": j.id, "title": j.title, "persona_model": j.persona_model,
             "synth_model": j.synth_model, "created_at": j.created_at} for j in c.jobs.list_all()]


@router.get("/jobs/{job_id}")
def get_job(job_id: str, c: Container = Depends(get_container)):
    job = c.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"job {job_id}")
    return job.model_dump()


# ---- estimate-first upload ------------------------------------------------

@router.post("/jobs/{job_id}/upload")
async def upload_resumes(
    job_id: str,
    files: list[UploadFile] = File(...),
    c: Container = Depends(get_container),
):
    """Stage a multi-resume batch and return the cost estimate. Nothing is screened
    (no LLM spend) until POST /jobs/{job_id}/confirm releases the staged tasks."""
    job = c.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"job {job_id}")
    s = c.settings
    validate_uploads(files, max_files=s.max_files_per_batch,
                     max_file_bytes=s.max_file_mb * 1024 * 1024)
    payloads = []
    for f in files:
        content = await f.read()
        check_size(content, f.filename or "resume", s.max_file_mb * 1024 * 1024)
        payloads.append((content, f.filename or "resume"))

    task_ids = c.service.enqueue_batch(job_id, payloads, staged=True)
    estimate = pricing.estimate_batch(
        len(payloads),
        pricing.resolve_model(job.persona_model, s.persona_model),
        pricing.resolve_model(job.synth_model, s.synth_model),
        hr_eval=s.hr_eval_enabled,
        coverage_audit=s.coverage_audit_enabled,
    )
    return {"job_id": job_id, "staged": len(task_ids), "task_ids": task_ids, "estimate": estimate}


@router.post("/jobs/{job_id}/optimize")
def api_optimize_resume(job_id: str, body: OptimizeIn, c: Container = Depends(get_container)):
    """Keyword-gap analysis + ATS-optimised resume rewrite. Returns ResumeOptimizationResult."""
    if c.jobs.get(job_id) is None:
        raise HTTPException(404, f"job {job_id}")
    if body.model != "auto":
        _validate_models(c.settings, body.model)
    # let failures propagate to the app's central _unhandled handler: it logs with the
    # request id and returns a generic 500 — never leak provider/quota/path detail to
    # the client (the old `str(e)` did exactly that).
    result = c.service.optimize_resume(job_id, body.resume_text, body.model)
    return result.model_dump()


class CoverLetterIn(BaseModel):
    resume_text: str
    tone: str = "professional"
    model: str = "auto"


@router.post("/jobs/{job_id}/cover-letter")
def api_cover_letter(job_id: str, body: CoverLetterIn, c: Container = Depends(get_container)):
    if c.jobs.get(job_id) is None:
        raise HTTPException(404, f"job {job_id}")
    if body.model != "auto":
        _validate_models(c.settings, body.model)
    # propagate to the central _unhandled handler (logs + generic 500); don't leak str(e).
    result = c.service.generate_cover_letter(job_id, body.resume_text, body.tone, body.model)
    return result.model_dump()


@router.post("/jobs/{job_id}/confirm")
def confirm_batch(job_id: str, body: ConfirmIn, c: Container = Depends(get_container)):
    released = c.service.confirm_batch(body.task_ids)
    return {"job_id": job_id, "released": released}


# ---- results & review -----------------------------------------------------

@router.get("/jobs/{job_id}/results")
def job_results(job_id: str, c: Container = Depends(get_container)):
    if c.jobs.get(job_id) is None:
        raise HTTPException(404, f"job {job_id}")
    return [r.model_dump() for r in c.results.list_for_job(job_id)]


@router.get("/results/{result_id}")
def result_breakdown(result_id: str, c: Container = Depends(get_container)):
    r = c.results.get(result_id)
    if r is None:
        raise HTTPException(404, f"result {result_id}")
    cand = c.candidates.get(r.candidate_id)
    job = c.jobs.get(r.job_id)
    readability = ats_readability.score(cand.profile, job.spec) if (cand and job) else None
    return {
        "result": r.model_dump(),
        "candidate": cand.model_dump() if cand else None,
        "ats_readability": readability,
        "near_duplicate_of": cand.near_duplicate_of if cand else "",
        "duplicate_similarity": cand.duplicate_similarity if cand else 0.0,
    }


@router.get("/tasks")
def list_tasks(status: str = "failed", c: Container = Depends(get_container)):
    """Inspect the durable queue (default: the dead-letter / failed tasks)."""
    try:
        st = TaskStatus(status)
    except ValueError:
        raise HTTPException(400, f"unknown status: {status}")
    return [t.model_dump() for t in c.queue.list_by_status(st)]


@router.post("/tasks/{task_id}/requeue")
def requeue_task(task_id: str, c: Container = Depends(get_container)):
    if not c.queue.requeue(task_id):
        raise HTTPException(400, "task not found or not in failed state")
    c.audit.record("task.requeued", task_id=task_id)
    return {"task_id": task_id, "requeued": True}


@router.post("/results/{result_id}/decision")
def set_decision(result_id: str, body: DecisionIn, c: Container = Depends(get_container)):
    try:
        c.results.set_decision(result_id, body.decision)
        c.audit.record("decision.set", result_id=result_id, decision=body.decision)
    except LookupError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"result_id": result_id, "decision": body.decision}


# ---- bias audit & ops -----------------------------------------------------

@router.get("/jobs/{job_id}/bias-audit")
def bias_audit(job_id: str, c: Container = Depends(get_container)):
    results = c.results.list_for_job(job_id)
    if not results:
        raise HTTPException(404, f"no results for job {job_id}")
    total = len(results)
    passed = sum(1 for r in results if r.passed_hard_filters)
    rec = Counter(r.synthesis.recommendation.value for r in results if r.synthesis)
    return {
        "job_id": job_id, "total": total,
        "hard_filter_pass_rate": round(passed / total, 3),
        "recommendations": dict(rec),
        "shortlist_rate": round(rec.get("shortlist", 0) / total, 3),
        "note": "screened on skills/experience only; PII stripped pre-evaluation. "
                "Attach consented cohort labels per candidate for true disparate-impact analysis.",
    }


@router.get("/queue")
def queue_stats(c: Container = Depends(get_container)):
    return c.queue.stats()


# ---- agents, runs (sessions), traces --------------------------------------

@router.get("/agents")
def list_agents(c: Container = Depends(get_container)):
    """The agent roster joined with live run stats (calls, cache-hit, cost)."""
    from app.candisift import agent_catalog
    stats = {s["agent"]: s for s in c.tracer.agent_stats()}
    roster = agent_catalog.agents(c.settings.persona_model, c.settings.synth_model)
    for a in roster:
        a["stats"] = stats.get(a["role"], {})
    return roster


@router.get("/runs")
def list_runs(limit: int = 100, candidate_id: str = "", job_id: str = "",
              c: Container = Depends(get_container)):
    return [r.model_dump() for r in c.tracer.list_runs(limit, candidate_id, job_id)]


@router.get("/runs/{trace_id}")
def get_run(trace_id: str, c: Container = Depends(get_container)):
    run = c.tracer.get_run(trace_id)
    if run is None:
        raise HTTPException(404, f"run {trace_id}")
    return run.model_dump()


# ---- instant match (no LLM) -----------------------------------------------

class MatchIn(BaseModel):
    resume_text: str
    jd_text: str


@router.post("/match")
def api_match(body: MatchIn):
    """Deterministic keyword match + resume quality analysis. Zero LLM cost."""
    import re as _re
    from app.candisift.domain import resume_analysis
    from app.candisift.adapters.llm.stub import StubJobSpecExtractor, StubProfileExtractor
    profile = StubProfileExtractor().extract(body.resume_text)
    jd = StubJobSpecExtractor().extract(body.jd_text)
    readability = ats_readability.score(profile, jd)
    resume_lower = body.resume_text.lower()
    gaps = [
        {"keyword": kw, "present": bool(_re.search(rf"\b{_re.escape(kw.lower())}\b", resume_lower))}
        for kw in jd.must_have_skills
    ]
    present = sum(1 for g in gaps if g["present"])
    return {
        "ats_score": readability["score"],
        "checks": readability["checks"],
        "keyword_gaps": gaps,
        "matched": present,
        "total_keywords": len(gaps),
        "coverage_pct": round(present / len(gaps) * 100) if gaps else 0,
        "jd_title": jd.title,
        "jd_min_years": jd.min_years,
        "resume_analysis": resume_analysis.full_analysis(body.resume_text),
    }


class AnalyzeIn(BaseModel):
    resume_text: str


@router.post("/resume/analyze")
def api_analyze(body: AnalyzeIn):
    """Standalone resume quality analysis: action verbs, quantification, sections, length, formatting."""
    from app.candisift.domain import resume_analysis
    return resume_analysis.full_analysis(body.resume_text)


# ---- candidates ---------------------------------------------------------------

@router.get("/candidates/{candidate_id}")
def get_candidate(candidate_id: str, c: Container = Depends(get_container)):
    cand = c.candidates.get(candidate_id)
    if cand is None:
        raise HTTPException(404, f"candidate {candidate_id}")
    return cand.model_dump()


@router.get("/jobs/{job_id}/candidates")
def list_job_candidates(job_id: str, c: Container = Depends(get_container)):
    if c.jobs.get(job_id) is None:
        raise HTTPException(404, f"job {job_id}")
    results = c.results.list_for_job(job_id)
    out = []
    for r in results:
        cand = c.candidates.get(r.candidate_id)
        if cand:
            out.append({
                "candidate_id": cand.id,
                "name": cand.profile.name or "",
                "source_filename": cand.source_filename,
                "result_id": r.id,
                "decision": r.decision.value,
                "passed_hard_filters": r.passed_hard_filters,
                "overall_fit": r.synthesis.overall_fit if r.synthesis else None,
                "recommendation": r.synthesis.recommendation.value if r.synthesis else None,
                "requires_human_review": r.requires_human_review,
                "bias_flags": r.bias_flags,
            })
    return out


# ---- analytics ---------------------------------------------------------------

@router.get("/analytics")
def analytics(c: Container = Depends(get_container)):
    """Full KPI summary — same data as the /dashboard UI view."""
    runs = c.tracer.list_runs(limit=2000)
    by_model = c.tracer.spend_by_model()
    stages = c.tracer.agent_stats()
    results = c.results.list_all()
    n_jobs = len(c.jobs.list_all())
    n_candidates = len(c.candidates.list_all())

    total_spend = round(sum(r.total_cost_usd for r in runs), 4)
    screen_runs = [r for r in runs if r.kind == "screen"]
    done_screens = [r for r in screen_runs if r.status == "done"]
    cached = [r for r in runs if r.cache_hit]
    real_done = [r for r in done_screens if not r.cache_hit]
    avg_screen_cost = (sum(r.total_cost_usd for r in real_done) / len(real_done)) if real_done else 0.0
    cache_rate = round(len(cached) / len(runs) * 100, 1) if runs else 0.0
    avg_latency_ms = round(sum(r.total_ms for r in real_done) / len(real_done)) if real_done else 0
    errors = sum(1 for r in runs if r.status == "error")
    passed = [r for r in results if r.passed_hard_filters]
    screened = [r for r in results if r.synthesis]
    recs: dict[str, int] = {"shortlist": 0, "maybe": 0, "reject": 0}
    fits = []
    for r in screened:
        key = r.synthesis.recommendation.value
        recs[key] = recs.get(key, 0) + 1
        fits.append(r.synthesis.overall_fit)
    decisions: dict[str, int] = {"pending": 0, "accepted": 0, "rejected": 0}
    for r in results:
        key = r.decision.value
        decisions[key] = decisions.get(key, 0) + 1
    return {
        "kpis": {
            "spend_usd": total_spend,
            "est_saved_usd": round(avg_screen_cost * len(cached), 4),
            "cache_rate_pct": cache_rate,
            "screened": len(done_screens),
            "candidates": n_candidates,
            "jobs": n_jobs,
            "shortlisted": recs["shortlist"],
            "avg_fit": round(sum(fits) / len(fits), 1) if fits else 0,
            "avg_latency_ms": avg_latency_ms,
            "errors": errors,
            "bias_flagged": sum(1 for r in results if r.bias_flags),
            "held_for_review": sum(1 for r in results if r.requires_human_review),
        },
        "by_model": by_model,
        "agent_stats": stages,
        "recommendations": recs,
        "decisions": decisions,
        "funnel": [
            {"label": "Uploaded", "n": n_candidates},
            {"label": "Passed hard filters", "n": len(passed)},
            {"label": "Screened (LLM)", "n": len(screened)},
            {"label": "Shortlisted", "n": recs["shortlist"]},
        ],
    }


# ---- audit & job progress ----------------------------------------------------

@router.get("/jobs/{job_id}/audit")
def job_audit(job_id: str, limit: int = 200, c: Container = Depends(get_container)):
    if c.jobs.get(job_id) is None:
        raise HTTPException(404, f"job {job_id}")
    return c.audit.events_for_job(job_id, limit)


@router.get("/jobs/{job_id}/progress")
def job_progress(job_id: str, c: Container = Depends(get_container)):
    if c.jobs.get(job_id) is None:
        raise HTTPException(404, f"job {job_id}")
    return c.queue.job_progress(job_id)

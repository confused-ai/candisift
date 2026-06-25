"""Recruiter UI (driving adapter) — server-rendered HTML, no JavaScript.

Flow: dashboard -> create job (pick models) -> upload resumes -> SEE COST ESTIMATE
-> confirm -> results table with status -> drill into a candidate's full breakdown
-> accept/reject (human-in-the-loop). No inline scripts/external assets, so the
strict CSP from SecurityHeadersMiddleware holds.
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.candisift import pricing
from app.candisift.domain import ats_readability
from .container import Container
from .deps import get_container
from .security import check_size, validate_uploads

log = logging.getLogger("candisift.ui")
router = APIRouter()
_templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def _render(request: Request, name: str, **ctx) -> HTMLResponse:
    return _templates.TemplateResponse(request, name, ctx)


async def _extract_upload(c: Container, f: UploadFile) -> str:
    """Validate + size-check one ad-hoc resume upload, then extract its text with the
    container's CONFIGURED parser (OCR page/time caps applied) — off the event loop so
    a slow scanned PDF can't block every concurrent request. The batch path already
    does this; these single-file optimize/match/cover-letter uploads must too, or they
    become an unbounded-type + OCR-resource DoS hole."""
    from starlette.concurrency import run_in_threadpool
    s = c.settings
    validate_uploads([f], max_files=1, max_file_bytes=s.max_file_mb * 1024 * 1024)
    content = await f.read()
    check_size(content, f.filename or "resume", s.max_file_mb * 1024 * 1024)
    return await run_in_threadpool(c.parser.extract, content, f.filename or "resume")


@router.get("/")
def landing():
    """Single product now — send the root straight to the CandiSift dashboard."""
    return RedirectResponse("/ats")


@router.get("/ats", response_class=HTMLResponse)
def dashboard(request: Request, c: Container = Depends(get_container)):
    return _render(request, "dashboard.html",
                   jobs=c.jobs.list_all(), models=pricing.catalog(),
                   queue=c.queue.stats(), llm=c.settings.has_llm)


@router.get("/dashboard", response_class=HTMLResponse)
def metrics_dashboard(request: Request, c: Container = Depends(get_container)):
    """Analytics overview: spend, cache savings, screening funnel, decisions, latency."""
    runs = c.tracer.list_runs(limit=2000)
    by_model = c.tracer.spend_by_model()
    stages = c.tracer.agent_stats()
    results = c.results.list_all()
    n_jobs = len(c.jobs.list_all())
    n_candidates = len(c.candidates.list_all())

    # --- spend / cache / latency from traced runs ---
    total_spend = round(sum(r.total_cost_usd for r in runs), 4)
    screen_runs = [r for r in runs if r.kind == "screen"]
    done_screens = [r for r in screen_runs if r.status == "done"]
    cached = [r for r in runs if r.cache_hit]
    real_done = [r for r in done_screens if not r.cache_hit]
    avg_screen_cost = (sum(r.total_cost_usd for r in real_done) / len(real_done)) if real_done else 0.0
    est_saved = round(avg_screen_cost * len(cached), 4)          # est: avg real-screen cost × cached runs
    cache_rate = round(len(cached) / len(runs) * 100, 1) if runs else 0.0
    avg_latency_ms = round(sum(r.total_ms for r in real_done) / len(real_done)) if real_done else 0
    errors = sum(1 for r in runs if r.status == "error")

    # --- screening funnel + decisions from results ---
    passed = [r for r in results if r.passed_hard_filters]
    screened = [r for r in results if r.synthesis]
    recs = {"shortlist": 0, "maybe": 0, "reject": 0}
    fits = []
    for r in screened:
        recs[r.synthesis.recommendation.value] = recs.get(r.synthesis.recommendation.value, 0) + 1
        fits.append(r.synthesis.overall_fit)
    decisions = {"pending": 0, "accepted": 0, "rejected": 0}
    for r in results:
        decisions[r.decision.value] = decisions.get(r.decision.value, 0) + 1

    funnel = [
        {"label": "Uploaded", "n": n_candidates},
        {"label": "Passed hard filters", "n": len(passed)},
        {"label": "Screened (LLM)", "n": len(screened)},
        {"label": "Shortlisted", "n": recs["shortlist"]},
    ]
    kpis = {
        "spend": total_spend, "saved": est_saved, "cache_rate": cache_rate,
        "screened": len(done_screens), "candidates": n_candidates, "jobs": n_jobs,
        "shortlist": recs["shortlist"], "avg_fit": round(sum(fits) / len(fits), 1) if fits else 0,
        "avg_latency_ms": avg_latency_ms, "errors": errors,
        "bias_flagged": sum(1 for r in results if r.bias_flags),
        "held_for_review": sum(1 for r in results if r.requires_human_review),
    }
    return _render(request, "analytics.html",
                   kpis=kpis, by_model=by_model,
                   model_max=max((m["cost_usd"] for m in by_model), default=0) or 1,
                   stages=stages,
                   stage_max=max((s["total_cost_usd"] for s in stages), default=0) or 1,
                   funnel=funnel, funnel_max=(funnel[0]["n"] or 1),
                   recs=recs, decisions=decisions, runs=runs[:15], queue=c.queue.stats())


@router.post("/jobs")
def create_job(
    jd_text: str = Form(...),
    persona_model: str = Form("auto"),
    synth_model: str = Form("auto"),
    c: Container = Depends(get_container),
):
    for m in (persona_model, synth_model):
        if not pricing.is_known_model(m):
            raise HTTPException(400, f"unknown model: {m}")
        if m != "auto" and c.settings.has_llm and not c.settings.has_key_for(m):
            raise HTTPException(400, f"{m} requires {pricing.provider_env_var(m)} to be set")
    job = c.service.create_job(jd_text, persona_model, synth_model)
    return RedirectResponse(f"/jobs/{job.id}", status_code=303)


@router.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_page(job_id: str, request: Request, watch: int = 0,
             c: Container = Depends(get_container)):
    job = c.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"job {job_id}")
    results = c.results.list_for_job(job_id)
    cand_names = {cd.id: (cd.profile.name or cd.source_filename or cd.id[:8])
                  for cd in c.candidates.list_all()}
    return _render(request, "job.html", job=job, results=results,
                   cand_names=cand_names, queue=c.queue.stats(),
                   progress=c.queue.job_progress(job_id), watch=watch,
                   events=c.audit.events_for_job(job_id))


@router.post("/jobs/{job_id}/upload", response_class=HTMLResponse)
async def upload(
    job_id: str,
    request: Request,
    files: list[UploadFile] = File(...),
    c: Container = Depends(get_container),
):
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
    return _render(request, "estimate.html", job=job, estimate=estimate, task_ids=task_ids)


@router.post("/jobs/{job_id}/confirm")
def confirm(job_id: str, task_ids: str = Form(...), c: Container = Depends(get_container)):
    ids = [t for t in task_ids.split(",") if t]
    c.service.confirm_batch(ids)
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@router.get("/optimize", response_class=HTMLResponse)
def optimize_picker(request: Request, c: Container = Depends(get_container)):
    """Global optimizer entry: pick a job, then paste a resume."""
    return _render(request, "optimize.html", job=None, result=None,
                   jobs=c.jobs.list_all(), models=pricing.catalog())


@router.get("/jobs/{job_id}/optimize", response_class=HTMLResponse)
def optimize_page(job_id: str, request: Request, c: Container = Depends(get_container)):
    job = c.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"job {job_id}")
    return _render(request, "optimize.html", job=job, result=None,
                   jobs=c.jobs.list_all(), models=pricing.catalog())


@router.post("/jobs/{job_id}/optimize", response_class=HTMLResponse)
async def run_optimize(
    job_id: str,
    request: Request,
    resume_text: str = Form(""),
    model: str = Form("auto"),
    resume_file: UploadFile = File(None),
    c: Container = Depends(get_container),
):
    job = c.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"job {job_id}")
    if resume_file and resume_file.filename:
        resume_text = await _extract_upload(c, resume_file)
    if not resume_text.strip():
        return _render(request, "optimize.html", job=job, result=None,
                       jobs=c.jobs.list_all(), models=pricing.catalog(),
                       error="Paste resume text or upload a file.")
    try:
        result = c.service.optimize_resume(job_id, resume_text, model)
    except Exception:
        log.exception("optimize_resume failed job_id=%s", job_id)
        return _render(request, "optimize.html", job=job, result=None,
                       jobs=c.jobs.list_all(), models=pricing.catalog(),
                       error="Optimization failed — please retry. If it persists the model may be unavailable.")
    return _render(request, "optimize.html", job=job, result=result,
                   jobs=c.jobs.list_all(), models=pricing.catalog())


@router.get("/match", response_class=HTMLResponse)
def match_page(request: Request):
    return _render(request, "match.html", result=None, resume_text="", jd_text="")


@router.post("/match", response_class=HTMLResponse)
async def run_match(
    request: Request,
    resume_text: str = Form(""),
    jd_text: str = Form(...),
    resume_file: UploadFile = File(None),
    c: Container = Depends(get_container),
):
    import re as _re
    from app.candisift.domain import resume_analysis
    from app.candisift.adapters.llm.stub import StubJobSpecExtractor, StubProfileExtractor
    if resume_file and resume_file.filename:
        resume_text = await _extract_upload(c, resume_file)
    if not resume_text.strip():
        return _render(request, "match.html", result=None, resume_text="", jd_text=jd_text,
                       error="Paste resume text or upload a file.")
    profile = StubProfileExtractor().extract(resume_text)
    jd = StubJobSpecExtractor().extract(jd_text)
    readability = ats_readability.score(profile, jd)
    resume_lower = resume_text.lower()
    gaps = [
        {"keyword": kw, "present": bool(_re.search(rf"\b{_re.escape(kw.lower())}\b", resume_lower))}
        for kw in jd.must_have_skills
    ]
    result = {
        "score": readability["score"],
        "checks": readability["checks"],
        "gaps": gaps,
        "present": sum(1 for g in gaps if g["present"]),
        "total": len(gaps),
        "title": jd.title,
        "min_years": jd.min_years,
        "analysis": resume_analysis.full_analysis(resume_text),
    }
    return _render(request, "match.html", result=result, resume_text=resume_text, jd_text=jd_text)


@router.get("/cover-letter", response_class=HTMLResponse)
def cover_letter_global(request: Request, c: Container = Depends(get_container)):
    return _render(request, "cover_letter.html", job=None, result=None,
                   jobs=c.jobs.list_all(), models=pricing.catalog())


@router.get("/jobs/{job_id}/cover-letter", response_class=HTMLResponse)
def cover_letter_page(job_id: str, request: Request, c: Container = Depends(get_container)):
    job = c.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"job {job_id}")
    return _render(request, "cover_letter.html", job=job, result=None,
                   jobs=c.jobs.list_all(), models=pricing.catalog())


@router.post("/jobs/{job_id}/cover-letter", response_class=HTMLResponse)
async def run_cover_letter(
    job_id: str, request: Request,
    resume_text: str = Form(""),
    tone: str = Form("professional"),
    model: str = Form("auto"),
    resume_file: UploadFile = File(None),
    c: Container = Depends(get_container),
):
    job = c.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"job {job_id}")
    if resume_file and resume_file.filename:
        resume_text = await _extract_upload(c, resume_file)
    if not resume_text.strip():
        return _render(request, "cover_letter.html", job=job, result=None,
                       jobs=c.jobs.list_all(), models=pricing.catalog(),
                       error="Paste resume text or upload a file.")
    try:
        result = c.service.generate_cover_letter(job_id, resume_text, tone, model)
    except Exception:
        log.exception("generate_cover_letter failed job_id=%s", job_id)
        return _render(request, "cover_letter.html", job=job, result=None,
                       jobs=c.jobs.list_all(), models=pricing.catalog(),
                       error="Cover letter generation failed — please retry. If it persists the model may be unavailable.")
    return _render(request, "cover_letter.html", job=job, result=result,
                   jobs=c.jobs.list_all(), models=pricing.catalog(), resume_text=resume_text)


@router.get("/agents", response_class=HTMLResponse)
def agents_page(request: Request, c: Container = Depends(get_container)):
    from app.candisift import agent_catalog
    stats = {s["agent"]: s for s in c.tracer.agent_stats()}
    roster = agent_catalog.agents(c.settings.persona_model, c.settings.synth_model)
    for a in roster:
        a["stats"] = stats.get(a["role"], {})
    return _render(request, "agents.html", agents=roster, runs=c.tracer.list_runs(limit=25),
                   queue=c.queue.stats())


@router.get("/results/{result_id}", response_class=HTMLResponse)
def breakdown(result_id: str, request: Request, c: Container = Depends(get_container)):
    r = c.results.get(result_id)
    if r is None:
        raise HTTPException(404, f"result {result_id}")
    cand = c.candidates.get(r.candidate_id)
    job = c.jobs.get(r.job_id)
    readability = ats_readability.score(cand.profile, job.spec) if (cand and job) else None
    trace = c.tracer.latest_run(r.candidate_id, r.job_id)
    return _render(request, "breakdown.html", r=r, cand=cand, job=job, readability=readability, trace=trace)


@router.post("/results/{result_id}/decision")
def decide(result_id: str, decision: str = Form(...), c: Container = Depends(get_container)):
    try:
        c.results.set_decision(result_id, decision)
        c.audit.record("decision.set", result_id=result_id, decision=decision)
    except LookupError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    r = c.results.get(result_id)
    return RedirectResponse(f"/jobs/{r.job_id}", status_code=303)

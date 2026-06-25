"""Application layer: the ScreeningService use-case — the funnel orchestrator.

Depends ONLY on domain models, domain rules, and ports (DIP). It knows the order
of the funnel and the model-selection policy ("auto" -> configured defaults); it
knows nothing about Agno, SQLite, FastAPI, or HTTP.

  ingest ─► dedup ─► [per screen] hard filter ─► rank ─► (survivor) personas ─► synthesis ─► persist+audit
"""
from __future__ import annotations

import base64
import contextvars
import hashlib
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from app.candisift.domain import ports
from app.candisift.domain.duplicate import find_near_duplicate
from app.candisift.domain.guardrails import (
    MAX_JD_CHARS, MAX_RESUME_CHARS, injection_score, sanitize_untrusted, scan_bias_proxies,
)
from app.candisift.domain.models import (
    AgentPersona, Candidate, Job, Recommendation, RolePersonas, ScreeningResult, TaskType,
)
from app.candisift.domain.services import dedup_key, hard_filter, strip_pii, validate_experience
from app.candisift.domain.verdict_guard import apply_guards

log = logging.getLogger("candisift.screening")

# below this many chars a resume is treated as unreadable (failed OCR / corrupt
# upload) and rejected rather than ingested as an empty profile.
_MIN_RESUME_CHARS = 20


class NotFoundError(LookupError):
    """A referenced candidate or job does not exist."""


class PermanentTaskError(Exception):
    """A task can never succeed — bad/corrupt/unreadable input. The worker
    dead-letters it immediately instead of wasting the retry budget on a
    deterministic failure. (Transient errors raise anything else -> retried.)"""


def _id() -> str:
    return uuid.uuid4().hex


def _result_id(job_id: str, candidate_id: str) -> str:
    return f"{job_id}.{candidate_id}"  # deterministic -> retried screen upserts same row


_MAX_PERSONA_INSTR_CHARS = 400
_MAX_PERSONA_ITEMS = 8


def _scrub_persona(ap: AgentPersona | None) -> AgentPersona | None:
    if ap is None:
        return None
    instr = [sanitize_untrusted(i, _MAX_PERSONA_INSTR_CHARS) for i in ap.instructions[:_MAX_PERSONA_ITEMS]]
    return AgentPersona(role=ap.role, title=sanitize_untrusted(ap.title, 200),
                        instructions=[i for i in instr if i])


def _scrub_personas(personas: RolePersonas | None) -> tuple[RolePersonas | None, bool]:
    """Sanitize + length-cap JD-derived personas, then injection-scan the assembled
    text. If anything trips the detector, drop personas entirely (-> generic agents)
    so a hostile JD can't steer the evaluators. Returns (personas_or_None, blocked)."""
    if personas is None:
        return None, False
    scrubbed = RolePersonas(
        domain=sanitize_untrusted(personas.domain, 120), seniority=personas.seniority,
        tech=_scrub_persona(personas.tech), risk=_scrub_persona(personas.risk),
        synth=_scrub_persona(personas.synth), hr=_scrub_persona(personas.hr),
    )
    combined = " ".join(p for p in (scrubbed.preamble("tech"), scrubbed.preamble("risk"),
                                    scrubbed.preamble("synth"), scrubbed.preamble("hr")) if p)
    if injection_score(combined) > 0:
        return None, True
    return scrubbed, False


def _models_fingerprint(persona_model: str, synth_model: str, spec,
                        hr_eval: bool = True, coverage_audit: bool = True) -> str:
    """Identity of the work a screen would do. Same fingerprint => the stored
    verdict is still valid, so we reuse it and skip the LLM. A JD edit (spec changes),
    a model change, OR a toggle change (hr_eval / coverage_audit) flips it and forces a
    recompute — otherwise a crash-resume could reuse evals produced under a different
    config than the operator now has set (e.g. resume without the HR lens)."""
    blob = f"{persona_model}|{synth_model}|hr={hr_eval}|cov={coverage_audit}|{spec.model_dump_json()}"
    return hashlib.sha256(blob.encode()).hexdigest()


class ScreeningService:
    """Implements ports.ScreeningUseCase."""

    def __init__(
        self,
        *,
        text_extractor: ports.TextExtractor,
        llm: ports.LLMProvider,
        ranker: ports.Ranker,
        candidates: ports.CandidateRepository,
        jobs: ports.JobRepository,
        results: ports.ResultRepository,
        audit: ports.AuditLog,
        queue: ports.TaskQueue,
        default_persona_model: str,
        default_synth_model: str,
        top_n: int = 30,
        tracer: ports.Tracer | None = None,
        memory: ports.AgentMemory | None = None,
        persona_designer: ports.PersonaDesigner | None = None,
        coverage_audit: bool = True,
        hr_eval: bool = True,
    ) -> None:
        self._extract = text_extractor
        self._llm = llm
        self._rank = ranker
        self._candidates = candidates
        self._jobs = jobs
        self._results = results
        self._audit = audit
        self._queue = queue
        self._default_persona = default_persona_model
        self._default_synth = default_synth_model
        self._top_n = top_n
        self._tracer = tracer
        self._memory = memory
        self._persona_designer = persona_designer
        self._coverage_audit = coverage_audit
        self._hr_eval = hr_eval

    # ---- model selection policy ------------------------------------------

    def _persona(self, model: str) -> str:
        return self._default_persona if model in ("", "auto") else model

    def _synth(self, model: str) -> str:
        return self._default_synth if model in ("", "auto") else model

    # ---- jobs -------------------------------------------------------------

    def create_job(self, jd_text: str, persona_model: str = "auto",
                   synth_model: str = "auto") -> Job:
        jd_text = sanitize_untrusted(jd_text, MAX_JD_CHARS)
        if injection_score(jd_text):
            self._audit.record("guardrail.injection_flag", surface="jd")
        spec = self._llm.jd_extractor(self._persona(persona_model)).extract(jd_text)

        # derive role-specialized agent personas from this JD (subject-matter
        # expert instructions the evaluators adopt for this role). Best-effort:
        # a failure must not block job creation -> generic agents.
        personas = None
        if self._persona_designer is not None:
            try:
                personas = self._persona_designer.design(jd_text, spec)
            except Exception:
                log.exception("persona design failed; using generic agents")
            # scrub + injection-scan before trusting/caching (JD is semi-trusted)
            personas, blocked = _scrub_personas(personas)
            if blocked:
                self._audit.record("guardrail.persona_injection_blocked", surface="jd")

        job = Job(id=_id(), title=spec.title or "Untitled role", raw_text=jd_text, spec=spec,
                  persona_model=persona_model, synth_model=synth_model, personas=personas)
        self._jobs.add(job)
        self._audit.record("job.created", job_id=job.id, title=job.title,
                           persona_model=persona_model, synth_model=synth_model,
                           persona_domain=personas.domain if personas else "",
                           persona_seniority=personas.seniority if personas else "")
        return job

    # ---- ingestion (with dedup) -------------------------------------------

    def ingest_resume(self, content: bytes, filename: str, model: str = "auto") -> Candidate:
        # Layer-1 cache: identical bytes seen before -> reuse the candidate and
        # skip OCR + extraction LLM entirely. This is the "never redo work" core.
        content_hash = hashlib.sha256(content).hexdigest()
        cached = self._candidates.find_by_content_hash(content_hash)
        if cached:
            self._audit.record("candidate.cache_hit", candidate_id=cached.id, filename=filename)
            if self._tracer:
                self._tracer.start_run("ingest", candidate_id=cached.id)
                self._tracer.end_run(status="done", cache_hit=True)
            return cached

        if self._tracer:
            self._tracer.start_run("ingest")
        try:
            text = sanitize_untrusted(self._extract.extract(content, filename), MAX_RESUME_CHARS)
            # Unreadable file (scanned PDF with no OCR available, blank image, corrupt
            # upload). Reject loudly — an empty profile would otherwise collapse every
            # unreadable resume onto one dedup_key and silently hide the failures.
            if len(text.strip()) < _MIN_RESUME_CHARS:
                raise PermanentTaskError(
                    f"{filename}: no readable text after extraction/OCR "
                    f"(<{_MIN_RESUME_CHARS} chars) — unreadable scan or corrupt file"
                )
            if injection_score(text):
                self._audit.record("guardrail.injection_flag", surface="resume", filename=filename)
            profile = self._llm.profile_extractor(self._persona(model)).extract(text)
            # deterministic post-extraction: cross-check total_years against
            # work_entry dates, detect concurrent employment and gaps
            profile = validate_experience(profile)

            # --- GITHUB ENRICHMENT ---
            if profile.github_url:
                try:
                    from app.candisift.adapters.github.github_enricher import GitHubEnricherAdapter
                    github_enricher = GitHubEnricherAdapter(self._llm, self._persona(model))
                    profile.github_projects = github_enricher.enrich(profile.github_url)
                    self._audit.record("candidate.github_enriched", filename=filename,
                                       projects=len(profile.github_projects))
                except Exception as e:
                    log.error(f"GitHub enrichment failed: {e}")
                    self._audit.record("candidate.github_enrich_failed", filename=filename,
                                       error=str(e)[:200])
            # --- LINKEDIN ENRICHMENT (resume-derived; no external API) ---
            if profile.linkedin_url:
                try:
                    from app.candisift.adapters.linkedin.linkedin_enricher import LinkedInEnricherAdapter
                    linkedin_enricher = LinkedInEnricherAdapter(self._llm, self._persona(model))
                    profile.linkedin_profile = linkedin_enricher.enrich(text, profile)
                    self._audit.record("candidate.linkedin_enriched", filename=filename,
                                       positions=len(profile.linkedin_profile.get("positions", [])))
                except Exception as e:
                    log.error(f"LinkedIn enrichment failed: {e}")
                    self._audit.record("candidate.linkedin_enrich_failed", filename=filename,
                                       error=str(e)[:200])
            # -------------------------

            key = dedup_key(profile)
            # key == "" => no extractable identity; do NOT dedup (a shared empty key
            # would collapse distinct identity-less applicants onto one another). Exact
            # byte re-uploads were already caught by the content_sha256 cache above.
            existing = self._candidates.find_by_dedup_key(key) if key else None
            if existing:
                self._audit.record("candidate.dedup_hit", candidate_id=existing.id, filename=filename)
                if self._tracer:
                    self._tracer.end_run(status="done")
                return existing

            # near-duplicate / resume-farming check (beyond exact dedup)
            near = find_near_duplicate(profile, [(c.id, c.profile) for c in self._candidates.list_all()])
            dup_of, dup_sim = (near[0], near[1]) if near else ("", 0.0)

            cand = Candidate(id=_id(), dedup_key=key, content_sha256=content_hash,
                             source_filename=filename, profile=profile,
                             near_duplicate_of=dup_of, duplicate_similarity=dup_sim)
            self._candidates.add(cand)
            if dup_of:
                self._audit.record("candidate.near_duplicate", candidate_id=cand.id,
                                   near_duplicate_of=dup_of, similarity=dup_sim)
            self._audit.record("candidate.ingested", candidate_id=cand.id, filename=filename)
            if self._tracer:
                self._tracer.end_run(status="done")
            return cand
        except Exception as e:
            if self._tracer:
                self._tracer.end_run(status="error", error=str(e))
            raise

    # ---- the funnel for one (candidate, job) ------------------------------

    def screen(self, candidate_id: str, job_id: str) -> ScreeningResult:
        cand = self._candidates.get(candidate_id)
        if cand is None:
            raise NotFoundError(f"candidate {candidate_id}")
        job = self._jobs.get(job_id)
        if job is None:
            raise NotFoundError(f"job {job_id}")

        rid = _result_id(job_id, candidate_id)
        fingerprint = _models_fingerprint(self._persona(job.persona_model),
                                          self._synth(job.synth_model), job.spec,
                                          self._hr_eval, self._coverage_audit)

        # Layer-2 cache + crash-resume: a stored result for this (candidate, job,
        # models, spec) is either TERMINAL (hard-rejected, or completed with a
        # synthesis) or a PARTIAL CHECKPOINT written after the personas ran but
        # before synthesis (a process that died in between). Terminal => reuse and
        # skip all LLM work. Partial => resume from it (reuse the completed
        # evaluators, pick up at synthesis). A spec/model change flips the
        # fingerprint and forces a fresh screen.
        prior = self._results.get(rid)
        resume = None
        if prior is not None and prior.models_fingerprint == fingerprint:
            terminal = (not prior.passed_hard_filters) or prior.synthesis is not None
            if terminal:
                self._audit.record("screen.cache_hit", result_id=rid, candidate_id=candidate_id)
                if self._tracer:
                    self._tracer.start_run("screen", candidate_id=candidate_id, job_id=job_id)
                    self._tracer.end_run(status="done", cache_hit=True)
                return prior
            resume = prior

        if self._tracer:
            self._tracer.start_run("screen", candidate_id=candidate_id, job_id=job_id)
        try:
            score = self._rank.score(cand.profile, job.spec)
            passed, reasons = hard_filter(cand.profile, job.spec)

            # stage 3: hard-filtered out -> no LLM spend
            if not passed:
                result = ScreeningResult(
                    id=rid, job_id=job_id, candidate_id=candidate_id,
                    passed_hard_filters=False, filter_reasons=reasons, semantic_score=score,
                    models_fingerprint=fingerprint,
                )
                self._results.upsert(result)
                self._audit.record("screen.rejected_hard_filter",
                                   result_id=rid, candidate_id=candidate_id, reasons=reasons)
                if self._tracer:
                    self._tracer.end_run(status="done")
                return result

            # stage 5: personas (parallel, PII-stripped) -> synthesis, on the job's models
            persona = self._persona(job.persona_model)
            synth_model = self._synth(job.synth_model)
            screened = strip_pii(cand.profile)
            # role-specialized persona preambles derived from the JD (empty if none)
            p = job.personas
            synth_persona = p.preamble("synth") if p else ""

            if resume is not None and resume.tech is not None and resume.risk is not None:
                # crash recovery: the personas already ran in a prior attempt and were
                # checkpointed. Reuse them and pick up at synthesis — no persona LLM
                # calls re-run. This is the "resume where it stopped" path.
                tech, risk, hr = resume.tech, resume.risk, resume.hr
                self._audit.record("screen.resumed", result_id=rid, candidate_id=candidate_id,
                                   stage="synthesis")
            else:
                tech_persona = p.preamble("tech") if p else ""
                risk_persona = p.preamble("risk") if p else ""
                hr_persona = p.preamble("hr") if p else ""
                tech_agent = self._llm.technical(persona)
                risk_agent = self._llm.risk(persona)
                # copy the context (carries the active trace) into each worker thread
                # so spans from the parallel personas attach to this run. HR is the
                # advisory people-fit lens — skip its call when hr_eval is off (cost).
                with ThreadPoolExecutor(max_workers=3) as ex:
                    f_tech = ex.submit(contextvars.copy_context().run,
                                       tech_agent.evaluate, screened, job.spec, tech_persona)
                    f_risk = ex.submit(contextvars.copy_context().run,
                                       risk_agent.evaluate, screened, risk_persona)
                    f_hr = (ex.submit(contextvars.copy_context().run,
                                      self._llm.hr(persona).evaluate, screened, job.spec, hr_persona)
                            if self._hr_eval else None)
                    tech = f_tech.result()
                    risk = f_risk.result()
                    hr = f_hr.result() if f_hr else None
                # checkpoint: the expensive evaluators are done. Persist them BEFORE
                # synthesis (synthesis=None marks the result partial) so a crash during
                # or after synthesis resumes here instead of re-running every persona.
                # ponytail: checkpoint at the persona barrier, not per-persona — the
                # parallel block is one unit; go per-persona only if one call dominates.
                self._results.upsert(ScreeningResult(
                    id=rid, job_id=job_id, candidate_id=candidate_id,
                    passed_hard_filters=True, semantic_score=score,
                    tech=tech, risk=risk, hr=hr, models_fingerprint=fingerprint,
                ))
                self._audit.record("screen.checkpoint", result_id=rid,
                                   candidate_id=candidate_id, stage="personas_done")

            # agent memory: recall the team's past decisions before the verdict
            self._recall_feedback(job_id)

            synthesis = self._llm.synthesizer(synth_model).synthesize(
                job.spec, tech, risk, hr, synth_persona)

            # bias guardrail: scan the verdict's OWN words for protected-class proxies.
            # A hit is flagged for human review (audit + on the result), never auto-acted.
            verdict_text = " ".join([synthesis.rationale] + [f.claim for f in synthesis.weaknesses])
            bias_flags = scan_bias_proxies(verdict_text)
            if bias_flags:
                self._audit.record("guardrail.bias_proxy_flag", result_id=rid,
                                   candidate_id=candidate_id, terms=bias_flags)

            # deterministic output guardrails (verdict_guard): reconcile the model's
            # verdict against the hard facts — cap an over-confident shortlist when a
            # must-have is unmet or a fraud signal is present, and flag claims whose
            # evidence doesn't trace to the profile. Caps/flags only; never upgrades.
            guard = apply_guards(synthesis, tech, risk, screened, bias_flagged=bool(bias_flags))
            synthesis.recommendation = guard.recommendation   # persist the capped verdict
            requires_review = guard.requires_human_review
            review_reasons = list(guard.review_reasons)
            if requires_review:
                self._audit.record("guardrail.verdict_review", result_id=rid,
                                   candidate_id=candidate_id, reasons=review_reasons)

            # §5 QA auditor (LLM-as-judge): a separate, cheaper second opinion that
            # checks the evaluation is complete + disciplined (coverage, grounding,
            # knockout, bias) WITHOUT re-scoring. An "unsafe" verdict is held for a
            # human. Best-effort: an auditor error routes to human review, it never
            # silently passes. Runs on the persona (cheap) tier — a different model
            # from the synthesizer, so it does not rubber-stamp its own work.
            coverage = None
            if self._coverage_audit:
                try:
                    coverage = self._llm.coverage_auditor(persona).audit(
                        job.spec, screened, tech, risk, hr, synthesis)
                except Exception:
                    log.exception("coverage audit failed; routing to human review")
                    requires_review = True
                    review_reasons.append("QA auditor error — verdict not independently verified")
                else:
                    if not coverage.safe_to_surface_to_recruiter:
                        requires_review = True
                        review_reasons.append("QA auditor flagged the verdict unsafe to surface")
                    self._audit.record("screen.coverage_audit", result_id=rid,
                                       candidate_id=candidate_id, overall=coverage.overall,
                                       safe=coverage.safe_to_surface_to_recruiter,
                                       failures=[f.check for f in coverage.failures])

            # Any human-review hold (bias, ungrounded claim, or QA-auditor-unsafe) also
            # caps an over-confident shortlist down to "maybe", so the recommendation a
            # recruiter scans — and the analytics shortlist tally — never shows a clean
            # green "shortlist" for a verdict we are holding back. (Knockout/fraud were
            # already capped inside apply_guards.)
            if requires_review and synthesis.recommendation is Recommendation.shortlist:
                synthesis.recommendation = Recommendation.maybe

            result = ScreeningResult(
                id=rid, job_id=job_id, candidate_id=candidate_id,
                passed_hard_filters=True, semantic_score=score,
                tech=tech, risk=risk, hr=hr, synthesis=synthesis,
                bias_flags=bias_flags,
                requires_human_review=requires_review,
                review_reasons=review_reasons,
                ungrounded_claims=guard.ungrounded,
                coverage=coverage,
                models_fingerprint=fingerprint,
            )
            self._results.upsert(result)
            if self._memory:
                self._memory.remember(
                    candidate_id=candidate_id, job_id=job_id, kind="synthesis",
                    content=synthesis.rationale,
                    data={"overall_fit": synthesis.overall_fit,
                          "recommendation": synthesis.recommendation.value},
                )
            self._audit.record(
                "screen.completed", result_id=rid, candidate_id=candidate_id,
                persona_model=persona, synth_model=synth_model,
                semantic_score=score, depth_score=tech.depth_score, risk_score=risk.risk_score,
                people_score=hr.people_score if hr else None,
                overall_fit=synthesis.overall_fit, recommendation=synthesis.recommendation.value,
            )
            if self._tracer:
                self._tracer.end_run(status="done")
            return result
        except Exception as e:
            if self._tracer:
                self._tracer.end_run(status="error", error=str(e))
            raise

    def _recall_feedback(self, job_id: str) -> list[dict]:
        """Agent retrieval tool: pull the team's recent decisions, recorded as a
        tool span so it shows in the run timeline. No-op without memory/tracer."""
        if not self._memory:
            return []
        t0 = time.monotonic()
        try:
            return self._memory.recall_recruiter_feedback(job_id)
        finally:
            if self._tracer:
                self._tracer.record_span(name="recall_recruiter_feedback", agent="tool",
                                         latency_ms=round((time.monotonic() - t0) * 1000.0, 1))

    # ---- batch: durable enqueue (multi-resume upload) ---------------------

    def enqueue_batch(self, job_id: str, files: list[tuple[bytes, str]],
                      staged: bool = False) -> list[str]:
        """Persist one durable ingest task per uploaded resume. The worker chains
        ingest -> screen. The job's chosen model rides in the payload. When staged,
        tasks are held (not claimable) until confirm_batch — the estimate-first flow."""
        job = self._jobs.get(job_id)
        if job is None:
            raise NotFoundError(f"job {job_id}")
        task_ids: list[str] = []
        for content, filename in files:
            tid = self._queue.enqueue(TaskType.ingest_resume, {
                "job_id": job_id,
                "filename": filename,
                "content_b64": base64.b64encode(content).decode(),
                "model": job.persona_model,
            }, staged=staged)
            task_ids.append(tid)
        self._audit.record("batch.enqueued", job_id=job_id, count=len(task_ids), staged=staged)
        return task_ids

    def confirm_batch(self, task_ids: list[str]) -> int:
        """Release a staged batch once the recruiter accepts the cost estimate."""
        n = self._queue.release(task_ids)
        self._audit.record("batch.confirmed", released=n)
        return n

    # ---- resume optimizer ---------------------------------------------------

    def optimize_resume(
        self, job_id: str, resume_text: str, model: str = "auto"
    ) -> "ResumeOptimizationResult":
        """Keyword-gap analysis + LLM resume rewrite targeted at a specific job."""
        from app.candisift.domain import ats_readability
        from app.candisift.domain.models import ResumeOptimizationResult
        from app.candisift.adapters.llm.stub import StubProfileExtractor

        job = self._jobs.get(job_id)
        if job is None:
            raise NotFoundError(f"job {job_id}")

        # resume is attacker-controlled like any upload — sanitize + injection-scan
        # before it reaches the optimizer LLM (the screen path does this; these
        # optimize/cover paths skipped it, leaving a prompt-injection hole).
        text = sanitize_untrusted(resume_text, MAX_RESUME_CHARS)
        if injection_score(text):
            self._audit.record("guardrail.injection_flag", surface="optimize", job_id=job_id)

        # ATS readability score BEFORE — deterministic, zero LLM cost
        stub_ex = StubProfileExtractor()
        score_before = ats_readability.score(stub_ex.extract(text), job.spec)["score"]

        # LLM optimize (one call; falls back to StubResumeOptimizer when no key)
        resolved = self._persona(model)
        llm_out = self._llm.resume_optimizer(resolved).optimize(
            text, job.spec, job.title
        )
        optimized = llm_out.optimized_resume or text

        # ATS readability score AFTER — deterministic
        score_after = ats_readability.score(stub_ex.extract(optimized), job.spec)["score"]

        self._audit.record(
            "optimize.completed", job_id=job_id, model=resolved,
            score_before=score_before, score_after=score_after,
            keywords_added=sum(1 for g in llm_out.keyword_gaps if g.status == "added"),
        )
        return ResumeOptimizationResult(
            job_id=job_id,
            original_resume=resume_text,
            optimized_resume=optimized,
            keyword_gaps=llm_out.keyword_gaps,
            changes=llm_out.changes,
            ats_score_before=score_before,
            ats_score_after=score_after,
            model_used=resolved,
        )

    # ---- cover letter generator -------------------------------------------

    def generate_cover_letter(
        self, job_id: str, resume_text: str, tone: str = "professional", model: str = "auto"
    ) -> "CoverLetterResult":
        from app.candisift.domain.models import CoverLetterResult
        job = self._jobs.get(job_id)
        if job is None:
            raise NotFoundError(f"job {job_id}")
        clean = sanitize_untrusted(resume_text, MAX_RESUME_CHARS)
        if injection_score(clean):
            self._audit.record("guardrail.injection_flag", surface="cover_letter", job_id=job_id)
        resolved = self._persona(model)
        text = self._llm.cover_letter_writer(resolved).write(
            clean, job.spec, job.title, tone
        )
        self._audit.record("cover_letter.generated", job_id=job_id, model=resolved, tone=tone)
        return CoverLetterResult(
            job_id=job_id, cover_letter=text, tone=tone, model_used=resolved,
        )

    # ---- task handlers (called by the durable worker) ---------------------

    def handle_ingest_task(self, payload: dict) -> None:
        try:
            content = base64.b64decode(payload["content_b64"])
            filename = payload["filename"]
            job_id = payload["job_id"]
        except (KeyError, ValueError) as e:  # binascii.Error subclasses ValueError
            raise PermanentTaskError(f"malformed ingest payload: {e}") from e
        cand = self.ingest_resume(content, filename, payload.get("model", "auto"))
        # Deterministic id => idempotent: if this ingest task re-runs (its complete()
        # was lost to a crash) it re-enqueues the SAME screen id, which the queue
        # ignores instead of creating a duplicate screen (which would double LLM spend).
        screen_id = f"screen:{job_id}:{cand.id}"
        self._queue.enqueue(TaskType.screen, {"candidate_id": cand.id, "job_id": job_id},
                            task_id=screen_id)

    def handle_screen_task(self, payload: dict) -> None:
        self.screen(payload["candidate_id"], payload["job_id"])

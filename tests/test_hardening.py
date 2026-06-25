"""Regression tests for the production-hardening pass (2026-06-22).

Each test pins a specific fix from the review so a future change can't silently
re-open the hole. Pure-domain + queue tests; no network, no LLM.
"""
from __future__ import annotations

import tempfile

import pytest

from app.candisift.config import Settings
from app.candisift.domain.models import (
    CandidateProfile, JDSpec, SkillItem, Finding, RiskEval, TaskType, TaskStatus,
)
from app.candisift.domain import services, duplicate, guardrails, verdict_guard, resume_analysis
from app.candisift import pricing


# ---- PII redaction (CRITICAL) --------------------------------------------

def test_strip_pii_redacts_identity_from_free_text():
    p = CandidateProfile(
        name="Jane Doe", email="jane@x.com", phone="555-123-9876",
        summary="Jane Doe led platform team; reach jane@x.com or linkedin.com/in/janedoe",
        titles=["Senior Engineer at Acme, Jane Doe"],
        skills=[SkillItem(name="python", evidence="Jane Doe shipped the Python service")],
        employment_gaps=["Jane Doe took leave 2021"],
    )
    clean = services.strip_pii(p)
    blob = " ".join([clean.summary, *clean.titles, *clean.employment_gaps,
                     *(s.evidence for s in clean.skills)]).lower()
    assert "jane" not in blob and "doe" not in blob
    assert "jane@x.com" not in blob and "linkedin.com" not in blob
    assert clean.name == "" and clean.email == "" and clean.phone == ""
    # job-relevant signal survives
    assert "python" in blob and "platform" in blob


# ---- dedup empty-PII no longer collapses ---------------------------------

def test_dedup_key_empty_identity_is_non_dedupable():
    a = CandidateProfile(summary="resume A")
    b = CandidateProfile(summary="resume B")
    assert services.dedup_key(a) == "" == services.dedup_key(b)   # no shared real hash
    real = CandidateProfile(name="X Y", email="x@y.com", phone="111")
    assert services.dedup_key(real) != ""


# ---- hard_filter fails CLOSED on missing required data -------------------

def test_hard_filter_missing_work_auth_rejects():
    jd = JDSpec(required_work_auth=["US Citizen"])
    ok, reasons = services.hard_filter(CandidateProfile(total_years=10), jd)
    assert not ok and any("work authorization" in r for r in reasons)


def test_hard_filter_knockout_keyword_rejects():
    jd = JDSpec(knockouts=["contractor only"])
    prof = CandidateProfile(summary="experienced contractor only, 10y", total_years=10)
    ok, reasons = services.hard_filter(prof, jd)
    assert not ok and any("knockout" in r for r in reasons)


# ---- fraud-cap negation veto ---------------------------------------------

def test_fraud_cap_ignores_negated_phrasing():
    clean = RiskEval(flags=[Finding(claim="no fabrication found in the resume", evidence="x")])
    assert not verdict_guard._risk_has_fraud(clean)
    real = RiskEval(flags=[Finding(claim="dates appear fabricated", evidence="x")])
    assert verdict_guard._risk_has_fraud(real)


# ---- near-dup sparse-profile false positive ------------------------------

def test_near_duplicate_skips_thin_profiles():
    thin = CandidateProfile(skills=[SkillItem(name="python")])
    other = CandidateProfile(skills=[SkillItem(name="python")])
    assert duplicate.find_near_duplicate(thin, [("c1", other)]) is None
    assert duplicate.jaccard(set(), set()) == 0.0


# ---- bias proxy no longer flags tech prose -------------------------------

def test_bias_scan_ignores_technical_adjectives():
    assert guardrails.scan_bias_proxies("migrated the old legacy system to a single-page app") == []
    assert "female" in guardrails.scan_bias_proxies("preferred a female candidate")


# ---- quantification no longer counts bare years --------------------------

def test_quantification_excludes_bare_years():
    res = resume_analysis.quantification_analysis("- Worked at Acme from 2018 to 2022")
    assert res["quantified"] == 0
    res2 = resume_analysis.quantification_analysis("- Cut latency by 40%")
    assert res2["quantified"] == 1


def test_action_verb_matches_present_tense():
    res = resume_analysis.action_verb_analysis("- Lead a team of 6\n- Drive adoption")
    assert res["strong"] == 2


# ---- pricing: counts HR + coverage, paid openai/* not $0 -----------------

def test_estimate_counts_optional_stages():
    base = pricing.estimate_batch(1, "claude-haiku-4-5", "claude-opus-4-8",
                                  hr_eval=False, coverage_audit=False)
    full = pricing.estimate_batch(1, "claude-haiku-4-5", "claude-opus-4-8",
                                  hr_eval=True, coverage_audit=True)
    assert full["per_resume_usd"] > base["per_resume_usd"]


def test_estimate_unknown_model_does_not_raise():
    out = pricing.estimate_batch(2, "totally-made-up-model", "claude-opus-4-8")
    assert "estimated_total_usd" in out               # resilient, no 500


def test_paid_openai_endpoint_not_reported_free():
    _, source = pricing._stage_price("openai/deepinfra-llama", 1000, 500)
    assert source == "unknown"                         # not "free"


# ---- prod config fails closed --------------------------------------------

def test_prod_validation_rejects_defaults():
    with pytest.raises(RuntimeError):
        Settings(env="prod").validate_runtime()       # default creds + cors + hsts
    ok = Settings(env="prod", basic_auth_user="admin2", basic_auth_pass="s3cret!",
                  hsts=True, cors_origins="https://ats.example.com")
    ok.validate_runtime()                              # explicit + safe -> boots


# ---- queue: idempotent enqueue + ownership-checked complete --------------

def _queue():
    from app.candisift.adapters.persistence.db import make_engine, init_db
    from app.candisift.adapters.persistence.queue import SqliteTaskQueue
    eng = make_engine(f"sqlite:///{tempfile.mktemp(suffix='.db')}")
    init_db(eng)
    return SqliteTaskQueue(eng)


def test_enqueue_idempotent_on_deterministic_id():
    q = _queue()
    a = q.enqueue(TaskType.screen, {"k": 1}, task_id="screen:j:c")
    b = q.enqueue(TaskType.screen, {"k": 1}, task_id="screen:j:c")
    assert a == b
    assert q.stats().get(TaskStatus.pending.value, 0) == 1   # only one row


def test_complete_requires_lease_ownership():
    import datetime as _dt
    q = _queue()
    q.enqueue(TaskType.screen, {"k": 1})
    t = q.claim_next(300)
    wrong = _dt.datetime(2000, 1, 1, tzinfo=_dt.timezone.utc)
    assert q.complete(t.id, lease_until=wrong) is False      # stale worker blocked
    assert q.complete(t.id, lease_until=t.lease_until) is True


def test_heartbeat_extends_then_blocks_stale():
    q = _queue()
    q.enqueue(TaskType.screen, {"k": 1})
    t = q.claim_next(300)
    new = q.heartbeat(t.id, t.lease_until, 300)
    assert new is not None and new != t.lease_until
    assert q.heartbeat(t.id, t.lease_until, 300) is None     # old lease no longer owns

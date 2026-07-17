"""Regression tests for the production-hardening pass (2026-06-22).

Each test pins a specific fix from the review so a future change can't silently
re-open the hole. Pure-domain + queue tests; no network, no LLM.
"""
from __future__ import annotations

import tempfile

import pytest

from app.candisift.config import Settings
from app.candisift.domain.models import (
    CandidateProfile, JDSpec, SkillItem, Finding, RiskEval, TaskType, TaskStatus, WorkEntry,
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
    ok, reasons, _ = services.hard_filter(CandidateProfile(total_years=10), jd)
    assert not ok and any("work authorization" in r for r in reasons)


def test_hard_filter_knockout_keyword_rejects():
    jd = JDSpec(knockouts=["contractor only"])
    prof = CandidateProfile(summary="experienced contractor only, 10y", total_years=10)
    ok, reasons, _ = services.hard_filter(prof, jd)
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


# ---- LLM structured-output recovery (fenced-JSON parse miss) ---------------
# Live failure: Agno's JSON mode handed back the raw model text (a ```json
# fenced block) instead of the schema instance; downstream crashed with
# "'str' object has no attribute 'rationale'/'work_entries'" on every screen
# of that candidate (any job). _structured must recover or raise cleanly.

def test_structured_recovers_fenced_json():
    from app.candisift.adapters.llm.agno_personas import _structured
    from app.candisift.domain.models import Synthesis
    raw = '```json\n{"overall_fit": 70, "recommendation": "maybe", "rationale": "ok"}\n```'
    s = _structured(raw, Synthesis)
    assert isinstance(s, Synthesis) and s.overall_fit == 70


def test_structured_passthrough_and_dict():
    from app.candisift.adapters.llm.agno_personas import _structured
    from app.candisift.domain.models import Synthesis
    inst = Synthesis(overall_fit=50, recommendation="maybe", rationale="x")
    assert _structured(inst, Synthesis) is inst
    d = _structured({"overall_fit": 10, "recommendation": "reject", "rationale": "y"}, Synthesis)
    assert isinstance(d, Synthesis)


def test_structured_raises_clear_error_on_garbage():
    from app.candisift.adapters.llm.agno_personas import _structured
    from app.candisift.domain.models import Synthesis
    with pytest.raises(ValueError):
        _structured("sorry, I cannot produce JSON today", Synthesis)
    with pytest.raises(ValueError):
        _structured(None, Synthesis)


def test_enqueue_revives_failed_task_with_same_id():
    # A failed screen task must not block a retry that re-enqueues the same
    # deterministic id (re-upload of the same resume to the same job).
    q = _queue()
    tid = q.enqueue(TaskType.screen, {"k": 1}, task_id="screen:j1:c1")
    t = q.claim_next(300)
    q.fail(t.id, "boom", retry=False, lease_until=t.lease_until)  # terminal failure
    assert q.stats().get(TaskStatus.failed.value, 0) == 1
    q.enqueue(TaskType.screen, {"k": 1}, task_id=tid)         # user retries
    assert q.stats().get(TaskStatus.failed.value, 0) == 0
    assert q.stats().get(TaskStatus.pending.value, 0) == 1    # claimable again


# ---- rate limiter: proxy-aware client key ---------------------------------
# request.client.host is the SOCKET peer. Behind a proxy/LB that peer is the proxy
# for EVERY user, so all of them share one bucket: one abusive client 429s the whole
# site, and the limiter stops being the brute-force fence in front of Basic auth.
# X-Forwarded-For is client-appendable, so it is trusted only as far as
# CANDISIFT_TRUSTED_PROXY_COUNT says real hops exist.

def _rl_client(per_minute: int, trusted_proxy_count: int):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.candisift.adapters.http.security import RateLimitMiddleware
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, per_minute=per_minute,
                       trusted_proxy_count=trusted_proxy_count)

    @app.get("/")
    def _root():
        return {"ok": True}

    return TestClient(app)


def test_rate_limit_ignores_spoofed_xff_when_not_behind_proxy():
    # count=0 (direct exposure): XFF is attacker-controlled noise. A client rotating
    # the header must not mint itself a fresh bucket per request.
    c = _rl_client(per_minute=1, trusted_proxy_count=0)
    assert c.get("/", headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
    assert c.get("/", headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 429


def test_rate_limit_separates_real_clients_behind_one_proxy():
    # count=1: our own LB appended the peer it saw, so the RIGHTMOST entry is the real
    # client. Two users behind one proxy get a bucket each...
    c = _rl_client(per_minute=1, trusted_proxy_count=1)
    assert c.get("/", headers={"X-Forwarded-For": "10.0.0.1"}).status_code == 200
    assert c.get("/", headers={"X-Forwarded-For": "10.0.0.2"}).status_code == 200
    assert c.get("/", headers={"X-Forwarded-For": "10.0.0.1"}).status_code == 429  # ...and only their own
    # entries LEFT of the trusted hop are forged and ignored: still keyed on 10.0.0.2
    assert c.get("/", headers={"X-Forwarded-For": "9.9.9.9, 10.0.0.2"}).status_code == 429
    # no header at all (a request that bypassed the proxy) -> fall back to the peer
    assert c.get("/").status_code == 200


def test_rate_limit_falls_back_to_peer_when_xff_shorter_than_chain():
    # 2 trusted hops but only 1 entry => the chain isn't what we were told; keying on
    # the lone (forgeable) entry would hand an attacker a bucket per forged value.
    c = _rl_client(per_minute=1, trusted_proxy_count=2)
    assert c.get("/", headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
    assert c.get("/", headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 429


def test_rate_limit_flattens_repeated_xff_header_lines():
    # A proxy may append its OWN X-Forwarded-For line rather than extending the first.
    # request.headers.get() returns only the first (attacker-forgeable) line; getlist +
    # flatten sees the real client the trusted hop appended. Also strips :port.
    from starlette.requests import Request
    from app.candisift.adapters.http.security import RateLimitMiddleware
    mw = RateLimitMiddleware(app=None, per_minute=1, trusted_proxy_count=1)
    scope = {"type": "http", "client": ("10.9.9.9", 5000),
             "headers": [(b"x-forwarded-for", b"1.1.1.1"),          # forged line
                         (b"x-forwarded-for", b"203.0.113.7:55")]}  # real, appended by LB
    assert mw._client_ip(Request(scope)) == "203.0.113.7"          # not the forged 1.1.1.1


# ---- health probe: real checks, not a static 200 --------------------------
# The Docker HEALTHCHECK hits /health. A static {"status":"ok"} stayed green with a
# corrupt/locked/unreachable DB, so the orchestrator never replaced a container that
# had stopped being able to do any work.

def test_health_probe_503_when_db_unreachable(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app.candisift.adapters.http.app import create_app
    from app.candisift.adapters.http.container import build_container

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)      # offline stub
    c = build_container(Settings(db_url=f"sqlite:///{tmp_path/'health.db'}",
                                 basic_auth_pass="testpass", worker_concurrency=1))
    with TestClient(create_app(c)) as client:
        r = client.get("/health")
        assert r.status_code == 200 and r.json() == {"status": "ok", "db": True, "worker": True}

        def _boom(*a, **k):
            raise OSError("disk I/O error")

        monkeypatch.setattr(c.queue._engine, "connect", _boom)   # DB goes away
        r = client.get("/health")
        assert r.status_code == 503
        assert r.json()["db"] is False


# ---- hard filter: reject on conflict, flag on missing evidence -------------
# The governing rule of the gate. "We could not parse it" is not the same fact as
# "they do not qualify", and only the second one may auto-reject. Each test below
# pins one gate that used to reject a qualified candidate on an extraction miss or
# a spelling coincidence.

def test_knockout_matches_whole_words_only():
    # "java" must not knock out a JavaScript dev; "intern" not an "international" one.
    jd = JDSpec(knockouts=["java", "intern"])
    prof = CandidateProfile(summary="international payments work", total_years=5,
                            skills=[SkillItem(name="JavaScript")])
    ok, reasons, _ = services.hard_filter(prof, jd)
    assert ok, reasons
    # the real term still knocks out
    ok, reasons, _ = services.hard_filter(
        CandidateProfile(total_years=5, skills=[SkillItem(name="Java")]), jd)
    assert not ok and any("knockout" in r for r in reasons)


def test_work_auth_alias_and_unparseable_routes_to_human():
    jd = JDSpec(required_work_auth=["US Citizen", "Green Card"])
    # a phrasing the alias table knows -> matches without a human
    ok, _, flags = services.hard_filter(
        CandidateProfile(work_authorization="U.S. citizen"), jd)
    assert ok and not flags
    # permanent resident == green card
    ok, _, _ = services.hard_filter(
        CandidateProfile(work_authorization="Lawful Permanent Resident"), jd)
    assert ok
    # a recognized CONFLICTING status still rejects for free
    ok, reasons, _ = services.hard_filter(
        CandidateProfile(work_authorization="H-1B, requires sponsorship"), jd)
    assert not ok and any("work auth" in r for r in reasons)
    # unrecognized free text -> flag, never an auto-reject on a legal gate
    ok, reasons, flags = services.hard_filter(
        CandidateProfile(work_authorization="eligible per company policy"), jd)
    assert ok and not reasons and any("could not be matched" in f for f in flags)


def test_work_auth_authorized_does_not_imply_citizenship():
    # the implication is one-directional: citizenship proves work authorization,
    # being authorized to work does not prove citizenship.
    ok, _, flags = services.hard_filter(
        CandidateProfile(work_authorization="Authorized to work in the United States"),
        JDSpec(required_work_auth=["Authorized to work in the US"]))
    assert ok and not flags
    ok, reasons, _ = services.hard_filter(
        CandidateProfile(work_authorization="Authorized to work in the United States"),
        JDSpec(required_work_auth=["US Citizen"]))
    assert not ok and any("work auth" in r for r in reasons)


def test_cert_gate_matches_aliases_and_flags_when_none_extracted():
    jd = JDSpec(required_certs=["Certified Kubernetes Administrator"])
    ok, _, _ = services.hard_filter(CandidateProfile(certifications=["CKA"]), jd)
    assert ok
    ok, _, _ = services.hard_filter(
        CandidateProfile(certifications=["AWS Solutions Architect (Associate)"]),
        JDSpec(required_certs=["AWS Certified Solutions Architect – Associate"]))
    assert ok
    # certs extracted but the required one genuinely absent -> reject
    ok, reasons, _ = services.hard_filter(CandidateProfile(certifications=["PMP"]), jd)
    assert not ok and any("certs" in r for r in reasons)
    # nothing extracted at all -> can't tell "has none" from "we missed it" -> flag
    ok, reasons, flags = services.hard_filter(CandidateProfile(), jd)
    assert ok and not reasons and any("no certifications extracted" in f for f in flags)


def test_years_gate_unknown_flags_near_miss_flags_real_shortfall_rejects():
    jd = JDSpec(min_years=5)
    # extraction produced nothing to measure -> flag, not "0y < 5y"
    ok, reasons, flags = services.hard_filter(CandidateProfile(), jd)
    assert ok and not reasons and any("not established" in f for f in flags)
    # 4.9y against "5+" is a recruiter call, not a knockout
    ok, reasons, flags = services.hard_filter(CandidateProfile(total_years=4.9), jd)
    assert ok and not reasons and any("near miss" in f for f in flags)
    # a real shortfall still rejects for free
    ok, reasons, _ = services.hard_filter(CandidateProfile(total_years=2), jd)
    assert not ok and any("experience" in r for r in reasons)


def test_location_gate_ignores_llm_inferred_candidate_remote_flag():
    # profile.remote_ok is a model guess with no evidence behind it; it must not arm
    # a deterministic reject on a role the JD itself marks remote-friendly.
    jd = JDSpec(locations=["Hyderabad"], remote_ok=True)
    ok, reasons, _ = services.hard_filter(
        CandidateProfile(location="Pune", remote_ok=False), jd)
    assert ok and not reasons


# ---- hard filter edge cases ----------------------------------------------
# A negated phrase contains the phrase it negates. Every gate that matches on text is
# exposed to it, and the failure is silent — the same class verdict_guard's fraud veto
# already handles.

def test_work_auth_negated_status_does_not_satisfy_the_gate():
    # "Not a US Citizen" contains "us citizen" -> a substring reading walks a candidate
    # straight through a LEGAL gate. It states what they are NOT, which doesn't tell us
    # what they ARE (a green-card holder may still qualify) -> unknown -> human.
    assert services.work_auth_satisfies("Not a US Citizen", ["US Citizen"]) is None
    assert services.work_auth_satisfies("US Citizen", ["US Citizen"]) is True
    assert services.work_auth_satisfies(
        "Not authorized to work in the US", ["Authorized to work in the US"]) is None
    # a negator inside an alias must not veto the alias itself
    assert services.work_auth_satisfies(
        "does not require sponsorship", ["Authorized to work in the US"]) is True
    # an affirmed conflicting status still rejects for free
    assert services.work_auth_satisfies("requires sponsorship", ["US Citizen"]) is False


def test_relocation_intent_ignores_negated_phrasing():
    assert services.states_relocation_intent("Willing to relocate")
    assert not services.states_relocation_intent("Not willing to relocate")
    assert not services.states_relocation_intent("Cannot relocate")
    # ...and the gate rejects the negated one rather than flagging it as willing
    jd = JDSpec(locations=["Hyderabad"], remote_ok=False)
    ok, reasons, _ = services.hard_filter(
        CandidateProfile(location="Pune", summary="Not willing to relocate"), jd)
    assert not ok and any("location" in r for r in reasons)


def test_location_matches_on_word_boundaries_and_skips_blank_entries():
    assert not services.location_matches("Bathinda", ["Bath"])          # substring != city
    assert services.location_matches("New York, NY", ["New York"])
    # containment is deliberate: a resume's "Greater Bangalore Area" must match
    assert services.location_matches("Greater Bangalore Area", ["Bangalore"])
    # a blank spec entry is a substring of everything -> would disable the gate
    assert not services.location_matches("Pune", [""])
    ok, reasons, _ = services.hard_filter(
        CandidateProfile(location="Pune"), JDSpec(locations=[""], remote_ok=False))
    assert ok and not reasons          # nothing to match on -> no reject


def test_work_mode_preference_in_location_field_is_unknown_not_conflict():
    # "Remote" in the location field states a preference, not a city: the candidate's
    # actual location is unknown, so it must flag rather than reject.
    assert not services.is_place("Remote") and services.is_place("Hyderabad")
    ok, reasons, flags = services.hard_filter(
        CandidateProfile(location="Remote"), JDSpec(locations=["Hyderabad"], remote_ok=False))
    assert ok and not reasons and any("location not stated" in f for f in flags)


def test_negation_is_clause_scoped_not_char_windowed():
    # one clause-boundary helper backs work-auth, relocation and knockout. A negation in
    # a DIFFERENT clause must not veto an affirmed phrase, and one in the SAME clause must.
    assert services.work_auth_satisfies(
        "Not an H1B holder. US citizen, no sponsorship required.", ["US Citizen"]) is True
    assert services.work_auth_satisfies("Not a US Citizen", ["US Citizen"]) is None
    assert services.states_relocation_intent("Cannot travel but willing to relocate")
    assert not services.states_relocation_intent("Not willing to relocate")
    assert not services._knockout_hit("java", "no experience with java")
    assert services._knockout_hit("java", "deep java expertise")


def test_visa_holders_satisfy_authorized_but_not_citizenship():
    # an H1B/OPT/TN holder IS authorized to work; rejecting them from an "authorized to
    # work" role is rejecting on non-conflicting evidence.
    for status in ("H1B visa", "STEM OPT", "TN visa", "L-1 visa"):
        assert services.work_auth_satisfies(status, ["Authorized to work in the US"]) is True
        assert services.work_auth_satisfies(status, ["US Citizen"]) is False   # citizenship not implied
    # hyphenated official spellings resolve
    assert "h1b" in services._auth_statuses("H-1B visa")


def test_cert_alias_expands_and_tiers_stay_distinct():
    assert services.cert_satisfied("Certified Kubernetes Administrator", ["CKA"])
    assert services.cert_satisfied("PMP", ["PMP (2021)"])          # trailing year tolerated
    assert services.cert_satisfied("AZ-900", ["Microsoft Azure Fundamentals"])
    # tier words are load-bearing: an Associate must not satisfy a Professional requirement
    assert not services.cert_satisfied("AWS SAP", ["AWS SAA"])


def test_strip_pii_keeps_signal_and_scrubs_github():
    from app.candisift.domain.models import SkillItem as SI
    p = CandidateProfile(
        name="Ruby Patel", location="Remote", github_url="https://github.com/rubypatel",
        github_projects=[{"name": "svc", "github_url": "https://github.com/rubypatel/svc",
                          "technologies": ["go"]}],
        summary="Remote-first. Shipped 2020-01-15; cut spend 12.500.000 EUR.",
        skills=[SI(name="ruby", evidence="10 years of Ruby on Rails")])
    clean = services.strip_pii(p)
    # name-token that is also a skill is NOT redacted (would gut the evidence)
    assert "Ruby" in clean.skills[0].evidence
    # a work-mode 'location' is not identity -> "remote" survives; metrics/ISO dates survive
    assert "remote" in clean.summary.lower()
    assert "2020-01-15" in clean.summary and "12.500.000" in clean.summary
    # github handle never reaches the model
    assert clean.github_url == "" and "github_url" not in clean.github_projects[0]
    assert clean.github_projects[0]["technologies"] == ["go"]


def test_missing_end_date_does_not_fabricate_experience():
    # a blank end on an OLD role must not run it to today (inflating years) or invent a
    # concurrent-employment fraud signal; only the latest-starting role is "present".
    prof = CandidateProfile(total_years=7.0, work_entries=[
        WorkEntry(company="Old", start_date="2010-01", end_date=""),
        WorkEntry(company="New", start_date="2015-01", end_date="2020-01")])
    out = services.validate_experience(prof)
    assert out.total_years == 7.0 and not out.concurrent_fulltime


def test_parse_date_rejects_whitespace_only():
    assert services._parse_date("  ") is None and services._parse_date(".") is None


def test_knockout_ignores_negated_mention():
    # "does not require sponsorship" contains the knockout and means the opposite —
    # knocking it out rejects the candidate for saying the reassuring thing.
    jd = JDSpec(knockouts=["requires sponsorship"])
    for stated in ("Does not require sponsorship", "No sponsorship required"):
        ok, reasons, _ = services.hard_filter(CandidateProfile(summary=stated), jd)
        assert ok, (stated, reasons)
    ok, reasons, _ = services.hard_filter(
        CandidateProfile(summary="Requires sponsorship for employment"), jd)
    assert not ok and any("knockout" in r for r in reasons)


def test_blank_spec_entries_gate_nobody():
    # extraction emits blank entries; a spec artefact must not decide anyone's screen.
    # A lone "" rejected EVERY candidate on the auth/cert gates (nothing to match) and
    # matched every candidate on location ("" is a substring of anything).
    prof = CandidateProfile(location="Pune", total_years=5)
    for jd in (JDSpec(required_work_auth=[""]), JDSpec(required_certs=[""]),
               JDSpec(locations=[""], remote_ok=False), JDSpec(knockouts=[""])):
        ok, reasons, flags = services.hard_filter(prof, jd)
        assert ok and not reasons and not flags, (jd, reasons, flags)


def test_blank_candidate_cert_entry_is_not_evidence_of_having_certs():
    # certifications=[""] is extraction noise, not "has certs" -> must flag, not reject
    ok, reasons, flags = services.hard_filter(
        CandidateProfile(certifications=[""]), JDSpec(required_certs=["PMP"]))
    assert ok and not reasons and any("no certifications extracted" in f for f in flags)


def test_zero_years_with_work_history_flags_rather_than_rejects():
    # work history present + 0 total = extraction failed (validate_experience leaves
    # total_years alone when no date parses). Rejecting there rejects on a parser miss.
    prof = CandidateProfile(total_years=0,
                            work_entries=[WorkEntry(company="A", start_date="sometime")])
    ok, reasons, flags = services.hard_filter(prof, JDSpec(min_years=5))
    assert ok and not reasons and any("not established" in f for f in flags)


# ---- experience extraction no longer undercounts ---------------------------

def test_parse_date_handles_common_resume_formats():
    assert services._parse_date("03/2021") == (2021, 3)
    assert services._parse_date("2021-03") == (2021, 3)
    assert services._parse_date("Jan. 2020") == (2020, 1)
    assert services._parse_date("January 2020") == (2020, 1)
    assert services._parse_date("Summer 2020") == (2020, 1)
    assert services._parse_date("garbage") is None


def test_validate_experience_does_not_undercount_on_partial_parse():
    # one entry's dates parse, one doesn't -> the computed total covers only part of
    # the career, so it must not overwrite a larger stated total into a rejection.
    prof = CandidateProfile(
        total_years=8,
        work_entries=[
            WorkEntry(company="A", start_date="2022-01", end_date="2024-01"),
            WorkEntry(company="B", start_date="sometime in the 2010s", end_date="???"),
        ],
    )
    out = services.validate_experience(prof)
    assert out.total_years == 8            # stated total kept, not clobbered to ~2
    # when EVERY entry parses, the computed total is authoritative (anti-hallucination)
    full = CandidateProfile(
        total_years=30,
        work_entries=[WorkEntry(company="A", start_date="2022-01", end_date="2024-01")],
    )
    assert services.validate_experience(full).total_years < 30


# ---- PII: location + phone shapes + education dates ------------------------

def test_strip_pii_removes_location_phone_shape_and_education_dates():
    from app.candisift.domain.models import EducationEntry
    p = CandidateProfile(
        name="Asha Rao", location="Lagos, Nigeria", phone="+15551234567",
        summary="Based in Lagos. Reach me on (555) 123-4567. Cut infra cost by 1500000 INR.",
        education=[EducationEntry(institution="Indian Institute of Technology",
                                  degree="BTech", start_date="2004", end_date="2008")],
    )
    clean = services.strip_pii(p)
    assert "lagos" not in clean.summary.lower()          # national-origin proxy gone
    assert "555" not in clean.summary                    # differently-formatted phone gone
    assert "1500000" in clean.summary                    # ...but the metric survives
    assert clean.education[0].start_date == "" and clean.education[0].end_date == ""
    # word-boundary redaction must not mangle a legitimate institution name
    assert clean.education[0].institution == "Indian Institute of Technology"


# ---- bias tripwire must not auto-act on prose ------------------------------

def test_pronoun_is_soft_bias_named_class_is_hard():
    assert guardrails.bias_hits_are_soft(["his"])
    assert not guardrails.bias_hits_are_soft(["female"])
    assert not guardrails.bias_hits_are_soft(["his", "female"])   # any hard hit -> hard
    assert not guardrails.bias_hits_are_soft([])


# ---- recruiter override of an auto-reject ---------------------------------
# An automated employment-decision tool a human cannot overrule is the thing
# regulators actually object to. The hard filter is a cost gate, not a verdict.

def test_recruiter_can_override_hard_filter_reject(tmp_path, monkeypatch):
    from app.candisift.adapters.http.container import build_container
    from app.candisift.domain.models import Candidate, Job

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)          # offline stub
    c = build_container(Settings(db_url=f"sqlite:///{tmp_path/'ovr.db'}",
                                 basic_auth_pass="testpass", worker_concurrency=1))
    # spec built directly: the offline stub extractor doesn't parse a city out of JD prose
    job = Job(id="job-ovr", title="Backend", raw_text="Backend engineer, Hyderabad",
              spec=JDSpec(title="Backend", must_have_skills=["python"],
                          locations=["Hyderabad"], remote_ok=False))
    c.jobs.add(job)
    # a candidate the location gate rejects and the recruiter disagrees with
    cand = Candidate(id="cand-ovr", dedup_key="k-ovr",
                     profile=CandidateProfile(location="Pune", total_years=9,
                                              skills=[SkillItem(name="Python")]))
    c.candidates.add(cand)

    rejected = c.service.screen(cand.id, job.id)
    assert not rejected.passed_hard_filters and rejected.synthesis is None

    out = c.service.screen(cand.id, job.id, override_hard_filter=True)
    assert out.passed_hard_filters and out.synthesis is not None    # cache busted, LLM ran
    assert out.hard_filter_overridden
    # the overruled reasons stay visible rather than being erased
    assert any("overridden" in f and "Pune" in f for f in out.filter_reasons)
    assert any(e["event"] == "screen.hard_filter_overridden"
               for e in c.audit.events_for_job(job.id))          # bypass is on the record


def _override_fixture(tmp_path, monkeypatch, dbname):
    from app.candisift.adapters.http.container import build_container
    from app.candisift.domain.models import Candidate, Job
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    c = build_container(Settings(db_url=f"sqlite:///{tmp_path/dbname}",
                                 basic_auth_pass="testpass", worker_concurrency=1))
    c.jobs.add(Job(id="j", title="B", raw_text="x",
                   spec=JDSpec(title="B", must_have_skills=["python"],
                               locations=["Hyderabad"], remote_ok=False)))
    c.candidates.add(Candidate(id="cd", dedup_key="k",
                               profile=CandidateProfile(location="Pune", total_years=9,
                                                        skills=[SkillItem(name="Python")])))
    return c


def test_override_survives_flagless_worker_retry(tmp_path, monkeypatch):
    # at-least-once delivery: handle_screen_task re-runs screen() with NO override flag.
    # The override is durable state on the result, so the retry must not re-reject.
    c = _override_fixture(tmp_path, monkeypatch, "retry.db")
    c.service.screen("cd", "j", override_hard_filter=True)
    c.service.handle_screen_task({"candidate_id": "cd", "job_id": "j"})   # worker retry
    r = c.results.get("j.cd")
    assert r.passed_hard_filters and r.hard_filter_overridden          # not reverted
    assert r.synthesis is not None                                    # paid work kept


def test_override_survives_fingerprint_change(tmp_path, monkeypatch):
    # a JD edit / model swap / toggle change flips the fingerprint and discards the
    # cached result; the re-screen carries no flag, but the human's decision must
    # outlive it. Flipping the hr_eval toggle changes the fingerprint (see
    # _models_fingerprint) without needing a job-mutation path jobs don't expose.
    c = _override_fixture(tmp_path, monkeypatch, "fp.db")
    c.service.screen("cd", "j", override_hard_filter=True)
    c.service._hr_eval = not c.service._hr_eval   # flips the fingerprint
    r = c.service.screen("cd", "j")               # re-screen, no flag
    assert r.passed_hard_filters and r.hard_filter_overridden


def test_ready_does_not_disclose_queue_depth_or_llm_key(tmp_path, monkeypatch):
    # /ready is unauthenticated: queue depth (pipeline volume) and whether an LLM key
    # is configured (whether spend is real) are free recon. Booleans only; the detailed
    # view stays behind auth at GET /api/queue.
    from fastapi.testclient import TestClient
    from app.candisift.adapters.http.app import create_app
    from app.candisift.adapters.http.container import build_container

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    c = build_container(Settings(db_url=f"sqlite:///{tmp_path/'ready.db'}",
                                 basic_auth_pass="testpass", worker_concurrency=1))
    with TestClient(create_app(c)) as client:
        body = client.get("/ready").json()
    assert "queue" not in body and "llm" not in body
    assert all(isinstance(v, bool) for k, v in body.items() if k != "status")

"""End-to-end TestClient run on the offline stub LLM. Exercises auth, security
headers, the estimate-first upload flow, the durable worker, results, decisions,
and upload guardrails."""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from app.candisift.config import Settings
from app.candisift.adapters.http.app import create_app
from app.candisift.adapters.http.container import build_container

AUTH = ("recruiter", "testpass")

STRONG = b"Asha Rao\nasha@x.com\nSenior Backend Engineer\n8 years Python. Led Kubernetes migration. PostgreSQL owner."
WEAK = b"Sam Lee\nJunior Developer\n2 years JavaScript. requires sponsorship."


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)  # force offline stub
    settings = Settings(db_url=f"sqlite:///{tmp_path/'ats.db'}", basic_auth_pass="testpass",
                        env="dev", rate_limit_per_min=10_000)
    app = create_app(build_container(settings))
    with TestClient(app) as c:
        yield c


def _poll_results(client, job_id, want, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/jobs/{job_id}/results", auth=AUTH)
        if r.status_code == 200 and len(r.json()) >= want:
            return r.json()
        time.sleep(0.3)
    raise AssertionError(f"timed out waiting for {want} results")


def test_auth_required(client):
    assert client.get("/").status_code == 401
    assert client.get("/api/models").status_code == 401


def test_security_headers_and_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert r.headers["content-security-policy"]
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"
    assert "x-request-id" in r.headers


def test_models_catalog(client):
    ids = [m["id"] for m in client.get("/api/models", auth=AUTH).json()]
    assert "auto" in ids and "claude-opus-4-8" in ids


def test_full_flow_estimate_confirm_screen(client):
    job = client.post("/api/jobs", auth=AUTH, json={
        "jd_text": "Senior Backend Engineer. Must have Python, Kubernetes, PostgreSQL. 5+ years.",
        "persona_model": "claude-haiku-4-5", "synth_model": "claude-opus-4-8",
    }).json()
    job_id = job["id"]

    # upload -> staged, estimate returned, NOTHING screened yet
    up = client.post(f"/api/jobs/{job_id}/upload", auth=AUTH, files=[
        ("files", ("strong.txt", STRONG, "text/plain")),
        ("files", ("weak.txt", WEAK, "text/plain")),
    ]).json()
    assert up["staged"] == 2
    assert up["estimate"]["estimated_total_usd"] > 0
    assert up["estimate"]["price_source"] in ("genai-prices", "static-fallback")
    assert client.get(f"/api/jobs/{job_id}/results", auth=AUTH).json() == []  # held

    # confirm -> worker processes
    rel = client.post(f"/api/jobs/{job_id}/confirm", auth=AUTH, json={"task_ids": up["task_ids"]}).json()
    assert rel["released"] == 2

    results = _poll_results(client, job_id, want=2)
    passed = [r for r in results if r["passed_hard_filters"]]
    rejected = [r for r in results if not r["passed_hard_filters"]]
    assert len(passed) == 1 and len(rejected) == 1
    assert passed[0]["synthesis"]["recommendation"] == "shortlist"
    assert "experience" in " ".join(rejected[0]["filter_reasons"])

    # breakdown + decision
    rid = passed[0]["id"]
    bd = client.get(f"/api/results/{rid}", auth=AUTH).json()
    assert bd["candidate"] is not None and bd["result"]["tech"]["depth_score"] > 0
    assert client.post(f"/api/results/{rid}/decision", auth=AUTH,
                       json={"decision": "accepted"}).status_code == 200

    audit = client.get(f"/api/jobs/{job_id}/bias-audit", auth=AUTH).json()
    assert audit["total"] == 2


def test_upload_rejects_bad_extension(client):
    job = client.post("/api/jobs", auth=AUTH, json={"jd_text": "Python role"}).json()
    r = client.post(f"/api/jobs/{job['id']}/upload", auth=AUTH,
                    files=[("files", ("malware.exe", b"MZ", "application/octet-stream"))])
    assert r.status_code == 400


def test_ui_dashboard_renders(client):
    r = client.get("/ats", auth=AUTH)          # ATS dashboard lives at /ats; "/" redirects here
    assert r.status_code == 200 and "New role" in r.text


def test_root_redirects_to_ats(client):
    r = client.get("/", auth=AUTH, follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/ats"


def test_hr_eval_flag_gates_the_call(tmp_path):
    """hr_eval_enabled=False skips the advisory HR persona call (cost), and the
    screen still completes; =True keeps it. Locks the cost knob's behaviour."""
    def screen(hr_on):
        s = Settings(db_url=f"sqlite:///{tmp_path/('on' if hr_on else 'off')}.db",
                     hr_eval_enabled=hr_on, env="dev")
        c = build_container(s)
        job = c.service.create_job("Backend Engineer. Must have Python. 1+ years.")
        cand = c.service.ingest_resume(STRONG, "strong.txt")
        return c.service.screen(cand.id, job.id)

    on, off = screen(True), screen(False)
    assert on.passed_hard_filters and off.passed_hard_filters
    assert on.hr is not None and off.hr is None        # the call is gated
    assert on.synthesis is not None and off.synthesis is not None  # screen still completes


def test_linkedin_enrichment_and_audit(tmp_path):
    """A resume with a LinkedIn URL gets a resume-derived linkedin_profile digest
    (no external API) and an audit event is recorded. No URL -> no enrichment, no PII."""
    s = Settings(db_url=f"sqlite:///{tmp_path/'li.db'}.db", env="dev")
    c = build_container(s)
    resume = (b"Asha Rao\nasha@x.com\nlinkedin.com/in/asha-rao\n"
              b"Senior Backend Engineer\n8 years Python. Led Kubernetes migration. PostgreSQL owner.")
    cand = c.service.ingest_resume(resume, "asha.txt")

    li = cand.profile.linkedin_profile
    assert li and (li.get("positions") or li.get("skills"))      # digest produced
    blob = str(li).lower()
    assert "asha@x.com" not in blob and "asha rao" not in blob   # professional content only, no contact PII

    events = [e["event"] for e in c.audit.events_for_job("", limit=200)]
    assert "candidate.linkedin_enriched" in events               # audit trail recorded

    # no LinkedIn URL -> enrichment skipped entirely
    plain = c.service.ingest_resume(b"Sam Lee\nDeveloper\n2 years Go.", "sam.txt")
    assert plain.profile.linkedin_profile == {}


def test_provider_routing_and_pricing():
    """Local (ollama/) and any-OpenAI-compatible (openai/) ids route to the right
    provider and don't crash cost estimation; bare ids keep their old routing."""
    from app.candisift import pricing
    from app.candisift.config import Settings

    assert pricing.provider_for("ollama/llama3.1") == "ollama"
    assert pricing.provider_for("openai/meta-llama/Meta-Llama-3.1-70B-Instruct") == "openai"
    assert pricing.provider_for("gpt-4o") == "openai"
    assert pricing.provider_for("claude-haiku-4-5") == "anthropic"
    assert pricing.provider_for("llama-3.3-70b-versatile") == "groq"   # bare llama still groq

    # local / custom-endpoint price is unknown -> 0, never raises on the estimate path
    assert pricing.call_cost("ollama/llama3.1", 4000, 4000) == 0.0
    est = pricing.estimate_batch(5, "ollama/llama3.1", "openai/foo/bar")
    assert est["estimated_total_usd"] == 0.0

    # a local Ollama model is "available" with no API key set
    assert Settings(env="dev").has_key_for("ollama/llama3.1") is True

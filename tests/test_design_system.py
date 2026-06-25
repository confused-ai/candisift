"""WS1 design-system foundation: ds.css served + Jinja macros render all states."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from jinja2 import Environment, FileSystemLoader

from app.candisift.config import Settings
from app.candisift.adapters.http.app import create_app
from app.candisift.adapters.http.container import build_container

AUTH = ("recruiter", "testpass")
CANDISIFT_TEMPLATES = Path("app/candisift/adapters/http/templates")


@pytest.fixture()
def client(tmp_path):
    settings = Settings(db_url=f"sqlite:///{tmp_path/'ats.db'}", basic_auth_pass="testpass",
                        env="dev", rate_limit_per_min=10_000)
    with TestClient(create_app(build_container(settings))) as c:
        yield c


def _render(macro_call: str) -> str:
    """Render a single _ds.html macro in isolation."""
    env = Environment(loader=FileSystemLoader(str(CANDISIFT_TEMPLATES)), autoescape=True)
    tmpl = env.from_string("{% import '_ds.html' as ds %}" + macro_call)
    return tmpl.render()


# --- Task 1: tokens served --------------------------------------------------

def test_ds_css_served_with_tokens(client):
    r = client.get("/static/ds.css", auth=AUTH)
    assert r.status_code == 200
    css = r.text
    for tok in ("--bg", "--surface", "--accent", "--ok", "--warn", "--bad", "--info",
                "--sp-1", "--sp-8", "--fs-base", "--r", "--font-sans"):
        assert tok in css, f"missing token {tok}"
    assert "prefers-color-scheme: dark" in css
    assert "prefers-reduced-motion" in css


# --- Task 2: status() -------------------------------------------------------

def test_status_known_kinds():
    out = _render("{{ ds.status('Completed', 'ok') }}")
    assert "status" in out and "is-ok" in out and "Completed" in out


def test_status_unknown_kind_falls_back_to_neutral():
    out = _render("{{ ds.status('Queued', 'queued') }}")
    assert "is-neutral" in out
    assert "is-queued" not in out


def test_status_escapes_text():
    out = _render("{{ ds.status('<script>', 'ok') }}")
    assert "<script>" not in out and "&lt;script&gt;" in out


# --- Task 3: btn() ----------------------------------------------------------

def test_btn_variants_and_states():
    assert "btn-primary" in _render("{{ ds.btn('Save') }}")
    assert "btn-danger" in _render("{{ ds.btn('Delete', variant='danger') }}")
    out = _render("{{ ds.btn('Saving', loading=True) }}")
    assert "is-loading" in out and 'aria-busy="true"' in out
    assert "disabled" in _render("{{ ds.btn('Off', disabled=True) }}")


def test_btn_submit_type_and_name():
    out = _render("{{ ds.btn('Go', type='submit', name='action', value='accept') }}")
    assert 'type="submit"' in out and 'name="action"' in out and 'value="accept"' in out


# --- Task 4: field() --------------------------------------------------------

def test_field_error_sets_aria_invalid():
    out = _render("{{ ds.field('email', 'Email', value='x', error='Required') }}")
    assert 'aria-invalid="true"' in out and "Required" in out and "field-error" in out


def test_field_help_and_required():
    out = _render("{{ ds.field('name', 'Name', help='Your full name', required=True) }}")
    assert "Your full name" in out and "field-required" in out and "required" in out


# --- Task 5: empty/flash/confirm/breadcrumb --------------------------------

def test_empty_with_cta():
    out = _render("{{ ds.ds_empty('No jobs yet', body='Create one to begin', cta_label='New job', cta_href='/jobs/new') }}")
    assert "ds-empty" in out and "No jobs yet" in out and 'href="/jobs/new"' in out


def test_flash_kinds_and_escape():
    assert "flash-bad" in _render("{{ ds.flash('bad', 'Upload failed') }}")
    out = _render("{{ ds.flash('ok', '<b>x</b>') }}")
    assert "<b>x</b>" not in out and "&lt;b&gt;" in out


def test_confirm_form_wraps_destructive_post():
    out = _render("{{ ds.confirm_form('/results/1/decide', 'Reject', 'Reject this candidate?', name='decision', value='reject') }}")
    assert 'action="/results/1/decide"' in out and 'method="post"' in out
    assert "data-confirm" in out and "Reject this candidate?" in out


def test_breadcrumb_renders_trail():
    out = _render("{{ ds.breadcrumb([('Jobs', '/ats'), ('Acme', '/jobs/1'), ('Optimize', None)]) }}")
    assert 'href="/ats"' in out and "Optimize" in out and "breadcrumb" in out


# --- Task 6: kpi/bar/ring ---------------------------------------------------

def test_kpi_null_shows_dash():
    out = _render("{{ ds.kpi('Spend', none) }}")
    assert "—" in out


def test_bar_clamps_over_max():
    out = _render("{{ ds.ds_bar(150, 100) }}")
    assert "width:100%" in out.replace(" ", "")
    out0 = _render("{{ ds.ds_bar(0, 0) }}")
    assert "width:0%" in out0.replace(" ", "")


def test_ring_not_scored_state():
    out = _render("{{ ds.ds_ring(none, 100, scored=False) }}")
    assert "is-not-scored" in out
    out2 = _render("{{ ds.ds_ring(120, 100) }}")
    assert "--p:100" in out2


# --- Task 8: app.js progressive enhancements -------------------------------

def test_app_js_has_progressive_enhancements(client):
    js = client.get("/static/app.js", auth=AUTH).text
    for marker in ("data-confirm", "data-flash-close", "is-dragover", "is-loading"):
        assert marker in js, f"app.js missing {marker} handler"


# --- Task 9: base links ds.css + deprecated aliases -------------------

def test_base_links_ds_css(client):
    ats = client.get("/dashboard", auth=AUTH).text
    assert "/static/ds.css" in ats, "CandiSift base does not link ds.css"


def test_pill_alias_present(client):
    css = client.get("/static/ds.css", auth=AUTH).text
    assert ".pill" in css and ".badge" in css

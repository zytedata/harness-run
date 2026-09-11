"""Run-scoped token expiry/renewal with fake clocks, stores and thread startup (B02)."""
import datetime as dt
from types import SimpleNamespace

import pytest
from google.auth import _helpers
from sandbox_fakes import FakeSandboxProvider, make_engine

from remote_agent_toolkit.runtime.gemini import backend, scoped_gcs


def test_client_preserves_expiry_and_writes_initial_refresh_before_thread(monkeypatch):
    expiry = dt.datetime(2099, 1, 1)
    seen = []
    monkeypatch.setattr(scoped_gcs, "mint_run_token", lambda *a: ("AUDIT_FAKE_TOKEN", expiry))
    monkeypatch.setattr(scoped_gcs, "write_run_token", lambda *a, **kw: seen.append(("write", a)))
    monkeypatch.setattr(scoped_gcs, "delete_run_token", lambda *a, **kw: None)
    monkeypatch.setattr(backend.threading, "Thread", lambda **kw: SimpleNamespace(
        start=lambda: seen.append(("thread", None))))
    session = make_engine(FakeSandboxProvider()).start_session()
    token, model_env = session._mint_tokens("t" * 32)
    assert token == "AUDIT_FAKE_TOKEN" and model_env["ANTHROPIC_AUTH_TOKEN"] == "fake-model-token"
    assert seen[0] == ("write", ("gs://out", session.session_id, "AUDIT_FAKE_TOKEN", expiry))
    assert seen[1][0] == "thread"
    assert session._gcs_token_expiry == expiry.isoformat()
    session._stop_refresh()
    assert session._gcs_token_expiry is None and session._refresh_stop is None


def test_worker_refreshes_before_transport_expiry(monkeypatch):
    initial = dt.datetime(2099, 1, 1, 1)
    fetched = []
    creds = scoped_gcs.worker_credentials(
        "AUDIT_OLD", "gs://audit-bucket", "audit", expiry=initial,
        fetch=lambda url, bearer: fetched.append(bearer) or {"token": "AUDIT_FRESH", "expiry": "2099-01-01T02:00:00"},
    )
    assert creds.expiry == initial
    monkeypatch.setattr(_helpers, "utcnow", lambda: initial - dt.timedelta(minutes=1))
    headers = {}
    creds.before_request(None, "GET", "https://audit.invalid", headers)
    assert fetched == ["AUDIT_OLD"]
    assert headers["authorization"] == "Bearer AUDIT_FRESH"


def test_worker_never_binds_a_non_expiring_credential():
    creds = scoped_gcs.worker_credentials("AUDIT_FAKE_TOKEN", "gs://audit-bucket", "audit")
    assert creds.expiry is not None  # a legacy/unknown expiry gets a conservative retry schedule


def test_invalid_expiry_cannot_be_silently_treated_as_unknown():
    assert scoped_gcs._parse_expiry("not-a-date") is None
    assert scoped_gcs._parse_expiry("2099-01-01T01:00:00+00:00") == dt.datetime(2099, 1, 1, 1)


def test_failed_initial_write_keeps_a_cleanup_handle_and_starts_no_thread(monkeypatch):
    def failed_write(*a, **kw):
        raise TimeoutError("dummy write failure")

    monkeypatch.setattr(scoped_gcs, "write_run_token", failed_write)
    monkeypatch.setattr(backend.threading, "Thread", lambda **kw: pytest.fail("unexpected thread"))
    session = make_engine(FakeSandboxProvider()).start_session()
    with pytest.raises(TimeoutError):
        session._mint_tokens("t" * 32)
    assert session._refresh_stop is None


def test_refresh_loop_re_mints_both_tokens_and_posts_the_model_token(monkeypatch):
    from remote_agent_toolkit.runtime.gemini import model_token

    minted = {"gcs": 0, "model": 0}
    written = []
    monkeypatch.setattr(scoped_gcs, "mint_run_token",
                        lambda *a: (minted.__setitem__("gcs", minted["gcs"] + 1) or f"gcs-{minted['gcs']}", None))
    monkeypatch.setattr(scoped_gcs, "write_run_token", lambda b, s, t, e, **kw: written.append(t))
    monkeypatch.setattr(scoped_gcs, "delete_run_token", lambda *a, **kw: None)
    monkeypatch.setattr(model_token, "mint_model_token",
                        lambda *a, **kw: (minted.__setitem__("model", minted["model"] + 1) or f"model-{minted['model']}", None))
    monkeypatch.setattr(backend, "TOKEN_REFRESH_S", 0.01)
    provider = FakeSandboxProvider()
    engine = make_engine(provider)
    session = engine.start_session()
    session._sandbox = provider.create(engine.resource, ttl_s=60, display_name="x").name
    turn_id = "t" * 32
    session._turn_id = turn_id
    session._mint_tokens(turn_id)
    import time
    deadline = time.time() + 2
    while not provider.worker(session._sandbox).tokens and time.time() < deadline:
        time.sleep(0.01)
    session._stop_refresh()
    assert written[0] == "gcs-1" and len(written) >= 2  # the initial write, then a refresh
    posted = provider.worker(session._sandbox).tokens[0]
    assert posted["turn_id"] == turn_id and posted["model_token"].startswith("model-")

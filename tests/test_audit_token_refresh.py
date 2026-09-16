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
    token, model_env, model_token = session._mint_tokens("t" * 32)
    assert token == "AUDIT_FAKE_TOKEN" and model_token == {"access_token": "fake-model-token", "expires_at": None}
    assert model_env["CLAUDE_CODE_USE_VERTEX"] == "1" and "ANTHROPIC_AUTH_TOKEN" not in model_env
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


def test_refresh_loop_re_mints_the_gcs_token(monkeypatch):
    minted = {"gcs": 0}
    written = []
    monkeypatch.setattr(scoped_gcs, "mint_run_token",
                        lambda *a: (minted.__setitem__("gcs", minted["gcs"] + 1) or f"gcs-{minted['gcs']}", None))
    monkeypatch.setattr(scoped_gcs, "write_run_token", lambda b, s, t, e, **kw: written.append(t))
    monkeypatch.setattr(scoped_gcs, "delete_run_token", lambda *a, **kw: None)
    monkeypatch.setattr(backend, "TOKEN_REFRESH_S", 0.01)
    session = make_engine(FakeSandboxProvider()).start_session()
    session._mint_tokens("t" * 32)
    import time
    deadline = time.time() + 2
    while len(written) < 2 and time.time() < deadline:
        time.sleep(0.01)
    session._stop_refresh()
    assert written[0] == "gcs-1" and len(written) >= 2  # the initial write, then a refresh


async def _open_turn(provider):
    """Start a turn on a worker that never finishes; return ``(session, run, worker)`` once dispatched."""
    import asyncio

    from sandbox_fakes import ScriptedWorker

    provider.worker_factory = lambda name: ScriptedWorker(auto=None)
    session = make_engine(provider).start_session()
    run = session.run("go")
    for _ in range(500):
        if session._sandbox is not None:
            break
        await asyncio.sleep(0.01)
    assert session._sandbox is not None
    return session, run, provider.worker(session._sandbox)


async def _wait(pred, timeout=3.0):
    import asyncio
    import time

    deadline = time.time() + timeout
    while not pred() and time.time() < deadline:
        await asyncio.sleep(0.01)
    return pred()


def test_refresh_loop_pushes_a_fresh_model_token_to_the_running_turns_worker(monkeypatch):
    import asyncio

    from remote_agent_toolkit.runtime.gemini import model_token

    minted = []
    monkeypatch.setattr(model_token, "mint_model_token",
                        lambda *a, **kw: minted.append(1) or (f"model-{len(minted)}", dt.datetime(2099, 1, 1)))
    monkeypatch.setattr(backend, "TOKEN_REFRESH_S", 0.02)
    provider = FakeSandboxProvider()

    async def go():
        session, run, worker = await _open_turn(provider)
        try:
            assert worker.turns[0]["model_token"] == {"access_token": "model-1", "expires_at": "2099-01-01T00:00:00"}
            assert "ANTHROPIC_AUTH_TOKEN" not in worker.turns[0]["model_env"]
            assert await _wait(lambda: len(worker.tokens) >= 2)
            first = worker.tokens[0]
            assert first["turn_id"] == session._turn_id
            assert first["model_token"]["access_token"] == "model-2"  # a NEW token, not the turn's
            assert first["model_token"]["expires_at"] == "2099-01-01T00:00:00"
        finally:
            session._stop_refresh()
            worker.finish()
            run.cancel()

    asyncio.run(go())


def test_a_model_token_push_the_worker_did_not_take_is_retried_soon(monkeypatch):
    import asyncio

    from remote_agent_toolkit.runtime.gemini import model_token

    monkeypatch.setattr(model_token, "mint_model_token", lambda *a, **kw: ("model", None))
    monkeypatch.setattr(backend, "TOKEN_REFRESH_S", 0.02)
    monkeypatch.setattr(backend, "TOKEN_REFRESH_RETRY_S", 0.02)
    provider = FakeSandboxProvider()

    async def go():
        session, run, worker = await _open_turn(provider)
        real_call, attempts = provider.call, []

        def flaky(sandbox, path, body=None, *, timeout_s=None):
            if path == "/token":
                attempts.append(1)
                if len(attempts) < 3:
                    raise ConnectionError("proxy hiccup")
            return real_call(sandbox, path, body, timeout_s=timeout_s)

        monkeypatch.setattr(provider, "call", flaky)
        try:
            # Two failures, then the worker takes it: the normal cadence is 60 s, the retry 20 ms.
            monkeypatch.setattr(backend, "TOKEN_REFRESH_S", 60.0)
            assert await _wait(lambda: len(worker.tokens) >= 1)
            assert len(attempts) == 3
        finally:
            session._stop_refresh()
            worker.finish()
            run.cancel()

    asyncio.run(go())


def test_an_api_key_engine_sends_no_model_token(monkeypatch):
    provider = FakeSandboxProvider()
    engine = make_engine(provider)
    engine._use_vertex = False
    session = engine.start_session()
    token, model_env, model_token = session._mint_tokens("t" * 32)
    assert model_env == {} and model_token is None
    assert session._refresh_model_token() is True  # nothing to push, nothing to retry
    session._stop_refresh()

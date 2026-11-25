"""Run-scoped token expiry/renewal with fake clocks, stores and thread startup."""
import datetime as dt
from types import SimpleNamespace

import pytest
from google.auth import _helpers
from sandbox_fakes import FakeSandboxProvider, make_engine

from harness_run.runtime.sandbox import backend, scoped_gcs


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
    # One fake clock for google-auth and for the worker's own horizon arithmetic.
    clock = {"now": dt.datetime(2026, 9, 18, 8, 0)}
    monkeypatch.setattr(_helpers, "utcnow", lambda: clock["now"])
    monkeypatch.setattr(scoped_gcs, "_now", lambda: clock["now"])
    initial = dt.datetime(2099, 1, 1, 1)
    fetched = []
    creds = scoped_gcs.worker_credentials(
        "AUDIT_OLD", "gs://audit-bucket", "audit", expiry=initial,
        fetch=lambda url, bearer: fetched.append(bearer) or {"token": "AUDIT_FRESH", "expiry": "2099-01-01T02:00:00"},
    )
    # A far-future expiry is clamped to the refresh horizon (#87): the replacement can only be
    # fetched WITH a live token, so the worker never parks on one token for that long.
    assert creds.expiry == clock["now"] + scoped_gcs.REFRESH_HORIZON
    clock["now"] = creds.expiry - dt.timedelta(minutes=1)
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

    from harness_run.runtime.sandbox import model_token

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

    from harness_run.runtime.sandbox import model_token

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


# -- which turns get a model token (only those whose model call goes to Vertex) ----------


def _both_harnesses_engine(provider):
    from harness_run.spec import AgentSpec

    return make_engine(provider, spec=AgentSpec(
        harness="claude-code", harnesses=("claude-code", "codex"), name="g", model="claude-sonnet-4-5"))


def _count_mints(monkeypatch):
    from harness_run.runtime.sandbox import model_token

    minted = []
    monkeypatch.setattr(model_token, "mint_model_token",
                        lambda *a, **kw: minted.append(1) or ("model", None))
    return minted


def test_model_calls_use_vertex_follows_where_the_model_call_goes():
    from harness_run.harness import model_calls_use_vertex
    from harness_run.spec import AgentSpec

    def spec(harness, model):
        return AgentSpec(harness=harness, name="g", model=model)

    assert model_calls_use_vertex(spec("claude-code", "claude-sonnet-4-5"))
    assert not model_calls_use_vertex(spec("codex", "gpt-5"))
    assert not model_calls_use_vertex(spec("claude-code", "openrouter/z-ai/glm-4.6"))
    assert not model_calls_use_vertex(spec("codex", "openrouter/openai/gpt-5"))


def test_a_claude_turn_on_a_vertex_engine_gets_the_model_token(monkeypatch):
    minted = _count_mints(monkeypatch)
    session = _both_harnesses_engine(FakeSandboxProvider()).start_session()
    _token, model_env, model_token = session._mint_tokens("t" * 32)
    assert model_token == {"access_token": "model", "expires_at": None}
    assert model_env["CLAUDE_CODE_USE_VERTEX"] == "1" and minted == [1]
    assert session._turn_has_model_token is True
    session._stop_refresh()


@pytest.mark.parametrize("config", [
    {"harness": "codex", "model": "gpt-5"},
    {"model": "openrouter/z-ai/glm-4.6"},
    {"harness": "codex", "model": "openrouter/openai/gpt-5"},
])
def test_a_turn_that_does_not_call_vertex_gets_no_model_token_and_none_is_minted(monkeypatch, config):
    from harness_run.config import SessionConfig

    minted = _count_mints(monkeypatch)
    session = _both_harnesses_engine(FakeSandboxProvider()).start_session()
    session._session_config = SessionConfig(**config)  # bound without the GCS record a real open writes
    _token, model_env, model_token = session._mint_tokens("t" * 32, session._client_spec())
    assert model_env == {} and model_token is None and minted == []
    # The 25-minute refresh has nothing to push either, even with a worker to push to.
    session._sandbox, session._turn_id = "sb", "t" * 32
    assert session._refresh_model_token() is True and minted == []
    session._stop_refresh()


def test_the_model_token_follows_each_turns_own_model(monkeypatch):
    """Turns of one session: Claude, then an OpenRouter model for one turn, then Claude again."""
    import asyncio

    from sandbox_fakes import ScriptedWorker, result_event

    from harness_run.config import TurnConfig

    minted = _count_mints(monkeypatch)
    provider = FakeSandboxProvider(lambda name: ScriptedWorker(auto=[result_event("done")]))
    engine = _both_harnesses_engine(provider)
    session = engine.start_session()

    async def go():
        await session.run("one")
        await session.run("two", config=TurnConfig(model="openrouter/z-ai/glm-4.6"))
        await session.run("three")

    asyncio.run(go())
    engine._join_background()
    bodies = [body for _sb, path, body in provider.calls if path == "/turn"]
    assert [b["model_token"] is not None for b in bodies] == [True, False, True]
    assert [bool(b["model_env"]) for b in bodies] == [True, False, True]
    assert len(minted) == 2


def test_an_adopter_stops_minting_once_the_worker_says_the_turn_has_no_model_token(monkeypatch):
    minted = _count_mints(monkeypatch)
    provider = FakeSandboxProvider()
    session = _both_harnesses_engine(provider).start_session()
    session._sandbox, session._turn_id = "sb", "t" * 32
    session._turn_has_model_token = None  # an adopted turn: started elsewhere
    pushes = []

    def call(sandbox, path, body=None, *, timeout_s=None):
        pushes.append(path)
        return {"ok": False, "error": "this turn runs without a model token", "no_model_token": True}

    monkeypatch.setattr(provider, "call", call)
    assert session._refresh_model_token() is True  # asks the worker once
    assert session._turn_has_model_token is False
    assert session._refresh_model_token() is True
    assert pushes == ["/token"] and minted == [1]


def test_a_handle_without_its_deploy_record_leaves_the_decision_to_the_worker(monkeypatch):
    minted = _count_mints(monkeypatch)
    engine = make_engine(FakeSandboxProvider())
    engine.spec = backend._fallback_spec("g")  # no harness known client-side
    session = engine.start_session()
    _token, _env, model_token = session._mint_tokens("t" * 32)
    assert model_token is not None and minted == [1]
    session._stop_refresh()

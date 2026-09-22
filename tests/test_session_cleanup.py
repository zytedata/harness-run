"""Cancellation and failed acquisition never leak the run's resources.

* a setup failure before dispatch (token minting, a config read) leaves no refresher and
  never reaches the platform;
* a dispatch failure ends the run with an explained error and releases every sandbox tried;
* interrupt on a turn whose worker cannot be reached releases the sandbox;
* a second submission while a turn runs is refused without touching the first's state;
* cleanup errors never mask the original failure or log secret material.
"""

import asyncio
from types import SimpleNamespace

import pytest
from sandbox_fakes import FakeSandboxProvider, ScriptedWorker, make_engine

from agent_run import TurnConfig
from agent_run.runtime.sandbox import backend, handoff, scoped_gcs
from agent_run.runtime.sandbox.provider import SandboxGone


async def _await(run):
    return await run


@pytest.mark.parametrize("phase", ["mint", "config", "model-token"])
def test_pre_dispatch_failure_leaves_no_refresher_and_never_dispatches(monkeypatch, phase):
    provider = FakeSandboxProvider()
    engine = make_engine(provider)
    session = engine.start_session()
    failure = RuntimeError("dummy setup failure")

    def fail(*a, **kw):
        raise failure

    if phase == "mint":
        monkeypatch.setattr(scoped_gcs, "mint_run_token", fail)
    if phase == "config":
        monkeypatch.setattr(session, "_resolve_session_config", fail)
    if phase == "model-token":
        from agent_run.runtime.sandbox import model_token

        monkeypatch.setattr(model_token, "mint_model_token", fail)
    with pytest.raises(RuntimeError) as caught:
        session.run("dummy", secrets={"KEY": "AUDIT_FAKE"}, config=TurnConfig(max_turns=1))
    assert caught.value is failure or caught.value.__cause__ is failure
    assert session._refresh_stop is None and session._current_run is None
    assert provider.calls == [] and provider.live() == []


def test_dispatch_failure_releases_every_sandbox_tried_and_stops_the_refresher(monkeypatch):
    provider = FakeSandboxProvider()
    # Every claimed sandbox is gone before the call reaches it: a definitive non-acceptance
    # (a lost answer is settled with the same sandbox instead; test_dispatch_ownership.py).
    provider.fail_next["/turn"] = [SandboxGone("expired")] * backend.DISPATCH_ATTEMPTS
    engine = make_engine(provider)
    session = engine.start_session()
    stops = []
    real_stop = session._stop_refresh
    monkeypatch.setattr(session, "_stop_refresh", lambda **kw: stops.append(1) or real_stop(**kw))
    result = asyncio.run(_await(session.run("dummy", secrets={"KEY": "AUDIT_FAKE"})))
    engine._join_background()
    assert result.is_error
    assert "could not hand the turn" in result.text and "AUDIT_FAKE" not in result.text
    assert len(provider.deleted) == backend.DISPATCH_ATTEMPTS and provider.live() == []
    assert stops and session._refresh_stop is None


def test_interrupt_on_an_unreachable_worker_releases_the_sandbox():
    provider = FakeSandboxProvider(lambda n: ScriptedWorker())  # never emits, deaf to control
    engine = make_engine(provider)
    session = engine.start_session()

    async def exercise():
        run = session.run("dummy")
        while session._sandbox is None:
            await asyncio.sleep(0.005)
        sandbox = session._sandbox
        provider.fail_next["/control"] = [RuntimeError("proxy down")]
        session._control_ready = True
        await asyncio.wait_for(session.interrupt(timeout=0.05), timeout=5)
        return run, sandbox

    run, sandbox = asyncio.run(exercise())
    engine._join_background()
    assert run.done and run.result.is_error and "could not be delivered" in run.result.warning
    assert sandbox in provider.deleted and session._sandbox is None


def test_active_run_is_not_cleaned_by_a_second_submission(monkeypatch):
    session = make_engine(FakeSandboxProvider()).start_session()
    session._current_run = SimpleNamespace(done=False)
    session._sandbox = "live-sandbox"
    monkeypatch.setattr(session, "_mint_tokens", lambda turn_id: pytest.fail("second minting"))
    with pytest.raises(RuntimeError, match="running a turn"):
        session.run("second")
    assert session._sandbox == "live-sandbox"


def test_cleanup_errors_preserve_original_exception_without_logging_secret(monkeypatch, caplog):
    session = make_engine(FakeSandboxProvider()).start_session()
    failure = RuntimeError("original setup failure")

    def mint(*a, **kw):
        raise failure

    def failed_cleanup(*a, **kw):
        raise PermissionError("AUDIT_FAKE_SECRET")

    monkeypatch.setattr(scoped_gcs, "mint_run_token", mint)
    monkeypatch.setattr(scoped_gcs, "delete_run_token", failed_cleanup)
    with pytest.raises(RuntimeError) as caught:
        session.run("dummy", secrets={"KEY": "AUDIT_FAKE"})
    assert caught.value.__cause__ is failure
    assert "AUDIT_FAKE_SECRET" not in caplog.text and "AUDIT_FAKE_SECRET" not in str(caught.value)


def test_initial_refresh_write_failure_is_rolled_back_before_thread_start(monkeypatch):
    session = make_engine(FakeSandboxProvider()).start_session()
    deleted = []

    def failed_write(*a, **kw):
        raise TimeoutError("dummy initial token write failed")

    monkeypatch.setattr(scoped_gcs, "write_run_token", failed_write)
    monkeypatch.setattr(scoped_gcs, "delete_run_token", lambda *a, **kw: deleted.append(a))
    monkeypatch.setattr(backend.threading, "Thread", lambda **kw: pytest.fail("unexpected thread"))
    with pytest.raises(TimeoutError):
        session.run("dummy")
    assert session._refresh_stop is None
    # Nothing was minted successfully, so nothing to delete; the token object was never written.
    assert deleted == []


def test_turn_config_record_failure_does_not_block_the_turn(monkeypatch):
    def fail(*a, **kw):
        raise PermissionError("bucket write denied")

    monkeypatch.setattr(handoff, "stage_turn_config", fail)
    provider = FakeSandboxProvider()
    session = make_engine(provider).start_session()
    result = asyncio.run(_await(session.run("dummy", config=TurnConfig(max_turns=1))))
    assert result.text == "done"
    body = [c[2] for c in provider.calls if c[1] == "/turn"][0]
    assert body["turn_config"] == {"max_turns": 1} and body["turn_config_gcs"] is None

"""Cancellation and failed acquisition with local tasks/fake remote resources (B01/B04)."""
import asyncio
import threading
from types import SimpleNamespace

import pytest

from remote_agent_toolkit import AgentSpec, TurnConfig
from remote_agent_toolkit.events import AgentEvent
from remote_agent_toolkit.runtime._run import DrivenRun
from remote_agent_toolkit.runtime.gemini import backend, handoff


def make_session(monkeypatch, *, warm=False):
    engine = backend.GeminiEngine("dummy", AgentSpec(name="audit", model="dummy"), "p", "l",
                                  output_bucket="gs://audit-bucket")
    engine._warm = warm
    monkeypatch.setattr(engine, "_in_background", lambda *a, **kw: None)
    return backend.GeminiSession(engine, "audit")


def test_warm_interrupt_preserves_job_before_local_completion(monkeypatch):
    session = make_session(monkeypatch, warm=True)
    cancelled = []
    monkeypatch.setattr(session._engine, "_agent_engines", lambda: SimpleNamespace(
        cancel_query_job=lambda **kw: cancelled.append(kw)))

    async def exercise():
        started = asyncio.Event()

        async def waiting():
            started.set()
            await asyncio.Event().wait()
            yield AgentEvent(kind="status", summary="unreachable")

        session._worker = SimpleNamespace(job_name="audit-worker-job")
        run = DrivenRun(waiting, "audit", session._engine.spec, session._on_complete)
        session._current_run = run
        run.ensure_started()
        await asyncio.wait_for(started.wait(), timeout=2)
        await asyncio.wait_for(session.interrupt(), timeout=2)
        assert run.done and session._worker is None

    asyncio.run(exercise())
    assert cancelled == [{"name": "dummy", "config": {"operation_name": "audit-worker-job"}}]


@pytest.mark.parametrize("phase", ["mint", "config", "turn-config", "claim"])
def test_pre_dispatch_failure_releases_owned_resources(monkeypatch, phase):
    session = make_session(monkeypatch, warm=phase == "claim")
    stop = threading.Event()  # no actual refresh thread
    deleted = []
    failure = RuntimeError("dummy setup failure")

    def fail(*a, **kw):
        raise failure

    def mint():
        session._gcs_token_stop = stop
        if phase == "mint":
            raise failure
        return "AUDIT_FAKE_TOKEN"

    monkeypatch.setattr(session, "_stage_secrets", lambda secrets: "gs://audit-bucket/dummy-secret")
    monkeypatch.setattr(session, "_mint_gcs_token", mint)
    monkeypatch.setattr(handoff, "delete_staged_secrets", lambda uri, **kw: deleted.append(uri))
    monkeypatch.setattr(session._engine, "_agent_engines", lambda: pytest.fail("unexpected dispatch"))
    if phase == "config":
        monkeypatch.setattr(session, "_resolve_session_config", fail)
    if phase == "turn-config":
        monkeypatch.setattr(handoff, "stage_turn_config", fail)
    if phase == "claim":
        monkeypatch.setattr(session._engine, "_claim_worker", fail)
    config = TurnConfig(max_turns=1) if phase == "turn-config" else None
    with pytest.raises(RuntimeError) as caught:
        session.run("dummy", secrets={"KEY": "AUDIT_FAKE"}, config=config)
    assert caught.value is failure
    assert stop.is_set()
    assert session._gcs_token_stop is None and session._staged_secrets_uri is None
    assert deleted == ["gs://audit-bucket/dummy-secret"]


@pytest.mark.parametrize("warm", [False, True])
def test_uncertain_remote_submission_keeps_handoff_resources(monkeypatch, warm):
    session = make_session(monkeypatch, warm=warm)
    session._engine._subscription = "legacy-audit-subscription" if warm else None
    stop = threading.Event()
    deleted = []

    def mint():
        session._gcs_token_stop = stop
        return "AUDIT_FAKE_TOKEN"

    def uncertain(*a, **kw):
        raise TimeoutError("acceptance is unknown")

    monkeypatch.setattr(session, "_stage_secrets", lambda secrets: "gs://audit-bucket/dummy-secret")
    monkeypatch.setattr(session, "_mint_gcs_token", mint)
    monkeypatch.setattr(handoff, "delete_staged_secrets", lambda uri, **kw: deleted.append(uri))
    monkeypatch.setattr(session._engine, "_agent_engines", lambda: SimpleNamespace(
        run_query_job=uncertain))
    monkeypatch.setattr(session._engine, "_dispatch", lambda: SimpleNamespace(publish=uncertain))
    with pytest.raises(TimeoutError):
        session.run("dummy", secrets={"KEY": "AUDIT_FAKE"})
    assert not stop.is_set() and deleted == []
    assert session._staged_secrets_uri == "gs://audit-bucket/dummy-secret"


def test_claimed_worker_is_retired_if_dispatch_client_creation_fails(monkeypatch):
    session = make_session(monkeypatch, warm=True)
    worker = SimpleNamespace(worker="audit-worker", job_name="audit-job")
    retired = []
    monkeypatch.setattr(session, "_mint_gcs_token", lambda: None)
    monkeypatch.setattr(session._engine, "_claim_worker", lambda: worker)
    monkeypatch.setattr(session._engine, "_retire_worker", lambda entry, **kw: retired.append((entry, kw)))

    def fail():
        raise RuntimeError("local dispatcher construction failed")

    monkeypatch.setattr(session._engine, "_dispatch", fail)
    with pytest.raises(RuntimeError, match="construction"):
        session.run("dummy")
    assert retired == [(worker, {"cancel_job": True})]
    assert session._worker is None


def test_active_run_is_not_cleaned_by_a_second_submission(monkeypatch):
    session = make_session(monkeypatch)
    session._current_run = SimpleNamespace(done=False)
    session._staged_secrets_uri = "gs://audit-bucket/first-turn-secret"
    monkeypatch.setattr(session, "_stage_secrets", lambda secrets: pytest.fail("second staging"))
    with pytest.raises(RuntimeError, match="already active"):
        session.run("second")
    assert session._staged_secrets_uri == "gs://audit-bucket/first-turn-secret"


def test_cleanup_errors_preserve_original_exception_without_logging_secret(monkeypatch, caplog):
    session = make_session(monkeypatch)
    failure = RuntimeError("original setup failure")

    def mint():
        raise failure

    def failed_cleanup(*a, **kw):
        raise PermissionError("AUDIT_FAKE_SECRET")

    monkeypatch.setattr(session, "_stage_secrets", lambda secrets: "gs://audit-bucket/dummy-secret")
    monkeypatch.setattr(session, "_mint_gcs_token", mint)
    monkeypatch.setattr(session, "_stop_gcs_token_refresh", failed_cleanup)
    monkeypatch.setattr(handoff, "delete_staged_secrets", failed_cleanup)
    with pytest.raises(RuntimeError) as caught:
        session.run("dummy", secrets={"KEY": "AUDIT_FAKE"})
    assert caught.value is failure
    assert "AUDIT_FAKE_SECRET" not in caplog.text

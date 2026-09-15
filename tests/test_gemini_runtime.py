"""Sandbox run plane, offline: the worker's turn runner and the client's Session/Run wiring.

The platform is a ``FakeSandboxProvider`` (``sandbox_fakes``); the worker is the REAL
``Worker`` class driven in-process with the harness faked, so a client turn here goes
client → provider.call → worker → harness fake → events → client, end to end.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import pytest
from sandbox_fakes import FakeSandboxProvider, ScriptedWorker, make_engine, result_event

from remote_agent_toolkit import AgentSpec, SessionConfig, TurnConfig
from remote_agent_toolkit.events import AgentEvent, RunStatus, StopReason
from remote_agent_toolkit.ports.blobstore import LocalBlobStore
from remote_agent_toolkit.runtime.gemini import backend, worker as worker_mod
from remote_agent_toolkit.runtime.gemini.worker import Worker


async def _await(run):
    return await run


def _real_worker_factory(tmp_path, spec, blobs=None):
    def factory(name):
        return Worker(
            workspace_root=str(tmp_path / "ws" / name.rsplit("/", 1)[-1]), baked=spec, baked_skills=None,
            blob_store_factory=(lambda bucket, prefix: blobs) if blobs is not None else None,
            mirror_factory=lambda uri, sid, tid: _NullMirror(),
        )
    return factory


class _NullMirror:
    def append(self, event):
        pass

    def close(self, timeout=None):
        pass


def _patch_harness(monkeypatch, harness_cls):
    import remote_agent_toolkit.harness.claude_code as harness_mod

    monkeypatch.setattr(harness_mod, "ClaudeCodeHarness", harness_cls)


class _DoneHarness:
    seen: list = []

    async def run(self, spec, rc):
        _DoneHarness.seen.append((spec, rc))
        yield AgentEvent(kind="message", summary="working")
        yield result_event("done", num_turns=4)


# -- the worker ------------------------------------------------------------------------


def _drive(worker: Worker, body: dict) -> list[AgentEvent]:
    """Run one turn to completion through the worker's endpoints; return its events."""
    from remote_agent_toolkit.runtime.gemini.history import event_from_mirror

    acc = worker.handle("/turn", body)
    assert acc["ok"], acc
    lines, since = [], 0
    while True:
        resp = worker.handle("/events", {"turn_id": body["turn_id"], "since": since, "wait": 5})
        lines.extend(resp["events"])
        since = resp["next"]
        if resp["done"]:
            break
    return [event_from_mirror(line) for line in lines]


def _body(session_id="sid-1", prompt="go", **extra) -> dict:
    return {"turn_id": uuid.uuid4().hex, "session_id": session_id, "prompt": prompt,
            "resume": False, "gcs": {"output_bucket": None}, **extra}


def test_worker_runs_the_turn_and_maps_the_session_id(tmp_path, monkeypatch):
    _DoneHarness.seen = []
    _patch_harness(monkeypatch, _DoneHarness)
    spec = AgentSpec(name="w", model="m")
    worker = _real_worker_factory(tmp_path, spec)("s1")

    events = _drive(worker, _body(session_id="1966652674296250368", secrets={"K": "v"},
                                  model_env={"ANTHROPIC_AUTH_TOKEN": "tok"}))
    kinds = [((e.raw or {}).get("event") or e.kind) for e in events]
    assert kinds == ["turn_started", "control_ready", "workspace_ready", "message", "result"]
    assert all((e.raw or {}).get("turn_id") for e in events)  # every event carries the turn id
    _spec, rc = _DoneHarness.seen[0]
    assert str(uuid.UUID(rc.session_id)) == rc.session_id  # canonical UUID for the SDK
    assert rc.secrets == {"K": "v"} and rc.env == {"ANTHROPIC_AUTH_TOKEN": "tok"}
    assert rc.control is not None and rc.workspace.name == "workspace"
    assert worker.handle("/health", {})["running_turn"] is None


def test_worker_refuses_a_second_concurrent_turn_and_unknown_turns(tmp_path, monkeypatch):
    import threading

    gate = threading.Event()

    class Slow:
        async def run(self, spec, rc):
            await asyncio.to_thread(gate.wait)
            yield result_event()

    _patch_harness(monkeypatch, Slow)
    worker = _real_worker_factory(tmp_path, AgentSpec(name="w", model="m"))("s1")
    first = _body()
    assert worker.handle("/turn", first)["ok"]
    second = worker.handle("/turn", _body())
    assert second == {"ok": False, "error": "a turn is running", "running": [first["turn_id"]]}
    assert worker.handle("/events", {"turn_id": "nope"}) == {"ok": False, "error": "unknown turn_id"}
    assert worker.handle("/turn", {"turn_id": "x"}) == {"ok": False, "error": "turn_id and session_id are required"}
    assert worker.handle("/health", {})["running_turn"] == first["turn_id"]
    gate.set()
    resp = worker.handle("/events", {"turn_id": first["turn_id"], "since": 0, "wait": 5})
    while not resp["done"]:
        resp = worker.handle("/events", {"turn_id": first["turn_id"], "since": resp["next"], "wait": 5})
    assert worker.handle("/turn", _body())["ok"]  # the sandbox takes another turn once idle


def test_worker_surfaces_a_harness_crash_as_a_terminal_error(tmp_path, monkeypatch):
    class Crashing:
        async def run(self, spec, rc):
            raise RuntimeError("Invalid session ID. Must be a valid UUID.")
            yield  # pragma: no cover

    _patch_harness(monkeypatch, Crashing)
    worker = _real_worker_factory(tmp_path, AgentSpec(name="w", model="m"))("s1")
    events = _drive(worker, _body())
    assert events[-1].kind == "result" and events[-1].raw["is_error"] is True
    assert "agent run failed" in events[-1].summary


def test_worker_late_crash_keeps_the_terminal_result(tmp_path, monkeypatch):
    class Dying:
        async def run(self, spec, rc):
            yield result_event("all done", cost=9.8, num_turns=40)
            raise RuntimeError("Command failed with exit code 1")

    _patch_harness(monkeypatch, Dying)
    worker = _real_worker_factory(tmp_path, AgentSpec(name="w", model="m"))("s1")
    events = _drive(worker, _body())
    assert [e.kind for e in events].count("result") == 1
    assert events[-1].kind == "status" and "result kept" in events[-1].summary
    assert next(e for e in events if e.kind == "result").cost_usd == 9.8


def test_worker_surfaces_a_workspace_prep_crash(tmp_path, monkeypatch):
    def boom(rc, prefer_baked_skills=True, baked_dir=None):
        raise RuntimeError("git clone failed: fatal: Authentication failed")

    monkeypatch.setattr(worker_mod, "_prepare_workspace", boom)
    worker = _real_worker_factory(tmp_path, AgentSpec(name="w", model="m"))("s1")
    events = _drive(worker, _body())
    assert [(e.kind, (e.raw or {}).get("event")) for e in events] == [
        ("status", "turn_started"), ("status", "control_ready"), ("result", "harness_error")]
    assert events[-1].raw["is_error"] is True and "git clone failed" in events[-1].summary


def test_worker_prep_falls_back_to_a_clean_workspace_when_the_snapshot_is_broken(tmp_path):
    """#74, worker side: a snapshot that fails to extract must not leave half the files for
    ``provision_repos`` to trip over, and the reason reaches ``workspace_ready``."""
    from types import SimpleNamespace

    from remote_agent_toolkit.checkpoint.workspace import snapshot

    class HalfThenFail(LocalBlobStore):
        def get_tree(self, key, local_dir):
            (tmp_path / "ws" / "repo").mkdir(parents=True)
            (tmp_path / "ws" / "repo" / "half.py").write_text("partial")
            raise OSError("archive aborted at member 162")

    blobs = HalfThenFail(str(tmp_path / "store"))
    snapshot(blobs, "sid", str(tmp_path))
    rc = SimpleNamespace(resume_sid="sid", blobs=blobs, workspace=tmp_path / "ws",
                         spec=AgentSpec(name="w", model="m", checkpoint=True), secrets={})
    ready = worker_mod._prepare_workspace(rc, prefer_baked_skills=False)
    assert ready["restored"] is False
    assert ready["restore_error"] == "OSError: archive aborted at member 162"
    assert (tmp_path / "ws").is_dir() and list((tmp_path / "ws").iterdir()) == []


def test_worker_merges_configs_over_the_baked_spec_and_echoes_them(tmp_path, monkeypatch):
    _DoneHarness.seen = []
    _patch_harness(monkeypatch, _DoneHarness)
    baked = AgentSpec(name="w", model="m-baked", max_budget_usd=5.0)
    worker = _real_worker_factory(tmp_path, baked)("s1")
    events = _drive(worker, _body(
        session_config=SessionConfig(model="m-session", system_prompt="SESSION").to_dict(),
        turn_config=TurnConfig(model="m-turn", max_budget_usd=1.0).to_dict(),
        session_config_gcs="gs://out/session-config/sid-1.json",
    ))
    spec, _rc = _DoneHarness.seen[0]
    assert (spec.model, spec.system_prompt, spec.max_budget_usd) == ("m-turn", "SESSION", 1.0)
    echo = events[1]
    assert echo.raw["event"] == "effective_spec" and echo.raw["spec"]["model"] == "m-turn"
    assert echo.raw["session_config_gcs"] == "gs://out/session-config/sid-1.json"


@pytest.mark.parametrize("cfg, needle", [
    ({"warp_drive": True}, "same toolkit revision"),
    (SessionConfig(harness="codex").to_dict(), "not baked into this engine"),
])
def test_worker_fails_the_turn_on_a_config_it_cannot_honor(tmp_path, monkeypatch, cfg, needle):
    _DoneHarness.seen = []
    _patch_harness(monkeypatch, _DoneHarness)
    worker = _real_worker_factory(tmp_path, AgentSpec(name="w", model="m-baked"))("s1")
    events = _drive(worker, _body(session_config=cfg))
    assert len(events) == 1 and events[0].kind == "result" and events[0].raw["is_error"] is True
    assert needle in events[0].summary
    assert _DoneHarness.seen == []


def test_worker_control_reaches_the_harness_and_dedupes(tmp_path, monkeypatch):
    class ControlAware:
        async def run(self, spec, rc):
            yield AgentEvent(kind="status", summary="started", raw={"subtype": "init"})
            msg = await rc.control.receive()
            yield msg.user_event()
            yield result_event(f"saw {msg.op}: {msg.message}")

    _patch_harness(monkeypatch, ControlAware)
    worker = _real_worker_factory(tmp_path, AgentSpec(name="w", model="m"))("s1")
    body = _body()
    assert worker.handle("/turn", body)["ok"]
    import time
    deadline = time.time() + 5
    while worker._turns[body["turn_id"]].control is None:
        assert time.time() < deadline
        time.sleep(0.005)
    ctl = {"turn_id": body["turn_id"], "op": "steer", "message": "skip the images", "message_id": "m-1"}
    assert worker.handle("/control", ctl) == {"ok": True, "delivered": True, "message_id": "m-1"}
    assert worker.handle("/control", ctl)["delivered"] is False  # a retry of m-1 is dropped
    assert worker.handle("/control", {"turn_id": "nope", "op": "stop"})["ok"] is False
    assert worker.handle("/control", {"turn_id": body["turn_id"], "op": "bogus"})["ok"] is False
    since, lines = 0, []
    while True:
        resp = worker.handle("/events", {"turn_id": body["turn_id"], "since": since, "wait": 5})
        lines += resp["events"]
        since = resp["next"]
        if resp["done"]:
            break
    assert [ln["kind"] for ln in lines][-2:] == ["user", "result"]
    assert lines[-1]["summary"] == "saw steer: skip the images"
    assert worker.handle("/control", ctl) == {"ok": False, "error": "turn already finished"}


def test_worker_events_are_paged_by_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_mod, "EVENTS_PAGE_BYTES", 300)

    class Chatty:
        async def run(self, spec, rc):
            for i in range(5):
                yield AgentEvent(kind="message", summary=f"line {i} " + "x" * 100)
            yield result_event()

    _patch_harness(monkeypatch, Chatty)
    worker = _real_worker_factory(tmp_path, AgentSpec(name="w", model="m"))("s1")
    body = _body()
    assert worker.handle("/turn", body)["ok"]
    pages, since = [], 0
    while True:
        resp = worker.handle("/events", {"turn_id": body["turn_id"], "since": since, "wait": 5})
        pages.append(resp["events"])
        since = resp["next"]
        if resp["done"]:
            break
    assert all(len(page) <= 2 for page in pages if page)  # ~150 B per event under a 300 B budget
    assert sum(len(p) for p in pages) == 9  # turn_started, control_ready, workspace_ready, 5 messages, result
    assert pages[-1][-1]["kind"] == "result" if pages[-1] else True


def test_worker_transcript_only_spec_does_not_resume_the_conversation(tmp_path, monkeypatch):
    seen = {}

    class Recording:
        async def run(self, spec, rc):
            seen["resume_sid"], seen["session_store"] = rc.resume_sid, rc.session_store
            yield result_event()

    _patch_harness(monkeypatch, Recording)
    blobs = LocalBlobStore(str(tmp_path / "blobs"))
    worker = _real_worker_factory(tmp_path, AgentSpec(name="w", model="m", transcript=True), blobs)("s1")
    _drive(worker, _body(resume=True, gcs={"output_bucket": "gs://out"}))
    assert seen["session_store"] is not None and seen["resume_sid"] is None


def test_worker_health_and_exec(tmp_path):
    worker = Worker(workspace_root=str(tmp_path), baked=AgentSpec(name="w", model="m"), baked_skills=None)
    health = worker.handle("/health", {})
    assert health["ok"] and health["workspace_writable"] and health["running_turn"] is None
    out = worker.handle("/exec", {"command": "echo hi; exit 3", "timeout": 5})
    assert (out["stdout"].strip(), out["returncode"]) == ("hi", 3)
    assert worker.handle("/exec", {})["ok"] is False
    assert worker.handle("/nope", {})["ok"] is False


def test_find_baked_skills(tmp_path):
    assert worker_mod.find_baked_skills(str(tmp_path / "spec.json")) is None
    (tmp_path / "skills").mkdir()
    assert worker_mod.find_baked_skills(str(tmp_path / "spec.json")) == tmp_path / "skills"


# -- the client: Session / Run over the fake provider -----------------------------------


def test_session_run_streams_from_the_worker_and_releases_the_sandbox(tmp_path, monkeypatch):
    _DoneHarness.seen = []
    _patch_harness(monkeypatch, _DoneHarness)
    spec = AgentSpec(name="g", model="m")
    provider = FakeSandboxProvider(_real_worker_factory(tmp_path, spec))
    engine = make_engine(provider, spec=spec)
    session = engine.start_session()

    async def go():
        run = session.run("go", secrets={"SH_APIKEY": "SUPERSECRET"})
        events = [ev async for ev in run]
        return run, events

    run, events = asyncio.run(go())
    engine._join_background()
    result = run.result
    assert result.text == "done" and result.num_turns == 4 and not result.is_error
    assert session.status == RunStatus.IDLE and session.stop_reason == StopReason.END_TURN
    assert [((e.raw or {}).get("event") or e.kind) for e in events] == [
        "turn_started", "control_ready", "workspace_ready", "message", "result"]
    # The turn went to the worker in one HTTP body: secrets as values, the tokens, no GCS staging.
    (sandbox, path, body), = [c for c in provider.calls if c[1] == "/turn"]
    assert body["secrets"] == {"SH_APIKEY": "SUPERSECRET"} and body["prompt"] == "go"
    assert body["gcs"]["token"] == "fake-run-token" and body["gcs"]["output_bucket"] == "gs://out"
    assert body["model_env"]["ANTHROPIC_AUTH_TOKEN"] == "fake-model-token"
    assert body["model_env"]["CLAUDE_CODE_USE_VERTEX"] == "1"
    assert body["session_config"] is None and body["turn_config"] is None
    # The sandbox was created for this turn (no pool) and deleted at the terminal event.
    assert provider.deleted == [sandbox] and provider.live() == []
    assert session._sandbox is None and session._refresh_stop is None


def test_session_run_returns_at_once_and_dispatches_in_the_driver():
    provider = FakeSandboxProvider()
    engine = make_engine(provider)
    session = engine.start_session()

    async def go():
        run = session.run("go")
        assert not provider.calls  # nothing dispatched synchronously
        assert session.busy
        return await run

    result = asyncio.run(go())
    assert result.text == "done"
    assert [c[1] for c in provider.calls][:3] == ["/health", "/turn", "/events"]


def test_session_send_resumes_on_a_fresh_sandbox_with_the_resume_flag():
    provider = FakeSandboxProvider()
    engine = make_engine(provider, spec=AgentSpec(name="g", model="m", checkpoint=True))
    session = engine.start_session()
    asyncio.run(_await(session.run("go")))
    asyncio.run(_await(session.send("answer", secrets={"K": "v2"})))
    turns = [c[2] for c in provider.calls if c[1] == "/turn"]
    assert [t["resume"] for t in turns] == [False, True]
    assert turns[1]["secrets"] == {"K": "v2"} and turns[1]["session_id"] == session.session_id
    assert len(provider.deleted) == 2  # one sandbox per turn, both released
    assert session.stop_reason == StopReason.NEEDS_INPUT  # checkpoint spec: awaiting the operator


def test_dispatch_failure_moves_to_another_sandbox_then_gives_up():
    provider = FakeSandboxProvider()
    provider.fail_next["/turn"] = [RuntimeError("proxy 502"), RuntimeError("proxy 502")]
    engine = make_engine(provider)
    session = engine.start_session()
    result = asyncio.run(_await(session.run("go")))
    engine._join_background()
    assert result.text == "done"
    assert len(provider.deleted) == 3  # two sandboxes that refused the turn + the one that ran it
    assert provider.live() == []

    provider.fail_next["/turn"] = [RuntimeError("proxy 502")] * backend.DISPATCH_ATTEMPTS
    result = asyncio.run(_await(engine.start_session().run("go")))
    assert result.is_error and "could not hand the turn to a sandbox" in result.text


def test_submit_rejects_hooks_and_requires_the_output_bucket():
    engine = make_engine(FakeSandboxProvider())
    with pytest.raises(ValueError, match="local-only"):
        engine.start_session().run("go", hooks={"PreToolUse": []})
    engine = make_engine(FakeSandboxProvider(), output_bucket=None)
    with pytest.raises(ValueError, match="output bucket"):
        engine.start_session().run("go")


def test_run_while_busy_raises_and_send_steers_through_control():
    def on_control(w, body):
        if body["op"] == "steer":
            w.emit(AgentEvent(kind="user", summary=body["message"],
                              raw={"event": "user_message", "message_id": body.get("message_id")}))
            w.emit(result_event(f"saw {body['message']}"))
            w.finish()

    def factory(name):
        w = ScriptedWorker(on_control=on_control)
        return w

    provider = FakeSandboxProvider(factory)
    engine = make_engine(provider)
    session = engine.start_session()

    async def go():
        run = session.run("build it")
        # Wait for the dispatch, then the worker's control_ready.
        while session._sandbox is None:
            await asyncio.sleep(0.005)
        w = provider.worker(session._sandbox)
        w.emit(AgentEvent(kind="status", summary="ready", raw={"event": "control_ready"}))
        while not session._control_ready:
            await asyncio.sleep(0.005)
        with pytest.raises(RuntimeError, match="running a turn"):
            session.run("another")
        assert session.send("skip the images", message_id="m-1") is run
        return await run, w

    result, w = asyncio.run(go())
    assert result.text == "saw skip the images"
    assert w.controls[0]["op"] == "steer" and w.controls[0]["message_id"] == "m-1"


def _acking_worker(name):
    """A worker that acks every control message with a ``user`` event (and records it)."""

    def on_control(w, body):
        w.emit(AgentEvent(kind="user", summary=body["message"] or "",
                          raw={"event": "user_message", "message_id": body.get("message_id"),
                               "interrupt": body["op"] == "interrupt"}))

    return ScriptedWorker(on_control=on_control)


def test_send_before_the_worker_is_ready_is_queued_and_delivered_in_order():
    # agentic-scraping (#84): a user message posted while the turn is being dispatched
    # (~1 s from the pool, 15–25 s on a fresh sandbox) used to raise ControlUnavailable and
    # every consumer needed a retry loop. The session queues it and posts it after control_ready.
    provider = FakeSandboxProvider(_acking_worker)
    engine = make_engine(provider)
    session = engine.start_session()

    async def go():
        run = session.run("go")
        assert session.send("first", message_id="m-1") is run       # no sandbox yet: queued
        assert session.send("second", interrupt=True, message_id="m-2") is run
        assert session._pending_control and session._sandbox is None
        while session._sandbox is None:
            await asyncio.sleep(0.005)
        w = provider.worker(session._sandbox)
        assert w.controls == []                                      # dispatched, not ready: still queued
        assert session.send("third", message_id="m-3") is run
        w.emit(AgentEvent(kind="status", summary="ready", raw={"event": "control_ready"}))
        while len(w.controls) < 3:
            await asyncio.sleep(0.005)
        assert session.send("fourth", message_id="m-4") is run       # ready and drained: posted directly
        while len(w.controls) < 4:
            await asyncio.sleep(0.005)
        w.emit(result_event("done"))
        w.finish()
        events = []
        async for ev in run:
            events.append(ev)
        return await run, w, events

    result, w, events = asyncio.run(go())
    assert [(c["op"], c["message_id"]) for c in w.controls] == [
        ("steer", "m-1"), ("interrupt", "m-2"), ("steer", "m-3"), ("steer", "m-4")]
    acks = [ev.raw["message_id"] for ev in events if ev.kind == "user"]
    assert acks == ["m-1", "m-2", "m-3", "m-4"]
    assert result.text == "done" and result.warning is None
    assert not session._pending_control and not session._flushing


def test_queued_messages_are_dropped_with_a_warning_when_the_turn_ends_before_the_worker_is_ready():
    engine = make_engine(FakeSandboxProvider(lambda n: ScriptedWorker()))
    session = engine.start_session()

    async def go():
        run = session.run("go")
        session.send("early")
        session.send("also early")
        while session._sandbox is None:
            await asyncio.sleep(0.005)
        engine._provider().worker(session._sandbox).finish()  # ends without control_ready or a result
        return await run

    result = asyncio.run(go())
    assert result.is_error and "without a terminal result" in result.text
    assert "2 message(s) sent into this turn were never delivered" in result.warning
    assert not session._pending_control


def test_queued_messages_when_the_dispatch_fails_land_on_the_dispatch_failed_result():
    provider = FakeSandboxProvider(lambda n: ScriptedWorker())
    provider.fail_next["/turn"] = [RuntimeError("proxy down")] * backend.DISPATCH_ATTEMPTS
    session = make_engine(provider).start_session()

    async def go():
        run = session.run("go")
        session.send("early")
        return await run

    result = asyncio.run(go())
    assert result.is_error and "could not hand the turn" in result.text
    assert "1 message(s) sent into this turn were never delivered" in result.warning


def test_send_to_a_ready_worker_that_does_not_take_it_raises_control_unavailable():
    from remote_agent_toolkit import ControlUnavailable

    provider = FakeSandboxProvider(lambda n: ScriptedWorker())
    engine = make_engine(provider)
    session = engine.start_session()

    async def go():
        run = session.run("go")
        while session._sandbox is None:
            await asyncio.sleep(0.005)
        w = provider.worker(session._sandbox)
        w.emit(AgentEvent(kind="status", summary="ready", raw={"event": "control_ready"}))
        while not session._control_ready:
            await asyncio.sleep(0.005)
        provider.fail_next["/control"] = [RuntimeError("proxy down")]
        with pytest.raises(ControlUnavailable, match="did not take"):
            session.send("now")
        w.finish()
        return await run

    result = asyncio.run(go())
    assert result.is_error and result.warning is None  # nothing was left queued


def test_interrupt_posts_stop_and_waits_for_the_interrupted_result():
    def on_control(w, body):
        if body["op"] == "stop":
            w.emit(result_event("(interrupted)", subtype="interrupted"))
            w.emit(AgentEvent(kind="status", summary="checkpoint saved", raw={"event": "checkpoint_saved"}))
            w.finish()

    provider = FakeSandboxProvider(lambda n: ScriptedWorker(on_control=on_control))
    engine = make_engine(provider)
    session = engine.start_session()

    async def go():
        run = session.run("build it")
        while session._sandbox is None:
            await asyncio.sleep(0.005)
        provider.worker(session._sandbox).emit(
            AgentEvent(kind="status", summary="ready", raw={"event": "control_ready"}))
        while not session._control_ready:
            await asyncio.sleep(0.005)
        await session.interrupt()
        return run

    run = asyncio.run(go())
    engine._join_background()
    assert run.done and run.result.is_error is False and run.result.cost_usd == 0.3
    assert session.stop_reason == StopReason.INTERRUPTED and session.status == RunStatus.IDLE
    assert provider.live() == []  # released at the terminal event


def test_interrupt_falls_back_to_cancel_and_deletes_the_sandbox_when_the_worker_never_stops():
    provider = FakeSandboxProvider(lambda n: ScriptedWorker())  # deaf to control
    engine = make_engine(provider)
    session = engine.start_session()

    async def go():
        run = session.run("build it")
        while session._sandbox is None:
            await asyncio.sleep(0.005)
        provider.worker(session._sandbox).emit(
            AgentEvent(kind="status", summary="ready", raw={"event": "control_ready"}))
        while not session._control_ready:
            await asyncio.sleep(0.005)
        await session.interrupt(timeout=0.05)
        return run

    run = asyncio.run(go())
    engine._join_background()
    assert run.done and run.result.is_error and "did not end the turn" in run.result.warning
    assert session.stop_reason == StopReason.ERROR
    assert provider.live() == []


def test_interrupt_before_control_ready_cancels_without_a_checkpoint():
    provider = FakeSandboxProvider(lambda n: ScriptedWorker())
    engine = make_engine(provider)
    session = engine.start_session()

    async def go():
        run = session.run("build it")
        await asyncio.sleep(0.01)
        await session.interrupt()
        return run

    run = asyncio.run(go())
    engine._join_background()
    assert run.done and run.result.is_error and "control channel yet" in run.result.warning
    assert provider.live() == []


def test_start_session_mints_a_canonical_uuid_and_binds_the_config(monkeypatch):
    from remote_agent_toolkit.runtime.gemini import handoff

    recorded = {}
    monkeypatch.setattr(handoff, "persist_session_config",
                        lambda bucket, sid, cfg, **kw: recorded.update(sid=sid, cfg=cfg) or f"gs://out/session-config/{sid}.json")
    engine = make_engine(FakeSandboxProvider())
    sid = engine.start_session().session_id
    assert str(uuid.UUID(sid)) == sid and "-" in sid
    session = engine.start_session(config=SessionConfig(model="m-session"))
    assert recorded == {"sid": session.session_id, "cfg": {"model": "m-session"}}
    asyncio.run(_await(session.run("go")))
    body = [c[2] for c in engine._provider().calls if c[1] == "/turn"][0]
    assert body["session_config"] == {"model": "m-session"}
    assert body["session_config_gcs"] == f"gs://out/session-config/{session.session_id}.json"
    with pytest.raises(ValueError, match="output bucket"):
        make_engine(FakeSandboxProvider(), output_bucket=None).start_session(config=SessionConfig(model="m"))


def test_turn_config_rides_the_body_and_is_recorded(monkeypatch):
    from remote_agent_toolkit.runtime.gemini import handoff

    recorded = []
    monkeypatch.setattr(handoff, "stage_turn_config",
                        lambda bucket, sid, cfg, **kw: recorded.append(cfg) or "gs://out/turn-config/x.json")
    engine = make_engine(FakeSandboxProvider())
    session = engine.start_session()
    asyncio.run(_await(session.run("go", config=TurnConfig(model="m-turn", max_turns=3))))
    asyncio.run(_await(session.send("more", config=TurnConfig())))  # all-INHERIT: nothing recorded
    bodies = [c[2] for c in engine._provider().calls if c[1] == "/turn"]
    assert bodies[0]["turn_config"] == {"model": "m-turn", "max_turns": 3}
    assert bodies[0]["turn_config_gcs"] == "gs://out/turn-config/x.json"
    assert bodies[1]["turn_config"] is None and bodies[1]["turn_config_gcs"] is None
    assert recorded == [{"model": "m-turn", "max_turns": 3}]


def test_api_key_mode_sends_no_model_env():
    engine = make_engine(FakeSandboxProvider(), use_vertex=False)
    asyncio.run(_await(engine.start_session().run("go", secrets={"ANTHROPIC_API_KEY": "sk"})))
    body = [c[2] for c in engine._provider().calls if c[1] == "/turn"][0]
    assert body["model_env"] == {} and body["secrets"] == {"ANTHROPIC_API_KEY": "sk"}


def test_vertex_mode_without_a_model_account_fails_before_dispatch():
    engine = make_engine(FakeSandboxProvider(), model_service_account=None)
    with pytest.raises(RuntimeError, match="model service account"):
        engine.start_session().run("go")
    assert engine._provider().calls == []


# -- records -----------------------------------------------------------------------------


def _transcript_engine(spec, monkeypatch, tmp_path):
    import remote_agent_toolkit.ports.blobstore as bs

    blobs = bs.LocalBlobStore(str(tmp_path))
    monkeypatch.setattr(bs, "GcsBlobStore", lambda bucket, prefix, **kw: blobs)
    return make_engine(FakeSandboxProvider(), spec=spec), blobs


def _attached(engine, session_id):
    session = backend.GeminiSession(engine, session_id)
    session._session_config_resolved = True
    return session


def test_transcripts_read_the_id_the_worker_wrote_under(tmp_path, monkeypatch):
    from remote_agent_toolkit.checkpoint.session_store import BlobSessionStore, _claude_session_id

    engine, blobs = _transcript_engine(AgentSpec(name="g", model="m", transcript=True), monkeypatch, tmp_path)
    raw_sid = "1966652674296250368"
    entries = [{"uuid": "u1", "type": "user"}, {"uuid": "u2", "type": "assistant"}]
    store = BlobSessionStore(blobs)
    asyncio.run(store.append({"session_id": _claude_session_id(raw_sid)}, entries))
    asyncio.run(store.append({"session_id": _claude_session_id(raw_sid), "subpath": "subagents/agent-X"},
                             [{"uuid": "u3", "type": "user"}]))
    assert asyncio.run(_attached(engine, raw_sid).transcripts()) == {
        "main": entries, "subagents/agent-X": [{"uuid": "u3", "type": "user"}]}


def test_transcripts_raise_only_when_the_spec_is_known_to_persist_nothing(tmp_path, monkeypatch):
    engine, _ = _transcript_engine(AgentSpec(name="g", model="m"), monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match="transcript=True"):
        asyncio.run(_attached(engine, "sid").transcripts())
    engine._spec_known = False
    assert asyncio.run(_attached(engine, "sid").transcripts()) == {}


def test_workspace_and_fork_are_remote_only():
    session = make_engine(FakeSandboxProvider()).start_session()
    with pytest.raises(NotImplementedError):
        session.workspace
    with pytest.raises(NotImplementedError):
        session.fork()


def test_deploy_rejects_the_local_only_workspace_argument():
    with pytest.raises(ValueError, match="local-only"):
        backend.deploy(AgentSpec(name="w", model="m"), project="p", location="l", workspace="/some/dir")


def test_worker_module_layout_does_not_shadow_the_package_exports():
    import pkgutil

    from remote_agent_toolkit import gemini

    submodules = {m.name for m in pkgutil.iter_modules(gemini.__path__)}
    assert not submodules & set(gemini.__all__)
    assert callable(gemini.deploy)
    assert Path(worker_mod.__file__).name == "worker.py"

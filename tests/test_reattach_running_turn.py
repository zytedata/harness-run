"""A session re-attached in another process adopts its turn running there (``_attach``).

Two engine objects over one fake provider stand for two processes: the *owner* starts a
turn on a held ``ScriptedWorker``; the *adopter* re-attaches by id (``get_session``) and
finds the turn through the event mirror the worker writes (written here by hand, as the
real worker would: a ``turn_started`` marker naming the sandbox, then ``control_ready``).
"""

from __future__ import annotations

import asyncio

import pytest
from sandbox_fakes import INSTANCE, FakeSandboxProvider, ScriptedWorker, make_engine, result_event

from remote_agent_toolkit import ControlUnavailable
from remote_agent_toolkit.events import AgentEvent, RunStatus, StopReason
from remote_agent_toolkit.ports.blobstore import LocalBlobStore
from remote_agent_toolkit.runtime.gemini import backend, handoff, history


def _started(turn_id, worker):
    return AgentEvent(kind="status", summary="turn started",
                      raw={"event": "turn_started", "turn_id": turn_id, "worker": worker})


def _ready(turn_id):
    return AgentEvent(kind="status", summary="control channel ready",
                      raw={"event": "control_ready", "turn_id": turn_id})


def _result(turn_id=None):
    raw = {"subtype": "success", "is_error": False}
    if turn_id is not None:
        raw["turn_id"] = turn_id
    return AgentEvent(kind="result", summary="done", raw=raw)


class _Bucket:
    """One blob store for both "processes" — the output bucket (mirror + session records)."""

    def __init__(self, tmp_path, monkeypatch):
        self.store = LocalBlobStore(str(tmp_path / "bucket"))
        self.mirror_reads: list[str] = []
        inner = self.store.list

        def counting(prefix):
            if prefix.startswith("events/"):
                self.mirror_reads.append(prefix)
            return inner(prefix)

        self.store.list = counting
        monkeypatch.setattr(history, "GcsBlobStore", lambda bucket, *a, **kw: self.store)
        monkeypatch.setattr(handoff, "GcsBlobStore", lambda bucket, *a, **kw: self.store)
        monkeypatch.setattr(backend, "ADOPTED_RELEASE_DELAY_S", 0.0)

    def write(self, sid, turn_id, *events, ms=1000):
        history.write_turn_mirror("gs://out/events", sid, [history.mirror_line(e) for e in events],
                                  now_ms=ms, turn_id=turn_id, store=self.store)


def _two_processes(provider):
    owner = make_engine(provider)
    session = owner.start_session()
    other = make_engine(provider)
    adopter = other.get_session(session.session_id)
    assert adopter is not session and adopter._foreign_check and not session._foreign_check
    return owner, session, other, adopter


async def _owner_turn(provider, session, bucket, *, ready=True):
    """Start the owner's turn, wait for dispatch, mirror what the worker records as it begins."""
    run = session.run("go")
    while session._sandbox is None:
        await asyncio.sleep(0.005)
    w = provider.worker(session._sandbox)
    turn_id, worker = w.turns[0]["turn_id"], w.turns[0]["sandbox"]
    assert "/" not in worker and session._sandbox.endswith("/" + worker)
    bucket.write(session.session_id, turn_id, _started(turn_id, worker), *([_ready(turn_id)] if ready else []))
    return run, w, turn_id


def _sandbox_of(w) -> str:
    return f"{INSTANCE}/sandboxEnvironments/{w.turns[0]['sandbox']}"


def _turn_calls(provider):
    return [sb for sb, path, _b in provider.calls if path == "/turn"]


def test_send_from_a_reattached_session_steers_the_turn_running_elsewhere(tmp_path, monkeypatch):
    bucket = _Bucket(tmp_path, monkeypatch)
    provider = FakeSandboxProvider(lambda n: ScriptedWorker())
    owner, session, other, adopter = _two_processes(provider)

    async def go():
        run, w, turn_id = await _owner_turn(provider, session, bucket)
        w.emit(AgentEvent(kind="message", summary="working on it"))
        adopted = adopter.send("skip the images", message_id="m1")
        assert adopter.busy and adopter._adopted and adopted is adopter._current_run
        assert adopter.status == RunStatus.RUNNING and adopter._refresh_stop is not None
        assert w.controls[-1] == {"turn_id": turn_id, "op": "steer", "message": "skip the images", "message_id": "m1"}
        seen = []

        async def drain():
            async for ev in adopted:
                seen.append(ev)

        drainer = asyncio.ensure_future(drain())
        await asyncio.sleep(0.05)
        w.emit(result_event("done"))
        w.finish()
        await run
        await drainer
        return seen, turn_id, _sandbox_of(w)

    seen, turn_id, sandbox = asyncio.run(go())
    owner._join_background()
    other._join_background()
    assert len(_turn_calls(provider)) == 1  # the owner's dispatch; no second turn was started
    assert [e.kind for e in seen] == ["message", "result"]  # replayed from the worker, then live
    assert all((e.raw or {}).get("turn_id") == turn_id for e in seen)
    assert adopter.last_result is not None and adopter.last_result.text == "done"
    assert adopter.status == RunStatus.IDLE and not adopter.busy and adopter._refresh_stop is None
    assert provider.deleted == [sandbox, sandbox]  # the owner at its result, the adopter after its delay
    assert len(bucket.mirror_reads) == 1


def test_interrupt_from_a_reattached_session_stops_the_turn_running_elsewhere(tmp_path, monkeypatch):
    bucket = _Bucket(tmp_path, monkeypatch)

    def on_control(w, body):
        if body["op"] == "stop":
            w.emit(result_event("stopped", subtype="interrupted"))
            w.finish()

    provider = FakeSandboxProvider(lambda n: ScriptedWorker(on_control=on_control))
    owner, session, other, adopter = _two_processes(provider)

    async def go():
        run, w, turn_id = await _owner_turn(provider, session, bucket)
        await adopter.interrupt(timeout=5)
        assert w.controls[-1]["op"] == "stop" and w.controls[-1]["turn_id"] == turn_id
        await run

    asyncio.run(go())
    owner._join_background()
    other._join_background()
    assert adopter.stop_reason == StopReason.INTERRUPTED and adopter.last_result.text == "stopped"
    assert adopter.last_result.warning is None  # a stop through the worker, not the cancel fallback
    assert session.stop_reason == StopReason.INTERRUPTED  # the owner's stream saw the same result


def test_interrupt_waits_for_a_replayed_control_ready_when_the_mirror_has_none_yet(tmp_path, monkeypatch):
    """The mirror may lag the worker by a flush: control_ready arrives with the replay instead."""
    bucket = _Bucket(tmp_path, monkeypatch)

    def on_control(w, body):
        if body["op"] == "stop":
            w.emit(result_event("stopped", subtype="interrupted"))
            w.finish()

    provider = FakeSandboxProvider(lambda n: ScriptedWorker(on_control=on_control))
    owner, session, other, adopter = _two_processes(provider)

    async def go():
        run, w, turn_id = await _owner_turn(provider, session, bucket, ready=False)
        w.emit(_ready(turn_id))  # the worker announced it; the mirror has not flushed it yet
        assert adopter._attach() is not None and adopter._control_ready is False
        await adopter.interrupt(timeout=5)
        await run
        return w

    w = asyncio.run(go())
    owner._join_background()
    other._join_background()
    assert w.controls[-1]["op"] == "stop"
    assert adopter.stop_reason == StopReason.INTERRUPTED and adopter.last_result.warning is None


def test_run_on_a_reattached_session_refuses_while_the_turn_runs_elsewhere(tmp_path, monkeypatch):
    bucket = _Bucket(tmp_path, monkeypatch)
    provider = FakeSandboxProvider(lambda n: ScriptedWorker())
    owner, session, other, adopter = _two_processes(provider)

    async def go():
        run, w, _ = await _owner_turn(provider, session, bucket)
        with pytest.raises(RuntimeError, match="running a turn"):
            adopter.run("another task")
        assert adopter.busy and adopter._adopted
        w.emit(result_event("done"))
        w.finish()
        await run
        await adopter._current_run

    asyncio.run(go())
    owner._join_background()
    other._join_background()
    assert len(_turn_calls(provider)) == 1


def test_exec_from_a_reattached_session_probes_the_adopted_turn_then_refuses(tmp_path, monkeypatch):
    bucket = _Bucket(tmp_path, monkeypatch)
    provider = FakeSandboxProvider(lambda n: ScriptedWorker())
    owner, session, other, adopter = _two_processes(provider)

    async def go():
        run, w, turn_id = await _owner_turn(provider, session, bucket)
        w.workspace = str(tmp_path)
        r1 = await adopter.exec("pwd", timeout=5)
        r2 = await adopter.exec("echo again", timeout=5)
        assert len(bucket.mirror_reads) == 1  # adopted once, reused
        w.emit(result_event("done"))
        w.finish()
        await run
        await adopter._current_run
        with pytest.raises(ControlUnavailable, match="no reachable worker"):
            await adopter.exec("pwd", timeout=5)
        return r1, r2, turn_id, _sandbox_of(w)

    r1, r2, turn_id, sandbox = asyncio.run(go())
    owner._join_background()
    other._join_background()
    assert r1.stdout.strip() == str(tmp_path) and r1.ok and r2.stdout == "again\n"
    probes = [(sb, body["turn_id"]) for sb, path, body in provider.calls if path == "/exec"]
    assert probes == [(sandbox, turn_id)] * 2
    assert len(bucket.mirror_reads) == 1  # no re-read once the adopted turn is over


def test_a_reattached_session_with_no_foreign_turn_checks_once_and_resumes_normally(tmp_path, monkeypatch):
    bucket = _Bucket(tmp_path, monkeypatch)
    provider = FakeSandboxProvider()  # workers answer every turn at once
    owner = make_engine(provider)
    sid = owner.start_session().session_id
    bucket.write(sid, "t-old", _started("t-old", "s-old"), _result("t-old"))  # a delivered turn
    other = make_engine(provider)
    adopter = other.get_session(sid)

    async def go():
        with pytest.raises(ControlUnavailable, match="no turn is running"):
            await adopter.exec("pwd")
        assert len(bucket.mirror_reads) == 1 and not adopter._foreign_check
        await adopter.interrupt()  # no-op, no read
        return await adopter.send("continue")

    r = asyncio.run(go())
    other._join_background()
    assert r.text == "done" and not adopter._adopted
    assert len(_turn_calls(provider)) == 1 and len(bucket.mirror_reads) == 1  # resumed; mirror read once


def test_a_session_started_here_never_looks_for_a_foreign_turn(tmp_path, monkeypatch):
    bucket = _Bucket(tmp_path, monkeypatch)
    session = make_engine(FakeSandboxProvider()).start_session()

    async def go():
        with pytest.raises(ControlUnavailable, match="no turn is running"):
            await session.exec("pwd")
        await session.interrupt()

    asyncio.run(go())
    assert bucket.mirror_reads == [] and not session._foreign_check


def test_an_adopter_whose_loop_shuts_down_leaves_the_turn_to_its_owner(tmp_path, monkeypatch):
    bucket = _Bucket(tmp_path, monkeypatch)
    provider = FakeSandboxProvider(lambda n: ScriptedWorker())
    owner, session, other, adopter = _two_processes(provider)

    async def go():
        run, w, _ = await _owner_turn(provider, session, bucket)
        adopted = adopter.send("a hint")
        await asyncio.sleep(0.02)
        adopted.task.cancel()  # what a closing event loop does to the adopter's stream
        await adopted
        assert not adopter.busy and adopted.cancelled and adopted.cancel_note is None
        assert provider.deleted == [] and session._sandbox in provider.sandboxes  # the owner's turn goes on
        w.emit(result_event("done"))
        w.finish()
        return await run, _sandbox_of(w)

    r, sandbox = asyncio.run(go())
    owner._join_background()
    other._join_background()
    assert r.text == "done" and provider.deleted == [sandbox]  # the owner's release only


def test_an_adopters_explicit_cancel_still_releases_the_sandbox(tmp_path, monkeypatch):
    """interrupt()'s fallback (the worker did not stop in time) cancels with a note: that is an order."""
    bucket = _Bucket(tmp_path, monkeypatch)
    provider = FakeSandboxProvider(lambda n: ScriptedWorker())  # ignores the stop
    owner, session, other, adopter = _two_processes(provider)
    monkeypatch.setattr(backend, "GONE_GRACE_S", 0.1)

    async def go():
        run, w, _ = await _owner_turn(provider, session, bucket)
        await adopter.interrupt(timeout=0.2)
        assert adopter.last_result.is_error and "did not end the turn" in (adopter.last_result.warning or "")
        return await run, _sandbox_of(w)

    r, sandbox = asyncio.run(go())
    owner._join_background()
    other._join_background()
    assert provider.deleted[0] == sandbox  # the adopter deleted it; the owner's turn ended on "sandbox is gone"
    assert r.is_error and "gone" in (r.text or "")


@pytest.mark.parametrize("events, expected", [
    ([], None),
    ([_started("t1", "s1")], (f"{INSTANCE}/sandboxEnvironments/s1", "t1", False)),
    ([_started("t1", "s1"), _ready("t1")], (f"{INSTANCE}/sandboxEnvironments/s1", "t1", True)),
    ([_started("t1", "s1"), _ready("t1"), _result("t1")], None),
    ([_started("t1", "s1"), _result()], None),  # a client-recorded end (no turn stamp) closes it too
    ([_started("t1", "s1"), _ready("t1"), _result("t1"), _started("t2", "s2")],
     (f"{INSTANCE}/sandboxEnvironments/s2", "t2", False)),  # the new turn's readiness is its own
    ([_started("t1", "s1"), _result("t0")], (f"{INSTANCE}/sandboxEnvironments/s1", "t1", False)),  # another turn's
    ([_started("t1", None)], None),  # a worker that recorded no sandbox: nothing to reach
    ([_started("t1", "projects/x/sandboxEnvironments/full")], ("projects/x/sandboxEnvironments/full", "t1", False)),
])
def test_find_running_turn_reads_the_last_started_turn_without_a_result(monkeypatch, events, expected):
    session = make_engine(FakeSandboxProvider()).get_session("old-sid")
    monkeypatch.setattr(backend.GeminiSession, "history", lambda self: list(events))
    assert session._find_running_turn() == expected

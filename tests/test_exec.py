"""``Session.exec()`` (issue #85): a read-only shell probe of the running turn's workspace.

Offline throughout. Three layers: ``control.run_shell`` (the one implementation), the
sandbox worker's ``/exec`` (cwd = the running turn's workspace), and the two runtimes'
``exec()`` — gemini over the fake provider (waits for dispatch, refuses once the turn is
over), local on the host (same rule).
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from typing import AsyncIterator

import pytest
from sandbox_fakes import INSTANCE, FakeSandboxProvider, ScriptedWorker, make_engine, result_event

from remote_agent_toolkit import AgentSpec, ControlUnavailable, ExecResult, local
from remote_agent_toolkit.control import EXEC_TIMEOUT_RC, run_shell
from remote_agent_toolkit.events import AgentEvent
from remote_agent_toolkit.ports.blobstore import LocalBlobStore
from remote_agent_toolkit.runtime.gemini import backend, history
from remote_agent_toolkit.runtime.gemini.provider import SandboxGone
from remote_agent_toolkit.runtime.gemini.worker import Worker

# -- control.run_shell ---------------------------------------------------------------------


def test_run_shell_captures_output_and_exit_code(tmp_path):
    r = run_shell("echo out; echo err >&2; pwd; exit 3", cwd=str(tmp_path), timeout=10)
    assert isinstance(r, ExecResult) and not r.ok
    assert r.stdout == f"out\n{tmp_path}\n" and r.stderr == "err\n" and r.returncode == 3
    assert r.truncated is False and r.duration_ms >= 0


def test_run_shell_kills_the_process_group_at_the_timeout():
    # ``sleep`` is a grandchild of the shell: killing bash alone would leave it holding the
    # pipe and communicate() would hang until it exits on its own.
    r = run_shell("echo started; sleep 30; echo never", cwd=None, timeout=0.3)
    assert r.returncode == EXEC_TIMEOUT_RC and r.stdout == "started\n"
    assert "exceeded 0.3s" in r.stderr and r.duration_ms < 5000


def test_run_shell_caps_each_stream_and_flags_it(monkeypatch):
    from remote_agent_toolkit import control

    monkeypatch.setattr(control, "EXEC_OUTPUT_CAP", 100)
    r = run_shell("head -c 500 /dev/zero | tr '\\0' x; echo short >&2", cwd=None, timeout=10)
    assert r.ok and len(r.stdout) == 100 and r.stderr == "short\n" and r.truncated is True


def test_run_shell_reports_a_missing_cwd_as_a_failed_command(tmp_path):
    r = run_shell("pwd", cwd=str(tmp_path / "nope"), timeout=10)
    assert r.returncode == 127 and "cwd does not exist" in r.stderr


def test_exec_result_round_trips_through_a_dict():
    r = ExecResult(stdout="a", stderr="b", returncode=1, truncated=True, duration_ms=7)
    assert ExecResult.from_dict(r.to_dict()) == r
    assert ExecResult.from_dict({}) == ExecResult(stdout="", stderr="", returncode=-1)


# -- the worker's /exec ------------------------------------------------------------------


def _patch_harness(monkeypatch, harness_cls):
    import remote_agent_toolkit.harness.claude_code as harness_mod

    monkeypatch.setattr(harness_mod, "ClaudeCodeHarness", harness_cls)


def test_worker_exec_runs_in_the_running_turns_workspace(tmp_path, monkeypatch):
    gate = threading.Event()

    class Held:
        async def run(self, spec, rc):
            await asyncio.to_thread(gate.wait)
            yield result_event()

    _patch_harness(monkeypatch, Held)
    worker = Worker(workspace_root=str(tmp_path), baked=AgentSpec(name="w", model="m"), baked_skills=None)
    # No turn: the workspace root is the cwd.
    idle = worker.handle("/exec", {"command": "pwd"})
    assert idle["ok"] and idle["stdout"].strip() == str(tmp_path) and idle["cwd"] == str(tmp_path)

    sid = str(uuid.uuid4())
    body = {"turn_id": "t-1", "session_id": sid, "prompt": "go", "resume": False, "gcs": {"output_bucket": None}}
    assert worker.handle("/turn", body)["ok"]
    try:
        # Wait for the workspace to be prepared (the harness is now held at the gate).
        since, ready = 0, False
        for _ in range(200):
            resp = worker.handle("/events", {"turn_id": "t-1", "since": since, "wait": 1})
            since = resp["next"]
            if any((ln.get("raw") or {}).get("event") == "workspace_ready" for ln in resp["events"]):
                ready = True
                break
        assert ready
        expected = tmp_path / "jobs" / sid / "workspace"
        out = worker.handle("/exec", {"command": "pwd", "turn_id": "t-1"})
        assert out["ok"] and out["stdout"].strip() == str(expected) and out["cwd"] == str(expected)
        (expected / "sub").mkdir()
        out = worker.handle("/exec", {"command": "pwd", "cwd": "sub"})  # relative → under the workspace
        assert out["stdout"].strip() == str(expected / "sub")
        out = worker.handle("/exec", {"command": "pwd", "cwd": str(tmp_path)})  # absolute stays absolute
        assert out["stdout"].strip() == str(tmp_path)
        stale = worker.handle("/exec", {"command": "pwd", "turn_id": "t-0"})
        assert stale["ok"] is False and "not running" in stale["error"]
        assert worker.handle("/exec", {"command": ""})["ok"] is False
    finally:
        gate.set()
    while not worker.handle("/events", {"turn_id": "t-1", "since": since, "wait": 1})["done"]:
        pass


# -- gemini: Session.exec over the fake provider -----------------------------------------


def test_gemini_exec_waits_for_dispatch_and_runs_on_the_turns_worker(tmp_path):
    provider = FakeSandboxProvider(lambda n: ScriptedWorker())
    engine = make_engine(provider)
    session = engine.start_session()

    async def go():
        run = session.run("go")
        assert session._sandbox is None
        r = await session.exec("echo probe; echo e >&2; exit 2", timeout=5)  # before the worker existed
        w = provider.worker(session._sandbox)
        w.workspace = str(tmp_path)
        r2 = await session.exec("pwd", cwd="", timeout=5)
        w.emit(result_event("done"))
        w.finish()
        await run
        return r, r2, w

    r, r2, w = asyncio.run(go())
    engine._join_background()
    assert (r.stdout, r.stderr, r.returncode, r.ok) == ("probe\n", "e\n", 2, False)
    assert r2.stdout.strip() == str(tmp_path)
    assert [c["command"] for c in w.execs] == ["echo probe; echo e >&2; exit 2", "pwd"]
    assert w.execs[0]["turn_id"] == w.turns[0]["turn_id"] and w.execs[0]["timeout"] == 5.0
    calls = [(path, body.get("timeout")) for _sb, path, body in provider.calls if path == "/exec"]
    assert calls == [("/exec", 5.0), ("/exec", 5.0)]


def test_gemini_exec_refuses_when_no_turn_runs_or_after_it_ended(monkeypatch):
    provider = FakeSandboxProvider(lambda n: ScriptedWorker(auto=[result_event("done")]))
    engine = make_engine(provider)
    session = engine.start_session()
    # No run in this process: exec() looks for a turn running elsewhere (the mirror; empty here).
    monkeypatch.setattr(backend.GeminiSession, "history", lambda self: [])

    async def go():
        with pytest.raises(ControlUnavailable, match="no turn is running"):
            await session.exec("pwd")
        run = session.run("go")
        await run
        with pytest.raises(ControlUnavailable, match="sandbox is gone"):
            await session.exec("pwd")

    asyncio.run(go())
    engine._join_background()
    assert not any(path == "/exec" for _sb, path, _b in provider.calls)


def test_gemini_exec_surfaces_a_dead_sandbox_and_a_refusing_worker_as_control_unavailable():
    class Refusing(ScriptedWorker):
        def handle(self, path, body):
            if path == "/exec":
                return {"ok": False, "error": "turn t-9 is not running on this worker"}
            return super().handle(path, body)

    provider = FakeSandboxProvider(lambda n: Refusing())
    engine = make_engine(provider)
    session = engine.start_session()

    async def go():
        run = session.run("go")
        while session._sandbox is None:
            await asyncio.sleep(0.005)
        provider.fail_next["/exec"] = [SandboxGone("expired")]
        with pytest.raises(ControlUnavailable, match="did not answer the probe: expired"):
            await session.exec("pwd")
        with pytest.raises(ControlUnavailable, match="refused the probe: turn t-9"):
            await session.exec("pwd")
        w = provider.worker(session._sandbox)
        w.emit(result_event("done"))
        w.finish()
        await run

    asyncio.run(go())
    engine._join_background()


def test_gemini_exec_validates_its_arguments():
    session = make_engine(FakeSandboxProvider(lambda n: ScriptedWorker())).start_session()

    async def go():
        with pytest.raises(ValueError, match="non-empty command"):
            await session.exec("")
        with pytest.raises(ValueError, match=f"at most {backend.EXEC_MAX_TIMEOUT_S:g}s"):
            await session.exec("pwd", timeout=backend.EXEC_MAX_TIMEOUT_S + 1)

    asyncio.run(go())


def test_gemini_exec_when_dispatch_fails_raises_instead_of_waiting_forever():
    provider = FakeSandboxProvider(lambda n: ScriptedWorker())
    provider.fail_next["/turn"] = [RuntimeError("proxy down")] * backend.DISPATCH_ATTEMPTS
    engine = make_engine(provider)
    session = engine.start_session()

    async def go():
        run = session.run("go")
        with pytest.raises(ControlUnavailable):
            await session.exec("pwd")
        return await run

    result = asyncio.run(go())
    engine._join_background()
    assert result.is_error


# -- gemini: exec() from a session re-attached in another process ------------------------


def _started(turn_id, worker):
    return AgentEvent(kind="status", summary="turn started",
                      raw={"event": "turn_started", "turn_id": turn_id, "worker": worker})


def _result(turn_id=None):
    raw = {"subtype": "success", "is_error": False}
    if turn_id is not None:
        raw["turn_id"] = turn_id
    return AgentEvent(kind="result", summary="done", raw=raw)


@pytest.mark.parametrize("events, expected", [
    ([], None),
    ([_started("t1", "s1")], (f"{INSTANCE}/sandboxEnvironments/s1", "t1")),
    ([_started("t1", "s1"), _result("t1")], None),
    ([_started("t1", "s1"), _result()], None),  # a client-recorded end (no turn stamp) closes it too
    ([_started("t1", "s1"), _result("t1"), _started("t2", "s2")], (f"{INSTANCE}/sandboxEnvironments/s2", "t2")),
    ([_started("t1", "s1"), _result("t0")], (f"{INSTANCE}/sandboxEnvironments/s1", "t1")),  # another turn's
    ([_started("t1", None)], None),  # a worker that recorded no sandbox: nothing to reach
    ([_started("t1", "projects/x/sandboxEnvironments/full")], ("projects/x/sandboxEnvironments/full", "t1")),
])
def test_find_running_turn_reads_the_last_started_turn_without_a_result(monkeypatch, events, expected):
    session = make_engine(FakeSandboxProvider()).get_session("old-sid")
    monkeypatch.setattr(backend.GeminiSession, "history", lambda self: list(events))
    assert session._find_running_turn() == expected


def test_gemini_exec_from_a_reattached_session_recovers_the_running_turns_sandbox(tmp_path, monkeypatch):
    store = LocalBlobStore(str(tmp_path / "mirror"))
    reads: list[str] = []
    inner = store.list
    store.list = lambda prefix: (reads.append(prefix), inner(prefix))[1]
    monkeypatch.setattr(history, "GcsBlobStore", lambda bucket, *a, **kw: store)

    provider = FakeSandboxProvider(lambda n: ScriptedWorker())
    owner = make_engine(provider)
    session = owner.start_session()
    other = make_engine(provider)  # a fresh process: no handle on the running turn
    adopted = other.get_session(session.session_id)
    assert adopted is not session

    def mirror(event, ms, turn_id):
        history.write_turn_mirror("gs://out/events", session.session_id, [history.mirror_line(event)],
                                  now_ms=ms, turn_id=turn_id, store=store)

    async def go():
        run = session.run("go")
        while session._sandbox is None:
            await asyncio.sleep(0.005)
        sandbox = session._sandbox
        w = provider.worker(sandbox)
        w.workspace = str(tmp_path)
        turn_id, worker = w.turns[0]["turn_id"], w.turns[0]["sandbox"]
        assert "/" not in worker and sandbox.endswith("/" + worker)
        mirror(_started(turn_id, worker), 1000, turn_id)  # what the real worker records as the turn begins
        r1 = await adopted.exec("pwd", timeout=5)
        r2 = await adopted.exec("echo again", timeout=5)
        assert len(reads) == 1  # resolved once, reused
        # The turn ends in the owner's process: the worker refuses further probes and the sandbox goes.
        w.emit(result_event("done"))
        w.finish()
        await run
        mirror(_result(turn_id), 2000, turn_id)
        with pytest.raises(ControlUnavailable, match="did not answer the probe|refused the probe"):
            await adopted.exec("pwd", timeout=5)
        assert adopted._attached is None  # forgotten: the next probe asks the mirror again
        with pytest.raises(ControlUnavailable, match=r"no turn is running on this session \(none recorded"):
            await adopted.exec("pwd", timeout=5)
        assert len(reads) == 2
        return r1, r2, sandbox, turn_id, w

    r1, r2, sandbox, turn_id, w = asyncio.run(go())
    owner._join_background()
    other._join_background()
    assert r1.stdout.strip() == str(tmp_path) and r1.ok
    assert r2.stdout == "again\n"
    probes = [(sb, body["turn_id"]) for sb, path, body in provider.calls if path == "/exec"]
    assert probes[:2] == [(sandbox, turn_id)] * 2  # reached the owner's sandbox, for its turn
    assert [c["command"] for c in w.execs[:2]] == ["pwd", "echo again"]


# -- local: Session.exec on the host ----------------------------------------------------


class _HeldHarness:
    """Yields nothing until ``release`` is set, then the result — keeps the turn running."""

    def __init__(self):
        self.release = asyncio.Event()

    async def run(self, spec, ctx) -> AsyncIterator[AgentEvent]:
        await self.release.wait()
        yield result_event("done")


def test_local_exec_runs_in_the_sessions_workspace_while_the_turn_runs(tmp_path):
    engine = local.deploy(AgentSpec(name="demo", model="m"), workdir=str(tmp_path / "wd"))
    session = engine.start_session()

    async def go():
        harness = _HeldHarness()
        engine._harness = harness
        with pytest.raises(ControlUnavailable, match="no turn is running"):
            await session.exec("pwd")
        run = session.run("go")
        (session.workspace / "sub").mkdir()
        r = await session.exec("pwd; echo hi >&2; exit 4", timeout=10)
        rel = await session.exec("pwd", cwd="sub", timeout=10)
        slow = await session.exec("sleep 5", timeout=0.2)
        with pytest.raises(ValueError):
            await session.exec("")
        harness.release.set()
        await run
        with pytest.raises(ControlUnavailable, match="no turn is running"):
            await session.exec("pwd")
        return r, rel, slow

    r, rel, slow = asyncio.run(go())
    assert r.stdout.strip() == str(session.workspace) and r.stderr == "hi\n" and r.returncode == 4
    assert rel.stdout.strip() == str(session.workspace / "sub")
    assert slow.returncode == EXEC_TIMEOUT_RC and "exceeded" in slow.stderr

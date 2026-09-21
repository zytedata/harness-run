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
from sandbox_fakes import FakeSandboxProvider, ScriptedWorker, make_engine, result_event

from agent_run import AgentSpec, ControlUnavailable, ExecResult, local
from agent_run.control import EXEC_TIMEOUT_RC, run_shell
from agent_run.events import AgentEvent
from agent_run.runtime.gemini import backend
from agent_run.runtime.gemini.provider import SandboxGone
from agent_run.runtime.gemini.worker import Worker

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
    from agent_run import control

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
    import agent_run.harness.claude_code as harness_mod

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


def test_gemini_exec_refuses_when_no_turn_runs_or_after_it_ended():
    provider = FakeSandboxProvider(lambda n: ScriptedWorker(auto=[result_event("done")]))
    engine = make_engine(provider)
    session = engine.start_session()

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
    provider.fail_next["/turn"] = [SandboxGone("expired")] * backend.DISPATCH_ATTEMPTS
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
        assert session.current_run is None
        run = session.run("go")
        assert session.current_run is run and session.busy
        (session.workspace / "sub").mkdir()
        r = await session.exec("pwd; echo hi >&2; exit 4", timeout=10)
        rel = await session.exec("pwd", cwd="sub", timeout=10)
        slow = await session.exec("sleep 5", timeout=0.2)
        with pytest.raises(ValueError):
            await session.exec("")
        harness.release.set()
        await run
        assert session.current_run is None and not session.busy
        with pytest.raises(ControlUnavailable, match="no turn is running"):
            await session.exec("pwd")
        return r, rel, slow

    r, rel, slow = asyncio.run(go())
    assert r.stdout.strip() == str(session.workspace) and r.stderr == "hi\n" and r.returncode == 4
    assert rel.stdout.strip() == str(session.workspace / "sub")
    assert slow.returncode == EXEC_TIMEOUT_RC and "exceeded" in slow.stderr

"""Claude CLI stderr capture (eval feedback: non-zero CLI exits were undiagnosable).

The SDK pipes the CLI's stderr ONLY when a callback is registered; without one it is
inherited/lost, while ProcessError still says "Check stderr output for details". The
harness registers ``_StderrCapture``: append-to-file beside the workspace + in-memory
tail surfaced with the failure. The SDK client is faked — no live model calls.
"""

from __future__ import annotations

import asyncio

import pytest
from fakes import make_sdk_client, result_msg

from remote_agent_toolkit import AgentSpec
from remote_agent_toolkit.harness.claude_code import ClaudeCodeHarness, _StderrCapture
from remote_agent_toolkit.harness.context import RunContext


def test_capture_appends_lazily_and_keeps_tail(tmp_path):
    cap = _StderrCapture(tmp_path / "job" / "stderr.log")
    assert not cap.path.exists()  # zero cost until the CLI actually writes

    cap("line one")
    cap("line two")
    assert cap.path.read_text() == "line one\nline two\n"
    assert cap.tail() == "line one\nline two"
    assert cap.tail(max_chars=8) == "line two"


def test_capture_file_is_capped_but_tail_survives(tmp_path):
    cap = _StderrCapture(tmp_path / "stderr.log")
    cap._FILE_CAP = 100
    for i in range(30):
        cap(f"line {i:04d} xxxxxxxxxx")
    text = cap.path.read_text()
    assert "capped" in text and len(text) < 300  # stopped near the cap, with a marker
    assert "line 0029" in cap.tail()  # the LAST lines (what diagnoses an exit) are kept


def _drive(script, tmp_path, monkeypatch):
    import claude_agent_sdk

    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", make_sdk_client(script))
    spec = AgentSpec(name="a", model="m")
    ctx = RunContext(spec=spec, prompt="hi", job_dir=tmp_path / "job", session_id="sid")
    events = []

    async def collect():
        async for ev in ClaudeCodeHarness().run(spec, ctx):
            events.append(ev)

    return ctx, events, collect


def test_cli_crash_surfaces_stderr_tail(tmp_path, monkeypatch):
    # A non-zero CLI exit raises out of the stream; the harness must surface what the
    # CLI said: a status event with the tail (reaches the stream / Cloud Logging on
    # gemini) and the tail + log path embedded in the raised error.
    script = [
        lambda c: c.options.stderr("node: something went wrong"),
        lambda c: c.options.stderr("Error: ENOMEM at finish line"),
        RuntimeError("Command failed with exit code 1"),
    ]
    ctx, events, collect = _drive(script, tmp_path, monkeypatch)

    with pytest.raises(RuntimeError) as ei:
        asyncio.run(collect())

    assert "ENOMEM at finish line" in str(ei.value)  # actual stderr, not a placeholder
    assert str(ctx.job_dir / "stderr.log") in str(ei.value)
    assert events[-1].kind == "status" and "ENOMEM" in events[-1].summary
    assert events[-1].raw["event"] == "claude_stderr"
    # The log lands at the bookkeeping level, NOT inside the agent-visible workspace.
    log = ctx.job_dir / "stderr.log"
    assert log.read_text().count("\n") == 2
    assert not (ctx.workspace / "stderr.log").exists()


def test_quiet_crash_raises_unwrapped(tmp_path, monkeypatch):
    # Nothing on stderr → nothing to add: the original exception propagates untouched.
    ctx, events, collect = _drive([ValueError("plain failure")], tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="plain failure"):
        asyncio.run(collect())
    assert not (ctx.job_dir / "stderr.log").exists()


def test_healthy_run_registers_callback_without_side_effects(tmp_path, monkeypatch):
    # The callback is always registered (options.stderr), but a clean run creates no file
    # and emits no stderr event — capture is free unless the CLI actually speaks.
    seen = {}
    script = [lambda c: seen.update(stderr=c.options.stderr), result_msg(result="done")]
    ctx, events, collect = _drive(script, tmp_path, monkeypatch)
    asyncio.run(collect())

    assert isinstance(seen["stderr"], _StderrCapture)
    assert [e.kind for e in events] == ["result"]
    assert not (ctx.job_dir / "stderr.log").exists()

"""Claude CLI stderr capture (eval feedback: non-zero CLI exits were undiagnosable).

The SDK pipes the CLI's stderr ONLY when a callback is registered; without one it is
inherited/lost, while ProcessError still says "Check stderr output for details". The
harness registers ``_StderrCapture``: append-to-file beside the workspace + in-memory
tail surfaced with the failure. ``query()`` is monkeypatched — no live model calls.
"""

from __future__ import annotations

import asyncio

import pytest

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


def _ctx(tmp_path, spec):
    return RunContext(spec=spec, prompt="hi", job_dir=tmp_path / "job", session_id="sid")


async def _collect(agen):
    events = []
    async for ev in agen:
        events.append(ev)
    return events


def test_cli_crash_surfaces_stderr_tail(tmp_path, monkeypatch):
    # A non-zero CLI exit raises out of query(); the harness must surface what the CLI
    # said: a status event with the tail (reaches the stream / Cloud Logging on gemini)
    # and the tail + log path embedded in the raised error.
    import claude_agent_sdk

    async def fake_query(*, prompt, options):
        options.stderr("node: something went wrong")
        options.stderr("Error: ENOMEM at finish line")
        raise RuntimeError("Command failed with exit code 1")
        yield  # pragma: no cover — makes this an async generator

    monkeypatch.setattr(claude_agent_sdk, "query", fake_query)
    spec = AgentSpec(name="a", model="m")
    ctx = _ctx(tmp_path, spec)
    events = []

    async def drive():
        async for ev in ClaudeCodeHarness().run(spec, ctx):
            events.append(ev)

    with pytest.raises(RuntimeError) as ei:
        asyncio.run(drive())

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
    import claude_agent_sdk

    async def fake_query(*, prompt, options):
        raise ValueError("plain failure")
        yield  # pragma: no cover

    monkeypatch.setattr(claude_agent_sdk, "query", fake_query)
    spec = AgentSpec(name="a", model="m")
    ctx = _ctx(tmp_path, spec)
    with pytest.raises(ValueError, match="plain failure"):
        asyncio.run(_collect(ClaudeCodeHarness().run(spec, ctx)))
    assert not (ctx.job_dir / "stderr.log").exists()


def test_healthy_run_registers_callback_without_side_effects(tmp_path, monkeypatch):
    # The callback is always registered (options.stderr), but a clean run creates no file
    # and emits no stderr event — capture is free unless the CLI actually speaks.
    import claude_agent_sdk
    from claude_agent_sdk import ResultMessage

    seen_options = {}

    async def fake_query(*, prompt, options):
        seen_options["stderr"] = options.stderr
        yield ResultMessage(
            subtype="success", duration_ms=10, duration_api_ms=8, is_error=False,
            num_turns=1, session_id="claude-sid", total_cost_usd=0.01, result="done",
        )

    monkeypatch.setattr(claude_agent_sdk, "query", fake_query)
    spec = AgentSpec(name="a", model="m")
    ctx = _ctx(tmp_path, spec)
    events = asyncio.run(_collect(ClaudeCodeHarness().run(spec, ctx)))

    assert isinstance(seen_options["stderr"], _StderrCapture)
    assert [e.kind for e in events] == ["result"]
    assert not (ctx.job_dir / "stderr.log").exists()

"""Run the Harness conformance suite against ClaudeCodeHarness (SDK client faked)."""

from __future__ import annotations

import asyncio

import pytest
from fakes import init_msg, make_sdk_client, result_msg

from remote_agent_toolkit import AgentSpec
from remote_agent_toolkit.conformance import run_harness_conformance
from remote_agent_toolkit.events import AgentEvent
from remote_agent_toolkit.harness.claude_code import ClaudeCodeHarness
from remote_agent_toolkit.harness.context import RunContext


def _run_turn(script, tmp_path, monkeypatch):
    import claude_agent_sdk

    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", make_sdk_client(script))
    spec = AgentSpec(name="a", model="m")
    ctx = RunContext(spec=spec, prompt="hi", job_dir=tmp_path / "job", session_id="sid")
    return lambda: ClaudeCodeHarness().run(spec, ctx)


def test_claude_code_harness_conformance(tmp_path, monkeypatch):
    run_turn = _run_turn([init_msg(), result_msg()], tmp_path, monkeypatch)
    assert asyncio.run(run_harness_conformance(run_turn)) is None


def test_conformance_catches_a_dropped_session_id(tmp_path, monkeypatch):
    # The failure the suite exists to catch: the init payload stops carrying the session
    # id embedders read out of raw["data"].
    async def missing_session_id():
        yield AgentEvent(kind="status", summary="started", raw={"subtype": "init", "data": {}})
        yield AgentEvent(
            kind="result", summary="done", cost_usd=0.01, usage={},
            raw={"num_turns": 1, "is_error": False},
        )

    with pytest.raises(AssertionError, match="session_id"):
        asyncio.run(run_harness_conformance(missing_session_id))

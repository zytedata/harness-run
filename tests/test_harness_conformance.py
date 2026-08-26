"""Run the Harness conformance suites: the generic one and Claude Code's stricter one."""

from __future__ import annotations

import asyncio

import pytest
from fakes import init_msg, make_sdk_client, result_msg

from remote_agent_toolkit import AgentSpec
from remote_agent_toolkit.conformance import (
    run_claude_code_harness_conformance,
    run_harness_conformance,
)
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
    assert asyncio.run(run_claude_code_harness_conformance(run_turn)) is None


# A well-formed normalized usage record: all five keys, int or None values.
NORMALIZED_USAGE = {
    "input_tokens": 5,
    "cache_read_input_tokens": None,
    "cache_creation_input_tokens": None,
    "output_tokens": 2,
    "reasoning_output_tokens": None,
}


def test_conformance_catches_a_dropped_session_id(tmp_path, monkeypatch):
    # The failure the stricter suite exists to catch: the init payload stops carrying the
    # session id embedders read out of raw["data"].
    async def missing_session_id():
        yield AgentEvent(kind="status", summary="started", raw={"subtype": "init", "data": {}})
        yield AgentEvent(
            kind="result", summary="done", cost_usd=0.01, usage=dict(NORMALIZED_USAGE),
            raw={"num_turns": 1, "is_error": False},
        )

    with pytest.raises(AssertionError, match="session_id"):
        asyncio.run(run_claude_code_harness_conformance(missing_session_id))


def test_generic_conformance_holds_for_a_codex_style_init_and_unpriced_result():
    # Codex's init event carries no raw["data"], and its cost_usd is None when the model
    # is unpriced — both must pass the backend-agnostic suite.
    async def codex_style():
        yield AgentEvent(kind="status", summary="started", raw={"subtype": "init"})
        yield AgentEvent(
            kind="result", summary="done", cost_usd=None, usage=dict(NORMALIZED_USAGE),
            raw={"num_turns": 1, "is_error": False},
        )

    assert asyncio.run(run_harness_conformance(codex_style)) is None


def test_generic_conformance_rejects_a_non_normalized_usage_dict():
    # A harness that leaks its own usage shape (extra/missing keys, or non-int values)
    # must fail the suite — the normalized record is the cross-harness contract.
    def turn_with_usage(usage):
        async def run_turn():
            yield AgentEvent(kind="status", summary="started", raw={"subtype": "init"})
            yield AgentEvent(
                kind="result", summary="done", cost_usd=None, usage=usage,
                raw={"num_turns": 1, "is_error": False},
            )

        return run_turn

    for bad in (
        {},  # pre-normalization verbatim passthrough
        {**NORMALIZED_USAGE, "total_tokens": 7},  # codex's old extra key
        {**NORMALIZED_USAGE, "output_tokens": 2.5},  # floats are not token counts
        {**NORMALIZED_USAGE, "input_tokens": True},  # bools are ints in Python; still wrong
    ):
        with pytest.raises(AssertionError, match="usage"):
            asyncio.run(run_harness_conformance(turn_with_usage(bad)))


def test_claude_conformance_pins_the_verbatim_usage_records(tmp_path, monkeypatch):
    # The claude layer promises the CLI's own records survive on raw: model_usage (the
    # complete per-model source of the normalized usage) and cli_usage (the flat dict).
    run_turn = _run_turn(
        [init_msg(), result_msg(model_usage={})], tmp_path, monkeypatch
    )
    with pytest.raises(AssertionError, match="model_usage"):
        asyncio.run(run_claude_code_harness_conformance(run_turn))

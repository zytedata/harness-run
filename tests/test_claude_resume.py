"""Resuming a stopped background task must not consume the new user's turn."""

import asyncio

import pytest
from claude_agent_sdk import AssistantMessage, TextBlock
from fakes import init_msg, make_sdk_client, result_msg, task_done_msg

from remote_agent_toolkit import AgentSpec
from remote_agent_toolkit.checkpoint.session_store import BlobSessionStore
from remote_agent_toolkit.harness.claude_code import ClaudeCodeHarness
from remote_agent_toolkit.harness.context import RunContext
from remote_agent_toolkit.ports.blobstore import LocalBlobStore


def noop(**kwargs):
    fields = dict(num_turns=0, cost=0, result=None, usage={}, model_usage={})
    return result_msg(**(fields | kwargs))


async def run(script, tmp_path, monkeypatch, *, resume=True):
    import claude_agent_sdk

    client_cls = make_sdk_client(script)
    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", client_cls)
    spec = AgentSpec(name="resume", model="m")
    blobs = LocalBlobStore(str(tmp_path / "blobs"))
    ctx = RunContext(
        spec=spec, prompt="Which files are here?", job_dir=tmp_path / "job",
        session_id="sid", resume_sid="sid" if resume else None,
        session_store=BlobSessionStore(blobs),
    )
    async with asyncio.timeout(2):
        events = [e async for e in ClaudeCodeHarness().run(spec, ctx)]
    return events, client_cls.instances[-1]


async def test_resume_waits_past_stale_task_noop_without_resubmitting_prompt(tmp_path, monkeypatch):
    events, client = await run([
        task_done_msg("old-task", "stopped"), init_msg(), noop(),
        init_msg(), AssistantMessage(content=[TextBlock(text="note.txt")], model="m"),
        result_msg(result="note.txt", num_turns=2, cost=0.12),
    ], tmp_path, monkeypatch)
    results = [e for e in events if e.kind == "result"]
    assert len(results) == 1 and results[0].summary == "note.txt"
    assert results[0].raw["num_turns"] == 2 and results[0].cost_usd == 0.12
    assert results[0].usage["output_tokens"] == 2
    assert client.prompts == ["Which files are here?"]
    assert client.disconnected
    assert any((e.raw or {}).get("event") == "awaiting_resume_prompt" for e in events)


@pytest.mark.parametrize("ending", ["timeout", "end", "crash"])
async def test_resume_noop_is_never_a_success_fallback(tmp_path, monkeypatch, ending):
    monkeypatch.setattr(ClaudeCodeHarness, "_RESUME_PROMPT_GRACE_S", 0.03, raising=False)
    tail = {"timeout": ["hang"], "end": [], "crash": [RuntimeError("CLI died")]}[ending]
    events, client = await run([
        task_done_msg("old-task", "stopped"), init_msg(), noop(), *tail,
    ], tmp_path, monkeypatch)
    results = [e for e in events if e.kind == "result"]
    assert len(results) == 1 and results[0].raw["is_error"] is True
    assert results[0].raw["subtype"] == "error_resume_prompt_unanswered"
    assert client.prompts == ["Which files are here?"] and client.disconnected


async def test_resume_grace_ends_when_next_invocation_starts(tmp_path, monkeypatch):
    monkeypatch.setattr(ClaudeCodeHarness, "_RESUME_PROMPT_GRACE_S", 0.03, raising=False)

    async def slow_model(_client):
        await asyncio.sleep(0.12)

    events, _ = await run([
        task_done_msg("old-task", "stopped"), init_msg(), noop(),
        init_msg(), slow_model, result_msg(result="note.txt"),
    ], tmp_path, monkeypatch)
    assert events[-1].summary == "note.txt" and not events[-1].raw["is_error"]


@pytest.mark.parametrize("result", [
    noop(result="A valid zero-turn answer"),
    noop(subtype="error_during_execution", is_error=True),
    noop(usage={"input_tokens": 20, "output_tokens": 0}),
    result_msg(result="A normal answer"),
])
async def test_resume_preserves_real_answers_usage_and_errors(tmp_path, monkeypatch, result):
    events, _ = await run([
        task_done_msg("old-task", "stopped"), init_msg(), result,
    ], tmp_path, monkeypatch)
    assert not any((e.raw or {}).get("event") == "awaiting_resume_prompt" for e in events)
    assert events[-1].raw["subtype"] == result.subtype


@pytest.mark.parametrize("resume,notification", [(False, True), (True, False)])
async def test_noop_guard_requires_resume_and_an_orphan_task(
    tmp_path, monkeypatch, resume, notification,
):
    script = [task_done_msg("old-task", "stopped")] if notification else []
    events, _ = await run([*script, init_msg(), noop()], tmp_path, monkeypatch, resume=resume)
    assert not any((e.raw or {}).get("event") == "awaiting_resume_prompt" for e in events)

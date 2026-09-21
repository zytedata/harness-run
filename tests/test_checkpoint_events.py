"""A caller stopping at the terminal result still receives checkpoint status."""

import asyncio

import pytest
from codex_fakes import make_async_codex, turn_completed, turn_started
from fakes import init_msg, make_sdk_client, result_msg

from agent_run import AgentSpec
from agent_run.checkpoint.session_store import BlobSessionStore
from agent_run.control import ControlMessage, LocalControlChannel
from agent_run.harness.claude_code import ClaudeCodeHarness
from agent_run.harness.codex import CodexHarness
from agent_run.harness.context import RunContext
from agent_run.ports.blobstore import LocalBlobStore


def context(tmp_path, harness):
    spec = AgentSpec(name="checkpoint", model="m", checkpoint=True, harness=harness)
    blobs = LocalBlobStore(str(tmp_path / "blobs"))
    ctx = RunContext(
        spec=spec, prompt="go", job_dir=tmp_path / "job", session_id="sid",
        blobs=blobs, session_store=BlobSessionStore(blobs),
        secrets={"OPENAI_API_KEY": "sk-test"}, control=LocalControlChannel(),
    )
    ctx.workspace.mkdir(parents=True)
    (ctx.workspace / "note.txt").write_text("partial work")
    return ctx


async def read_until_result(harness, ctx):
    events = []
    stream = harness.run(ctx.spec, ctx)
    try:
        async for event in stream:
            events.append(event)
            if event.kind == "result":
                break
    finally:
        await stream.aclose()
    return events


@pytest.mark.parametrize("save_fails", [False, True])
@pytest.mark.parametrize("is_error", [False, True])
async def test_claude_checkpoint_reaches_consumer_before_success_or_error(
    tmp_path, monkeypatch, save_fails, is_error,
):
    import claude_agent_sdk

    ctx = context(tmp_path, "claude-code")
    if save_fails:
        def denied(*args, **kwargs):
            raise PermissionError("archive denied")
        monkeypatch.setattr(ctx.blobs, "put_tree", denied)
    client_cls = make_sdk_client([
        init_msg(),
        result_msg(subtype="error_max_turns" if is_error else "success",
                   is_error=is_error),
    ])
    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", client_cls)
    events = await read_until_result(ClaudeCodeHarness(), ctx)
    checkpoint = [e for e in events if (e.raw or {}).get("event", "").startswith("checkpoint_")]
    assert len(checkpoint) == 1
    assert checkpoint[0].raw["event"] == ("checkpoint_error" if save_fails else "checkpoint_saved")
    assert events[-1].kind == "result" and events[-1].raw["is_error"] is is_error
    assert ctx.blobs.exists("workspace/sid.tar.gz") is not save_fails


async def test_codex_stop_before_first_rollout_reports_partial_checkpoint(tmp_path, monkeypatch):
    import openai_codex

    ctx = context(tmp_path, "codex")
    ctx.control.send(ControlMessage(op="stop"))

    async def wait_for_interrupt(client):
        async with asyncio.timeout(2):
            while not client.interrupted:
                await asyncio.sleep(0)

    client_cls = make_async_codex([
        turn_started(), wait_for_interrupt, turn_completed("interrupted"),
    ])
    monkeypatch.setattr(openai_codex, "AsyncCodex", client_cls)
    events = await read_until_result(CodexHarness(), ctx)
    checkpoint = next(e for e in events if (e.raw or {}).get("event") == "checkpoint_error")
    assert checkpoint.raw["conversation_saved"] is False
    assert checkpoint.raw["workspace_key"] == "workspace/sid.tar.gz"
    assert events[-1].raw["subtype"] == "interrupted"
    assert events[-1].raw["is_error"] is False
    assert events[-1].raw["num_turns"] == 0
    restored = tmp_path / "restored"
    ctx.blobs.get_tree(checkpoint.raw["workspace_key"], restored)
    assert (restored / "note.txt").read_text() == "partial work"

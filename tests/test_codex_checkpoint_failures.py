"""Codex checkpoint save and resume failures remain visible to callers."""

import json

import pytest
from codex_fakes import agent_message, token_usage, turn_completed, turn_started
from test_codex_harness import _events_of, _fake_rollout

from harness_run import AgentSpec
from harness_run.checkpoint.session_store import BlobSessionStore
from harness_run.harness.context import RunContext
from harness_run.ports.blobstore import LocalBlobStore


def context(tmp_path):
    spec = AgentSpec(name="checkpoint-test", model="gpt-5.6-luna", checkpoint=True,
                     harness="codex")
    blobs = LocalBlobStore(str(tmp_path / "blobs"))
    ctx = RunContext(spec=spec, prompt="dummy", job_dir=tmp_path / "job",
                     session_id="sid", secrets={"OPENAI_API_KEY": "sk-test"},
                     blobs=blobs, session_store=BlobSessionStore(blobs))
    ctx.workspace.mkdir(parents=True)
    return ctx


@pytest.mark.parametrize("failure", ["missing_rollout", "meta.json", "rollout.jsonl"])
async def test_conversation_save_failure_is_visible_before_the_result(
    tmp_path, monkeypatch, failure
):
    ctx = context(tmp_path)
    put_bytes = ctx.blobs.put_bytes

    def write(key, data):
        if key == f"codex-threads/sid/{failure}":
            raise PermissionError("DUMMY_STORAGE_CREDENTIAL must not be echoed")
        put_bytes(key, data)

    monkeypatch.setattr(ctx.blobs, "put_bytes", write)

    def write_rollout(_client):
        if failure != "missing_rollout":
            _fake_rollout(ctx.job_dir / "codex_home")

    script = [turn_started(), write_rollout, agent_message("done"), token_usage(),
              turn_completed()]
    events, _ = await _events_of(script, tmp_path, monkeypatch, spec=ctx.spec, ctx=ctx)
    checkpoint = [e for e in events if (e.raw or {}).get("event", "").startswith("checkpoint_")]
    assert len(checkpoint) == 1
    event = checkpoint[0]
    assert event.raw["event"] == "checkpoint_error"
    assert event.raw["conversation_saved"] is False
    assert event.raw["workspace_saved"] is True
    assert event.raw["workspace_key"] == "workspace/sid.tar.gz"
    assert ctx.blobs.exists("workspace/sid.tar.gz")
    result_index = next(i for i, e in enumerate(events) if e.kind == "result")
    assert events.index(event) < result_index  # clients may stop reading at the result
    assert "DUMMY_STORAGE_CREDENTIAL" not in json.dumps([e.raw for e in events])
    assert all("DUMMY_STORAGE_CREDENTIAL" not in (e.summary or "") for e in events)


async def test_conversation_error_does_not_claim_a_failed_workspace_was_saved(
    tmp_path, monkeypatch
):
    ctx = context(tmp_path)

    def denied(*_args):
        raise PermissionError("DUMMY_STORAGE_CREDENTIAL must not be echoed")

    monkeypatch.setattr(ctx.blobs, "put_tree", denied)
    script = [turn_started(), agent_message("done"), token_usage(), turn_completed()]
    events, _ = await _events_of(script, tmp_path, monkeypatch, spec=ctx.spec, ctx=ctx)
    event = next(e for e in events if (e.raw or {}).get("event") == "checkpoint_error")
    assert event.raw["conversation_saved"] is False
    assert event.raw["workspace_saved"] is False
    assert "workspace_key" not in event.raw
    assert "DUMMY_STORAGE_CREDENTIAL" not in event.summary


def stage_metadata(ctx, data=None):
    meta = {"thread_id": "thr-fake", "relpath": "sessions/rollout-x-thr-fake.jsonl"}
    ctx.blobs.put_bytes("codex-threads/sid/meta.json",
                        json.dumps(meta).encode() if data is None else data)
    ctx.blobs.put_bytes("codex-threads/sid/rollout.jsonl", b'{"line": 1}\n')
    ctx.resume_sid = ctx.session_id


@pytest.mark.parametrize("failure", ["metadata_denied", "rollout_denied", "rollout_missing"])
async def test_existing_checkpoint_read_failure_does_not_start_a_fresh_thread(
    tmp_path, monkeypatch, failure
):
    import openai_codex

    ctx = context(tmp_path)
    stage_metadata(ctx)
    if failure == "rollout_missing":
        ctx.blobs.delete("codex-threads/sid/rollout.jsonl")
    get_bytes = ctx.blobs.get_bytes
    leaf = "meta.json" if failure == "metadata_denied" else "rollout.jsonl"

    def read(key):
        if failure.endswith("_denied") and key == f"codex-threads/sid/{leaf}":
            raise PermissionError("DUMMY_STORAGE_CREDENTIAL must not be echoed")
        return get_bytes(key)

    monkeypatch.setattr(ctx.blobs, "get_bytes", read)
    part = "metadata" if failure == "metadata_denied" else "rollout"
    with pytest.raises(RuntimeError, match=f"checkpoint {part} could not be read") as error:
        await _events_of([], tmp_path, monkeypatch, spec=ctx.spec, ctx=ctx)
    assert "DUMMY_STORAGE_CREDENTIAL" not in str(error.value)
    client = openai_codex.AsyncCodex.instances[-1]
    assert not client.thread_starts and not client.thread_resumes
    assert not client.turns


@pytest.mark.parametrize("metadata", [
    b"invalid JSON containing DUMMY_STORAGE_CREDENTIAL",
    b"[]",
    b'{"relpath": "sessions/rollout.jsonl"}',
    b'{"thread_id": "", "relpath": "sessions/rollout.jsonl"}',
    b'{"thread_id": 123, "relpath": "sessions/rollout.jsonl"}',
])
async def test_corrupt_checkpoint_metadata_has_a_clear_redacted_error(
    tmp_path, monkeypatch, metadata
):
    import openai_codex

    ctx = context(tmp_path)
    stage_metadata(ctx, metadata)
    with pytest.raises(RuntimeError, match="checkpoint metadata is invalid") as error:
        await _events_of([], tmp_path, monkeypatch, spec=ctx.spec, ctx=ctx)
    assert "DUMMY_STORAGE_CREDENTIAL" not in str(error.value)
    client = openai_codex.AsyncCodex.instances[-1]
    assert not client.thread_starts and not client.thread_resumes
    assert not client.turns

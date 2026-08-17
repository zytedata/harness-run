"""Tests for workspace snapshot/restore over a LocalBlobStore."""

from __future__ import annotations

from remote_agent_toolkit.checkpoint import restore, snapshot
from remote_agent_toolkit.ports.blobstore import LocalBlobStore


def test_snapshot_restore_round_trip(tmp_path) -> None:
    work = tmp_path / "work"
    (work / "proj").mkdir(parents=True)
    (work / "out.json").write_text('{"ok": true}')
    (work / "proj" / "spider.py").write_text("class S: pass")

    store = LocalBlobStore(str(tmp_path / "store"))
    key = snapshot(store, "sess-1", str(work))
    assert key == "workspace/sess-1.tar.gz"

    dest = tmp_path / "resumed"
    assert restore(store, "sess-1", str(dest)) is True
    assert (dest / "out.json").read_text() == '{"ok": true}'
    assert (dest / "proj" / "spider.py").read_text() == "class S: pass"


def test_restore_missing_session_returns_false(tmp_path) -> None:
    store = LocalBlobStore(str(tmp_path / "store"))
    assert restore(store, "never-snapshotted", str(tmp_path / "dest")) is False


def _ctx(tmp_path, store, **kwargs):
    from remote_agent_toolkit.harness.context import RunContext
    from remote_agent_toolkit.spec import AgentSpec

    return RunContext(
        spec=AgentSpec(name="a", model="m", checkpoint=True),
        prompt="go",
        job_dir=tmp_path / "job",
        session_id="sess-1",
        session_store=object(),
        blobs=store,
        **kwargs,
    )


def test_checkpoint_snapshots_and_scrubs_the_toolkit_owned_workspace(tmp_path) -> None:
    from remote_agent_toolkit.harness._shared import finalize_checkpoint

    store = LocalBlobStore(str(tmp_path / "store"))
    ctx = _ctx(tmp_path, store)
    (ctx.workspace / "repo" / ".git").mkdir(parents=True)
    cfg = ctx.workspace / "repo" / ".git" / "config"
    cfg.write_text('[remote "origin"]\n\turl = https://u:tok@example.com/p.git\n')

    event = finalize_checkpoint(ctx.spec, ctx)
    assert event.raw["workspace_key"] == "workspace/sess-1.tar.gz"
    assert store.exists("workspace/sess-1.tar.gz")
    assert "tok" not in cfg.read_text()  # never archive a push token


def test_checkpoint_leaves_a_caller_owned_workspace_alone(tmp_path) -> None:
    # workspace_dir is the caller's directory: it is already durable, holds files no
    # session owns, and its repos are the caller's too — so the conversation is
    # checkpointed and the files are neither archived nor rewritten.
    from remote_agent_toolkit.harness._shared import finalize_checkpoint

    store = LocalBlobStore(str(tmp_path / "store"))
    theirs = tmp_path / "theirs"
    (theirs / "repo" / ".git").mkdir(parents=True)
    cfg = theirs / "repo" / ".git" / "config"
    cfg.write_text('[remote "origin"]\n\turl = https://u:tok@example.com/p.git\n')
    ctx = _ctx(tmp_path, store, workspace_dir=theirs)

    event = finalize_checkpoint(ctx.spec, ctx)
    assert event.raw["workspace_key"] is None
    assert not store.exists("workspace/sess-1.tar.gz")
    assert "https://u:tok@example.com/p.git" in cfg.read_text()

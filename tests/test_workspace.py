"""Tests for workspace snapshot/restore over a LocalBlobStore."""

from __future__ import annotations

import pytest

from agent_run.checkpoint import restore, snapshot
from agent_run.ports.blobstore import LocalBlobStore


def test_snapshot_restore_round_trip(tmp_path) -> None:
    work = tmp_path / "work"
    (work / "proj").mkdir(parents=True)
    (work / "out.json").write_text('{"ok": true}')
    (work / "proj" / "spider.py").write_text("class S: pass")

    store = LocalBlobStore(str(tmp_path / "store"))
    key = snapshot(store, "sess-1", str(work))
    assert key == "workspace/sess-1.tar.gz"

    dest = tmp_path / "resumed"
    result = restore(store, "sess-1", str(dest))
    assert result and result.found is True and result.skipped == ()
    assert (dest / "out.json").read_text() == '{"ok": true}'
    assert (dest / "proj" / "spider.py").read_text() == "class S: pass"


def test_restore_missing_session_returns_false(tmp_path) -> None:
    store = LocalBlobStore(str(tmp_path / "store"))
    assert not restore(store, "never-snapshotted", str(tmp_path / "dest"))


def _ctx(tmp_path, store, **kwargs):
    from agent_run.harness.context import RunContext
    from agent_run.spec import AgentSpec

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
    from agent_run.harness._shared import finalize_checkpoint

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
    from agent_run.harness._shared import finalize_checkpoint

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


class _FailingMidway:
    """A store whose ``get_tree`` writes one file into the destination and then fails, the
    shape of a ``tarfile`` abort partway through an archive (#74)."""

    def __init__(self, inner):
        self._inner = inner

    def exists(self, key):
        return self._inner.exists(key)

    def get_tree(self, key, local_dir):
        from pathlib import Path

        (Path(local_dir) / "repo").mkdir(parents=True)
        (Path(local_dir) / "repo" / "half.py").write_text("partial")
        raise RuntimeError("archive aborted at member 162")


def test_restore_failure_removes_the_half_restored_workspace_it_created(tmp_path) -> None:
    # #74: an aborted extraction used to leave the members before the bad one on disk, and
    # the fresh-workspace fallback then failed on ``git clone`` into the non-empty dir.
    store = LocalBlobStore(str(tmp_path / "store"))
    snapshot(store, "sess-1", str(tmp_path))  # any snapshot, so exists() is True
    dest = tmp_path / "fresh-worker-ws"
    with pytest.raises(RuntimeError, match="member 162"):
        restore(_FailingMidway(store), "sess-1", str(dest))
    assert not dest.exists()  # nothing half-restored for the fallback to trip over


def test_restore_failure_keeps_a_workspace_that_already_had_content(tmp_path) -> None:
    # A same-host resume restores over the live directory; on failure it stays as it was.
    store = LocalBlobStore(str(tmp_path / "store"))
    snapshot(store, "sess-1", str(tmp_path))
    dest = tmp_path / "same-host-ws"
    dest.mkdir()
    (dest / "mine.txt").write_text("from the previous turn")
    with pytest.raises(RuntimeError):
        restore(_FailingMidway(store), "sess-1", str(dest))
    assert (dest / "mine.txt").read_text() == "from the previous turn"


def test_try_restore_reports_the_failure_instead_of_swallowing_it(tmp_path, caplog) -> None:
    from agent_run.checkpoint.workspace import try_restore, workspace_ready_event

    store = LocalBlobStore(str(tmp_path / "store"))
    snapshot(store, "sess-1", str(tmp_path))
    with caplog.at_level("WARNING", logger="agent_run.checkpoint.workspace"):
        result, error = try_restore(_FailingMidway(store), "sess-1", str(tmp_path / "ws"))
    assert not result and error == "RuntimeError: archive aborted at member 162"
    assert "restore failed for session sess-1" in caplog.text
    assert "RuntimeError" in caplog.text  # with the traceback

    event = workspace_ready_event(result, error, [], ["repo"])
    assert event["restored"] is False and event["restore_error"] == error
    assert "restore failed: RuntimeError" in event["summary"]
    # A plain fresh start (no snapshot) has no error to report.
    ok = workspace_ready_event(restore(store, "nope", str(tmp_path / "ws2")), None, [], [])
    assert ok["restored"] is False and ok["restore_error"] is None and ok["skipped"] == []

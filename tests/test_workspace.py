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

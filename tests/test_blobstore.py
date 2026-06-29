"""Tests for the LocalBlobStore adapter (bytes, trees, list, traversal guard)."""

from __future__ import annotations

import pytest

from remote_agent_toolkit.ports.blobstore import LocalBlobStore


def test_put_get_bytes_round_trip(tmp_path) -> None:
    store = LocalBlobStore(str(tmp_path / "store"))
    store.put_bytes("a/b/c.bin", b"hello")
    assert store.get_bytes("a/b/c.bin") == b"hello"
    assert store.exists("a/b/c.bin")
    assert not store.exists("a/b/missing.bin")


def test_get_bytes_missing_raises_keyerror(tmp_path) -> None:
    store = LocalBlobStore(str(tmp_path / "store"))
    with pytest.raises(KeyError):
        store.get_bytes("nope")


def test_put_get_tree_round_trip(tmp_path) -> None:
    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    (src / "top.txt").write_text("top")
    (src / "sub" / "nested.txt").write_text("nested")

    store = LocalBlobStore(str(tmp_path / "store"))
    store.put_tree("snap.tar.gz", str(src))

    dest = tmp_path / "dest"
    store.get_tree("snap.tar.gz", str(dest))

    # arcname="." means contents restore without an extra nesting level.
    assert (dest / "top.txt").read_text() == "top"
    assert (dest / "sub" / "nested.txt").read_text() == "nested"


def test_get_tree_missing_raises_keyerror(tmp_path) -> None:
    store = LocalBlobStore(str(tmp_path / "store"))
    with pytest.raises(KeyError):
        store.get_tree("missing.tar.gz", str(tmp_path / "out"))


def test_list_returns_keys_relative_to_root_sorted(tmp_path) -> None:
    store = LocalBlobStore(str(tmp_path / "store"))
    store.put_bytes("p/2.txt", b"2")
    store.put_bytes("p/1.txt", b"1")
    store.put_bytes("q/3.txt", b"3")

    assert store.list("p/") == ["p/1.txt", "p/2.txt"]
    assert store.list("") == ["p/1.txt", "p/2.txt", "q/3.txt"]
    assert store.list("q/") == ["q/3.txt"]


def test_list_empty_store(tmp_path) -> None:
    store = LocalBlobStore(str(tmp_path / "empty"))
    assert store.list("") == []


def test_path_traversal_rejected(tmp_path) -> None:
    store = LocalBlobStore(str(tmp_path / "store"))
    with pytest.raises(ValueError):
        store.put_bytes("../escape.txt", b"x")
    with pytest.raises(ValueError):
        store.get_bytes("../../etc/passwd")

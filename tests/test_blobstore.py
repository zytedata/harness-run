"""Tests for the LocalBlobStore adapter (bytes, trees, list, traversal guard)."""

from __future__ import annotations

import os
import tarfile

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


def test_parse_gcs_uri_variants() -> None:
    from remote_agent_toolkit.ports.blobstore import parse_gcs_uri

    assert parse_gcs_uri("gs://bkt/some/prefix") == ("bkt", "some/prefix")
    assert parse_gcs_uri("bkt/some/prefix/") == ("bkt", "some/prefix")
    assert parse_gcs_uri("gs://bkt") == ("bkt", "")


def test_get_tree_restores_symlinks_whatever_they_point_at(tmp_path) -> None:
    """#74: a workspace snapshot always carries ``.venv/bin/python -> /usr/local/bin/python3``.
    ``tarfile``'s ``data`` filter rejects absolute (and escaping) link targets and aborted the
    whole restore at that member. A symlink is only followed at use, so it is recreated."""
    src = tmp_path / "src"
    (src / "repo").mkdir(parents=True)
    (src / "repo" / "a.py").write_text("x")
    (src / "repo" / ".venv" / "bin").mkdir(parents=True)
    os.symlink("/usr/local/bin/python3", src / "repo" / ".venv" / "bin" / "python")  # absolute
    os.symlink("python", src / "repo" / ".venv" / "bin" / "python3")  # relative, inside
    os.symlink("../../../elsewhere", src / "repo" / ".venv" / "escaping")  # relative, outside

    store = LocalBlobStore(str(tmp_path / "store"))
    store.put_tree("snap.tar.gz", str(src))
    dest = tmp_path / "dest"
    assert store.get_tree("snap.tar.gz", str(dest)) == []  # nothing skipped

    venv = dest / "repo" / ".venv"
    assert (dest / "repo" / "a.py").read_text() == "x"
    assert os.readlink(venv / "bin" / "python") == "/usr/local/bin/python3"
    assert os.readlink(venv / "bin" / "python3") == "python"
    assert os.readlink(venv / "escaping") == "../../../elsewhere"


def _crafted_archive(path, members):
    with tarfile.open(str(path), mode="w:gz") as tar:
        for info, data in members:
            tar.addfile(info, data)


def _member(name, kind=tarfile.REGTYPE, linkname="", data=b""):
    import io

    info = tarfile.TarInfo(name)
    info.type = kind
    info.linkname = linkname
    info.size = len(data)
    return info, (io.BytesIO(data) if kind == tarfile.REGTYPE else None)


def test_get_tree_skips_unsafe_members_and_reports_them(tmp_path) -> None:
    """Symlinks are the only relaxation: everything else the ``data`` filter refuses is left
    out — and reported, not fatal — so a stray FIFO cannot cost the rest of the snapshot."""
    store = LocalBlobStore(str(tmp_path / "store"))
    (tmp_path / "store").mkdir()
    _crafted_archive(
        tmp_path / "store" / "bad.tar.gz",
        [
            _member("./ok.txt", data=b"fine"),
            _member("./fifo", kind=tarfile.FIFOTYPE),
            _member("../escape.txt", data=b"outside"),
            _member("./hard", kind=tarfile.LNKTYPE, linkname="/etc/hostname"),
            _member("./link", kind=tarfile.SYMTYPE, linkname="/etc"),  # allowed
            _member("./link/through.txt", data=b"written outside via the symlink"),  # refused
            _member("./also-ok.txt", data=b"still restored"),
        ],
    )
    dest = tmp_path / "dest"
    skipped = store.get_tree("bad.tar.gz", str(dest))
    assert skipped == ["./fifo", "../escape.txt", "./hard", "./link/through.txt"]
    assert (dest / "ok.txt").read_text() == "fine"
    assert (dest / "also-ok.txt").read_text() == "still restored"
    assert os.readlink(dest / "link") == "/etc"
    assert not (tmp_path / "escape.txt").exists() and not (dest / "fifo").exists()
    assert not os.path.lexists("/etc/through.txt")

"""Workspace snapshot / restore over a ``BlobStore`` (DESIGN.md §6).

Tar the per-job cwd to the BlobStore at a checkpoint; restore it on resume. The full
uncommitted workspace (generated project files, downloaded artifacts, etc.) is captured so
a fresh worker can resume with the files the agent already produced — conversation state is
handled separately by :class:`~remote_agent_toolkit.checkpoint.session_store.BlobSessionStore`.

Lifted & generalized from the PoC ``checkpoint/workspace.py`` (DESIGN.md §8): these are thin
wrappers over the ``BlobStore`` tree ops, which own the tar.gz mechanics. ``BlobStore`` is the
only dependency; nothing third-party is imported here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..ports.blobstore import BlobStore


def _workspace_key(session_id: str) -> str:
    return f"workspace/{session_id}.tar.gz"


def snapshot(blobs: BlobStore, session_id: str, workdir: str) -> str:
    """Tar ``workdir`` into the BlobStore under a session-keyed key; return the key."""
    key = _workspace_key(session_id)
    blobs.put_tree(key, workdir)
    return key


def restore(blobs: BlobStore, session_id: str, workdir: str) -> bool:
    """Restore the workspace snapshot for ``session_id`` into ``workdir``.

    Returns whether a snapshot existed.
    """
    key = _workspace_key(session_id)
    if not blobs.exists(key):
        return False
    blobs.get_tree(key, workdir)
    return True

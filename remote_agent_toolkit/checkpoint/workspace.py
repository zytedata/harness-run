"""Workspace snapshot / restore over a ``BlobStore`` (DESIGN.md §6) — P1.

Tar the per-job cwd to the BlobStore at a checkpoint; restore it on resume. Lifted &
generalized from the PoC ``checkpoint/`` (DESIGN.md §8). ``BlobStore`` is the only
dependency; nothing third-party is imported here. P0: stubs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..ports.blobstore import BlobStore


def snapshot(blobs: BlobStore, session_id: str, workdir: str) -> str:
    """Tar ``workdir`` into the BlobStore, returning the blob key (P1)."""
    raise NotImplementedError("P1: put_tree(workdir) under a session-keyed blob key; return it.")


def restore(blobs: BlobStore, session_id: str, workdir: str) -> bool:
    """Restore the latest workspace snapshot for ``session_id`` into ``workdir`` (P1).

    Returns whether a snapshot existed.
    """
    raise NotImplementedError("P1: get_tree the session's workspace snapshot into workdir.")

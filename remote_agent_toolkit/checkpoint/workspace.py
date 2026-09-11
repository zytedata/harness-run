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

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..ports.blobstore import BlobStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RestoreResult:
    """What :func:`restore` did; truthy iff a snapshot existed and was extracted."""

    found: bool
    skipped: tuple[str, ...] = ()  # archive members left out as unsafe (BlobStore.get_tree)

    def __bool__(self) -> bool:
        return self.found


def _workspace_key(session_id: str) -> str:
    return f"workspace/{session_id}.tar.gz"


def snapshot(blobs: BlobStore, session_id: str, workdir: str) -> str:
    """Tar ``workdir`` into the BlobStore under a session-keyed key; return the key."""
    key = _workspace_key(session_id)
    blobs.put_tree(key, workdir)
    return key


def restore(blobs: BlobStore, session_id: str, workdir: str) -> RestoreResult:
    """Restore the workspace snapshot for ``session_id`` into ``workdir``.

    The result is truthy iff a snapshot existed (it then also names any members the store
    skipped as unsafe). If extraction fails, a ``workdir`` this call created or found empty
    is removed again before the error propagates, so a caller that falls back to a fresh
    workspace never stages into half-restored files (a ``git clone`` into a directory the
    aborted extraction had already half-filled fails on "already exists"). A ``workdir``
    that already had content — a same-host resume — is left as it is.
    """
    key = _workspace_key(session_id)
    if not blobs.exists(key):
        return RestoreResult(found=False)
    path = Path(workdir)
    pristine = not path.exists() or not any(path.iterdir())
    try:
        skipped = blobs.get_tree(key, workdir)
    except BaseException:
        if pristine:
            shutil.rmtree(workdir, ignore_errors=True)
        raise
    return RestoreResult(found=True, skipped=tuple(skipped or ()))


def try_restore(
    blobs: BlobStore, session_id: str, workdir: str
) -> tuple[RestoreResult, str | None]:
    """:func:`restore` for the runtimes' workspace prep: ``(result, error)``.

    A failed restore (a corrupt or unreadable snapshot, a store error) is not fatal to the
    turn: the runtime falls back to a fresh workspace. But it must not be silent either —
    the caller of ``send()`` is told, through ``workspace_ready``, why it did not get its
    files back. The exception is logged with its traceback and summarized in ``error``;
    ``result`` is then a not-found one.
    """
    try:
        return restore(blobs, session_id, workdir), None
    except Exception as exc:  # noqa: BLE001 — the turn goes on in a fresh workspace
        logger.warning(
            "workspace restore failed for session %s; starting from a fresh workspace",
            session_id,
            exc_info=True,
        )
        return RestoreResult(found=False), f"{type(exc).__name__}: {str(exc)[:200]}"


def workspace_ready_event(
    result: RestoreResult, error: str | None, skills: list[str], repos: list[str]
) -> dict:
    """The ``workspace_ready`` status event both runtimes emit before the first turn event.

    ``restored`` says whether the agent has its previous files back. When it is ``False``
    on a resume, ``restore_error`` tells a failed restore (the workspace is fresh, as after
    ``checkpoint=False``) from a session that simply had no snapshot (``None``).
    ``skipped`` lists snapshot members that were left out as unsafe to extract.
    """
    restored = bool(result)
    detail = f"restored={restored}"
    if error:
        detail += f", restore failed: {error}"
    if result.skipped:
        detail += f", skipped={len(result.skipped)}"
    return {
        "event": "workspace_ready",
        "restored": restored,
        "restore_error": error,
        "skipped": list(result.skipped),
        "skills": skills,
        "repos": repos,
        "summary": (
            f"workspace ready ({detail}, skills staged={len(skills)}, "
            f"repos cloned={len(repos)})"
        ),
    }

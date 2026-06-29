"""``GcsSessionStore`` — the Claude SDK ``SessionStore`` protocol over GCS (DESIGN.md §6) — P1.

Keys by **``session_id`` alone** (ignores ``project_key``) so any worker, any cwd can
resume — the SDK's default cwd-keyed local store is fatal for serverless. Validated by
``conformance.run_session_store_conformance``.

``claude_agent_sdk`` types and ``google.cloud.storage`` are imported lazily. P0: stub.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..ports.blobstore import BlobStore


class GcsSessionStore:
    """Claude SDK ``SessionStore`` backed by a :class:`BlobStore`, keyed by session_id (P1)."""

    def __init__(self, blobs: BlobStore) -> None:
        self.blobs = blobs

    async def append(self, session_id: str, project_key: str, message: Any) -> None:
        raise NotImplementedError(
            "P1: append a message to the session log keyed by session_id alone "
            "(ignore project_key) so any worker/cwd can resume."
        )

    async def load(self, session_id: str, project_key: str) -> list[Any]:
        raise NotImplementedError("P1: load the message log for session_id from the BlobStore.")

    async def list_subkeys(self, session_id: str) -> list[str]:
        raise NotImplementedError("P1: list stored subkeys for session_id.")

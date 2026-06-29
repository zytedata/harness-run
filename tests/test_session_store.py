"""Run the SessionStore conformance suite against BlobSessionStore over LocalBlobStore."""

from __future__ import annotations

import asyncio

from remote_agent_toolkit.checkpoint import BlobSessionStore
from remote_agent_toolkit.conformance import run_session_store_conformance
from remote_agent_toolkit.ports.blobstore import LocalBlobStore


def test_blob_session_store_conformance(tmp_path) -> None:
    root = str(tmp_path / "sessions")
    # All factory-produced stores share the same backing dir so the keying invariant
    # (a fresh store with the same session_id sees the data) is exercised.
    result = asyncio.run(
        run_session_store_conformance(lambda: BlobSessionStore(LocalBlobStore(root)))
    )
    assert result is None

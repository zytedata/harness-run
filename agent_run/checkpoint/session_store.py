"""``BlobSessionStore`` — the Claude SDK ``SessionStore`` protocol over a ``BlobStore`` (DESIGN.md §6).

Keys by **``session_id`` alone** (ignores the SDK's cwd-derived ``project_key``) so any
worker, any cwd can resume — the SDK's default cwd-keyed local store is fatal for
serverless. Backed by any :class:`~agent_run.ports.blobstore.BlobStore`, so a
``LocalBlobStore`` gives local checkpointing and a ``GcsBlobStore`` gives cloud resume.
Validated by :func:`agent_run.conformance.run_session_store_conformance`.

Design (lifted from the PoC ``gcs_session_store.py``):

* **Append-only batch objects, never read-modify-write.** Each ``append`` writes a new,
  time-ordered blob under ``<prefix>/<session_id>/<leaf>/``; ``load`` concatenates them in
  key order (lexical == chronological). Avoids O(n^2) rewrites for long sessions and is safe
  for the single-writer-per-session model.
* **``uuid`` dedup in ``load``** makes re-materialization idempotent.

No third-party imports here — only the ``BlobStore`` port and stdlib ``json``/``time``.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..ports.blobstore import BlobStore


def _claude_session_id(session_id: str) -> str:
    """Map a runtime session id to the canonical UUID the Claude Agent SDK requires.

    The gemini cold path's session id comes from ADK ``sessions.create`` and is NUMERIC
    (e.g. ``1966652674296250368``); pinning it as the Claude session id makes the ``claude``
    CLI exit 1 with "Invalid session ID. Must be a valid UUID" — before emitting any event.
    A sid that is already a canonical UUID (the warm path's client-chosen id, every local
    one) passes through unchanged; anything else maps via uuid5, which is DETERMINISTIC so
    the store keying and resume stay stable across turns and workers.

    Everything reading or writing the store must key by this, the worker driving the SDK
    and a client reading transcripts back alike.

    The ``ratk-session:`` salt below is FROZEN and deliberately kept through the rename to
    ``agent-run``: it is a uuid5 input, never displayed or typed, so renaming it buys
    nothing and silently orphans stored data — the derived id is the blob path prefix
    (``sessions/<csid>/main/``) and the run-scoped GCS credential's scope prefix
    (``scoped_gcs.py``), and uuid5 is one-way, so old sessions could not be found again.
    Should it ever have to change, dual-read on ``load``/``load_all`` (new id first, old id
    as fallback) plus granting both prefixes for a transition window is the cheap route.
    """
    try:
        uuid.UUID(session_id)
        return session_id
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"ratk-session:{session_id}"))


class BlobSessionStore:
    """Claude SDK ``SessionStore`` backed by a :class:`BlobStore`, keyed by ``session_id``.

    The SDK passes a ``key`` dict with ``session_id`` and an optional ``subpath`` (for
    subagent transcripts). The main transcript lands under ``.../<sid>/main/``; subagents
    keep their real subpath so it round-trips losslessly in :meth:`list_subkeys`.
    """

    def __init__(self, blobs: BlobStore, prefix: str = "sessions") -> None:
        self.blobs = blobs
        self._prefix = prefix.strip("/")
        self._seq = 0

    # -- internal key helpers ------------------------------------------------
    def _dir(self, key: dict) -> str:
        """Blob-key prefix for a transcript (main or a subagent subpath)."""
        sid = key["session_id"]
        sub = key.get("subpath")
        leaf = "main" if not sub else sub
        return f"{self._prefix}/{sid}/{leaf}"

    # -- required SessionStore methods --------------------------------------
    async def append(self, key: dict, entries: list[dict]) -> None:
        if not entries:
            return
        self._seq += 1
        # Lexical key order == chronological: ms timestamp + per-instance seq.
        blobkey = f"{self._dir(key)}/{int(time.time() * 1000):013d}-{self._seq:06d}.jsonl"
        body = "\n".join(json.dumps(e) for e in entries) + "\n"
        self.blobs.put_bytes(blobkey, body.encode())

    async def load(self, key: dict) -> list[dict] | None:
        blobkeys = self.blobs.list(self._dir(key) + "/")
        if not blobkeys:
            return None
        out: list[dict] = []
        seen: set[str] = set()
        for bk in blobkeys:  # already sorted lexically by the BlobStore
            for line in self.blobs.get_bytes(bk).decode().splitlines():
                if not line.strip():
                    continue
                e = json.loads(line)
                u = e.get("uuid")
                if u is not None:
                    if u in seen:
                        continue
                    seen.add(u)
                out.append(e)
        return out or None

    # -- beyond the protocol: read a whole session back ----------------------
    async def load_all(self, session_id: str) -> dict[str, list[dict]]:
        """Every transcript of ``session_id``, keyed by ``"main"`` and each subagent subpath.

        Leaves with no entries are omitted, so a session that was never persisted reads
        back as ``{}``. What the runtimes' ``Session.transcripts()`` is built on.
        """
        out: dict[str, list[dict]] = {}
        for leaf in ("main", *await self.list_subkeys({"session_id": session_id})):
            key = {"session_id": session_id}
            if leaf != "main":
                key["subpath"] = leaf
            entries = await self.load(key)
            if entries:
                out[leaf] = entries
        return out

    # -- optional: lets resume rehydrate subagent transcripts ---------------
    async def list_subkeys(self, key: dict) -> list[str]:
        sid = key["session_id"]
        base = f"{self._prefix}/{sid}/"
        subs: set[str] = set()
        for bk in self.blobs.list(base):
            rel = bk[len(base):]
            container = rel.rsplit("/", 1)[0] if "/" in rel else ""
            if container and container != "main":
                subs.add(container)
        return sorted(subs)

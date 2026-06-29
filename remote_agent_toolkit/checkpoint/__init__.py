"""Checkpoint / resume (DESIGN.md §6) — conversation (SessionStore) + workspace (tar).

Keyed by ``session_id`` alone so any worker, any cwd can resume (the default cwd-keyed
local store is fatal for serverless).
"""

from __future__ import annotations

from .session_store import BlobSessionStore
from .workspace import restore, snapshot

__all__ = ["BlobSessionStore", "snapshot", "restore"]

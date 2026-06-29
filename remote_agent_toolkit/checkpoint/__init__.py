"""Checkpoint / resume (DESIGN.md §6) — conversation (SessionStore) + workspace (tar).

Keyed by ``session_id`` alone so any worker, any cwd can resume (the default cwd-keyed
local store is fatal for serverless). P1.
"""

from __future__ import annotations

__all__: list[str] = []

"""Run-plane namespaces: ``local`` (in-process) and ``gemini`` (Google Agent Sandbox).

Both expose the *same* Engine/Session/Run surface (DESIGN.md §3.2): develop locally,
ship remotely, with zero code change — swap ``local`` ↔ ``gemini``.
"""

from __future__ import annotations

from . import gemini, local

__all__ = ["local", "gemini"]

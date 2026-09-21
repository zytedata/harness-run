"""Run-plane namespaces: ``local`` (in-process) and ``sandbox`` (Google Agent Sandbox).

Both expose the *same* Engine/Session/Run surface (DESIGN.md §3.2): develop locally,
ship remotely, with zero code change — swap ``local`` ↔ ``sandbox``.
"""

from __future__ import annotations

from . import local, sandbox

__all__ = ["local", "sandbox"]

"""The harness seam (DESIGN.md §1, §7): the abstraction over "a coding agent loop".

Only the Claude Code (Agent SDK) binding ships (``claude_code``); the protocol
(``base.Harness``) exists so a second harness can slot in behind it later.
"""

from __future__ import annotations

from .base import Harness

__all__ = ["Harness"]

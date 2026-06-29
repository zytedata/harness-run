"""GitHub MCP attach + optional REST helpers (DESIGN.md §8) — P1.

Builds the GitHub MCP server config (token resolved at runtime from the
``SecretResolver``) for the harness, and optional thin REST helpers. ``claude_agent_sdk``
MCP config types are imported lazily. P0: stub.
"""

from __future__ import annotations

from typing import Any


def github_mcp(token: str) -> Any:
    """Build the GitHub MCP server config using a runtime-resolved ``token`` (P1)."""
    raise NotImplementedError(
        "P1: construct the GitHub MCP server config with the runtime-resolved token "
        "(secret value, not name) for the harness."
    )

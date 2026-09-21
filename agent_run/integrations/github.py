"""GitHub MCP config builder (DESIGN.md §8).

Builds the GitHub MCP server config for the harness. The token is resolved at runtime from
the ``SecretResolver`` (a secret *value*, not a name — never baked into the deployed engine,
DESIGN.md §3.5). Tiny and dependency-free; ported from the PoC ``agent.py::_mcp_servers``.
"""

from __future__ import annotations


def github_mcp_config(token: str) -> dict:
    """Build the GitHub MCP server config using a runtime-resolved ``token``."""
    return {
        "type": "http",
        "url": "https://api.githubcopilot.com/mcp/",
        "headers": {"Authorization": f"Bearer {token}"},
    }

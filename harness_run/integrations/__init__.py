"""Integrations (DESIGN.md §8): git cloning + token injection, GitHub MCP attach.

Lifted & generalized from the PoC ``git_repo.py`` / ``_mcp_servers``.
"""

from __future__ import annotations

from .git import (
    clone,
    provision_repo,
    provision_repos,
    reauth_repos,
    redact,
    repo_name,
    scrub_repo_tokens,
)
from .github import github_mcp_config

__all__ = [
    "clone",
    "provision_repo",
    "provision_repos",
    "reauth_repos",
    "scrub_repo_tokens",
    "repo_name",
    "redact",
    "github_mcp_config",
]

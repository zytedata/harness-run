"""Integrations (DESIGN.md §8): git cloning + PAT injection, GitHub MCP attach.

Lifted & generalized from the PoC ``git_repo.py`` / ``_mcp_servers``.
"""

from __future__ import annotations

from .git import clone, provision_repo, provision_repos, redact, repo_name
from .github import github_mcp_config

__all__ = [
    "clone",
    "provision_repo",
    "provision_repos",
    "repo_name",
    "redact",
    "github_mcp_config",
]

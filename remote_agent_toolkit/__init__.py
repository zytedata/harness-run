"""remote-agent-toolkit — define and run remote/background Claude-Code agents.

Public surface (see DESIGN.md §5). Importing this package pulls in *only* stdlib;
all third-party dependencies (claude-agent-sdk, google-adk, google-cloud-*, agentplatform,
yaml) are imported lazily inside the function/method that needs them.
"""

from __future__ import annotations

from .config import INHERIT, SessionConfig, TurnConfig
from .events import AgentEvent, RunResult, RunStatus, StopReason
from .runtime import gemini, local
from .spec import AgentSpec, McpServer, RepoSource, SkillSource, SystemPrompt

__all__ = [
    "AgentSpec",
    "SessionConfig",
    "TurnConfig",
    "INHERIT",
    "SystemPrompt",
    "SkillSource",
    "RepoSource",
    "McpServer",
    "AgentEvent",
    "RunResult",
    "RunStatus",
    "StopReason",
    "gemini",
    "local",
]

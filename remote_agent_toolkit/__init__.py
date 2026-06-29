"""remote-agent-toolkit — define and run remote/background Claude-Code agents.

Public surface (see DESIGN.md §5). Importing this package pulls in *only* stdlib;
all third-party dependencies (claude-agent-sdk, google-adk, google-cloud-*, vertexai,
yaml) are imported lazily inside the function/method that needs them.
"""

from __future__ import annotations

from .events import AgentEvent, RunResult, RunStatus, StopReason
from .runtime import gemini, local
from .spec import AgentSpec, McpServer, SkillSource, SystemPrompt

__all__ = [
    "AgentSpec",
    "SystemPrompt",
    "SkillSource",
    "McpServer",
    "AgentEvent",
    "RunResult",
    "RunStatus",
    "StopReason",
    "gemini",
    "local",
]

"""harness-run — define and run remote/background Claude-Code agents.

Public surface (see DESIGN.md §5). Importing this package pulls in *only* stdlib;
all third-party dependencies (claude-agent-sdk, google-adk, google-cloud-*, agentplatform,
yaml) are imported lazily inside the function/method that needs them.
"""

from __future__ import annotations

from .config import INHERIT, SessionConfig, TurnConfig
from .control import ControlUnavailable, ExecResult
from .events import AgentEvent, RunResult, RunStatus, StopReason
from .runtime import sandbox, local
from .spec import (
    DEFAULT_MAX_BUFFER_SIZE,
    AgentSpec,
    McpServer,
    RepoSource,
    SkillSource,
    SystemPrompt,
)

__all__ = [
    "AgentSpec",
    "DEFAULT_MAX_BUFFER_SIZE",
    "SessionConfig",
    "TurnConfig",
    "INHERIT",
    "SystemPrompt",
    "SkillSource",
    "RepoSource",
    "McpServer",
    "AgentEvent",
    "ControlUnavailable",
    "ExecResult",
    "RunResult",
    "RunStatus",
    "StopReason",
    "sandbox",
    "local",
]

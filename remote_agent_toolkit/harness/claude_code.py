"""``ClaudeCodeHarness`` — the Claude Agent SDK binding (DESIGN.md §7, §8) — P1.

An ADK ``BaseAgent`` wrapping the Agent SDK ``query()``: builds ``ClaudeAgentOptions``
from the ``AgentSpec``, translates SDK messages → generic ``AgentEvent``s (via
``translate``), and runs so it can deploy to Agent Engine. This is the generalized
``ClaudeCodeAgent`` from the PoC ``agent.py``.

``claude_agent_sdk`` and ``google.adk`` are imported lazily inside methods so importing
this module needs no third-party deps. P0: signatures/stubs only.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, TYPE_CHECKING

from ..events import AgentEvent

if TYPE_CHECKING:
    from ..spec import AgentSpec


class ClaudeCodeHarness:
    """The Claude Agent SDK harness (implements ``harness.base.Harness``; P1)."""

    def build_options(self, spec: AgentSpec, ctx: Any) -> Any:
        """Build ``ClaudeAgentOptions`` from ``spec`` (lazy ``import claude_agent_sdk``; P1)."""
        raise NotImplementedError(
            "P1: build ClaudeAgentOptions from the spec — model, system prompt "
            "(inherit/append), staged skills, MCP servers, tool allow/deny lists, "
            "permission mode, session id, and the SessionStore."
        )

    async def run(self, spec: AgentSpec, ctx: Any) -> AsyncIterator[AgentEvent]:
        """Drive the Agent SDK ``query()`` loop, yielding ``AgentEvent``s (P1)."""
        raise NotImplementedError(
            "P1: run claude_agent_sdk.query() and translate each SDK message into an "
            "AgentEvent; finalize checkpoint/artifacts inline at the terminal result event."
        )
        yield  # pragma: no cover - makes this an async generator for typing

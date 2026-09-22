"""The ``Harness`` protocol (DESIGN.md §7) — stdlib-only.

A harness is the abstraction over a coding-agent loop: it turns an ``AgentSpec`` into
backend options and runs the loop, yielding generic ``AgentEvent``s (so the rest of the
toolkit is harness-agnostic).
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Protocol, runtime_checkable, TYPE_CHECKING

from ..events import AgentEvent

if TYPE_CHECKING:
    from ..spec import AgentSpec


@runtime_checkable
class Harness(Protocol):
    """Abstraction over a coding-agent loop (Claude Code binding ships; §1, §9)."""

    def build_options(self, spec: AgentSpec, ctx: Any) -> Any:
        """Translate ``spec`` (+ runtime ``ctx``) into backend-specific run options.

        For the Claude binding this builds ``ClaudeAgentOptions``; ``ctx`` carries
        resolved secrets, staged skills, the session id, the cwd, and the port adapters.
        """
        ...

    async def run(self, spec: AgentSpec, ctx: Any) -> AsyncIterator[AgentEvent]:
        """Run the agent loop, yielding generic :class:`AgentEvent`s.

        Finalize (checkpoint/artifacts) inline at the terminal ``result`` event — the
        async executor stops draining the generator afterward (DESIGN.md §6).
        """
        ...

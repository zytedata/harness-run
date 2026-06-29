"""``EventSink`` port + adapters (DESIGN.md §6, §7).

The only near-real-time async channel on Gemini Agent Runtime: the harness emits a
per-step structured log (write side), the client tails it filtered by ``session_id``
(read side). Protocol is stdlib-only; ``CloudLoggingSink`` imports
``google.cloud.logging`` lazily. P0: protocol + stubs (``InMemorySink`` is P1).
"""

from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable

from ..events import AgentEvent


@runtime_checkable
class EventSink(Protocol):
    """Per-step event channel: ``emit`` on the runtime side, ``tail`` on the client side."""

    def emit(self, event: AgentEvent) -> None:
        """Write ``event`` to the sink (runtime side)."""
        ...

    def tail(self, session_id: str, since: float | None = None) -> AsyncIterator[AgentEvent]:
        """Stream events for ``session_id`` (optionally from ``since``) (client side)."""
        ...


class CloudLoggingSink:
    """Cloud Logging-backed :class:`EventSink` — the near-real-time channel (P2).

    ``emit`` uses ``log_struct``; ``tail`` polls the logging API filtered by
    ``session_id``. ``google.cloud.logging`` is imported lazily.
    """

    def __init__(self, session_id: str, log_name: str = "remote-agent-toolkit") -> None:
        self.session_id = session_id
        self.log_name = log_name

    def emit(self, event: AgentEvent) -> None:
        raise NotImplementedError("P2: log_struct the event tagged with session_id.")

    def tail(self, session_id: str, since: float | None = None) -> AsyncIterator[AgentEvent]:
        raise NotImplementedError(
            "P2: poll Cloud Logging filtered by session_id (since), yielding AgentEvents."
        )


class InMemorySink:
    """In-process :class:`EventSink` for local dev / tests (P1)."""

    def __init__(self) -> None:
        self._events: dict[str, list[AgentEvent]] = {}

    def emit(self, event: AgentEvent) -> None:
        raise NotImplementedError("P1: append the event to an in-memory per-session buffer.")

    def tail(self, session_id: str, since: float | None = None) -> AsyncIterator[AgentEvent]:
        raise NotImplementedError(
            "P1: async-iterate the in-memory buffer for session_id, awaiting new events."
        )

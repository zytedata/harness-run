"""Offline round-trip tests for the EventSink adapters (InMemorySink only — no Cloud Logging).

Verifies the emit/tail contract that the Cloud Logging channel must also honour: events
round-trip in order, tail STOPS after the terminal ``"result"`` event, emit guards a missing
session_id, and tailing an unknown session is bounded (does not hang).
"""

from __future__ import annotations

import asyncio

import pytest

from remote_agent_toolkit.events import AgentEvent
from remote_agent_toolkit.ports.eventsink import InMemorySink


async def _collect(sink: InMemorySink, session_id: str) -> list[AgentEvent]:
    return [e async for e in sink.tail(session_id)]


def test_inmemory_roundtrip_stops_after_result() -> None:
    sink = InMemorySink(session_id="s1")
    sink.emit(AgentEvent(kind="thinking", summary="planning"))
    sink.emit(AgentEvent(kind="tool_use", summary="run bash", raw={"tool": "Bash"}))
    sink.emit(
        AgentEvent(
            kind="result",
            summary="done",
            cost_usd=0.1,
            raw={"is_error": False, "num_turns": 3},
        )
    )

    events = asyncio.run(_collect(sink, "s1"))

    # Iteration stopped right after the terminal "result" event (length 3, not blocking).
    assert [e.kind for e in events] == ["thinking", "tool_use", "result"]
    assert len(events) == 3
    result = events[-1]
    assert result.cost_usd == 0.1
    assert result.raw == {"is_error": False, "num_turns": 3}


def test_emit_without_session_id_raises() -> None:
    sink = InMemorySink()
    with pytest.raises(ValueError):
        sink.emit(AgentEvent(kind="thinking", summary="no session"))


def test_tail_unknown_session_returns_nothing() -> None:
    # No events ever emitted for "ghost" — tail must terminate (idle bound), not hang.
    # Tighten the idle bound so this exercises the timeout without a 5s wait.
    sink = InMemorySink(session_id="s1")
    sink._IDLE_TIMEOUT_S = 0.1
    events = asyncio.run(_collect(sink, "ghost"))
    assert events == []

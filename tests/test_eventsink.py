"""Offline tests for the EventSink adapters (no GCP; Cloud Logging is faked at the client).

Verifies the emit/tail contract that the Cloud Logging channel must also honour: events
round-trip in order, tail STOPS after the terminal ``"result"`` event, emit guards a missing
session_id, and tailing an unknown session is bounded (does not hang). The CloudLoggingSink
tests cover the one-shot ``read`` backstop: read-quota rejections (the shared
60-reads/min per-project budget) are retried with backoff inside the caller's timeout,
while other errors surface immediately. (The live tail is the GCS event stream —
``runtime/gemini/stream.py`` — tested in ``test_stream.py``; Cloud Logging is emit-only.)
"""

from __future__ import annotations

import asyncio
import datetime

import pytest
from google.api_core import exceptions as gax

from remote_agent_toolkit.events import AgentEvent
from remote_agent_toolkit.ports import eventsink as eventsink_mod
from remote_agent_toolkit.ports.eventsink import CloudLoggingSink, InMemorySink


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


# -- CloudLoggingSink poll-failure policy (fake client; no GCP) -----------------


class _Entry:
    """Stand-in for a google.cloud.logging entry (attribute access only)."""

    def __init__(self, insert_id: str, payload: dict):
        self.insert_id = insert_id
        self.payload = payload
        self.timestamp = datetime.datetime(2026, 8, 4, tzinfo=datetime.timezone.utc)


class _FlakyClient:
    """Raises scripted exceptions for the first polls, then serves a result entry."""

    project = "p"

    def __init__(self, errors: list[Exception]):
        self._errors = list(errors)
        self.calls = 0

    def list_entries(self, *, filter_, order_by):
        self.calls += 1
        if self._errors:
            raise self._errors.pop(0)
        return iter([_Entry("i1", {"kind": "result", "summary": "done"})])


def test_read_retries_quota_rejections_within_timeout(monkeypatch) -> None:
    client = _FlakyClient([gax.TooManyRequests("read quota")] * 2)
    sink = CloudLoggingSink(project="p")
    sink._client = client
    monkeypatch.setattr(eventsink_mod.random, "uniform", lambda a, b: 0.01)

    events = sink.read("sid", timeout=5.0)

    assert [e.kind for e in events] == ["result"]
    assert client.calls == 3


def test_read_still_raises_non_quota_errors() -> None:
    sink = CloudLoggingSink(project="p")
    sink._client = _FlakyClient([RuntimeError("permission denied")])
    with pytest.raises(RuntimeError, match="permission denied"):
        sink.read("sid", timeout=2.0)

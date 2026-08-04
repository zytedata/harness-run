"""Offline tests for the EventSink adapters (no GCP; Cloud Logging is faked at the client).

Verifies the emit/tail contract that the Cloud Logging channel must also honour: events
round-trip in order, tail STOPS after the terminal ``"result"`` event, emit guards a missing
session_id, and tailing an unknown session is bounded (does not hang). The CloudLoggingSink
tests cover the poll-failure policy: read-quota rejections (the shared 60-reads/min
per-project budget) back off and never kill a tail, while other persistent errors still
surface after ``_MAX_POLL_FAILURES``.
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


def _tail_all(sink: CloudLoggingSink) -> list[AgentEvent]:
    async def go():
        return [e async for e in sink.tail("sid")]

    return asyncio.run(go())


def _sink_with(client, monkeypatch, sleeps: list[float]) -> CloudLoggingSink:
    sink = CloudLoggingSink(project="p")
    sink._client = client
    # Record backoff sleeps instead of actually waiting (keeps the test instant).
    real_sleep = asyncio.sleep

    async def fake_sleep(s):
        sleeps.append(s)
        await real_sleep(0)

    monkeypatch.setattr(eventsink_mod.asyncio, "sleep", fake_sleep)
    # Deterministic jitter: take the upper bound of the jitter window.
    monkeypatch.setattr(eventsink_mod.random, "uniform", lambda a, b: b)
    return sink


def test_tail_survives_sustained_quota_rejections(monkeypatch) -> None:
    # 15 consecutive 429s — well past _MAX_POLL_FAILURES — must NOT kill the tail: quota
    # exhaustion is an expected operating condition (60 reads/min shared by the project).
    sleeps: list[float] = []
    client = _FlakyClient([gax.TooManyRequests("read quota")] * 15)
    sink = _sink_with(client, monkeypatch, sleeps)

    events = _tail_all(sink)

    assert [e.kind for e in events] == ["result"]
    # Backoff grew exponentially (2, 4, 8, 16) and capped at _QUOTA_BACKOFF_CAP_S.
    assert sleeps[:5] == [2.0, 4.0, 8.0, 16.0, 30.0]
    assert max(sleeps) == eventsink_mod._QUOTA_BACKOFF_CAP_S


def test_tail_treats_grpc_resource_exhausted_as_quota(monkeypatch) -> None:
    sleeps: list[float] = []
    client = _FlakyClient([gax.ResourceExhausted("read quota")] * 12)
    sink = _sink_with(client, monkeypatch, sleeps)

    assert [e.kind for e in _tail_all(sink)] == ["result"]


def test_tail_still_raises_on_persistent_non_quota_errors(monkeypatch) -> None:
    # The fatal threshold must stay: a bad filter / revoked perms should surface, not spin.
    sleeps: list[float] = []
    client = _FlakyClient([RuntimeError("permission denied")] * 20)
    sink = _sink_with(client, monkeypatch, sleeps)

    with pytest.raises(RuntimeError, match="permission denied"):
        _tail_all(sink)
    assert client.calls == eventsink_mod._MAX_POLL_FAILURES


def test_tail_quota_rejections_do_not_feed_the_fatal_counter(monkeypatch) -> None:
    # Alternating 429s and real errors: only the real errors may count toward the fatal
    # threshold, and a success resets both counters.
    sleeps: list[float] = []
    errors: list[Exception] = []
    for _ in range(9):  # 9 (quota, error) pairs — 9 real errors < _MAX_POLL_FAILURES
        errors += [gax.TooManyRequests("q"), RuntimeError("blip")]
    client = _FlakyClient(errors)
    sink = _sink_with(client, monkeypatch, sleeps)

    assert [e.kind for e in _tail_all(sink)] == ["result"]
    assert client.calls == 19  # 18 scripted failures + the succeeding poll
    # Both policies ran: the 1s retry pause for real errors, the backoff ladder for 429s
    # (2, 4, ... — consecutive across interleaved non-quota blips, reset only by a success).
    assert sleeps.count(eventsink_mod._POLL_INTERVAL_S) == 9
    assert {2.0, 4.0, 8.0}.issubset(set(sleeps))


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

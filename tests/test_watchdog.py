"""Run-robustness: the cold-run job watchdog and the bounded Cloud Logging polls.

Both exist because of a live incident: a dropped connection wedged `list_entries` forever
(no per-call timeout in the logging client) while the job had long finished — the client
hung ~1 h past completion. Layer 1 bounds every poll; layer 2 ends a run whose job died
without a terminal event.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

import pytest

from remote_agent_toolkit.events import AgentEvent
from remote_agent_toolkit.ports import eventsink
from remote_agent_toolkit.runtime.gemini.backend import _watched_tail


def _result(text="done"):
    return AgentEvent(kind="result", summary=text, raw={"subtype": "success", "is_error": False})


# -- layer 2: the watchdog -------------------------------------------------------


def test_watchdog_is_passthrough_while_events_flow():
    probes = []

    async def tail():
        yield AgentEvent(kind="status", summary="working")
        yield _result()

    async def probe():
        probes.append(1)
        return "RUNNING"

    async def drive():
        return [e async for e in _watched_tail(tail, probe, "sid", quiet_s=5, grace_s=5)]

    events = asyncio.run(drive())
    assert [e.kind for e in events] == ["status", "result"]
    assert probes == []  # a flowing stream never probes


def test_watchdog_ends_run_when_job_dies_eventless():
    async def tail():
        yield AgentEvent(kind="status", summary="started")
        await asyncio.sleep(3600)  # the tail never sees a terminal event

    async def probe():
        return "FAILED"

    async def drive():
        return [e async for e in _watched_tail(tail, probe, "sid", quiet_s=0.02, grace_s=0.05)]

    events = asyncio.run(drive())
    assert events[0].kind == "status"
    terminal = events[-1]
    assert terminal.kind == "result" and terminal.raw["is_error"] is True
    assert terminal.raw["event"] == "job_terminated_without_result"
    assert terminal.raw["job_state"] == "FAILED"
    assert "no terminal event" in terminal.summary


def test_watchdog_tolerates_flaky_probe_then_fires():
    # None (probe unreachable) and RUNNING keep the run alive; only a terminal state + grace ends it.
    states = iter([None, "RUNNING", "SUCCESS", "SUCCESS", "SUCCESS"])

    async def tail():
        yield AgentEvent(kind="status", summary="started")
        await asyncio.sleep(3600)

    async def probe():
        return next(states, "SUCCESS")

    async def drive():
        return [e async for e in _watched_tail(tail, probe, "sid", quiet_s=0.02, grace_s=0.03)]

    events = asyncio.run(drive())
    assert events[-1].raw["job_state"] == "SUCCESS"


def test_watchdog_recovers_real_result_from_durable_history():
    # Observed live: Cloud Logging ingestion lag starved the tail while the job SUCCEEDED and
    # the worker had already mirrored everything to GCS. The watchdog must deliver the REAL
    # remainder from the durable record — not a synthetic error.
    async def tail():
        yield AgentEvent(kind="status", summary="workspace ready")  # only this got ingested
        await asyncio.sleep(3600)

    async def probe():
        return "SUCCESS"

    full_history = [
        AgentEvent(kind="status", summary="workspace ready"),
        AgentEvent(kind="message", summary="DONE"),
        _result("DONE"),
    ]

    async def history_reader():
        return full_history

    async def drive():
        return [e async for e in _watched_tail(tail, probe, "sid", history_reader,
                                               quiet_s=0.02, grace_s=0.03)]

    events = asyncio.run(drive())
    # The already-delivered event is not duplicated; the missing remainder arrives verbatim.
    assert [e.summary for e in events] == ["workspace ready", "DONE", "DONE"]
    assert events[-1].kind == "result" and not (events[-1].raw or {}).get("is_error")


def test_watchdog_synthesizes_error_when_history_has_no_result():
    # A durable record without a terminal result (worker killed mid-turn) is no recovery —
    # the synthetic error still fires.
    async def tail():
        await asyncio.sleep(3600)
        yield  # pragma: no cover

    async def probe():
        return "FAILED"

    async def history_reader():
        return [AgentEvent(kind="status", summary="workspace ready")]  # partial, no result

    async def drive():
        return [e async for e in _watched_tail(tail, probe, "sid", history_reader,
                                               quiet_s=0.02, grace_s=0.03)]

    events = asyncio.run(drive())
    assert len(events) == 1 and events[0].raw["event"] == "job_terminated_without_result"


def test_watchdog_plain_end_of_stream_passes_through():
    async def tail():
        yield AgentEvent(kind="status", summary="only")
        # generator ends with no result event (e.g. tail hit its own cap)

    async def probe():
        return "RUNNING"

    async def drive():
        return [e async for e in _watched_tail(tail, probe, "sid", quiet_s=5, grace_s=5)]

    events = asyncio.run(drive())
    assert [e.kind for e in events] == ["status"]  # no synthetic invention


# -- layer 1: bounded polls ------------------------------------------------------


class _Entry:
    def __init__(self, payload, insert_id):
        self.payload = payload
        self.insert_id = insert_id
        self.timestamp = datetime.now(timezone.utc)


def _sink_with(client):
    sink = eventsink.CloudLoggingSink(project="p")
    sink._client = client
    return sink


def test_tail_rides_out_transient_poll_failures(monkeypatch):
    monkeypatch.setattr(eventsink, "_POLL_INTERVAL_S", 0.001)

    class FlakyClient:
        project = "p"
        calls = 0

        def list_entries(self, filter_, order_by):
            FlakyClient.calls += 1
            if FlakyClient.calls <= 2:
                raise ConnectionError("transient network blip")
            return [_Entry({"kind": "result", "summary": "ok"}, "i1")]

    async def drive():
        return [e async for e in _sink_with(FlakyClient()).tail("sid", since=time.time())]

    events = asyncio.run(drive())
    assert [e.kind for e in events] == ["result"] and FlakyClient.calls == 3


def test_tail_surfaces_permanent_poll_failure(monkeypatch):
    monkeypatch.setattr(eventsink, "_POLL_INTERVAL_S", 0.001)
    monkeypatch.setattr(eventsink, "_MAX_POLL_FAILURES", 3)

    class DeadClient:
        project = "p"

        def list_entries(self, filter_, order_by):
            raise PermissionError("permanently forbidden")

    async def drive():
        return [e async for e in _sink_with(DeadClient()).tail("sid", since=time.time())]

    with pytest.raises(PermissionError):
        asyncio.run(drive())


def test_tail_times_out_wedged_poll_and_retries(monkeypatch):
    # A wedged connection blocks list_entries indefinitely; the per-poll bound abandons it
    # and the NEXT poll (fresh call) succeeds.
    monkeypatch.setattr(eventsink, "_POLL_INTERVAL_S", 0.001)
    monkeypatch.setattr(eventsink, "_POLL_TIMEOUT_S", 0.05)

    class WedgedOnceClient:
        project = "p"
        calls = 0

        def list_entries(self, filter_, order_by):
            WedgedOnceClient.calls += 1
            if WedgedOnceClient.calls == 1:
                time.sleep(0.5)  # wedged well past the poll bound
                return []
            return [_Entry({"kind": "result", "summary": "ok"}, "i1")]

    async def drive():
        # Measure inside the loop: asyncio.run()'s SHUTDOWN waits for the abandoned worker
        # thread (loop.shutdown_default_executor), which is fine in production where the loop
        # is long-lived — the stream itself must not wait out the wedge.
        t0 = time.monotonic()
        events = [e async for e in _sink_with(WedgedOnceClient()).tail("sid", since=time.time())]
        return events, time.monotonic() - t0

    events, elapsed = asyncio.run(drive())
    assert [e.kind for e in events] == ["result"]
    assert elapsed < 0.4  # did NOT wait out the 0.5 s wedge


def test_read_bounded_by_timeout():
    import concurrent.futures

    class WedgedClient:
        project = "p"

        def list_entries(self, filter_, order_by):
            time.sleep(1.0)
            return []

    with pytest.raises(concurrent.futures.TimeoutError):
        _sink_with(WedgedClient()).read("sid", timeout=0.05)

"""Run-robustness: the cold-run job watchdog and the bounded Cloud Logging polls.

Both exist because of a live incident: a dropped connection wedged `list_entries` forever
(no per-call timeout in the logging client) while the job had long finished — the client
hung ~1 h past completion. Layer 1 bounds every poll; layer 2 ends a run whose job died
without a terminal event.
"""

from __future__ import annotations

import asyncio
import time

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


def test_watchdog_recovers_result_when_tail_arrives_out_of_order():
    # Live regression: Cloud Logging delivered checkpoint before result. Four tail events
    # had arrived, but the result was history item 4 (index 3), so count-based prefix slicing
    # skipped it and the client finalized a successful run as an error.
    workspace = AgentEvent(kind="status", summary="workspace ready")
    started = AgentEvent(kind="status", summary="session started")
    message = AgentEvent(kind="message", summary="ok")
    result = _result("ok")
    checkpoint = AgentEvent(kind="status", summary="checkpoint saved")

    async def tail():
        for event in (workspace, started, message, checkpoint):
            yield event
        await asyncio.sleep(3600)

    async def probe():
        return "SUCCESS"

    async def history_reader():
        return [workspace, started, message, result, checkpoint]

    async def drive():
        return [e async for e in _watched_tail(
            tail, probe, "sid", history_reader, quiet_s=0.02, grace_s=0.03
        )]

    events = asyncio.run(drive())
    assert [e.summary for e in events] == [
        "workspace ready", "session started", "ok", "checkpoint saved", "ok"
    ]
    assert events[-1].kind == "result"


def test_watchdog_history_reconciliation_preserves_duplicate_counts():
    # Two identical status events in history are legitimate. If the tail delivered one,
    # multiset subtraction must recover exactly the other one, not drop both.
    repeated = AgentEvent(kind="status", summary="still working")

    async def tail():
        yield repeated
        await asyncio.sleep(3600)

    async def probe():
        return "SUCCESS"

    async def history_reader():
        return [repeated, repeated, _result()]

    async def drive():
        return [e async for e in _watched_tail(
            tail, probe, "sid", history_reader, quiet_s=0.02, grace_s=0.03
        )]

    events = asyncio.run(drive())
    assert [e.kind for e in events] == ["status", "status", "result"]


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


# -- layer 1: bounded one-shot read ------------------------------------------------------


def _sink_with(client):
    sink = eventsink.CloudLoggingSink(project="p")
    sink._client = client
    return sink


def test_read_bounded_by_timeout():
    import concurrent.futures

    class WedgedClient:
        project = "p"

        def list_entries(self, filter_, order_by):
            time.sleep(1.0)
            return []

    with pytest.raises(concurrent.futures.TimeoutError):
        _sink_with(WedgedClient()).read("sid", timeout=0.05)

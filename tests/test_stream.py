"""Offline tests for the GCS-mirror event stream (LocalBlobStore; no GCP).

Covers the write side (batching, immediate terminal flush, close-drains, best-effort on
storage failure), the read side (ordering across files, watermark dedup, ``since`` floor,
stop-at-result, transient-error tolerance), the live writer→tail round-trip, and the
wedge protection, and the client-side wiring (an output bucket is required to run).
"""

from __future__ import annotations

import asyncio
import time

import pytest

from agent_run.events import AgentEvent
from agent_run.ports.blobstore import LocalBlobStore
from agent_run.runtime.gemini import stream as stream_mod
from agent_run.runtime.gemini.stream import MirrorStream, tail_stream

URI = "gs://bkt/events"  # store= overrides the bucket; only the prefix ("events") is used


def _ev(kind: str, summary: str = "") -> AgentEvent:
    return AgentEvent(kind=kind, summary=summary)


def _tail(store, sid, since=None) -> list[AgentEvent]:
    async def go():
        return [e async for e in tail_stream(URI, sid, since=since, store=store)]

    return asyncio.run(go())


# -- write side -------------------------------------------------------------------


def test_writer_flushes_result_immediately_and_close_drains(tmp_path):
    store = LocalBlobStore(str(tmp_path))
    writer = MirrorStream(URI, "s1", store=store)
    writer.append(_ev("status", "workspace ready"))
    writer.append(_ev("result", "done"))  # terminal => immediate flush, no interval wait
    deadline = time.monotonic() + 5
    while not store.list("events/s1/") and time.monotonic() < deadline:
        time.sleep(0.01)
    assert store.list("events/s1/"), "terminal event not flushed promptly"

    writer.append(_ev("status", "post-terminal bookkeeping"))
    writer.close()
    events = _tail(store, "s1")
    # Everything before + including the result arrives; tail stops at the result, so the
    # post-terminal line (flushed by close) is durably recorded but not streamed.
    assert [e.kind for e in events] == ["status", "result"]


def test_writer_batches_are_ordered_across_files(tmp_path):
    store = LocalBlobStore(str(tmp_path))
    writer = MirrorStream(URI, "s1", store=store)
    for i in range(5):
        writer.append(_ev("message", f"m{i}"))
        time.sleep(0.6)  # force one flush interval per event => several files
    writer.append(_ev("result", "done"))
    writer.close()

    assert len(store.list("events/s1/")) >= 3  # genuinely incremental, not one blob
    events = _tail(store, "s1")
    assert [e.summary for e in events[:-1]] == [f"m{i}" for i in range(5)]
    assert events[-1].kind == "result"


def test_writer_swallows_storage_failures(tmp_path):
    class Boom(LocalBlobStore):
        def put_bytes(self, key, data):
            raise RuntimeError("bucket gone")

    writer = MirrorStream(URI, "s1", store=Boom(str(tmp_path)))
    writer.append(_ev("result", "done"))
    writer.close()  # must not raise — the mirror is telemetry, never load-bearing


# -- read side --------------------------------------------------------------------


def test_tail_since_skips_previous_turns(tmp_path):
    store = LocalBlobStore(str(tmp_path))
    cutoff = time.time() - 100
    # A previous turn's file, named well before the cutoff (old per-turn naming).
    store.put_bytes(
        f"events/s1/{int((cutoff - 50) * 1000):015d}.jsonl",
        b'{"kind": "result", "summary": "old turn"}',
    )
    writer = MirrorStream(URI, "s1", store=store)
    writer.append(_ev("result", "new turn"))
    writer.close()

    events = _tail(store, "s1", since=cutoff)
    assert [e.summary for e in events] == ["new turn"]
    # Without a floor, the old turn's terminal result would end the tail first.
    assert _tail(store, "s1")[0].summary == "old turn"


def test_tail_rides_out_transient_errors_and_raises_on_persistent(tmp_path):
    class Flaky(LocalBlobStore):
        def __init__(self, root, fail_first):
            super().__init__(root)
            self.failures_left = fail_first

        def list(self, prefix):
            if self.failures_left:
                self.failures_left -= 1
                raise RuntimeError("transient")
            return super().list(prefix)

    store = Flaky(str(tmp_path), fail_first=3)
    store.put_bytes("events/s1/000000000000001-0001.jsonl", b'{"kind": "result"}')
    assert [e.kind for e in _tail(store, "s1")] == ["result"]

    persistent = Flaky(str(tmp_path), fail_first=10_000)
    with pytest.raises(RuntimeError, match="transient"):
        _tail(persistent, "s1")


def test_live_round_trip_writer_feeding_running_tail(tmp_path):
    """The real shape: a tail already polling while the writer produces events."""
    store = LocalBlobStore(str(tmp_path))

    async def go():
        writer = MirrorStream(URI, "s1", store=store)

        def produce():
            writer.append(_ev("status", "workspace ready"))
            writer.append(_ev("tool_use", "Bash"))
            time.sleep(0.7)  # across a flush boundary
            writer.append(_ev("result", "42"))

        seen: list[AgentEvent] = []

        async def consume():
            async for e in tail_stream(URI, "s1", store=store):
                seen.append(e)

        await asyncio.gather(consume(), asyncio.to_thread(produce))
        writer.close()
        return seen

    events = asyncio.run(go())
    assert [e.kind for e in events] == ["status", "tool_use", "result"]


def test_tail_start_after_beats_the_clock_floor(tmp_path):
    """Back-to-back turns: the previous turn's files can be SECONDS old — inside the
    5s `since` slack — so the clock floor alone would replay its terminal result as the
    new turn's first event (live-caught bug). The explicit key watermark must win."""
    store = LocalBlobStore(str(tmp_path))
    watermark: dict = {"key": ""}

    # Turn 1 streams and its tail records the consumed watermark.
    w1 = MirrorStream(URI, "s1", store=store)
    w1.append(_ev("result", "turn1"))
    w1.close()

    async def turn1():
        return [e async for e in tail_stream(URI, "s1", store=store, watermark=watermark)]

    events = asyncio.run(turn1())
    assert [e.summary for e in events] == ["turn1"]
    assert watermark["key"], "tail must record the last consumed object"

    # Turn 2 submits immediately: since = now - 5 covers turn 1's just-written file.
    since = time.time() - 5
    w2 = MirrorStream(URI, "s1", store=store)
    w2.append(_ev("result", "turn2"))
    w2.close()

    async def turn2(**kw):
        return [e async for e in tail_stream(URI, "s1", since=since, store=store, **kw)]

    # Without the watermark the stale result comes back first (the bug)...
    assert asyncio.run(turn2())[0].summary == "turn1"
    # ...with it, turn 2 sees only its own events.
    events = asyncio.run(turn2(start_after=watermark["key"], watermark=watermark))
    assert [e.summary for e in events] == ["turn2"]


def test_tail_times_out_wedged_list_and_retries(tmp_path, monkeypatch):
    # A wedged connection blocks a list indefinitely; the per-call bound abandons it and
    # the NEXT poll (a fresh call) succeeds — the stream must not wait out the wedge.
    monkeypatch.setattr(stream_mod, "_TAIL_POLL_S", 0.001)
    monkeypatch.setattr(stream_mod, "_POLL_TIMEOUT_S", 0.05)

    class WedgedOnce(LocalBlobStore):
        calls = 0

        def list(self, prefix):
            WedgedOnce.calls += 1
            if WedgedOnce.calls == 1:
                time.sleep(0.5)  # wedged well past the poll bound
            return super().list(prefix)

    store = WedgedOnce(str(tmp_path))
    store.put_bytes("events/s1/000000000000001-0001.jsonl", b'{"kind": "result"}')

    async def drive():
        # Measure inside the loop: asyncio.run()'s shutdown waits for the abandoned worker
        # thread, which is fine in production where the loop is long-lived.
        t0 = time.monotonic()
        events = [e async for e in tail_stream(URI, "s1", store=store)]
        return events, time.monotonic() - t0

    events, elapsed = asyncio.run(drive())
    assert [e.kind for e in events] == ["result"]
    assert elapsed < 0.4  # did NOT wait out the 0.5s wedge


# -- client-side wiring --------------------------------------------------------------


def test_submit_requires_an_output_bucket():
    from sandbox_fakes import FakeSandboxProvider, make_engine

    from agent_run.runtime.gemini import backend

    engine = make_engine(FakeSandboxProvider(), output_bucket=None)
    with pytest.raises(ValueError, match="output bucket"):
        backend.GeminiSession(engine, "sid").run("go")


def test_tail_poll_cadence_is_subsecond():
    # The stream replaces a ~1s log tail; keep it at least as responsive.
    assert stream_mod._TAIL_POLL_S <= 1.0
    assert stream_mod._FLUSH_INTERVAL_S <= 1.0

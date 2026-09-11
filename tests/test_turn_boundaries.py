"""The turn's live event source: the worker's /events first, the mirror as the fallback."""

from __future__ import annotations

import asyncio

import pytest
from sandbox_fakes import FakeSandboxProvider, ScriptedWorker

from remote_agent_toolkit.events import AgentEvent
from remote_agent_toolkit.ports.blobstore import LocalBlobStore
from remote_agent_toolkit.runtime.gemini import backend, history, stream
from remote_agent_toolkit.runtime.gemini.provider import SandboxGone

SID = "sid-1"
URI = "gs://out/events"
TURN = "t" * 32


def event(kind, tag, turn_id=TURN):
    return AgentEvent(kind=kind, summary=tag,
                      raw={"event": tag, "turn_id": turn_id, "session_id": SID,
                           "subtype": "success", "is_error": False, "num_turns": 1})


def result_event(text="done", turn_id=TURN):
    return event("result", text, turn_id)


def write(store, events, turn_id=TURN, now_ms=1000):
    history.write_turn_mirror(URI, SID, [history.mirror_line(e) for e in events],
                              now_ms=now_ms, turn_id=turn_id, store=store)


@pytest.fixture
def storage(tmp_path, monkeypatch):
    store = LocalBlobStore(str(tmp_path / "blobs"))
    monkeypatch.setattr(stream, "_TAIL_POLL_S", 0.005)
    monkeypatch.setattr(backend, "GONE_GRACE_S", 0.1)
    monkeypatch.setattr(backend, "DIRECT_RETRY_SLEEP_S", 0.0)
    return store


def _provider_with(worker):
    provider = FakeSandboxProvider(lambda n: worker)
    template = provider.add_template("g")
    handle = provider.create(template, ttl_s=60, display_name="ratk-g-x")
    return provider, handle.name


async def _collect(provider, sandbox, store, **kw):
    return [e async for e in backend._stream_turn(
        provider, sandbox, TURN, SID, events_uri=URI, store_kwargs={"store": store}, **kw)]


def test_direct_stream_delivers_in_order_and_stops_at_the_result(storage):
    worker = ScriptedWorker()
    provider, sandbox = _provider_with(worker)
    worker.turns.append({"turn_id": TURN})
    worker.emit(event("status", "turn_started"), event("message", "hi"), result_event("done"))
    worker.emit(event("status", "checkpoint_saved"))  # after the result: never delivered
    events = asyncio.run(_collect(provider, sandbox, storage))
    assert [e.kind for e in events] == ["status", "message", "result"]
    assert all(c[1] == "/events" for c in provider.calls)
    assert provider.calls[0][2]["since"] == 0 and provider.calls[0][2]["wait"] == backend.EVENTS_WAIT_S


def test_direct_stream_ignores_events_of_another_turn(storage):
    worker = ScriptedWorker()
    provider, sandbox = _provider_with(worker)
    worker.turns.append({"turn_id": TURN})
    worker.emit(event("result", "stale", turn_id="other"), event("message", "mine"), result_event("done"))
    events = asyncio.run(_collect(provider, sandbox, storage))
    assert [e.summary for e in events] == ["mine", "done"]


def test_worker_ending_without_a_result_yields_an_explained_error(storage):
    worker = ScriptedWorker()
    provider, sandbox = _provider_with(worker)
    worker.turns.append({"turn_id": TURN})
    worker.emit(event("message", "half"))
    worker.finish(error="Traceback: MemoryError")
    events = asyncio.run(_collect(provider, sandbox, storage))
    assert [e.kind for e in events] == ["message", "result"]
    assert events[-1].raw["is_error"] and "MemoryError" in events[-1].summary
    assert events[-1].raw["event"] == "worker_ended_without_result"


def test_gone_sandbox_falls_back_to_the_mirror_and_skips_delivered_events(storage):
    worker = ScriptedWorker()
    provider, sandbox = _provider_with(worker)
    worker.turns.append({"turn_id": TURN})
    worker.emit(event("status", "turn_started"), event("message", "one"))
    # The worker flushed everything to the mirror, then the sandbox died (OOM) before the
    # client's next poll: the mirror has the whole turn, the client already saw two events.
    write(storage, [event("status", "turn_started"), event("message", "one"),
                    event("message", "two"), result_event("done")])

    async def go():
        gen = backend._stream_turn(provider, sandbox, TURN, SID, events_uri=URI, store_kwargs={"store": storage})
        out = [await gen.__anext__(), await gen.__anext__()]
        provider.vanish(sandbox)
        out += [e async for e in gen]
        return out

    events = asyncio.run(go())
    assert [e.summary for e in events] == ["turn_started", "one", "two", "done"]
    assert events[-1].kind == "result" and not events[-1].raw["is_error"]


def test_gone_sandbox_without_a_terminal_record_ends_the_run_after_the_grace(storage):
    worker = ScriptedWorker()
    provider, sandbox = _provider_with(worker)
    worker.turns.append({"turn_id": TURN})
    write(storage, [event("status", "turn_started")])  # partial mirror, no result
    provider.vanish(sandbox)
    events = asyncio.run(_collect(provider, sandbox, storage))
    assert [e.kind for e in events] == ["status", "result"]
    assert events[-1].raw["event"] == "sandbox_unreachable" and "gone" in events[-1].summary


def test_repeated_proxy_failures_fall_back_to_the_mirror(storage, monkeypatch):
    worker = ScriptedWorker()
    provider, sandbox = _provider_with(worker)
    worker.turns.append({"turn_id": TURN})
    provider.fail_next["/events"] = [RuntimeError("502")] * backend.DIRECT_MAX_FAILURES
    write(storage, [event("message", "from-mirror"), result_event("done")])
    events = asyncio.run(_collect(provider, sandbox, storage))
    assert [e.summary for e in events] == ["from-mirror", "done"]


def test_one_proxy_blip_is_retried_directly(storage, monkeypatch):
    worker = ScriptedWorker()
    provider, sandbox = _provider_with(worker)
    worker.turns.append({"turn_id": TURN})
    provider.fail_next["/events"] = [RuntimeError("502")]
    worker.emit(result_event("done"))
    events = asyncio.run(_collect(provider, sandbox, storage))
    assert [e.summary for e in events] == ["done"]
    assert storage.list("") == []  # never touched the mirror


def test_mirror_fallback_without_a_result_ends_with_an_explained_error(storage, monkeypatch):
    worker = ScriptedWorker()
    provider, sandbox = _provider_with(worker)
    worker.turns.append({"turn_id": TURN})
    provider.fail_next["/events"] = [RuntimeError("502")] * backend.DIRECT_MAX_FAILURES
    events = asyncio.run(_collect(provider, sandbox, storage, mirror_max_wait_s=0.05))
    assert len(events) == 1 and events[0].raw["event"] == "sandbox_unreachable"
    assert "stopped answering" in events[0].summary


def test_tail_stream_selects_the_turn_by_identity(storage):
    write(storage, [event("result", "old", turn_id="previous")], turn_id="previous", now_ms=900)
    write(storage, [event("message", "now"), result_event("done")])
    # A key that names the turn but whose payload belongs to another turn is filtered too.
    history.write_turn_mirror(URI, SID, [history.mirror_line(event("result", "spoof", turn_id="x"))],
                              now_ms=1001, turn_id=TURN, store=storage)

    async def go():
        return [e async for e in stream.tail_stream(URI, SID, turn_id=TURN, store=storage, max_wait_s=1.0)]

    events = asyncio.run(go())
    assert [e.summary for e in events] == ["now", "done"]


def test_read_history_scoped_to_a_turn(storage):
    write(storage, [event("result", "old", turn_id="previous")], turn_id="previous", now_ms=900)
    write(storage, [event("message", "now"), result_event("done")])
    assert [e.summary for e in history.read_history("gs://out", SID, store=storage, turn_id=TURN)] == ["now", "done"]
    assert len(history.read_history("gs://out", SID, store=storage)) == 3


def test_stream_turn_handles_a_gone_sandbox_raised_by_the_provider_type(storage):
    class Gone:
        def call(self, *a, **kw):
            raise SandboxGone("expired")

    write(storage, [result_event("done")])

    async def go():
        return [e async for e in backend._stream_turn(Gone(), "s", TURN, SID, events_uri=URI,
                                                      store_kwargs={"store": storage})]

    assert [e.summary for e in asyncio.run(go())] == ["done"]

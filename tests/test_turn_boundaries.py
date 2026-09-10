"""Turn identity survives dispatch, pickup, live tailing and history recovery."""

import asyncio
import json
from types import SimpleNamespace as NS

import pytest
from test_warm_pool import _per_worker_engine

from remote_agent_toolkit import AgentSpec
from remote_agent_toolkit.events import AgentEvent
from remote_agent_toolkit.ports import eventsink
from remote_agent_toolkit.ports.blobstore import LocalBlobStore
from remote_agent_toolkit.runtime.gemini import adk_agent, backend, handoff, history, stream

SID = "warm-sid"
URI = "gs://out/events"


def event(kind, tag, turn_id="current", worker="worker"):
    return AgentEvent(
        kind=kind, summary=tag,
        raw={"event": tag, "turn_id": turn_id, "worker": worker, "session_id": SID,
             "subtype": "success", "is_error": False, "num_turns": 1},
    )


def write(store, events, turn_id="current", now_ms=1000):
    history.write_turn_mirror(URI, SID, [history.mirror_line(e) for e in events],
                              now_ms=now_ms, turn_id=turn_id, store=store)


@pytest.fixture
def storage(tmp_path, monkeypatch):
    store = LocalBlobStore(str(tmp_path / "blobs"))
    monkeypatch.setattr(stream, "GcsBlobStore", lambda *a, **kw: store)
    monkeypatch.setattr(history, "GcsBlobStore", lambda *a, **kw: store)
    monkeypatch.setattr(handoff, "load_session_config", lambda *a, **kw: None)
    monkeypatch.setattr(eventsink, "CloudLoggingSink", lambda **kw: NS(emit=lambda e: None,
                                                                        read=lambda sid: []))
    monkeypatch.setattr(stream, "_TAIL_POLL_S", 0.005)
    return store


async def test_pickup_ignores_checkpoint_and_wrong_worker_until_matching_start():
    picked_up = []

    async def source():
        yield event("status", "checkpoint_saved", "previous")
        yield event("status", "turn_started", worker="abandoned")
        await asyncio.sleep(0.02)
        assert not picked_up, "an unrelated event retired the addressed subscription"
        yield event("status", "turn_started")
        yield event("result", "answer")

    events = [e async for e in backend._pickup_watched_tail(
        source, 1, lambda attempt: 1, SID, on_pickup=lambda: picked_up.append(True),
        is_pickup=lambda e: e.raw.get("event") == "turn_started" and e.raw["worker"] == "worker",
    )]
    assert picked_up == [True]
    assert [e.summary for e in events] == ["turn_started", "answer"]


async def test_unrelated_traffic_cannot_extend_pickup_deadline():
    async def source():
        while True:
            yield event("status", "checkpoint_saved", "previous")
            await asyncio.sleep(0.003)

    async with asyncio.timeout(0.5):
        events = [e async for e in backend._pickup_watched_tail(
            source, 0.03, lambda attempt: 1, SID, max_redispatch=0,
        )]
    assert len(events) == 1 and events[0].raw["event"] == "pool_pickup_timeout"


async def test_tail_rejects_old_results_and_verifies_payload_identity(storage):
    write(storage, [event("result", "old answer", "old")], turn_id="old")
    # Even a wrong event in a current-turn object must not terminate the tail.
    write(storage, [event("result", "wrong payload", "old"),
                    event("result", "retired worker", worker="abandoned"),
                    event("result", "new answer")], now_ms=1001)
    events = [e async for e in stream.tail_stream(
        URI, SID, store=storage, turn_id="current",
        accept_event=lambda e: e.raw["worker"] == "worker",
    )]
    assert [e.summary for e in events] == ["new answer"]


async def test_tail_accepts_later_flush_with_an_earlier_clock(storage):
    write(storage, [event("status", "turn_started")], now_ms=2000)
    events = []
    async for e in stream.tail_stream(URI, SID, store=storage, turn_id="current"):
        events.append(e)
        if len(events) == 1:
            write(storage, [event("result", "answer")], now_ms=1000)
    assert [e.summary for e in events] == ["turn_started", "answer"]


@pytest.mark.parametrize("fallback", ["old", "current", "abandoned"])
def test_recovery_filters_mirror_job_and_logging(storage, monkeypatch, fallback):
    write(storage, [event("result", "old mirror", "old")], turn_id="old")
    write(storage, [event("status", "turn_started")])
    candidate = event("result", "fallback", "old" if fallback == "old" else "current",
                      "abandoned" if fallback == "abandoned" else "worker")
    storage.put_bytes(f"jobs/{SID}.jsonl", json.dumps({
        "custom_metadata": {"kind": candidate.kind, "raw": candidate.raw},
        "content": {"parts": [{"text": candidate.summary}]},
    }).encode())
    monkeypatch.setattr(eventsink, "CloudLoggingSink", lambda **kw: NS(read=lambda sid: [candidate]))
    recovered = history.read_history("gs://out", SID, store=storage,
                                     turn_id="current", worker="worker")
    assert [e.summary for e in recovered] == (["fallback"] if fallback == "current" else ["turn_started"])
    assert any(e.summary == "old mirror" for e in history.read_history("gs://out", SID, store=storage))


async def test_reattached_back_to_back_sends_have_distinct_results(storage, monkeypatch):
    engine, dispatch, _ = _per_worker_engine(monkeypatch)
    publish = dispatch.publish
    previous = []

    def complete(payload, attributes=None):
        publish(payload, attributes)
        if previous:
            write(storage, [event("status", "checkpoint_saved", previous[-1])],
                  turn_id=previous[-1])
        tid, wid = payload["turn_id"], attributes["worker"]
        previous.append(tid)
        write(storage, [event("status", "turn_started", tid, wid),
                        event("result", payload["message"], tid, wid)], turn_id=tid)

    monkeypatch.setattr(dispatch, "publish", complete)
    results = []
    async with asyncio.timeout(2):
        for prompt in ["FIRST", "SECOND", "THIRD"]:
            session = backend.GeminiSession(engine, SID)
            results.append(await session.send(prompt))
    engine._join_background()
    assert [r.text for r in results] == ["FIRST", "SECOND", "THIRD"]
    assert len(set(previous)) == 3


async def test_redispatch_rejects_abandoned_workers_result(storage, monkeypatch):
    engine, dispatch, ae = _per_worker_engine(monkeypatch)
    publish = dispatch.publish

    def complete(payload, attributes=None):
        publish(payload, attributes)
        if len(dispatch.published) < 2:
            return
        tid = payload["turn_id"]
        old = dispatch.published[0][1]["worker"]
        current = attributes["worker"]
        write(storage, [event("status", "turn_started", tid, old),
                        event("result", "ABANDONED", tid, old),
                        event("status", "turn_started", tid, current),
                        event("result", "CURRENT", tid, current)], turn_id=tid)

    monkeypatch.setattr(dispatch, "publish", complete)
    monkeypatch.setattr(backend, "pickup_deadline_s", lambda *a, **kw: 0.03)
    async with asyncio.timeout(2):
        result = await backend.GeminiSession(engine, SID).run("go")
    engine._join_background()
    assert result.text == "CURRENT" and not result.is_error
    assert len(dispatch.published) == 2 and len(ae.cancelled) == 1


async def test_dead_current_job_cannot_recover_previous_success(storage, monkeypatch):
    engine, dispatch, ae = _per_worker_engine(monkeypatch)
    write(storage, [event("result", "OLD SUCCESS", "old")], turn_id="old")
    publish = dispatch.publish

    def start_only(payload, attributes=None):
        publish(payload, attributes)
        tid = payload["turn_id"]
        write(storage, [event("status", "turn_started", tid, attributes["worker"])], turn_id=tid)

    monkeypatch.setattr(dispatch, "publish", start_only)
    monkeypatch.setattr(ae, "check_query_job", lambda **kw: NS(status="SUCCESS"))
    watched = backend._watched_tail
    monkeypatch.setattr(backend, "_watched_tail", lambda *a, **kw: watched(*a, **kw, quiet_s=0.01, grace_s=0.02))
    async with asyncio.timeout(2):
        result = await backend.GeminiSession(engine, SID).run("go")
    engine._join_background()
    assert result.is_error and "no terminal" in result.text


@pytest.mark.parametrize("config_error", [False, True])
async def test_worker_identifies_every_event_including_early_errors(storage, monkeypatch, tmp_path, config_error):
    spec = AgentSpec(name="worker", model="m")
    agent = adk_agent.build_agent(spec)
    monkeypatch.setenv("AGENT_EVENTS_GCS", URI)
    monkeypatch.setenv("AGENT_JOBS_ROOT", str(tmp_path / "jobs"))
    monkeypatch.setattr(adk_agent, "_checkpoint_ports", lambda spec: (None, None))
    monkeypatch.setattr(adk_agent, "_prepare_workspace", lambda *a: {"summary": "ready"})
    monkeypatch.setattr(handoff, "fetch_config", lambda *a: None)

    class Harness:
        async def run(self, spec, ctx):
            yield AgentEvent(kind="result", summary="done", raw={"subtype": "success"})

    import remote_agent_toolkit.harness as harness
    monkeypatch.setattr(harness, "resolve_harness", lambda spec: Harness())
    events = [e async for e in agent._run_turn(
        spec, SID, "go", None, worker="worker", turn_id="current",
        session_config_uri="gs://out/missing" if config_error else None,
    )]
    assert events
    for e in events:
        assert e.custom_metadata["raw"]["turn_id"] == "current"
        assert e.custom_metadata["raw"]["worker"] == "worker"
    durable = history.read_history("gs://out", SID, store=storage, turn_id="current")
    assert durable[-1].kind == "result"
    assert (durable[-1].raw.get("event") == "config_error") is config_error


def test_incompatible_engine_fails_before_dispatch_or_secret_staging(monkeypatch):
    engine, dispatch, ae = _per_worker_engine(monkeypatch)
    engine._turn_event_ids = False
    session = backend.GeminiSession(engine, SID)
    monkeypatch.setattr(session, "_stage_secrets", lambda *a: pytest.fail("secrets staged"))
    with pytest.raises(RuntimeError, match="redeploy"):
        session.run("go")
    assert not dispatch.published and not ae.jobs


@pytest.mark.parametrize("targets,supported", [([], True), (["old"], False),
                                             (["new"], True), (["old", "new"], False)])
def test_protocol_discovery_checks_serving_revisions(targets, supported):
    resource = "projects/p/locations/l/reasoningEngines/1"

    def api(version, enabled):
        env = [NS(name="AGENT_EVENT_TURN_IDS", value="1")] if enabled else []
        return NS(name=f"{resource}/runtimeRevisions/{version}",
                  spec=NS(deployment_spec=NS(env=env)))

    latest = api("new", True)
    latest.traffic_config = {"traffic_split_manual": {"targets": [
        {"runtime_revision_name": f"{resource}/runtimeRevisions/{version}",
         "percent": 100 // len(targets)} for version in targets
    ]}} if targets else None
    client = NS(agent_engines=NS(
        get=lambda **kw: NS(api_resource=latest),
        runtimes=NS(revisions=NS(list=lambda **kw: [NS(api_resource=api("old", False)),
                                                   NS(api_resource=api("new", True))])),
    ))
    assert backend._supports_turn_event_ids(client, resource) is supported


async def test_cold_directive_reaches_worker_without_becoming_prompt(monkeypatch):
    agent = adk_agent.build_agent(AgentSpec(name="cold", model="m"))
    seen = {}
    tid = "a" * 32

    async def run_turn(spec, sid, prompt, resume_sid, *args, **kwargs):
        seen.update(sid=sid, prompt=prompt, resume_sid=resume_sid, **kwargs)
        yield "ran"

    monkeypatch.setattr(agent, "_run_turn", run_turn)
    ctx = NS(invocation_id="inv", session=None, user_content=NS(parts=[NS(text=(
        f"AGENT_SESSION={SID}\nAGENT_RESUME={SID}\nAGENT_GCS_TOKEN=fake\n"
        f"AGENT_TURN_ID={tid}\nanswer this"
    ))]))
    assert [e async for e in agent._run_async_impl(ctx)] == ["ran"]
    assert seen["turn_id"] == tid and seen["prompt"] == "answer this"
    assert seen["sid"] == SID and seen["resume_sid"] == SID


async def test_early_correlated_error_ends_warm_turn_without_redispatch(storage, monkeypatch):
    engine, dispatch, ae = _per_worker_engine(monkeypatch)
    publish = dispatch.publish

    def fail(payload, attributes=None):
        publish(payload, attributes)
        tid = payload["turn_id"]
        error = event("result", "config_error", tid, attributes["worker"])
        error.raw.update(subtype="error", is_error=True)
        write(storage, [error], turn_id=tid)

    monkeypatch.setattr(dispatch, "publish", fail)
    async with asyncio.timeout(2):
        result = await backend.GeminiSession(engine, SID).run("go")
    engine._join_background()
    assert result.is_error and result.text == "config_error"
    assert len(dispatch.published) == 1 and not ae.cancelled

"""Session history: mirror round-trip, ADK job-output parsing, layered reads, enumeration.

Everything runs offline against LocalBlobStore; the live GCS path is validated separately.
"""

from __future__ import annotations

import asyncio
import json

from remote_agent_toolkit import AgentSpec
from remote_agent_toolkit.events import AgentEvent, RunStatus, StopReason
from remote_agent_toolkit.ports.blobstore import LocalBlobStore
from remote_agent_toolkit.runtime.gemini import adk_agent, backend, history


def _result_event(text="done"):
    return AgentEvent(kind="result", summary=text, cost_usd=0.1,
                      raw={"subtype": "success", "is_error": False, "num_turns": 2})


def test_mirror_round_trip():
    ev = _result_event()
    line = history.mirror_line(ev)
    back = history.event_from_mirror(json.loads(json.dumps(line)))  # via real JSON
    assert back == ev


def test_adk_line_parsing_real_shape():
    # The exact shape the platform writes to jobs/<sid>.jsonl (from a live job output).
    line = {
        "content": {"parts": [{"text": "workspace ready (restored=False, ...)"}], "role": "model"},
        "partial": True,
        "custom_metadata": {"kind": "status", "raw": {"event": "workspace_ready", "repos": ["r"]}},
        "author": "agent", "id": "x", "timestamp": 1784053119.1,
    }
    ev = history.event_from_adk_line(line)
    assert ev.kind == "status" and ev.summary.startswith("workspace ready")
    assert ev.raw == {"event": "workspace_ready", "repos": ["r"]}
    # Foreign ADK events (no toolkit metadata) are skipped.
    assert history.event_from_adk_line({"content": {"parts": [{"text": "hi"}]}}) is None


def test_read_history_prefers_mirror_and_keeps_turn_order(tmp_path):
    store = LocalBlobStore(str(tmp_path))
    sid = "warm-sid"
    # Two mirrored turns (older + newer) — and a platform jobs file that must NOT win.
    history.write_turn_mirror("gs://bkt/events", sid,
                              [history.mirror_line(AgentEvent(kind="message", summary="turn1"))],
                              now_ms=1000, store=store)
    history.write_turn_mirror("gs://bkt/events", sid,
                              [history.mirror_line(_result_event("turn2-done"))],
                              now_ms=2000, store=store)
    store.put_bytes(f"jobs/{sid}.jsonl", b'{"custom_metadata": {"kind": "status"}, "content": null}')

    events = history.read_history("gs://bkt", sid, store=store)
    assert [e.summary for e in events] == ["turn1", "turn2-done"]  # all turns, in order


def test_read_history_falls_back_to_platform_job_output(tmp_path):
    store = LocalBlobStore(str(tmp_path))
    line = {"content": {"parts": [{"text": "done"}]},
            "custom_metadata": {"kind": "result", "raw": {"is_error": False}, "cost_usd": 0.2}}
    store.put_bytes("jobs/cold-sid.jsonl", json.dumps(line).encode())
    events = history.read_history("gs://bkt", "cold-sid", store=store)
    assert len(events) == 1 and events[0].kind == "result" and events[0].cost_usd == 0.2


def test_list_sessions_merges_sources(tmp_path):
    store = LocalBlobStore(str(tmp_path))
    store.put_bytes("events/warm-1/000000000001000.jsonl", b"{}")
    store.put_bytes("jobs/cold-2.jsonl", b"{}")
    store.put_bytes("jobs/cold-2_input.jsonl", b"{}")  # inputs are not sessions

    class FakeAdkSessions:
        def list(self, name):
            assert name == "projects/p/locations/l/reasoningEngines/1"
            return [type("S", (), {"name": "projects/p/.../sessions/cold-2"})(),
                    type("S", (), {"name": "projects/p/.../sessions/adk-only-3"})()]

    got = history.list_sessions(
        output_bucket="gs://bkt",
        resource="projects/p/locations/l/reasoningEngines/1",
        adk_sessions=FakeAdkSessions(),
        store=store,
    )
    by_id = {s["session_id"]: s for s in got}
    assert set(by_id) == {"warm-1", "cold-2", "adk-only-3"}
    assert by_id["cold-2"]["sources"] == ["jobs", "adk"]
    assert by_id["warm-1"]["sources"] == ["events"]


def test_run_turn_writes_turn_mirror(tmp_path, monkeypatch):
    # The worker streams every surfaced event into session-keyed mirror files AS THE TURN
    # RUNS (stream.MirrorStream) — the durable record Session.history() reads (warm turns
    # have no session-keyed job output) and the live channel the client tails.
    from remote_agent_toolkit.runtime.gemini import stream as stream_mod

    monkeypatch.setenv("AGENT_JOBS_ROOT", str(tmp_path / "jobs"))
    monkeypatch.setenv("AGENT_EVENTS_GCS", "gs://bkt/events")
    monkeypatch.chdir(tmp_path)
    store = LocalBlobStore(str(tmp_path / "blobs"))
    monkeypatch.setattr(stream_mod, "GcsBlobStore", lambda bucket: store)

    class OneEventHarness:
        async def run(self, spec, rc):
            yield _result_event("mirrored")

    import remote_agent_toolkit.harness.claude_code as harness_mod
    import remote_agent_toolkit.ports.eventsink as eventsink_mod
    from remote_agent_toolkit.ports.eventsink import InMemorySink
    monkeypatch.setattr(harness_mod, "ClaudeCodeHarness", OneEventHarness)
    monkeypatch.setattr(eventsink_mod, "CloudLoggingSink", lambda **kw: InMemorySink(session_id="77"))

    spec = AgentSpec(name="w", model="m")
    agent = adk_agent.build_agent(spec)

    async def drive():
        return [ev async for ev in agent._run_turn(spec, "77", "go", None)]

    asyncio.run(drive())

    assert store.list("events/77/")  # streamed incrementally; file count is timing-dependent
    events = history.read_history("gs://bkt", "77", store=store)
    # turn_started + workspace_ready + the result
    assert [e.kind for e in events] == ["status", "status", "result"]
    assert events[-1].summary == "mirrored"


def test_reattached_session_reconstructs_last_result(monkeypatch):
    spec = AgentSpec(name="g", model="m")
    engine = backend.GeminiEngine(resource="r/reasoningEngines/1", spec=spec,
                                  project="p", location="l", output_bucket="gs://out")
    session = backend.GeminiSession(engine, "old-sid")  # re-attached: no run in this process
    monkeypatch.setattr(
        backend.GeminiSession, "history",
        lambda self: [AgentEvent(kind="message", summary="hi"), _result_event("final answer")],
    )
    result = session.last_result
    assert result is not None and result.text == "final answer" and result.num_turns == 2
    assert session.stop_reason == StopReason.END_TURN and session.status == RunStatus.IDLE


def test_reattached_session_without_history_has_no_result(monkeypatch):
    spec = AgentSpec(name="g", model="m")
    engine = backend.GeminiEngine(resource="r/reasoningEngines/1", spec=spec,
                                  project="p", location="l", output_bucket="gs://out")
    session = backend.GeminiSession(engine, "old-sid")
    monkeypatch.setattr(backend.GeminiSession, "history", lambda self: [])
    assert session.last_result is None and session.status == RunStatus.PENDING

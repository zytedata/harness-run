"""Gemini run-plane offline tests: AgentEvent->ADK translation + Session/Run wiring.

The GCP-touching control plane (deploy / run_query_job / sessions.create) is validated
live; here we test the pure-Python glue with fakes — the AgentEvent->ADK Event mapping and
the GeminiSession -> DrivenRun -> EventSink.tail mechanism that drives a run.
"""

from __future__ import annotations

import asyncio

from remote_agent_toolkit import AgentSpec
from remote_agent_toolkit.events import AgentEvent, RunStatus, StopReason
from remote_agent_toolkit.ports.eventsink import InMemorySink
from remote_agent_toolkit.runtime.gemini import adk_agent, backend
from remote_agent_toolkit.runtime.gemini.translate import to_adk_event


async def _await(run):
    return await run


def test_to_adk_event_kinds():
    msg = to_adk_event(AgentEvent(kind="message", summary="hi"), "agent")
    assert msg.partial is True and (msg.turn_complete in (None, False))
    assert msg.custom_metadata["kind"] == "message"

    think = to_adk_event(AgentEvent(kind="thinking", summary="hmm"), "agent")
    assert think.content.parts[0].thought is True

    result = to_adk_event(
        AgentEvent(kind="result", summary="done", cost_usd=0.3, usage={"t": 1},
                   raw={"is_error": False, "num_turns": 2}),
        "agent",
    )
    assert result.partial is False and result.turn_complete is True
    assert result.custom_metadata["cost_usd"] == 0.3 and result.error_message is None

    err = to_adk_event(AgentEvent(kind="result", summary="boom", raw={"is_error": True}), "agent")
    assert err.error_message == "boom"


def test_build_agent_sanitizes_node_name():
    # ADK BaseAgent.name must be a valid Python identifier; spec.name may have hyphens.
    agent = adk_agent.build_agent(AgentSpec(name="hello-coder", model="m"))
    assert agent.name == "hello_coder"
    assert agent.spec_data["name"] == "hello-coder"  # the real name is preserved in the spec
    assert adk_agent._safe_node_name("9to5") == "a_9to5"


def test_split_resume_directive_and_parse_gcs():
    sid, rest = adk_agent._split_resume_directive("AGENT_RESUME=abc123\nplease continue")
    assert sid == "abc123" and rest == "please continue"
    assert adk_agent._split_resume_directive("just a prompt") == (None, "just a prompt")
    assert adk_agent._parse_gcs_uri("gs://bkt/some/prefix") == ("bkt", "some/prefix")


def test_find_baked_skills(tmp_path, monkeypatch):
    # No baked skills dir on the candidate paths -> None.
    monkeypatch.chdir(tmp_path)
    assert adk_agent._find_baked_skills() is None
    # A skills/ dir beside cwd is discovered.
    (tmp_path / "skills").mkdir()
    assert adk_agent._find_baked_skills() == tmp_path / "skills"


def _seed_sink(session_id, events):
    sink = InMemorySink(session_id=session_id)
    for ev in events:
        sink.emit(ev)
    return sink


def test_gemini_session_run_drives_from_sink(monkeypatch):
    spec = AgentSpec(name="g", model="m")
    engine = backend.GeminiEngine(
        resource="projects/p/locations/l/reasoningEngines/123",
        spec=spec, project=None, location=None, output_bucket="gs://out",
    )

    captured = {}

    class FakeAE:
        def run_query_job(self, name, config):
            captured["name"] = name
            captured["config"] = config
            return {"job": "fake"}

    monkeypatch.setattr(engine, "_agent_engines", lambda: FakeAE())

    seed = _seed_sink("sid-1", [
        AgentEvent(kind="message", summary="working"),
        AgentEvent(kind="result", summary="done", cost_usd=0.2, usage={"t": 1},
                   raw={"subtype": "success", "is_error": False, "num_turns": 4, "session_id": "sid-1"}),
    ])
    # _submit constructs CloudLoggingSink(...) for tailing — return our pre-seeded sink instead.
    import remote_agent_toolkit.ports.eventsink as eventsink_mod
    monkeypatch.setattr(eventsink_mod, "CloudLoggingSink", lambda **kw: seed)

    session = backend.GeminiSession(engine, "sid-1")
    result = asyncio.run(_await(session.run("go")))

    assert result.text == "done" and result.num_turns == 4 and result.cost_usd == 0.2
    assert result.is_error is False
    assert session.status == RunStatus.IDLE and session.stop_reason == StopReason.END_TURN
    # the run was actually submitted, to the right engine, with the message in the payload
    assert captured["name"] == engine.resource
    assert "go" in captured["config"]["query"]
    assert captured["config"]["output_gcs_uri"] == "gs://out/jobs/sid-1.jsonl"


def test_gemini_interrupt_cancels_remote_job(monkeypatch):
    import types

    spec = AgentSpec(name="g", model="m")
    engine = backend.GeminiEngine(resource="r/reasoningEngines/1", spec=spec, project=None, location=None)
    cancelled = {}

    class FakeAE:
        def cancel_query_job(self, name, config):
            cancelled["name"] = name
            cancelled["op"] = config["operation_name"]

    monkeypatch.setattr(engine, "_agent_engines", lambda: FakeAE())
    session = backend.GeminiSession(engine, "sid")
    session._last_job = types.SimpleNamespace(job_name="jobX")  # as run_query_job would set it
    asyncio.run(session.interrupt())
    assert cancelled == {"name": engine.resource, "op": "jobX"}


def test_gemini_session_send_prefixes_resume(monkeypatch):
    spec = AgentSpec(name="g", model="m", checkpoint=True)
    engine = backend.GeminiEngine(resource="r/reasoningEngines/9", spec=spec,
                                  project=None, location=None, output_bucket=None)
    captured = {}

    class FakeAE:
        def run_query_job(self, name, config):
            captured["config"] = config
            return {}

    monkeypatch.setattr(engine, "_agent_engines", lambda: FakeAE())
    seed = _seed_sink("sid-2", [
        AgentEvent(kind="result", summary="resumed", raw={"subtype": "success", "is_error": False,
                                                          "num_turns": 1, "session_id": "sid-2"}),
    ])
    import remote_agent_toolkit.ports.eventsink as eventsink_mod
    monkeypatch.setattr(eventsink_mod, "CloudLoggingSink", lambda **kw: seed)

    session = backend.GeminiSession(engine, "sid-2")
    asyncio.run(_await(session.send("answer")))
    assert "AGENT_RESUME=sid-2" in captured["config"]["query"]
    # checkpoint spec => a clean turn end is NEEDS_INPUT (awaiting the next operator message)
    assert session.stop_reason == StopReason.NEEDS_INPUT

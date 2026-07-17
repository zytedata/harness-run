"""TurnTracer: AgentEvent stream → OTel spans (the engine Traces tab's content).

Uses a real in-process TracerProvider with an in-memory exporter — what's locked in is
the span structure (turn root + paired tool children), attributes, and the never-raise
contract. The OTLP export pipe itself is the AdkApp's (not ours) and is validated live.
"""

from __future__ import annotations

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from remote_agent_toolkit.events import AgentEvent
from remote_agent_toolkit.runtime.gemini.tracing import TurnTracer

# The global tracer provider can only be set once per process; share one + clear between
# tests via the exporter.
_EXPORTER = InMemorySpanExporter()
_PROVIDER = TracerProvider()
_PROVIDER.add_span_processor(SimpleSpanProcessor(_EXPORTER))
trace.set_tracer_provider(_PROVIDER)


@pytest.fixture(autouse=True)
def _clear_spans():
    _EXPORTER.clear()
    yield


def _spans():
    return {s.name: s for s in _EXPORTER.get_finished_spans()}


def test_turn_span_wraps_paired_tool_spans():
    t = TurnTracer(agent_name="crawler", model="claude-x", session_id="sid-1")
    t.observe(AgentEvent(kind="tool_use", summary="Bash ls",
                         raw={"tool": "Bash", "id": "tu_1", "input": {"command": "ls"}}))
    t.observe(AgentEvent(kind="tool_result", summary="Bash",
                         raw={"tool": "Bash", "id": "tu_1", "is_error": False}))
    t.observe(AgentEvent(kind="result", summary="done", cost_usd=0.01,
                         usage={"input_tokens": 10, "output_tokens": 5},
                         raw={"is_error": False, "num_turns": 2}))
    t.close()

    spans = _spans()
    turn = spans["invoke_agent crawler"]
    tool = spans["execute_tool Bash"]
    assert tool.parent.span_id == turn.context.span_id  # tool nests under the turn
    assert tool.attributes["gen_ai.tool.name"] == "Bash"
    assert tool.attributes["gen_ai.tool.call.id"] == "tu_1"
    assert turn.attributes["gen_ai.conversation.id"] == "sid-1"
    assert turn.attributes["gen_ai.request.model"] == "claude-x"
    assert turn.attributes["gen_ai.usage.input_tokens"] == 10
    assert turn.attributes["rat.cost_usd"] == 0.01
    assert turn.attributes["rat.num_turns"] == 2
    # tool span's real duration: closed at tool_result time, before the turn ended
    assert tool.end_time <= turn.end_time


def test_error_result_and_error_tool_set_error_status():
    t = TurnTracer(agent_name="a", model="m", session_id="s")
    t.observe(AgentEvent(kind="tool_use", summary="Bash boom",
                         raw={"tool": "Bash", "id": "tu_e"}))
    t.observe(AgentEvent(kind="tool_result", summary="Bash (error)",
                         raw={"tool": "Bash", "id": "tu_e", "is_error": True}))
    t.observe(AgentEvent(kind="result", summary="run errored", raw={"is_error": True}))
    t.close()

    spans = _spans()
    assert spans["execute_tool Bash"].status.status_code == trace.StatusCode.ERROR
    assert spans["invoke_agent a"].status.status_code == trace.StatusCode.ERROR


def test_orphaned_tool_span_closed_on_close():
    # A crash between tool_use and tool_result must not leak an unended span.
    t = TurnTracer(agent_name="a", model="m", session_id="s")
    t.observe(AgentEvent(kind="tool_use", summary="Bash hang",
                         raw={"tool": "Bash", "id": "tu_orphan"}))
    t.close()

    tool = _spans()["execute_tool Bash"]
    assert tool.status.status_code == trace.StatusCode.ERROR  # marked, not silently ok
    assert tool.end_time is not None


def test_messages_become_span_events_and_long_text_is_clipped():
    t = TurnTracer(agent_name="a", model="m", session_id="s")
    t.observe(AgentEvent(kind="message", summary="x" * 5000))
    t.observe(AgentEvent(kind="thinking", summary="hmm"))
    t.observe(AgentEvent(kind="status", summary="workspace ready", raw={"event": "workspace_ready"}))
    t.close()

    turn = _spans()["invoke_agent a"]
    events = {e.name: e for e in turn.events}
    assert set(events) == {"message", "thinking", "status"}
    assert len(events["message"].attributes["summary"]) == 400  # clipped, not a transcript


def test_never_raises_on_malformed_events_or_after_close():
    t = TurnTracer(agent_name="a", model="m", session_id="s")
    t.observe(AgentEvent(kind="tool_result", summary="no matching use", raw={"id": "??"}))
    t.observe(AgentEvent(kind="tool_use", summary="no id", raw={"tool": "Bash"}))
    t.observe(AgentEvent(kind="result", summary="ok", raw=None))
    t.close()
    t.close()  # idempotent
    t.observe(AgentEvent(kind="message", summary="after close"))  # no-op, no crash


def test_ensure_export_pipe_noop_off_platform_and_respects_real_provider(monkeypatch):
    from remote_agent_toolkit.runtime.gemini import tracing as tracing_mod

    # Off the platform (no engine id): never installs anything.
    monkeypatch.setattr(tracing_mod, "_pipe_ready", False)
    monkeypatch.delenv("GOOGLE_CLOUD_AGENT_ENGINE_ID", raising=False)
    tracing_mod.ensure_export_pipe()
    assert trace.get_tracer_provider() is _PROVIDER

    # On the platform but a real provider is already active: left untouched.
    monkeypatch.setattr(tracing_mod, "_pipe_ready", False)
    monkeypatch.setenv("GOOGLE_CLOUD_AGENT_ENGINE_ID", "123")
    tracing_mod.ensure_export_pipe()
    assert trace.get_tracer_provider() is _PROVIDER


def test_ensure_export_pipe_installs_over_proxy_provider(monkeypatch):
    # The observed job-worker state: engine id set, ambient provider is the proxy.
    from opentelemetry.trace import ProxyTracerProvider

    from remote_agent_toolkit.runtime.gemini import tracing as tracing_mod

    monkeypatch.setattr(tracing_mod, "_pipe_ready", False)
    monkeypatch.setenv("GOOGLE_CLOUD_AGENT_ENGINE_ID", "eng-1")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj-1")
    monkeypatch.setenv("GOOGLE_CLOUD_AGENT_ENGINE_LOCATION", "us-central1")
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: ProxyTracerProvider())
    installed = {}
    monkeypatch.setattr(trace, "set_tracer_provider", lambda p: installed.update(p=p))

    import google.auth

    monkeypatch.setattr(google.auth, "default", lambda: (object(), "proj-1"))
    # AuthorizedSession would try to use the fake credentials lazily — fine for a unit test.
    tracing_mod.ensure_export_pipe()

    provider = installed.get("p")
    assert provider is not None
    attrs = provider.resource.attributes
    assert attrs["cloud.resource_id"] == (
        "//aiplatform.googleapis.com/projects/proj-1/locations/us-central1/reasoningEngines/eng-1"
    )
    assert attrs["service.name"] == "eng-1"
    assert attrs["cloud.platform"] == "gcp.agent_engine"
    # One-shot: a second call doesn't build another provider.
    installed.clear()
    tracing_mod.ensure_export_pipe()
    assert not installed


def test_disabled_when_otel_unusable(monkeypatch):
    # Simulate a broken/missing provider path: constructor failure → permanent no-op.
    monkeypatch.setattr(trace, "get_tracer", lambda *a, **k: (_ for _ in ()).throw(RuntimeError))
    t = TurnTracer(agent_name="a", model="m", session_id="s")
    t.observe(AgentEvent(kind="message", summary="hi"))
    t.close()
    assert _EXPORTER.get_finished_spans() == ()

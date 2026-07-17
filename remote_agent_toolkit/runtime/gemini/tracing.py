"""Cloud Trace spans from the harness event stream (the engine's ``Traces`` tab).

Two halves, both needed because the platform provides neither where it matters:

**The pipe.** ``AdkApp(enable_tracing=True)`` is supposed to register a global OTel
``TracerProvider`` exporting OTLP → ``telemetry.googleapis.com``, resource-tagged with the
engine's ``cloud.resource_id`` (what the console's Agent Platform **Traces** tab filters
on). Observed live: that setup only takes effect on the *sync* serving path — the only
traces ever exported by any engine were the deploy-time validation queries; **query-job
workers (every run the toolkit does, cold and warm) never exported a span**. So
:func:`ensure_export_pipe` installs the same provider/exporter/resource shape ourselves
whenever the ambient provider is a no-op/proxy.

**The spans.** ADK only auto-instruments its own LLM/tool calls, and this agent's real
work happens inside the Claude Code subprocess where ADK can't see it. :class:`TurnTracer`
rebuilds the structure from the generic ``AgentEvent`` stream instead:

* one root span per turn (``invoke_agent <name>``), parented under ADK's ``invocation``
  span when one is active, closed with result status/usage/cost;
* one child span per tool call — opened on ``tool_use``, closed on the matching
  ``tool_result`` (paired by tool-use id), so durations are real;
* ``message``/``thinking``/``status`` as point-in-time span events on the root.

Attribute names follow the OTel GenAI semconv where one fits (``gen_ai.conversation.id``
carries the session id, which is what groups turns in the console's session view);
toolkit-specific facts use the ``rat.`` prefix. Values are truncated summaries — the same
text already flowing to Cloud Logging and the GCS event mirror, so tracing adds no new
exposure surface (and events never carry secret values to begin with).

Tracing is strictly best-effort: every public method swallows everything. A tracing bug
or a missing/no-op provider (e.g. running locally) must never take a run down.
"""

from __future__ import annotations

from typing import Any

from ...events import AgentEvent

_ATTR_MAX = 400  # span attribute / event values are summaries, not transcripts

_TELEMETRY_ENDPOINT = "https://telemetry.googleapis.com/v1/traces"
_pipe_ready = False  # process-wide: ensure_export_pipe is one-shot


def _clip(text: str) -> str:
    return text[:_ATTR_MAX]


def ensure_export_pipe() -> None:
    """Install an OTLP → Cloud Trace ``TracerProvider`` if none is active. Never raises.

    Mirrors what ``AdkApp(enable_tracing=True)`` builds for the sync serving path (same
    endpoint, auth, and resource attributes — the ``cloud.resource_id`` is what associates
    spans with the engine in the console), because query-job workers don't get that setup
    (observed live; see the module docstring). One-shot per process; a real provider
    already registered (e.g. the AdkApp's, or a consumer's own) is left untouched. Off the
    platform (no ``GOOGLE_CLOUD_AGENT_ENGINE_ID``) this is a no-op, so local/unit runs
    never grow an exporter.
    """
    global _pipe_ready
    if _pipe_ready:
        return
    _pipe_ready = True  # even on failure: don't retry (and re-fail) every turn
    try:
        import os

        engine_id = os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_ID")
        if not engine_id:
            return

        from opentelemetry import trace
        from opentelemetry.trace import NoOpTracerProvider, ProxyTracerProvider

        current = trace.get_tracer_provider()
        if not isinstance(current, (NoOpTracerProvider, ProxyTracerProvider)):
            return  # a real provider is already exporting; don't fight it

        import google.auth
        from google.auth.transport.requests import AuthorizedSession
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        credentials, adc_project = google.auth.default()
        project = os.environ.get("GOOGLE_CLOUD_PROJECT") or adc_project
        location = os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_LOCATION") or os.environ.get(
            "GOOGLE_CLOUD_LOCATION", ""
        )
        resource = Resource.create(
            {
                "gcp.project_id": project or "",
                "cloud.account.id": project or "",
                "cloud.provider": "gcp",
                "cloud.platform": "gcp.agent_engine",
                "cloud.region": location,
                "service.name": engine_id,
                "cloud.resource_id": (
                    f"//aiplatform.googleapis.com/projects/{project}"
                    f"/locations/{location}/reasoningEngines/{engine_id}"
                ),
            }
        )
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(
                    session=AuthorizedSession(credentials=credentials),
                    endpoint=_TELEMETRY_ENDPOINT,
                )
            )
        )
        trace.set_tracer_provider(provider)
    except Exception:  # noqa: BLE001 — tracing must never take a worker down
        pass


class TurnTracer:
    """Bridge one turn's ``AgentEvent`` stream into OTel spans. Never raises."""

    def __init__(self, agent_name: str, model: str, session_id: str) -> None:
        self._turn: Any = None
        self._tools: dict[str, Any] = {}
        ensure_export_pipe()  # job workers don't get the AdkApp's provider; bring our own
        try:
            from opentelemetry import trace

            self._otel = trace
            tracer = trace.get_tracer("remote-agent-toolkit")
            # start_span (not start_as_current_span): this object outlives any single
            # ``await`` in the surrounding async generator, and mutating the ambient
            # context across yields would leak it into unrelated tasks. The active span
            # at construction time (ADK's ``invocation``) still becomes the parent.
            self._turn = tracer.start_span(f"invoke_agent {agent_name}")
            self._turn.set_attribute("gen_ai.operation.name", "invoke_agent")
            self._turn.set_attribute("gen_ai.agent.name", agent_name)
            self._turn.set_attribute("gen_ai.request.model", model)
            self._turn.set_attribute("gen_ai.conversation.id", session_id)
            self._ctx = trace.set_span_in_context(self._turn)
            self._tracer = tracer
        except Exception:  # noqa: BLE001 — no provider / no otel: tracing silently off
            self._turn = None

    def observe(self, event: AgentEvent) -> None:
        """Fold one surfaced event into the trace (open/close tool spans, annotate)."""
        if self._turn is None:
            return
        try:
            self._observe(event)
        except Exception:  # noqa: BLE001 — tracing must never break the run
            pass

    def _observe(self, event: AgentEvent) -> None:
        raw = event.raw or {}
        if event.kind == "tool_use" and raw.get("id"):
            span = self._tracer.start_span(
                f"execute_tool {raw.get('tool', 'tool')}", context=self._ctx
            )
            span.set_attribute("gen_ai.operation.name", "execute_tool")
            span.set_attribute("gen_ai.tool.name", str(raw.get("tool", "")))
            span.set_attribute("gen_ai.tool.call.id", str(raw["id"]))
            span.set_attribute("rat.summary", _clip(event.summary))
            self._tools[str(raw["id"])] = span
        elif event.kind == "tool_result" and str(raw.get("id")) in self._tools:
            span = self._tools.pop(str(raw["id"]))
            if raw.get("is_error"):
                span.set_status(self._otel.StatusCode.ERROR, _clip(event.summary))
            span.end()
        elif event.kind == "result":
            usage = event.usage or {}
            for key in ("input_tokens", "output_tokens"):
                if isinstance(usage.get(key), int):
                    self._turn.set_attribute(f"gen_ai.usage.{key}", usage[key])
            if event.cost_usd is not None:
                self._turn.set_attribute("rat.cost_usd", event.cost_usd)
            if isinstance(raw.get("num_turns"), int):
                self._turn.set_attribute("rat.num_turns", raw["num_turns"])
            if raw.get("is_error"):
                self._turn.set_status(self._otel.StatusCode.ERROR, _clip(event.summary))
        else:  # message / thinking / status → point-in-time annotations on the turn
            self._turn.add_event(event.kind, {"summary": _clip(event.summary)})

    def close(self) -> None:
        """End the turn: close orphaned tool spans (crash mid-tool), then the root."""
        if self._turn is None:
            return
        try:
            for span in self._tools.values():
                span.set_status(self._otel.StatusCode.ERROR, "tool span never closed")
                span.end()
            self._tools.clear()
            self._turn.end()
            # A cold job's worker can be torn down right after the terminal event; flush
            # so the turn's spans aren't lost in a batch buffer. Sync on purpose (this is
            # reached from an async generator's finally, where awaiting is illegal).
            provider = self._otel.get_tracer_provider()
            flush = getattr(provider, "force_flush", None)
            if flush is not None:
                flush(10_000)  # ms; bounded — never wedge a worker on telemetry
        except Exception:  # noqa: BLE001
            pass
        self._turn = None

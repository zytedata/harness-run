"""Cloud Trace spans from the harness event stream (the engine's ``Traces`` tab).

Two halves, both needed because the platform provides neither where it matters:

**The flush.** The platform initializes OpenTelemetry in query-job workers (a real global
``TracerProvider`` exporting to ``telemetry.googleapis.com``, resource-tagged with the
engine's ``cloud.resource_id`` — what the console's Traces tab filters on), **but never
flushes it on the job path**: the sync serving path force-flushes after each query stream,
while a job worker is torn down with the batch processor's buffer unexported. Verified by
a standalone repro (identical stock-ADK engine: sync span exports, job span — recorded on
a live provider — never does). So :meth:`TurnTracer.close` force-flushes at the end of
every turn; that flush is what makes toolkit traces exist at all. Spans that end *after*
it (the platform's own outermost runner span) may still be lost to teardown — the
"(Missing span ID …)" placeholder the console then shows is cosmetic and theirs to fix.
An earlier belt-and-braces ``ensure_export_pipe`` (self-installed provider for the case
where the ambient one is a no-op) was removed once live evidence showed every cloud
worker already has the platform's provider — see DESIGN §6 and git history if a platform
regression ever brings that case back.

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


def _clip(text: str) -> str:
    return text[:_ATTR_MAX]


class TurnTracer:
    """Bridge one turn's ``AgentEvent`` stream into OTel spans. Never raises."""

    def __init__(self, agent_name: str, model: str, session_id: str) -> None:
        self._turn: Any = None
        self._tools: dict[str, Any] = {}
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
        """End the turn: close orphaned tool spans, end the root, and FLUSH.

        The flush is load-bearing, not hygiene: the platform never flushes OTel on the
        query-job path and tears the worker down with the batch buffer unexported (see
        the module docstring) — without this call no toolkit span would ever reach Cloud
        Trace. Sync on purpose (reached from an async generator's finally, where awaiting
        is illegal); bounded so telemetry can never wedge a worker.
        """
        if self._turn is None:
            return
        try:
            for span in self._tools.values():
                span.set_status(self._otel.StatusCode.ERROR, "tool span never closed")
                span.end()
            self._tools.clear()
            self._turn.end()
            provider = self._otel.get_tracer_provider()
            flush = getattr(provider, "force_flush", None)
            if flush is not None:
                flush(10_000)  # ms
        except Exception:  # noqa: BLE001
            pass
        self._turn = None

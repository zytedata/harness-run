"""Generic ``AgentEvent`` → ADK ``Event`` (the gemini deploy side).

On Gemini Agent Runtime the deployed agent is an ADK ``BaseAgent``, so each generic
:class:`AgentEvent` the harness emits is also surfaced as an ADK ``Event`` for the engine's
event stream (persisted to the job's GCS output). The near-real-time channel is the
``EventSink`` (Cloud Logging) — these ADK events are the durable, drain-on-completion copy.

Structured tool detail already rides on the ``EventSink`` via ``event.raw``; here we emit a
text event carrying the summary plus ``custom_metadata`` (kind + raw + cost/usage), and mark
the terminal ``result`` event non-partial / turn-complete. ``google.adk`` / ``google.genai``
are imported lazily.
"""

from __future__ import annotations

from typing import Any

from ...events import AgentEvent


def to_adk_event(event: AgentEvent, author: str) -> Any:
    """Build an ADK ``Event`` from a generic :class:`AgentEvent`."""
    from google.adk.events import Event
    from google.genai import types

    is_result = event.kind == "result"
    meta: dict[str, Any] = {"kind": event.kind}
    if event.raw is not None:
        meta["raw"] = event.raw
    if is_result:
        meta["cost_usd"] = event.cost_usd
        meta["usage"] = event.usage

    part = types.Part(text=event.summary, thought=True if event.kind == "thinking" else None)
    ev = Event(
        author=author,
        content=types.Content(role="model", parts=[part]),
        partial=not is_result,
        turn_complete=True if is_result else None,
        custom_metadata=meta,
    )
    if is_result and (event.raw or {}).get("is_error"):
        ev.error_message = event.summary
    return ev

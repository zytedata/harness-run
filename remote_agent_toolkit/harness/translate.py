"""Claude SDK message → generic ``AgentEvent`` translation (DESIGN.md §7, §8).

The point is to surface the *inner* Claude Code loop upward rather than flattening it
to one final blob: each streamed SDK message becomes zero or more ``AgentEvent``s that
keep tool name, inputs, results, and final usage distinct. Lifted & generalized from
the PoC ``event_mapping.py`` (which targeted ADK ``Event``s); here the target is the
harness-agnostic ``AgentEvent`` so the vocabulary is reusable and testable.

Mapping (SDK message type → ``AgentEvent.kind``):

* assistant text   → ``message``
* thinking         → ``thinking``
* tool use         → ``tool_use``   (raw: tool name, id, input)
* tool result      → ``tool_result`` (raw: tool name, id, is_error, content)
* system ``init``  → ``status``
* final result     → ``result``     (cost_usd/usage on the event; raw: subtype,
                                      is_error, num_turns, duration_ms, session_id)

``claude_agent_sdk`` types are imported lazily so importing this module needs no deps.
"""

from __future__ import annotations

from typing import Any, Iterator

from ..events import AgentEvent

# Tool-input keys worth surfacing in a one-line ``tool_use`` summary, in priority order.
_SUMMARY_KEYS = (
    "command", "file_path", "skill", "pattern", "url",
    "description", "subagent_type", "prompt", "query",
)


def _summary_arg(args: dict) -> str:
    for key in _SUMMARY_KEYS:
        val = args.get(key)
        if val:
            return " ".join(str(val).split())[:160]
    return ""


def _coerce(content: Any) -> Any:
    """Keep tool-result content JSON-ish for ``raw`` without exploding its size."""
    if isinstance(content, (dict, list, str, int, float, bool)) or content is None:
        return content
    return str(content)


class EventTranslator:
    """Stateful translator: remembers tool_use ids so results can be labelled.

    One instance per run (a ``tool_result`` arrives in a later message than its
    ``tool_use``, so the id → name mapping must persist across ``translate`` calls).
    """

    def __init__(self) -> None:
        self._tool_names: dict[str, str] = {}

    def translate(self, message: Any) -> Iterator[AgentEvent]:
        """Yield zero or more :class:`AgentEvent`s for one Claude SDK message."""
        from claude_agent_sdk import (
            AssistantMessage,
            ResultMessage,
            SystemMessage,
            TextBlock,
            ThinkingBlock,
            ToolResultBlock,
            ToolUseBlock,
            UserMessage,
        )

        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    if block.text.strip():
                        yield AgentEvent(kind="message", summary=block.text)
                elif isinstance(block, ThinkingBlock):
                    if block.thinking.strip():
                        yield AgentEvent(kind="thinking", summary=block.thinking)
                elif isinstance(block, ToolUseBlock):
                    self._tool_names[block.id] = block.name
                    args = block.input or {}
                    summary = f"{block.name} {_summary_arg(args)}".strip()
                    yield AgentEvent(
                        kind="tool_use",
                        summary=summary,
                        raw={"tool": block.name, "id": block.id, "input": args},
                    )
        elif isinstance(message, UserMessage):
            # UserMessage content carries tool results back from the engine.
            content = message.content
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, ToolResultBlock):
                        name = self._tool_names.get(block.tool_use_id, "tool")
                        yield AgentEvent(
                            kind="tool_result",
                            summary=f"{name}{' (error)' if block.is_error else ''}",
                            raw={
                                "tool": name,
                                "id": block.tool_use_id,
                                "is_error": bool(block.is_error),
                                "content": _coerce(block.content),
                            },
                        )
        elif isinstance(message, SystemMessage):
            if message.subtype == "init":
                data = message.data or {}
                yield AgentEvent(
                    kind="status",
                    summary=f"session started (model={data.get('model')})",
                    raw={"subtype": "init", "data": data},
                )
        elif isinstance(message, ResultMessage):
            usage = message.usage or {}
            if message.is_error:
                text = message.result or "(run errored)"
            else:
                text = message.result or "(no final text)"
            yield AgentEvent(
                kind="result",
                summary=text,
                cost_usd=message.total_cost_usd,
                usage=usage,
                raw={
                    "subtype": message.subtype,
                    "is_error": bool(message.is_error),
                    "num_turns": message.num_turns,
                    "duration_ms": message.duration_ms,
                    "session_id": message.session_id,
                },
            )
        # Other SDK message types (StreamEvent, RateLimitEvent) are intentionally dropped.

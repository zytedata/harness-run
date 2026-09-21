"""EventTranslator: synthetic Claude SDK messages → generic AgentEvents."""

from __future__ import annotations

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

from agent_run.harness.translate import EventTranslator


def test_assistant_text_thinking_and_tool_use():
    tr = EventTranslator()
    msg = AssistantMessage(
        content=[
            TextBlock(text="Hello there"),
            TextBlock(text="   "),  # whitespace-only → dropped
            ThinkingBlock(thinking="let me reason", signature="sig"),
            ToolUseBlock(id="t1", name="Bash", input={"command": "ls -la"}),
        ],
        model="claude-sonnet-4-6",
    )
    events = list(tr.translate(msg))
    kinds = [e.kind for e in events]
    assert kinds == ["message", "thinking", "tool_use"]
    assert events[0].summary == "Hello there"
    assert events[2].summary == "Bash ls -la"
    assert events[2].raw == {"tool": "Bash", "id": "t1", "input": {"command": "ls -la"}}


def test_tool_result_is_labelled_from_prior_tool_use():
    tr = EventTranslator()
    list(
        tr.translate(
            AssistantMessage(
                content=[ToolUseBlock(id="t1", name="Bash", input={"command": "true"})],
                model="m",
            )
        )
    )
    out = list(
        tr.translate(
            UserMessage(
                content=[ToolResultBlock(tool_use_id="t1", content="ok", is_error=False)],
            )
        )
    )
    assert len(out) == 1
    assert out[0].kind == "tool_result"
    assert out[0].summary == "Bash"  # name recovered from the remembered tool_use id
    assert out[0].raw["is_error"] is False


def test_result_carries_cost_usage_and_metadata():
    tr = EventTranslator()
    msg = ResultMessage(
        subtype="success",
        duration_ms=1234,
        duration_api_ms=1000,
        is_error=False,
        num_turns=7,
        session_id="sess-abc",
        total_cost_usd=0.42,
        usage={"input_tokens": 10},
        result="final answer",
        model_usage={"m": {"inputTokens": 10, "contextWindow": 1_000_000}},
    )
    (ev,) = list(tr.translate(msg))
    assert ev.kind == "result"
    assert ev.summary == "final answer"
    assert ev.cost_usd == 0.42
    assert ev.usage == {"input_tokens": 10}
    assert ev.raw["num_turns"] == 7 and ev.raw["session_id"] == "sess-abc"
    assert ev.raw["is_error"] is False
    assert ev.raw["model_usage"]["m"]["contextWindow"] == 1_000_000


def test_result_prefers_sdk_structured_output():
    tr = EventTranslator()
    msg = ResultMessage(
        subtype="success",
        duration_ms=10,
        duration_api_ms=8,
        is_error=False,
        num_turns=1,
        session_id="s",
        total_cost_usd=0.01,
        result=None,
        structured_output={"answer": 42},
    )
    (event,) = list(tr.translate(msg))
    assert event.summary == '{"answer":42}'
    assert event.raw["structured_output"] == {"answer": 42}


def test_system_init_becomes_status_and_other_subtypes_drop():
    tr = EventTranslator()
    assert [e.kind for e in tr.translate(SystemMessage(subtype="init", data={"model": "m"}))] == [
        "status"
    ]
    assert list(tr.translate(SystemMessage(subtype="other", data={}))) == []

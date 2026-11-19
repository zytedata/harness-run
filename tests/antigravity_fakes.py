"""Fake ``google.antigravity.Agent`` + ``Step`` factories for offline harness tests.

The steps and usage records are the REAL SDK pydantic types (``google-antigravity`` is a dev
dependency), so field names and enum values stay honest; only the agent, its conversation
and the transport are faked. ``make_agent(turns)`` returns an ``Agent`` class whose
conversation plays one script per ``send()``: an item that is a ``Step`` is yielded, a
:class:`Usage` updates the cumulative counters, :data:`CANCEL_POINT` waits until the harness
cancels the turn, an Exception is raised, and ``("stop", reason)`` sets the turn's stop
reason.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from google.antigravity import types

from harness_run.harness.antigravity import conversation_id_for

# The main trajectory id of a test turn: the conversation id the harness derives from the
# tests' session id ("sid").
MAIN = conversation_id_for("sid")
CANCEL_POINT = object()


@dataclass
class Usage:
    """Set the cumulative usage of a trajectory (and the session total moves with it)."""

    trajectory: str
    prompt: int
    cached: int = 0
    candidates: int = 0
    thoughts: int = 0


def step(
    kind: str,
    *,
    index: int,
    status: str = "DONE",
    source: str = "MODEL",
    content: str = "",
    thinking: str = "",
    tools: list[tuple[str, dict]] | None = None,
    trajectory: str = MAIN,
    parent: str = "",
    error: str = "",
    structured: Any = None,
) -> types.Step:
    return types.Step(
        id=f"{trajectory}-{index}",
        step_index=index,
        trajectory_id=trajectory,
        parent_trajectory_id=parent,
        type=types.StepType(kind),
        source=types.StepSource(source),
        status=types.StepStatus(status),
        content=content,
        thinking=thinking,
        tool_calls=[
            types.ToolCall(name=name, args=dict(args), id=f"call-{index}-{i}")
            for i, (name, args) in enumerate(tools or [])
        ],
        error=error,
        structured_output=structured,
    )


def tool_call(index: int, name: str, args: dict, result: dict, **kw: Any) -> list[types.Step]:
    """A tool call as the SDK streams it: announced (ACTIVE, twice), then DONE with results."""
    return [
        step("TOOL_CALL", index=index, status="ACTIVE", tools=[(name, args)], **kw),
        step("TOOL_CALL", index=index, status="ACTIVE", tools=[(name, args)], **kw),
        step("TOOL_CALL", index=index, status="DONE", tools=[(name, {**args, **result})], **kw),
    ]


class FakeConversation:
    def __init__(self, turns: list[list[Any]], start: dict[str, Usage] | None = None) -> None:
        self._turns = list(turns)
        self._script: list[Any] = []
        self._per_traj: dict[str, types.UsageMetadata] = {}
        for traj, u in (start or {}).items():
            self._per_traj[traj] = self._usage(u)
        self._cancelled = asyncio.Event()
        self._last_turn_stop_reason = types.StopReason.UNSPECIFIED
        self.sent: list[Any] = []
        self.cancels = 0

    @staticmethod
    def _usage(u: Usage) -> types.UsageMetadata:
        return types.UsageMetadata(
            prompt_token_count=u.prompt,
            cached_content_token_count=u.cached,
            candidates_token_count=u.candidates,
            thoughts_token_count=u.thoughts,
            total_token_count=u.prompt + u.candidates + u.thoughts,
        )

    @property
    def total_usage(self) -> types.UsageMetadata:
        total = types.UsageMetadata(
            prompt_token_count=0, cached_content_token_count=0, candidates_token_count=0,
            thoughts_token_count=0, total_token_count=0,
        )
        for u in self._per_traj.values():
            total = total + u
        return total

    @property
    def trajectory_usages(self) -> dict[str, types.UsageMetadata]:
        return {k: v.model_copy() for k, v in self._per_traj.items()}

    async def send(self, prompt: Any) -> None:
        self.sent.append(prompt)
        self._script = self._turns.pop(0) if self._turns else []
        self._cancelled.clear()
        self._last_turn_stop_reason = types.StopReason.UNSPECIFIED

    async def cancel(self) -> None:
        self.cancels += 1
        self._cancelled.set()

    async def receive_steps(self):
        for item in self._script:
            if self._cancelled.is_set():
                raise types.AntigravityCancelledError()
            if item is CANCEL_POINT:
                await self._cancelled.wait()
                raise types.AntigravityCancelledError()
            if isinstance(item, Usage):
                self._per_traj[item.trajectory] = self._usage(item)
                continue
            if isinstance(item, tuple) and item[0] == "stop":
                self._last_turn_stop_reason = types.StopReason(item[1])
                continue
            if isinstance(item, BaseException):
                raise item
            await asyncio.sleep(0)
            yield item
        if self._cancelled.is_set():
            raise types.AntigravityCancelledError()


def make_agent(turns: list[list[Any]], start: dict[str, Usage] | None = None):
    """An ``Agent`` stand-in; ``.instances`` records each one (config, conversation)."""

    class FakeAgent:
        instances: list[FakeAgent] = []

        def __init__(self, config: Any) -> None:
            self.config = config
            self.conversation = FakeConversation(turns, start)
            FakeAgent.instances.append(self)

        async def __aenter__(self) -> FakeAgent:
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

    return FakeAgent

"""Fake ``openai_codex.AsyncCodex`` + notification factories for offline harness tests.

The notifications are built from the REAL generated pydantic models (the ``openai-codex``
package is a runtime dep), so field names/enum values stay honest; only the client and
its transport are faked. ``make_async_codex(script)`` mirrors ``fakes.make_sdk_client``:
the returned class plays ``script`` from ``TurnHandle.stream()`` — an item that is an
Exception is raised, a callable is invoked with the client (its return, if any, is
yielded), anything else is yielded as-is.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from openai_codex.generated.notification_registry import (
    ErrorNotification,
    ItemCompletedNotification,
    ItemStartedNotification,
    ThreadTokenUsageUpdatedNotification,
    TurnCompletedNotification,
    TurnStartedNotification,
)
from openai_codex.generated.v2_all import (
    AgentMessageThreadItem,
    CollabAgentState,
    CollabAgentToolCallThreadItem,
    CommandExecutionThreadItem,
    SubAgentActivityThreadItem,
    ThreadItem,
    ThreadTokenUsage,
    TokenUsageBreakdown,
    Turn,
    TurnError,
)

_TID = "thr-fake"
# What `thread.read()` reports unless a test overrides it (mirrors an OpenAI-path thread).
_DEFAULT_RESOLVED = {"model_provider": "openai", "model": "gpt-5.6-luna"}
_TURN_ID = "turn-fake"


def note(method: str, payload: Any) -> SimpleNamespace:
    """The SDK's ``Notification`` is a (method, payload) dataclass; a namespace suffices."""
    return SimpleNamespace(method=method, payload=payload)


def turn_started() -> SimpleNamespace:
    turn = Turn(id=_TURN_ID, items=[], status="inProgress")
    return note("turn/started", TurnStartedNotification(thread_id=_TID, turn=turn))


def turn_completed(status: str = "completed", error: str | None = None) -> SimpleNamespace:
    turn = Turn(
        id=_TURN_ID,
        items=[],
        status=status,
        duration_ms=1234,
        error=TurnError(message=error) if error else None,
    )
    return note("turn/completed", TurnCompletedNotification(thread_id=_TID, turn=turn))


def agent_message(text: str, phase: str | None = "final_answer") -> SimpleNamespace:
    item = ThreadItem(
        AgentMessageThreadItem(id="m1", type="agentMessage", text=text, phase=phase)
    )
    return note(
        "item/completed",
        ItemCompletedNotification(
            completed_at_ms=0, item=item, thread_id=_TID, turn_id=_TURN_ID
        ),
    )


def command_execution(
    *, started: bool, command: str = "echo hi", exit_code: int | None = 0,
    output: str = "hi", status: str = "completed",
) -> SimpleNamespace:
    item = ThreadItem(
        CommandExecutionThreadItem(
            id="c1", type="commandExecution", command=command, command_actions=[],
            cwd="/w", status="inProgress" if started else status,
            exit_code=None if started else exit_code,
            aggregated_output=None if started else output,
        )
    )
    if started:
        return note(
            "item/started",
            ItemStartedNotification(
                started_at_ms=0, item=item, thread_id=_TID, turn_id=_TURN_ID
            ),
        )
    return note(
        "item/completed",
        ItemCompletedNotification(
            completed_at_ms=0, item=item, thread_id=_TID, turn_id=_TURN_ID
        ),
    )


def token_usage(
    *, in_tok: int = 1000, cached: int = 0, out_tok: int = 100, reasoning: int = 0,
    total_in: int | None = None, cache_write: int | None = None,
) -> SimpleNamespace:
    last = TokenUsageBreakdown(
        input_tokens=in_tok, cached_input_tokens=cached, output_tokens=out_tok,
        reasoning_output_tokens=reasoning, total_tokens=in_tok + out_tok,
        cache_write_input_tokens=cache_write,
    )
    total = TokenUsageBreakdown(
        input_tokens=total_in if total_in is not None else in_tok,
        cached_input_tokens=cached, output_tokens=out_tok,
        reasoning_output_tokens=reasoning,
        total_tokens=(total_in if total_in is not None else in_tok) + out_tok,
        cache_write_input_tokens=cache_write,
    )
    return note(
        "thread/tokenUsage/updated",
        ThreadTokenUsageUpdatedNotification(
            thread_id=_TID, turn_id=_TURN_ID,
            token_usage=ThreadTokenUsage(last=last, total=total),
        ),
    )


def collab_agent_tool_call(
    *, started: bool = False, agents: dict[str, str] | None = None,
    tool: str = "spawnAgent",
) -> SimpleNamespace:
    """A collab-agent (subagent) tool call carrying per-thread CollabAgentStatus values."""
    agents = agents if agents is not None else {"thr-sub-1": "completed"}
    item = ThreadItem(
        CollabAgentToolCallThreadItem(
            id="ca1", type="collabAgentToolCall", tool=tool,
            status="inProgress" if started else "completed",
            sender_thread_id=_TID, receiver_thread_ids=list(agents),
            agents_states={tid: CollabAgentState(status=s) for tid, s in agents.items()},
        )
    )
    if started:
        return note(
            "item/started",
            ItemStartedNotification(
                started_at_ms=0, item=item, thread_id=_TID, turn_id=_TURN_ID
            ),
        )
    return note(
        "item/completed",
        ItemCompletedNotification(
            completed_at_ms=0, item=item, thread_id=_TID, turn_id=_TURN_ID
        ),
    )


def sub_agent_activity(
    *, agent_thread_id: str = "thr-sub-1", activity: str = "started",
) -> SimpleNamespace:
    item = ThreadItem(
        SubAgentActivityThreadItem(
            id="sa1", type="subAgentActivity", agent_path=f"{_TID}/0",
            agent_thread_id=agent_thread_id, kind=activity,
        )
    )
    return note(
        "item/completed",
        ItemCompletedNotification(
            completed_at_ms=0, item=item, thread_id=_TID, turn_id=_TURN_ID
        ),
    )


def codex_error(message: str, will_retry: bool = True) -> SimpleNamespace:
    return note(
        "error",
        ErrorNotification(
            error=TurnError(message=message), thread_id=_TID, turn_id=_TURN_ID,
            will_retry=will_retry,
        ),
    )


def make_async_codex(
    script: list,
    resolved: dict | None = _DEFAULT_RESOLVED,
    follow_up_scripts: list[list] = (),
) -> type:
    """A fake ``AsyncCodex`` class whose turn stream plays ``script``.

    Records every instance on ``instances``; each instance records ``login_keys``,
    ``thread_starts`` / ``thread_resumes`` (kwargs), ``turns`` (prompt, kwargs),
    ``steers`` (the steer inputs) and ``interrupted``. The thread id is ``"thr-fake"``.
    A callable script entry is invoked with the client (awaited if it returns a
    coroutine) and its result, if any, is yielded. Each further ``thread.turn()`` on the
    thread (the harness continuing after an interrupt) plays the next entry of
    ``follow_up_scripts``.
    """

    class FakeHandle:
        def __init__(self, client: FakeAsyncCodex, turn_script: list) -> None:
            self._client = client
            self._script = turn_script

        async def stream(self):
            for entry in self._script:
                if isinstance(entry, Exception):
                    raise entry
                if callable(entry):
                    result = entry(self._client)
                    if asyncio.iscoroutine(result):
                        result = await result
                    if result is not None:
                        yield result
                    continue
                yield entry

        async def interrupt(self) -> None:
            self._client.interrupted = True

        async def steer(self, input):
            self._client.steers.append(input)

    class FakeThread:
        def __init__(self, client: FakeAsyncCodex, thread_id: str = _TID) -> None:
            self._client = client
            self.id = thread_id

        async def turn(self, prompt, **kwargs):
            self._client.turns.append((prompt, kwargs))
            n = len(self._client.turns) - 1
            turn_script = script if n == 0 else list(follow_up_scripts)[n - 1]
            return FakeHandle(self._client, turn_script)

        async def read(self, **_kw):
            """The app-server's record of this thread — routing attribution.

            Shaped like ``ThreadReadResponse``: ``.thread.model_provider`` plus
            ``.thread.settings.model``. ``make_async_codex(..., resolved=...)`` sets what
            it reports; ``resolved=None`` makes it raise, standing in for a server that
            does not answer — attribution must degrade, not fail the turn.
            """
            resolved = self._client.resolved
            if resolved is None:
                raise RuntimeError("thread.read unavailable")
            return SimpleNamespace(thread=SimpleNamespace(
                model_provider=resolved.get("model_provider"),
                settings=SimpleNamespace(model=resolved.get("model")),
            ))

    class FakeAsyncCodex:
        instances: list[FakeAsyncCodex] = []

        def __init__(self, config=None) -> None:
            self.config = config
            self.resolved = resolved
            self.login_keys: list[str] = []
            self.thread_starts: list[dict] = []
            self.thread_resumes: list[tuple[str, dict]] = []
            self.turns: list[tuple[Any, dict]] = []
            self.steers: list = []
            self.interrupted = False
            self.closed = False
            FakeAsyncCodex.instances.append(self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc) -> None:
            self.closed = True

        async def login_api_key(self, api_key: str) -> None:
            self.login_keys.append(api_key)

        async def thread_start(self, **kwargs):
            self.thread_starts.append(kwargs)
            return FakeThread(self)

        async def thread_resume(self, thread_id: str, **kwargs):
            self.thread_resumes.append((thread_id, kwargs))
            return FakeThread(self, thread_id)

    return FakeAsyncCodex

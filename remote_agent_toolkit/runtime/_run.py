"""``DrivenRun`` — the shared, backend-agnostic ``Run`` handle (DESIGN.md §4, §5).

A run is driven by a background task that pulls an async ``AgentEvent`` source (the
in-process harness for ``local``; a Cloud-Logging tail for ``gemini``) into a queue while
accumulating the terminal :class:`RunResult`. That single driver feeds all three
consumption modes uniformly:

* **await** it       → wait for completion, returns :class:`RunResult`.
* **async-iterate**  → stream :class:`AgentEvent`s as they happen.
* **poll**           → check ``done`` / ``status``, read ``result`` when done.

Both runtimes build the *same* result from the terminal ``result`` event, so status and
``stop_reason`` semantics are identical local vs. remote.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Callable, TYPE_CHECKING

from ..events import AgentEvent, RunResult, RunStatus, StopReason
from ..structured import parse_structured_output

if TYPE_CHECKING:
    from ..spec import AgentSpec

_SENTINEL = object()


def stop_reason_for(result_raw: dict, checkpoint: bool) -> StopReason:
    """Map a result event's subtype/error onto a :class:`StopReason`."""
    subtype = str(result_raw.get("subtype") or "")
    if "max_turn" in subtype:
        return StopReason.MAX_TURNS
    if "budget" in subtype:
        return StopReason.BUDGET_EXCEEDED
    if result_raw.get("is_error"):
        return StopReason.ERROR
    # A clean turn end: in interactive/checkpoint mode the session is paused awaiting the
    # operator's next message; otherwise the turn is simply complete.
    return StopReason.NEEDS_INPUT if checkpoint else StopReason.END_TURN


def build_result(result_ev: AgentEvent, session_id: str, spec: AgentSpec) -> tuple[RunResult, StopReason]:
    raw = result_ev.raw or {}
    structured = (
        parse_structured_output(result_ev.summary, spec.output_schema)
        if spec.output_schema is not None
        else None
    )
    result = RunResult(
        text=result_ev.summary,
        structured_output=structured,
        is_error=bool(raw.get("is_error")),
        num_turns=int(raw.get("num_turns") or 0),
        cost_usd=float(result_ev.cost_usd or 0.0),
        usage=result_ev.usage,
        session_id=raw.get("session_id") or session_id,
    )
    return result, stop_reason_for(raw, spec.checkpoint)


class DrivenRun:
    """A single turn, driven by a background task feeding a queue + a result slot."""

    def __init__(
        self,
        agen_factory: Callable[[], AsyncIterator[AgentEvent]],
        session_id: str,
        spec: AgentSpec,
        on_complete: Callable[[RunResult, StopReason], None],
    ) -> None:
        self._agen_factory = agen_factory
        self._session_id = session_id
        self._spec = spec
        self._on_complete = on_complete
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._status = RunStatus.PENDING
        self._result: RunResult | None = None
        self._stop_reason: StopReason | None = None

    # -- driving ---------------------------------------------------------------

    def ensure_started(self) -> None:
        if self._task is None:
            self._task = asyncio.ensure_future(self._drive())

    async def _drive(self) -> None:
        self._status = RunStatus.RUNNING
        result_ev: AgentEvent | None = None
        try:
            async for event in self._agen_factory():
                if event.kind == "result" and result_ev is None:
                    result_ev = event
                await self._queue.put(event)
        except asyncio.CancelledError:
            self._finalize(result_ev, error="interrupted")
            await self._queue.put(_SENTINEL)
            raise
        except Exception as exc:  # noqa: BLE001 — surface as an error result, don't crash callers
            await self._queue.put(
                AgentEvent(kind="status", summary=f"run failed: {str(exc)[:200]}", raw={"event": "error"})
            )
            self._finalize(result_ev, error=str(exc))
            await self._queue.put(_SENTINEL)
            return
        self._finalize(result_ev, error=None)
        await self._queue.put(_SENTINEL)

    def _finalize(self, result_ev: AgentEvent | None, error: str | None) -> None:
        if error is not None:
            text = None if error == "interrupted" else f"run failed: {error}"
            self._result = RunResult(text=text, is_error=True, session_id=self._session_id)
            self._stop_reason = StopReason.ERROR
        elif result_ev is not None:
            self._result, self._stop_reason = build_result(result_ev, self._session_id, self._spec)
        else:
            self._result = RunResult(text=None, is_error=True, session_id=self._session_id)
            self._stop_reason = StopReason.ERROR
        self._status = RunStatus.IDLE
        self._on_complete(self._result, self._stop_reason)

    # -- the three consumption modes ------------------------------------------

    def __await__(self):
        return self._wait().__await__()

    async def _wait(self) -> RunResult | None:
        self.ensure_started()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        return self._result

    async def __aiter__(self) -> AsyncIterator[AgentEvent]:
        self.ensure_started()
        while True:
            item = await self._queue.get()
            if item is _SENTINEL:
                break
            yield item

    @property
    def task(self) -> asyncio.Task | None:
        return self._task

    @property
    def done(self) -> bool:
        return self._status in (RunStatus.IDLE, RunStatus.TERMINATED)

    @property
    def status(self) -> RunStatus:
        return self._status

    @property
    def result(self) -> RunResult | None:
        return self._result

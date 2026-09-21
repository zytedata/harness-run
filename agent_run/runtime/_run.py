"""``DrivenRun`` — the shared, backend-agnostic ``Run`` handle (DESIGN.md §4, §5).

A run is driven by a background task that pulls an async ``AgentEvent`` source (the
in-process harness for ``local``; a Cloud-Logging tail for ``sandbox``) into a queue while
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
    if subtype == "interrupted":
        # The operator's interrupt() ended the turn through the normal end-of-turn path.
        return StopReason.INTERRUPTED
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
    structured = None
    recovered = False
    if spec.output_schema is not None:
        structured = parse_structured_output(result_ev.summary, spec.output_schema)
        if structured is None:
            # The turn's final text can lose the deliverable: a stale background-task
            # notification re-invokes the model after it already answered, and its
            # reply to the notification becomes the final message. The harness carries
            # the displaced segment-boundary results' texts on the result event
            # (newest first) — recover the structured output from those.
            for text in raw.get("segment_summaries") or ():
                structured = parse_structured_output(text, spec.output_schema)
                if structured is not None:
                    recovered = True
                    break
    result = RunResult(
        text=result_ev.summary,
        structured_output=structured,
        structured_output_recovered=recovered,
        is_error=bool(raw.get("is_error")),
        num_turns=int(raw.get("num_turns") or 0),
        # Passed through as the harness reported it. ``None`` means the spend is
        # unknown; coercing it to 0.0 here would read as a free run.
        cost_usd=result_ev.cost_usd,
        usage=result_ev.usage,
        session_id=raw.get("session_id") or session_id,
        resources=_resources(raw),
    )
    return result, stop_reason_for(raw, spec.checkpoint)


_RESOURCE_KEYS = ("memory_peak_bytes", "memory_limit_bytes", "cpu_usec")


def _resources(raw: dict) -> dict[str, int] | None:
    """The worker-sampled high-water marks on a result event's ``raw`` (``None`` if none)."""
    found = {k: int(raw[k]) for k in _RESOURCE_KEYS if isinstance(raw.get(k), int) and not isinstance(raw.get(k), bool)}
    return found or None


class DrivenRun:
    """A single turn, driven by a background task feeding a queue + a result slot."""

    def __init__(
        self,
        agen_factory: Callable[[], AsyncIterator[AgentEvent]],
        session_id: str,
        spec: AgentSpec,
        on_complete: Callable[[RunResult, StopReason], None],
        on_event: Callable[[AgentEvent], None] | None = None,
    ) -> None:
        self._agen_factory = agen_factory
        self._session_id = session_id
        self._spec = spec
        self._on_complete = on_complete
        # Lets the owning session watch the stream (e.g. for the worker's control_ready
        # marker) without consuming it; called before the event is queued.
        self._on_event = on_event
        self._cancel_note: str | None = None
        self._cancelled = False
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
                if self._on_event is not None:
                    self._on_event(event)
                await self._queue.put(event)
        except asyncio.CancelledError:
            self._cancelled = True
            self._finalize(result_ev, error="interrupted")
            await self._queue.put(_SENTINEL)
            raise
        except Exception as exc:  # noqa: BLE001 — surface as an error result, don't crash callers
            summary = (
                f"harness exited abnormally after the terminal result (result kept): {str(exc)[:200]}"
                if result_ev is not None
                else f"run failed: {str(exc)[:200]}"
            )
            await self._queue.put(
                AgentEvent(
                    kind="status", summary=summary,
                    raw={"event": "late_harness_error" if result_ev is not None else "error"},
                )
            )
            self._finalize(result_ev, error=str(exc))
            await self._queue.put(_SENTINEL)
            return
        self._finalize(result_ev, error=None)
        await self._queue.put(_SENTINEL)

    def _finalize(self, result_ev: AgentEvent | None, error: str | None) -> None:
        if result_ev is not None:
            # A terminal result event is authoritative even if the harness died afterwards
            # (e.g. the claude CLI exiting non-zero while shutting down — which it also does
            # on error results like error_max_turns). The run's text, cost_usd, usage and
            # turn count are all real; discarding them turned a flaky exit at the finish
            # line into a total loss and made failed runs read as free (eval feedback).
            self._result, self._stop_reason = build_result(result_ev, self._session_id, self._spec)
            if error is not None and error != "interrupted":
                self._result.warning = f"harness exited abnormally after the result: {error[:500]}"
        elif error is not None:
            text = None if error == "interrupted" else f"run failed: {error}"
            self._result = RunResult(
                text=text, is_error=True, session_id=self._session_id,
                warning=self._cancel_note if error == "interrupted" else None,
            )
            self._stop_reason = StopReason.ERROR
        else:
            self._result = RunResult(text=None, is_error=True, session_id=self._session_id)
            self._stop_reason = StopReason.ERROR
        self._status = RunStatus.IDLE
        self._on_complete(self._result, self._stop_reason)

    @property
    def cancelled(self) -> bool:
        """Whether the driver task was cancelled (``cancel()``, or the event loop shutting down)."""
        return self._cancelled

    @property
    def cancel_note(self) -> str | None:
        """The reason a caller gave ``cancel()``; None for a cancellation nobody asked for."""
        return self._cancel_note

    def cancel(self, note: str | None = None) -> None:
        """Cancel the driver task (the fallback when the harness cannot be stopped cleanly).

        The run ends as an error result with no text; ``note`` lands on
        ``RunResult.warning`` so the result says why it ended this way.
        """
        self._cancel_note = note
        if self._task is not None and not self._task.done():
            self._task.cancel()

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

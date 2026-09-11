"""Turn control: sending into a running turn, and interrupting it (DESIGN.md §4, §7).

The session API is turn-based: ``run()``/``send()`` start a turn on an idle session. An
interactive loop needs two more things while a turn is RUNNING:

* **steer** — deliver a message into the running turn; the model sees it at its next step
  and the turn keeps going as the same ``Run``.
* **interrupt** — stop what the model is doing now. With a message, the turn continues
  from that message in the same harness session and workspace. Without one, the turn ends
  normally (checkpoint taken) with ``StopReason.INTERRUPTED`` and the session is resumable.

Both travel as a :class:`ControlMessage` over a :class:`ControlChannel` the runtime hands
the harness in ``RunContext.control``. The harness reads the channel while the harness
stream runs (:class:`ControlledStream`) and acts on each message. ``local`` uses the
in-process :class:`LocalControlChannel`; ``gemini`` posts the message to the sandbox
worker's ``/control`` endpoint, which feeds the same kind of queue on the worker's loop
(``runtime/gemini/worker.py``).

Every delivered message is surfaced as a ``user`` event whose ``raw["message_id"]`` is the
caller's id — that event is the acknowledgement that the model has the message.

Stdlib-only.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal, Protocol, runtime_checkable

from .events import AgentEvent

ControlOp = Literal["steer", "interrupt", "stop"]


class ControlUnavailable(RuntimeError):
    """``send()`` was called on a RUNNING session whose worker cannot receive it.

    The turn runs somewhere that reads no control channel: an engine revision deployed
    before this feature, or a channel that could not be confirmed. Nothing was sent and no
    second turn was started. Callers that want a fallback can queue the message and send
    it once the session is idle.
    """


@dataclass
class ControlMessage:
    """One operator action on a running turn.

    ``op`` is ``"steer"`` (deliver ``message``, keep going), ``"interrupt"`` (interrupt,
    then deliver ``message`` in the same harness session) or ``"stop"`` (interrupt and end
    the turn; ``message`` is ``None``). ``message_id`` is the caller's id for the message;
    it is stamped on the ``user`` event the harness emits when the model gets the message,
    and the gemini inbox dedupes on it.
    """

    op: ControlOp
    message: str | None = None
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict:
        return {"op": self.op, "message": self.message, "message_id": self.message_id}

    @classmethod
    def from_dict(cls, d: dict) -> ControlMessage:
        op = d.get("op")
        if op not in ("steer", "interrupt", "stop"):
            raise ValueError(f"unknown control op {op!r}")
        message = d.get("message")
        if op != "stop" and not isinstance(message, str):
            raise ValueError(f"control op {op!r} needs a message")
        return cls(
            op=op,
            message=message if op != "stop" else None,
            message_id=str(d.get("message_id") or uuid.uuid4()),
        )

    def user_event(self) -> AgentEvent:
        """The ``user`` event the harness emits once the model has this message."""
        return AgentEvent(
            kind="user",
            summary=self.message or "",
            raw={
                "event": "user_message",
                "message_id": self.message_id,
                "interrupt": self.op == "interrupt",
            },
        )


@runtime_checkable
class ControlChannel(Protocol):
    """The harness side of turn control: wait for the next operator message."""

    async def receive(self) -> ControlMessage:
        """Block until a control message arrives, then return it."""
        ...


class LocalControlChannel:
    """In-process :class:`ControlChannel` (the ``local`` runtime): an ``asyncio.Queue``.

    The session ``send()``s from the same event loop the harness runs on.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[ControlMessage] = asyncio.Queue()

    def send(self, message: ControlMessage) -> None:
        self._queue.put_nowait(message)

    async def receive(self) -> ControlMessage:
        return await self._queue.get()

    def pending(self) -> int:
        return self._queue.qsize()


class ControlledStream:
    """Read a harness stream and a control channel together.

    ``next(timeout)`` returns whichever arrives first: ``("message", m)`` from the stream,
    ``("control", ControlMessage)`` from the channel, ``("end", None)`` when the stream is
    exhausted, or ``("timeout", None)``. Neither pending read is cancelled by a timeout, so
    no item is ever lost. ``replace_stream`` swaps the stream when the harness starts a
    follow-up turn (Codex: a new turn handle after an interrupt). ``close`` drops the
    pending reads.
    """

    def __init__(self, stream: AsyncIterator[Any], control: ControlChannel | None) -> None:
        self._stream = stream
        self._control = control
        self._stream_task: asyncio.Task | None = None
        self._control_task: asyncio.Task | None = None

    def replace_stream(self, stream: AsyncIterator[Any]) -> None:
        if self._stream_task is not None:
            self._stream_task.cancel()
            self._stream_task = None
        self._stream = stream

    def _tasks(self) -> set[asyncio.Task]:
        if self._stream_task is None:
            self._stream_task = asyncio.ensure_future(anext(self._stream))
        tasks = {self._stream_task}
        if self._control is not None and self._control_task is None:
            self._control_task = asyncio.ensure_future(self._control.receive())
        if self._control_task is not None:
            tasks.add(self._control_task)
        return tasks

    async def next(self, timeout: float | None) -> tuple[str, Any]:
        done, _ = await asyncio.wait(
            self._tasks(), timeout=timeout, return_when=asyncio.FIRST_COMPLETED
        )
        if not done:
            return "timeout", None
        # A control message is handed over before a stream message that arrived in the
        # same tick: an operator's interrupt should not wait behind a stream item.
        if self._control_task is not None and self._control_task in done:
            task, self._control_task = self._control_task, None
            return "control", task.result()
        task, self._stream_task = self._stream_task, None
        try:
            return "message", task.result()
        except StopAsyncIteration:
            return "end", None

    async def close(self) -> None:
        for task in (self._stream_task, self._control_task):
            if task is not None and not task.done():
                task.cancel()
        self._stream_task = self._control_task = None

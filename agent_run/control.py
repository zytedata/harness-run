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
in-process :class:`LocalControlChannel`; ``sandbox`` posts the message to the sandbox
worker's ``/control`` endpoint, which feeds the same kind of queue on the worker's loop
(``runtime/sandbox/worker.py``).

Every delivered message is surfaced as a ``user`` event whose ``raw["message_id"]`` is the
caller's id — that event is the acknowledgement that the model has the message.

A third operator action LOOKS at the running turn instead of talking to it: ``Session.exec()``
runs a shell command in the turn's workspace and returns an :class:`ExecResult` — the
read-only probe an application uses to show the agent's work while it is still working (a
``git diff`` of the workspace, a file listing). Both runtimes run it through :func:`run_shell`
here (``local`` on the host, ``sandbox`` inside the sandbox via the worker's ``/exec``), so
the output cap and the timeout semantics are the same everywhere.

Stdlib-only.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import time
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
    and the sandbox inbox dedupes on it.
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


# -- exec: a read-only probe of the running turn's workspace ----------------------------

# Longest ``exec`` runs when the caller gives no timeout. Bounded so a probe can never pin a
# worker (or the sandbox proxy call, which the platform cuts at ~300 s) indefinitely.
EXEC_DEFAULT_TIMEOUT_S = 60.0
# Kept per stream (characters, head of the output). The sandbox proxy rejects answers past
# ~2 MB serialized; two streams under this cap fit with JSON escaping to spare.
EXEC_OUTPUT_CAP = 500_000
# Exit code reported when the command hit the timeout (the shell convention of ``timeout(1)``).
EXEC_TIMEOUT_RC = 124


@dataclass
class ExecResult:
    """What one :meth:`Session.exec` returned: the command's output and exit code.

    ``stdout`` / ``stderr`` are text (invalid UTF-8 replaced), each cut to
    :data:`EXEC_OUTPUT_CAP` characters from the head with ``truncated`` set when either was
    cut — a probe never fails for producing too much. ``returncode`` is the command's, or
    :data:`EXEC_TIMEOUT_RC` (124) when it was killed at the timeout, with a note in
    ``stderr``. ``duration_ms`` is the wall time of the command itself (the transport is on
    top of it).
    """

    stdout: str
    stderr: str
    returncode: int
    truncated: bool = False
    duration_ms: int = 0

    @property
    def ok(self) -> bool:
        """``returncode == 0``."""
        return self.returncode == 0

    def to_dict(self) -> dict:
        return {
            "stdout": self.stdout, "stderr": self.stderr, "returncode": self.returncode,
            "truncated": self.truncated, "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ExecResult:
        return cls(
            stdout=str(d.get("stdout") or ""), stderr=str(d.get("stderr") or ""),
            returncode=int(d.get("returncode", -1)), truncated=bool(d.get("truncated", False)),
            duration_ms=int(d.get("duration_ms") or 0),
        )


def _cap(text: str) -> tuple[str, bool]:
    if len(text) <= EXEC_OUTPUT_CAP:
        return text, False
    return text[:EXEC_OUTPUT_CAP], True


def run_shell(command: str, *, cwd: str | os.PathLike | None, timeout: float | None) -> ExecResult:
    """Run ``command`` under ``/bin/bash -c`` in ``cwd``; the one implementation behind ``exec``.

    Sync and blocking (callers on an event loop wrap it in ``asyncio.to_thread``). ``timeout``
    ``None`` means :data:`EXEC_DEFAULT_TIMEOUT_S`; at the timeout the command's whole process
    group is killed (a ``sleep`` forked by the shell included, so the pipes close) and the
    result carries what was captured so far with ``returncode`` 124. A missing ``cwd`` is
    reported as a failed command (exit 127 with a note), not as an exception — probing a
    workspace that is not there yet is a normal thing to ask.
    """
    limit = EXEC_DEFAULT_TIMEOUT_S if timeout is None else float(timeout)
    if cwd is not None and not os.path.isdir(cwd):
        return ExecResult(stdout="", stderr=f"cwd does not exist: {cwd}", returncode=127)
    t0 = time.monotonic()
    proc = subprocess.Popen(
        ["/bin/bash", "-c", command], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=limit)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, err = proc.communicate()
        note = f"command exceeded {limit:g}s and was killed".encode()
        err = (err.rstrip(b"\n") + b"\n" + note) if err else note
        rc = EXEC_TIMEOUT_RC
    duration_ms = int((time.monotonic() - t0) * 1000)
    stdout, cut_out = _cap(out.decode("utf-8", errors="replace"))
    stderr, cut_err = _cap(err.decode("utf-8", errors="replace"))
    return ExecResult(stdout=stdout, stderr=stderr, returncode=rc,
                      truncated=cut_out or cut_err, duration_ms=duration_ms)


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

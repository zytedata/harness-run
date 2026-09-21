"""Run-plane value types: status/stop-reason enums, events, and results (DESIGN.md §5).

Stdlib-only. The ``Run`` *handle* protocol lives in ``runtime/base.py``; this module
holds the plain data carried over the wire / event stream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class RunStatus(str, Enum):
    """Session/run lifecycle state (CMA-style; DESIGN.md §4).

    ``IDLE`` is *not* "done" — it carries a :class:`StopReason`. ``IDLE`` +
    ``StopReason.NEEDS_INPUT`` is the interactive checkpoint-and-resume pause.
    """

    PENDING = "pending"
    RUNNING = "running"
    IDLE = "idle"
    TERMINATED = "terminated"


class StopReason(str, Enum):
    """Why a run/turn stopped."""

    NEEDS_INPUT = "needs_input"
    END_TURN = "end_turn"
    MAX_TURNS = "max_turns"
    BUDGET_EXCEEDED = "budget_exceeded"
    # The operator stopped the turn with ``Session.interrupt()``. The turn ended through
    # the normal end-of-turn path (checkpoint taken, accounting kept), so the session is
    # idle and a later ``send()`` resumes it. Not an error.
    INTERRUPTED = "interrupted"
    ERROR = "error"


@dataclass
class AgentEvent:
    """A single generic event from a run (harness-agnostic).

    ``cost_usd`` and ``usage`` are populated only on the terminal ``"result"`` event.
    ``raw`` carries the original harness/SDK payload for callers that need detail.

    A ``"user"`` event is an operator message delivered INTO a running turn
    (``Session.send()`` while the session is running): ``summary`` is the message text and
    ``raw`` carries ``event: user_message``, the caller's ``message_id`` and whether it
    came with an ``interrupt``. It is emitted when the harness hands the message to the
    model, so it doubles as the acknowledgement that the model has it. The turn's opening
    prompt is not echoed this way.
    """

    kind: Literal["thinking", "tool_use", "tool_result", "message", "user", "status", "result"]
    summary: str
    raw: dict | None = None
    cost_usd: float | None = None
    usage: dict | None = None


@dataclass
class RunResult:
    """The terminal result of a run.

    Attributes:
        text: Final assistant text, or ``None``.
        structured_output: Parsed structured output if an ``output_schema`` was set.
        structured_output_recovered: ``structured_output`` came from an earlier
            segment-boundary result of the turn, not from ``text`` — a background-task
            notification re-invoked the model after it delivered the answer, and its
            reply to the stale notification displaced the deliverable from the final
            message. The recovered value is real; this flags that ``text`` doesn't
            contain it.
        is_error: Whether the run terminated in error.
        num_turns: Model calls made by the main agent — the same unit on every
            harness. Cumulative across background-task re-invocation segments;
            subagent (Task/collab-agent) calls are NOT counted, matching what
            ``max_turns`` caps.
        cost_usd: Total spend for the run, or ``None`` when the spend is unknown:
            the harness could not price the model, or OpenRouter reported no charge.
            A ``cost_unknown`` status event fires in that case and ``max_budget_usd``
            could not be enforced. ``0.0`` means the run really was free.
        usage: Normalized token accounting for the WHOLE turn — subagent sessions and
            background-task re-invocation segments included — identical in shape and
            meaning on every harness (see ``harness/_usage.py``). A flat dict whose
            five keys are always present: ``input_tokens``, ``cache_read_input_tokens``,
            ``cache_creation_input_tokens``, ``output_tokens``,
            ``reasoning_output_tokens`` — the input buckets disjoint, reasoning a share
            of output — ``None`` where the harness reports no such number (never a
            fake ``0``). The harness's own verbatim
            records stay on the result event's ``raw`` (``model_usage``/``cli_usage``
            on claude-code, ``subagent_usage`` on codex).
        session_id: The session this result belongs to (for re-attach/resume).
        artifacts: Blob keys/URIs of artifacts produced by the run.
        warning: Non-fatal anomaly note — e.g. the harness process exited abnormally
            *after* emitting this result. The result (text, spend, usage) is real and
            kept; the warning records that the run didn't shut down cleanly.
        resources: The turn's resource high-water marks as the worker sampled them
            inside its container (``sandbox``): ``memory_peak_bytes``,
            ``memory_limit_bytes``, ``cpu_usec`` — whichever were readable. ``None``
            when nothing was sampled (``local``, sampling disabled, no cgroup).
    """

    text: str | None
    structured_output: Any = None
    structured_output_recovered: bool = False
    is_error: bool = False
    num_turns: int = 0
    cost_usd: float | None = None
    usage: dict | None = None
    session_id: str | None = None
    artifacts: tuple[str, ...] = field(default=())
    warning: str | None = None
    resources: dict[str, int] | None = None

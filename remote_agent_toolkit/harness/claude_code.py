"""``ClaudeCodeHarness`` — the Claude Agent SDK binding (DESIGN.md §7, §8).

Drives ``claude_agent_sdk.query()`` against the per-run working directory and translates
each streamed SDK message into a generic :class:`AgentEvent` (via :mod:`translate`), so
the rest of the toolkit — and every runtime (``local`` / ``gemini``) — stays harness-
agnostic. This is the generalized, ADK-free core of the PoC ``ClaudeCodeAgent``: the
ADK / Agent Engine wrapping is a ``gemini``-side concern, not the harness's.

``claude_agent_sdk`` is imported lazily inside methods, so importing this module needs no
third-party deps.
"""

from __future__ import annotations

import asyncio
from collections import deque
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Sequence, TYPE_CHECKING

from ..events import AgentEvent
from ..spec import SystemPrompt
from ..structured import parse_structured_output
from ._shared import (
    GITHUB_MCP_TOKEN_KEYS as _GITHUB_MCP_TOKEN_KEYS,
    INTERACTIVE_SUFFIX as _INTERACTIVE_SUFFIX,
    finalize_checkpoint,
    harness_consumed_secret_names,
    runtime_env,
)

__all__ = ["ClaudeCodeHarness", "DEFAULT_ALLOWED_TOOLS", "harness_consumed_secret_names"]

if TYPE_CHECKING:
    from ..spec import AgentSpec
    from .context import RunContext

# Sensible coding-agent default when ``spec.allowed_tools`` is unset: the filesystem/exec
# tools skills depend on. The Skill tool is enabled separately via the ``skills`` option,
# not listed here (passing "Skill" in allowed_tools is deprecated in the SDK).
DEFAULT_ALLOWED_TOOLS = ("Read", "Write", "Edit", "Bash", "Glob", "Grep", "TodoWrite")


class _StderrCapture:
    """Captures the claude CLI's stderr — the SDK pipes it ONLY when a callback is set.

    Without a callback the CLI inherits the parent's stderr (interleaved unattributed, or
    lost), yet on a non-zero exit the SDK's ``ProcessError`` says "Check stderr output for
    details" — promising output nobody captured, making CLI-level failures undiagnosable
    (eval feedback). Each line is appended to ``<job_dir>/stderr.log`` — beside, not inside,
    the agent workspace, so it is never agent-visible or snapshotted — and an in-memory
    tail is kept for embedding into the failure event/error. Zero cost on healthy runs:
    the file is only created when the CLI actually writes a line. Never raises.
    """

    _FILE_CAP = 2_000_000  # bytes appended per run; beyond it only the in-memory tail grows

    def __init__(self, path: Path) -> None:
        self.path = path
        self._written = 0
        self._capped = False
        self._tail: deque[str] = deque(maxlen=60)

    def __call__(self, line: str) -> None:
        try:
            self._tail.append(line[-2000:])
            if self._capped:
                return
            data = line + "\n"
            if self._written + len(data) > self._FILE_CAP:
                data = "... [stderr.log capped; the failure event carries the tail]\n"
                self._capped = True
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(data)
            self._written += len(data)
        except Exception:  # noqa: BLE001 — diagnostics must never break the run
            pass

    def tail(self, max_chars: int = 2000) -> str:
        return "\n".join(self._tail)[-max_chars:]


class _TaskTracker:
    """Tracks the CLI's background tasks (Bash ``run_in_background``, Monitor) in a turn.

    The CLI honors those tools' semantics itself — it re-invokes the model when a task
    reaches a terminal state — but only while the message stream stays open. The tracker
    tells the run loop when a ``result`` is a *segment boundary* rather than the end of
    the turn: either tasks are still ``pending`` (running), or one just went terminal and
    its notification hasn't been ``delivered`` yet (no re-invocation observed since — the
    CLI is about to re-invoke). Driven by translated events, not raw SDK messages.
    """

    def __init__(self) -> None:
        self._pending: dict[str, str] = {}  # task_id -> description
        self._undelivered: set[str] = set()  # terminal; no tool-result boundary seen since
        self._staged: set[str] = set()  # terminal + boundary seen; next model call has it

    def observe(self, event: AgentEvent) -> None:
        # Delivery proof (verified live, CLI 2.1.223): the CLI injects a terminal
        # notification into the NEXT model call it assembles — never into a call already
        # in flight. Assistant output therefore proves delivery only when a tool_result
        # boundary (where the next call is assembled) separates the notification from
        # that output. Two stages:
        #   task_terminal  → undelivered (the CLI has not had a call to inject it into)
        #   tool_result    → staged      (the call being assembled will carry it)
        #   model output   → delivered   (that call ran; drop the staged ids)
        # Without this, a task completing *during* a long invocation stayed
        # "undelivered" forever — even after Claude discussed its result — and the
        # stale ledger at the final result kept the CLI open long enough for old
        # ScheduleWakeup prompts to create a spurious follow-up turn. Clearing on bare
        # assistant output is NOT enough: output emitted after the notification can
        # come from a call that was already in flight when it arrived, and the CLI
        # then re-invokes ~40 ms after the result — the turn must stay open for it.
        if event.kind == "tool_result":
            self._staged |= self._undelivered
            self._undelivered.clear()
        elif event.kind in {"message", "thinking", "tool_use"}:
            self._staged.clear()

        raw = event.raw or {}
        kind = raw.get("event")
        if kind == "task_started":
            self._pending[raw["task_id"]] = raw.get("description") or ""
        elif kind == "task_terminal":
            if raw.get("task_id") in self._pending:
                del self._pending[raw["task_id"]]
                self._undelivered.add(raw["task_id"])
        elif raw.get("subtype") == "init":
            # A (re-)invocation started: terminal notifications so far were delivered.
            self._undelivered.clear()
            self._staged.clear()

    @property
    def pending(self) -> dict[str, str]:
        return dict(self._pending)

    @property
    def undelivered(self) -> set[str]:
        return self._undelivered | self._staged

    def waiting(self) -> str | None:
        """Why the turn must stay open at a result event (``None`` = truly done)."""
        if self._pending:
            return "pending"
        if self._undelivered or self._staged:
            return "undelivered"
        return None


class _StreamPhase(Enum):
    """Whether the CLI is running the model or is between model invocations."""

    ACTIVE = "active"
    BETWEEN_INVOCATIONS = "between_invocations"


class ClaudeCodeHarness:
    """The Claude Agent SDK harness (implements ``harness.base.Harness``)."""

    # After a task went terminal but before the CLI's re-invocation: how long to wait for
    # that re-invocation. Short — the CLI re-invokes within moments; the fallback exists
    # because a task that completed *mid-turn* is delivered inside the current invocation
    # (no re-invoke follows) and must not hang the run.
    _UNDELIVERED_GRACE_S = 20.0

    # How many demoted (segment-boundary) result texts ride the final result event as
    # structured-output recovery candidates (``raw["segment_summaries"]``, newest first).
    _SEGMENT_FALLBACK_MAX = 3

    # -- option building -------------------------------------------------------

    def _system_prompt(self, spec: AgentSpec, interactive: bool) -> Any:
        """Resolve ``spec.system_prompt`` (+ interactive suffix) to an SDK value.

        * ``SystemPrompt`` → inherit Claude Code's preset and append.
        * plain ``str``    → replace the prompt entirely (suffix appended in interactive).
        * ``None``         → Claude Code's preset. The SDK's own ``None`` means an *empty*
          system prompt (``--system-prompt ""``): no preset, no ``CLAUDE.md``, no
          environment block, so the agent would not be Claude Code at all.
        """
        suffix = _INTERACTIVE_SUFFIX if interactive else ""
        sp = spec.system_prompt
        if isinstance(sp, str):
            return sp + suffix if suffix else sp
        append = (sp.append or "" if isinstance(sp, SystemPrompt) else "") + suffix
        preset: dict[str, Any] = {"type": "preset", "preset": "claude_code"}
        if append:
            preset["append"] = append
        return preset

    def _mcp_servers(self, spec: AgentSpec, ctx: RunContext) -> dict:
        """Translate ``spec.mcp_servers`` into SDK MCP config, pulling tokens from secrets."""
        from ..integrations.github import github_mcp_config

        servers: dict[str, Any] = {}
        for srv in spec.mcp_servers:
            if srv.kind == "github":
                token = next((ctx.secrets[k] for k in _GITHUB_MCP_TOKEN_KEYS if ctx.secrets.get(k)), None)
                if token:
                    servers["github"] = github_mcp_config(token)
                # else: no token resolved — skip silently (never log the token).
            elif srv.kind == "remote":
                cfg: dict[str, Any] = {"type": "http", "url": srv.url}
                if srv.headers:
                    cfg["headers"] = dict(srv.headers)
                servers[srv.name or "remote"] = cfg
            elif srv.kind == "stdio":
                cfg = {"type": "stdio", "command": srv.command}
                if srv.args:
                    cfg["args"] = list(srv.args)
                servers[srv.name or "stdio"] = cfg
        return servers

    def build_options(self, spec: AgentSpec, ctx: RunContext) -> Any:
        """Build ``ClaudeAgentOptions`` from ``spec`` + runtime ``ctx``."""
        from claude_agent_sdk import ClaudeAgentOptions

        if (spec.model or "").startswith("openrouter/"):
            # Fail closed rather than hand the id to the Claude CLI, which would try it
            # against the Anthropic API and fail with an opaque model error. OpenRouter
            # routing lives in the codex binding (harness="codex").
            raise ValueError(
                f"model {spec.model!r} routes through OpenRouter, which the claude-code "
                'harness cannot reach; run this turn with harness="codex"'
            )

        extra: dict[str, Any] = {}
        if ctx.session_store is not None:
            # Mirror the transcript to the store (flush at end) so this conversation is
            # resumable on another worker; resume_sid rehydrates a prior conversation.
            extra["session_store"] = ctx.session_store
            extra["session_store_flush"] = True
            if ctx.resume_sid:
                extra["resume"] = ctx.resume_sid
            elif ctx.session_id:
                # New conversation: pin the session id up front so checkpoint keying is
                # deterministic and never depends on scraping it from the message stream.
                extra["session_id"] = ctx.session_id

        allowed = (
            list(spec.allowed_tools)
            if spec.allowed_tools is not None
            else list(DEFAULT_ALLOWED_TOOLS)
        )
        if spec.disallowed_tools:
            extra["disallowed_tools"] = list(spec.disallowed_tools)
        if spec.reasoning_effort is not None:
            # Claude's scale has no minimal/none (Codex-only levels); low is the floor.
            # Set via `extra` so None never reaches an SDK without the kwarg.
            extra["effort"] = {"minimal": "low", "none": "low"}.get(
                spec.reasoning_effort, spec.reasoning_effort
            )

        return ClaudeAgentOptions(
            cwd=str(ctx.workspace),
            model=spec.model or None,
            allowed_tools=allowed,
            # Load project-level settings (.claude/ in the job cwd) so staged skills are
            # discovered, while staying isolated from the host's ~/.claude user settings.
            setting_sources=["project"],
            skills="all",
            permission_mode=spec.permission_mode,
            max_turns=spec.max_turns,
            max_budget_usd=spec.max_budget_usd,
            # The SDK's own default is 1 MiB per stdout message, and it raises from inside
            # the read loop when a message exceeds it — the turn ends with no result and
            # the run's work is lost (an in-context image Read is enough to trip it). See
            # `spec.DEFAULT_MAX_BUFFER_SIZE`.
            max_buffer_size=spec.max_buffer_size,
            env=runtime_env(spec, ctx),
            mcp_servers=self._mcp_servers(spec, ctx),
            # MCP config is loaded outside the setting sources, so `spec.mcp_servers` is
            # the only channel that reaches the agent: no project `.mcp.json`, no host
            # config, no plugin-provided server.
            strict_mcp_config=True,
            hooks=ctx.hooks,
            system_prompt=self._system_prompt(spec, ctx.interactive),
            **extra,
        )

    # -- run loop --------------------------------------------------------------

    def _finalize(self, spec: AgentSpec, ctx: RunContext) -> AgentEvent | None:
        """Checkpoint the workspace inline at the terminal result event (see ``_shared``)."""
        return finalize_checkpoint(spec, ctx)

    def _final_result(
        self, event: AgentEvent, turns_total: int, segment_summaries: Sequence[str] = ()
    ) -> AgentEvent:
        """Stamp the cumulative turn count onto the turn's final result event."""
        if event.raw is not None:
            event.raw["num_turns"] = turns_total
            if segment_summaries:
                event.raw["segment_summaries"] = list(segment_summaries)
        return event

    async def run(self, spec: AgentSpec, ctx: RunContext) -> AsyncIterator[AgentEvent]:
        """Drive one turn over a ``ClaudeSDKClient`` stream, yielding ``AgentEvent``s.

        Background-task semantics are HONORED (DESIGN.md §7): when the model ends its
        turn with background tasks (Bash ``run_in_background``, Monitor) still running,
        the stream stays open — the CLI re-invokes the model itself when a task reaches
        a terminal state (proven live; the SDK's one-shot ``query()`` instead tears the
        CLI down at the first result, firing those advertised notifications into the
        void). Interim results are demoted to ``awaiting_tasks`` status events; the turn
        ends at a result with no pending work, where the checkpoint finalizes inline
        (before the trailing checkpoint status event, so the snapshot happens while a
        non-draining executor is still pulling). ``spec.background_task_timeout`` bounds
        the wait; on expiry the last produced result stands.

        A deliverable a demoted result carried is not lost to the demotion: the final
        result rides with the demoted results' schema-valid texts
        (``raw["segment_summaries"]``, newest first), so ``build_result`` can recover
        the structured output when the model's reply to a stale task notification
        displaced it from the turn's final message.
        """
        from claude_agent_sdk import ClaudeSDKClient

        from .translate import EventTranslator

        options = self.build_options(spec, ctx)
        # Capture the CLI's stderr to <job_dir>/stderr.log + an in-memory tail; without a
        # registered callback the SDK doesn't pipe it at all and CLI-exit failures are
        # undiagnosable by construction (the ProcessError text promises stderr details).
        stderr_log = _StderrCapture(ctx.job_dir / "stderr.log")
        options.stderr = stderr_log
        translator = EventTranslator()
        tracker = _TaskTracker()
        finalized = False
        # Every demoted (segment-boundary) result, oldest first; the newest is also the
        # crash/timeout fallback result when the turn never reaches a clean final one.
        demoted: list[AgentEvent] = []
        turns_total = 0  # num_turns resets per re-invocation; RunResult reports the sum
        wait_deadline: float | None = None
        phase = _StreamPhase.ACTIVE  # the initial query has been submitted to the model

        def segment_summaries(final_ev: AgentEvent) -> list[str]:
            # Recovery candidates carried on the final result for build_result: each
            # demoted result was the agent's intended end-of-turn answer until a
            # background-task notification re-invoked the model past it, and the reply
            # to a stale notification can displace a schema-valid deliverable from the
            # turn's final text. Newest first, capped to keep the persisted result
            # event small — and filtered with the reader's own schema-validating parse,
            # so the cap can never fill with junk JSON (e.g. a quoted API response) and
            # evict a real answer.
            if spec.output_schema is None:
                return []
            out: list[str] = []
            for ev in reversed(demoted):
                if ev is final_ev or not ev.summary:
                    continue
                if parse_structured_output(ev.summary, spec.output_schema) is None:
                    continue
                out.append(ev.summary)
                if len(out) >= self._SEGMENT_FALLBACK_MAX:
                    break
            return out

        client = ClaudeSDKClient(options=options)
        try:
            await client.connect()
            await client.query(ctx.prompt)
            stream = client.receive_messages()
            while True:
                if not demoted or phase is _StreamPhase.ACTIVE:
                    timeout = None  # model working; a foreground tool call may run long
                elif tracker.waiting() == "pending" and wait_deadline is not None:
                    timeout = max(1.0, wait_deadline - asyncio.get_running_loop().time())
                else:  # terminal notification observed; re-invocation due momentarily
                    timeout = self._UNDELIVERED_GRACE_S
                try:
                    message = await asyncio.wait_for(anext(stream), timeout)
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    reason = tracker.waiting()
                    if reason == "pending":
                        summary = (
                            "gave up waiting on background tasks "
                            f"({len(tracker.pending)} still pending); finalizing with the "
                            "result already produced"
                        )
                    else:
                        summary = (
                            "gave up waiting for Claude Code to re-invoke after a background "
                            "task notification; finalizing with the result already produced"
                        )
                    yield AgentEvent(
                        kind="status",
                        summary=summary,
                        raw={
                            "event": "task_wait_timeout",
                            "reason": reason,
                            "pending": tracker.pending,
                        },
                    )
                    break
                for event in translator.translate(message):
                    tracker.observe(event)
                    if (event.raw or {}).get("subtype") == "init":
                        # The notification grace ends as soon as the CLI starts the next
                        # invocation. Keep the demoted result only as a crash fallback; it
                        # must not impose a per-message timeout on an actively working model.
                        phase = _StreamPhase.ACTIVE
                        wait_deadline = None
                    if event.kind != "result":
                        yield event
                        continue
                    turns_total += int((event.raw or {}).get("num_turns") or 0)
                    reason = None if (event.raw or {}).get("is_error") else tracker.waiting()
                    if reason is None:
                        fin = self._finalize(spec, ctx)
                        finalized = True
                        yield self._final_result(event, turns_total, segment_summaries(event))
                        if fin is not None:
                            yield fin
                        return
                    # Segment boundary, not the end of the turn: hold the stream open
                    # for the CLI's task-completion re-invocation.
                    demoted.append(event)
                    phase = _StreamPhase.BETWEEN_INVOCATIONS
                    if reason == "pending":
                        wait_deadline = (
                            asyncio.get_running_loop().time() + spec.background_task_timeout
                        )
                    else:
                        wait_deadline = None
                    yield AgentEvent(
                        kind="status",
                        summary=(
                            f"turn paused awaiting background tasks ({reason}: "
                            f"{len(tracker.pending) or len(tracker.undelivered)})"
                        ),
                        raw={
                            "event": "awaiting_tasks",
                            "reason": reason,
                            "pending": tracker.pending,
                            "undelivered": sorted(tracker.undelivered),
                        },
                    )
        except Exception as exc:
            # Surface the captured stderr with the failure: as a status event (reaches the
            # stream / Cloud Logging on gemini) and embedded in the raised error, so a
            # non-zero CLI exit is classifiable post-mortem instead of "check stderr".
            tail = stderr_log.tail()
            if tail:
                yield AgentEvent(
                    kind="status",
                    summary=f"claude stderr (tail): {tail}",
                    raw={"event": "claude_stderr", "log": str(stderr_log.path)},
                )
            if demoted and not finalized:
                # A real result exists — the crash while waiting on tasks must not void
                # it (same principle as the runtimes' late-harness-death handling).
                yield AgentEvent(
                    kind="status",
                    summary=f"harness failed while awaiting background tasks: {str(exc)[:200]}",
                    raw={"event": "late_harness_error"},
                )
                fin = self._finalize(spec, ctx)
                finalized = True
                last = demoted[-1]
                yield self._final_result(last, turns_total, segment_summaries(last))
                if fin is not None:
                    yield fin
                return
            if not tail:
                raise
            raise RuntimeError(
                f"{exc} [claude stderr tail: {tail[-500:]}] (full log: {stderr_log.path})"
            ) from exc
        finally:
            try:
                await client.disconnect()
            except Exception:  # noqa: BLE001 — teardown must not mask the run's outcome
                pass

        # Stream ended / wait timed out without a clean final result: the last produced
        # result stands (then the defensive no-result fallback, as before).
        if demoted and not finalized:
            fin = self._finalize(spec, ctx)
            finalized = True
            last = demoted[-1]
            yield self._final_result(last, turns_total, segment_summaries(last))
            if fin is not None:
                yield fin
        elif not finalized:
            fin = self._finalize(spec, ctx)
            if fin is not None:
                yield fin

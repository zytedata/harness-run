"""``ClaudeCodeHarness`` — the Claude Agent SDK binding (DESIGN.md §7, §8).

Drives ``claude_agent_sdk.query()`` against the per-run working directory and translates
each streamed SDK message into a generic :class:`AgentEvent` (via :mod:`translate`), so
the rest of the toolkit — and every runtime (``local`` / ``gemini``) — stays harness-
agnostic. This is the generalized, ADK-free core of the PoC ``ClaudeCodeAgent``: the
ADK / Agent Engine wrapping is a ``gemini``-side concern, not the harness's.

``claude_agent_sdk`` is imported lazily inside methods, so importing this module needs no
third-party deps.

OpenRouter models
-----------------

An ``openrouter/<vendor>/<model>`` id runs on this harness too. OpenRouter serves an
Anthropic-compatible endpoint, so the CLI can talk to it (see :func:`_openrouter_env`).

Those turns go through the toolkit's own proxy (:mod:`._openrouter_proxy`).
:meth:`ClaudeCodeHarness.run` starts one per turn and closes it at the end. The CLI is
pointed at ``127.0.0.1`` and gets a random token, so the OpenRouter key never leaves this
process. What the proxy sees is the only record of what OpenRouter did:

* ``spec.openrouter_provider`` reaches the request body through it. The CLI has no field
  for it.
* Each response becomes an ``openrouter_request`` event: selected provider, model, region,
  HTTP status, and the charge. The result reports the summed charges as ``cost_usd`` with
  ``price_source="openrouter"``. The CLI's own estimate is wrong here, because it prices
  these ids from a catalogue that has no entry for them. It is kept as
  ``cli_reported_cost_usd``.
* ``max_budget_usd`` is checked against that total between responses. If a request gets
  through anyway, the proxy answers it with 402. Either way the turn ends as
  ``error_budget_exceeded``.

A turn that ends with no final assistant message becomes ``error_no_final_text``. Some
model and harness pairings finish cleanly at the protocol level without answering.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections import deque
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Sequence, TYPE_CHECKING

from ..events import AgentEvent
from ..spec import SystemPrompt
from ..structured import parse_structured_output
from . import pricing
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

# OpenRouter support (see "OpenRouter models" in the module docstring). OpenRouter serves an
# Anthropic-compatible endpoint, so the Claude CLI can talk to it: point it at the base URL,
# hand it the OpenRouter key as a bearer token, and register the model as a custom catalogue
# entry so the CLI stops rejecting the id as unrecognized.
_OPENROUTER_PREFIX = "openrouter/"
_OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"
# The Anthropic-compatible base: the CLI appends /v1/messages itself.
_OPENROUTER_ANTHROPIC_BASE = "https://openrouter.ai/api"
_OPENROUTER_SHELL_WRAPPER = """#!/bin/sh
unset ANTHROPIC_AUTH_TOKEN OPENROUTER_API_KEY
exec /bin/bash -c "$1"
"""


# OpenRouter receives the SDK's native output format, but its selected model may still be
# responsible for following it. Measured: GLM-5.3 answered "`answer`: 42" prose with
# `output_schema` set. Stating the schema in the prompt produced a bare JSON object, so the
# OpenRouter path uses both forms, matching the Codex binding.
_OPENROUTER_SCHEMA_INSTRUCTION = (
    "\n\nFINAL MESSAGE FORMAT: your last message of the turn must be ONLY a single JSON "
    "object conforming to this schema — no prose before or after it, no code fence, no "
    "explanation:\n{schema}"
)


def _openrouter_cost_unknown(model: str | None) -> AgentEvent:
    return AgentEvent(
        kind="status",
        summary=(
            f"OpenRouter did not report a charge for model {model!r}; cost_usd is "
            "unknown and max_budget_usd could not be enforced"
        ),
        raw={"event": "cost_unknown", "model": model},
    )


def _openrouter_env(
    model: str,
    api_key: str,
    shell_wrapper: Path,
    base_url: str = _OPENROUTER_ANTHROPIC_BASE,
) -> dict[str, str]:
    """Env that points the Claude CLI at OpenRouter for ``model``.

    ``ANTHROPIC_CUSTOM_MODEL_OPTION`` is what lets the real id through: without it the CLI
    refuses the turn with ``unrecognized_model``. The Vertex/Bedrock/Foundry switches and
    ``ANTHROPIC_API_KEY`` are blanked because they outrank ``ANTHROPIC_AUTH_TOKEN`` in the
    CLI's auth order, and a deployed engine bakes the Vertex ones in.

    ``api_key`` is the proxy's per-run token, not the account key: ``run`` always starts a
    proxy when a key is available. The default ``base_url`` is only reachable by a caller
    that builds options directly.
    """
    bare = model[len(_OPENROUTER_PREFIX) :]
    env = {
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_AUTH_TOKEN": api_key,
        "ANTHROPIC_CUSTOM_MODEL_OPTION": bare,
        "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME": bare,
        # The CLI needs the token, while Bash tools do not. This executable receives the
        # original shell command as its sole argument and removes auth before running it.
        "CLAUDE_CODE_SHELL_PREFIX": str(shell_wrapper),
        # The SDK layers this env OVER the worker's own, so an ambient OpenRouter key
        # would otherwise reach the CLI process even though the proxy holds the real one.
        _OPENROUTER_KEY_ENV: "",
        "ANTHROPIC_API_KEY": "",
        "CLAUDE_CODE_USE_VERTEX": "",
        "CLAUDE_CODE_USE_BEDROCK": "",
        "CLAUDE_CODE_USE_FOUNDRY": "",
    }
    window = pricing.OPENROUTER_CONTEXT_WINDOWS.get(model)
    if window is not None:
        # Claude Code otherwise assumes 200k for an unknown model and compacts early.
        env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(window)
    return env


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

    def _schema_steer(self, spec: AgentSpec) -> str:
        """Prompt text asking for a bare JSON object, for OpenRouter models only."""
        if not (spec.model or "").startswith(_OPENROUTER_PREFIX) or spec.output_schema is None:
            return ""
        from ..spec import _output_schema_to_dict

        schema = _output_schema_to_dict(spec.output_schema)
        if not isinstance(schema, dict):
            return ""
        return _OPENROUTER_SCHEMA_INSTRUCTION.format(
            schema=json.dumps(schema, separators=(",", ":"))
        )

    @staticmethod
    def _output_format(spec: AgentSpec) -> dict[str, Any] | None:
        """Map the public output schema to the Claude Agent SDK's native option."""
        if spec.output_schema is None:
            return None
        from ..spec import _output_schema_to_dict

        schema = _output_schema_to_dict(spec.output_schema)
        return {"type": "json_schema", "schema": schema} if isinstance(schema, dict) else None

    def _system_prompt(self, spec: AgentSpec, interactive: bool) -> Any:
        """Resolve ``spec.system_prompt`` (+ interactive suffix) to an SDK value.

        * ``SystemPrompt`` → inherit Claude Code's preset and append.
        * plain ``str``    → replace the prompt entirely (suffix appended in interactive).
        * ``None``         → Claude Code's preset. The SDK's own ``None`` means an *empty*
          system prompt (``--system-prompt ""``): no preset, no ``CLAUDE.md``, no
          environment block, so the agent would not be Claude Code at all.
        """
        suffix = (_INTERACTIVE_SUFFIX if interactive else "") + self._schema_steer(spec)
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
                token = next(
                    (ctx.secrets[k] for k in _GITHUB_MCP_TOKEN_KEYS if ctx.secrets.get(k)), None
                )
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

    def build_options(
        self,
        spec: AgentSpec,
        ctx: RunContext,
        *,
        openrouter_base_url: str | None = None,
        openrouter_client_token: str | None = None,
    ) -> Any:
        """Build ``ClaudeAgentOptions`` from ``spec`` + runtime ``ctx``."""
        from claude_agent_sdk import ClaudeAgentOptions

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

        model = spec.model or ""
        openrouter = model.startswith(_OPENROUTER_PREFIX)
        env = runtime_env(spec, ctx)
        if openrouter:
            api_key = ctx.secrets.get(_OPENROUTER_KEY_ENV) or os.environ.get(_OPENROUTER_KEY_ENV)
            if not api_key:
                raise RuntimeError(
                    "claude-code harness has no OpenRouter credentials: pass "
                    f"{_OPENROUTER_KEY_ENV} in the per-invocation secrets (or set it in the "
                    "environment)"
                )
            ctx.job_dir.mkdir(parents=True, exist_ok=True)
            shell_wrapper = ctx.job_dir / "openrouter-shell"
            shell_wrapper.write_text(_OPENROUTER_SHELL_WRAPPER, encoding="utf-8")
            shell_wrapper.chmod(0o700)
            env.update(
                _openrouter_env(
                    model,
                    openrouter_client_token or api_key,
                    shell_wrapper,
                    openrouter_base_url or _OPENROUTER_ANTHROPIC_BASE,
                )
            )
            # The CLI wants the provider-relative id; the prefix is the toolkit's own.
            model = model[len(_OPENROUTER_PREFIX) :]

        return ClaudeAgentOptions(
            cwd=str(ctx.workspace),
            model=model or None,
            allowed_tools=allowed,
            # Load project-level settings (.claude/ in the job cwd) so staged skills are
            # discovered, while staying isolated from the host's ~/.claude user settings.
            setting_sources=["project"],
            skills="all",
            permission_mode=spec.permission_mode,
            max_turns=spec.max_turns,
            # Claude Code prices unknown models with its own fallback rates. OpenRouter
            # can differ by orders of magnitude, so its native cap would stop valid runs.
            # `_final_result` checks OpenRouter's exact streamed charge instead.
            max_budget_usd=None if openrouter else spec.max_budget_usd,
            # The SDK's own default is 1 MiB per stdout message, and it raises from inside
            # the read loop when a message exceeds it — the turn ends with no result and
            # the run's work is lost (an in-context image Read is enough to trip it). See
            # `spec.DEFAULT_MAX_BUFFER_SIZE`.
            max_buffer_size=spec.max_buffer_size,
            env=env,
            mcp_servers=self._mcp_servers(spec, ctx),
            # MCP config is loaded outside the setting sources, so `spec.mcp_servers` is
            # the only channel that reaches the agent: no project `.mcp.json`, no host
            # config, no plugin-provided server.
            strict_mcp_config=True,
            hooks=ctx.hooks,
            system_prompt=self._system_prompt(spec, ctx.interactive),
            output_format=self._output_format(spec),
            **extra,
        )

    # -- run loop --------------------------------------------------------------

    def _finalize(self, spec: AgentSpec, ctx: RunContext) -> AgentEvent | None:
        """Checkpoint the workspace inline at the terminal result event (see ``_shared``)."""
        return finalize_checkpoint(spec, ctx)

    def _final_result(
        self,
        event: AgentEvent,
        turns_total: int,
        segment_summaries: Sequence[str] = (),
        openrouter: bool = False,
        exact_openrouter_cost: float | None = None,
        max_budget_usd: float | None = None,
        budget_blocked: bool = False,
    ) -> AgentEvent:
        """Stamp the cumulative turn count onto the turn's final result event.

        On an ``openrouter`` turn the cost is the charge OpenRouter reported, or nothing.
        The CLI's own ``total_cost_usd`` cannot be trusted here — it prices from its
        catalogue, which has no entry for these ids, so it bills them at a default rate
        (measured 1.67x over Kimi K3's real rate) — and there is no estimate to fall back
        on (see :mod:`pricing`). It is kept as ``cli_reported_cost_usd`` either way.
        """
        if event.raw is not None:
            event.raw["num_turns"] = turns_total
            if segment_summaries:
                event.raw["segment_summaries"] = list(segment_summaries)
        if exact_openrouter_cost is not None:
            if event.raw is not None:
                event.raw["cli_reported_cost_usd"] = event.cost_usd
                event.raw["price_source"] = "openrouter"
            event.cost_usd = exact_openrouter_cost
        elif openrouter:
            # OpenRouter reported no charge for this turn. Report nothing rather than
            # something wrong — same contract as the codex binding.
            if event.raw is not None:
                event.raw["cli_reported_cost_usd"] = event.cost_usd
            event.cost_usd = None
        if (
            max_budget_usd is not None
            and event.summary == "(no final text)"
            and event.raw is not None
            and not event.raw.get("is_error")
        ):
            # Some OpenRouter model/harness pairs end successfully at the protocol
            # level without returning the requested answer. Surface that as a failed
            # run so callers can retry or select another pairing.
            event.raw["cli_reported_subtype"] = event.raw.get("subtype")
            event.raw["subtype"] = "error_no_final_text"
            event.raw["is_error"] = True
        if (
            max_budget_usd is not None
            and event.cost_usd is not None
            and event.cost_usd >= max_budget_usd
            and event.raw is not None
        ):
            # The exact cost is known after a model response. This can exceed the cap by
            # one response, but it prevents a wrong CLI price from stopping much earlier.
            event.raw["cli_reported_subtype"] = event.raw.get("subtype")
            event.raw["subtype"] = "error_budget_exceeded"
            event.raw["is_error"] = True
            event.raw["budget_enforcement"] = "after_model_response"
        elif budget_blocked and event.raw is not None:
            # The proxy refused a request with 402 after the cap was spent. Whatever the
            # CLI made of that error, the run stopped because of the budget.
            event.raw["cli_reported_subtype"] = event.raw.get("subtype")
            event.raw["subtype"] = "error_budget_exceeded"
            event.raw["is_error"] = True
            event.raw["budget_enforcement"] = "proxy_402"
        return event

    async def run(self, spec: AgentSpec, ctx: RunContext) -> AsyncIterator[AgentEvent]:
        """Run one Claude Code turn, with a local metadata proxy for OpenRouter turns."""
        from ._openrouter_proxy import OpenRouterProxy

        proxy = None
        if (spec.model or "").startswith(_OPENROUTER_PREFIX):
            api_key = ctx.secrets.get(_OPENROUTER_KEY_ENV) or os.environ.get(_OPENROUTER_KEY_ENV)
            if api_key:
                model = (spec.model or "").removeprefix(_OPENROUTER_PREFIX)
                proxy = OpenRouterProxy(
                    api_key,
                    spec.max_budget_usd,
                    model,
                    provider=spec.openrouter_provider,
                ).start()
        try:
            async for event in self._run(spec, ctx, proxy):
                yield event
        finally:
            if proxy is not None:
                proxy.close()

    async def _run(
        self, spec: AgentSpec, ctx: RunContext, proxy: Any | None
    ) -> AsyncIterator[AgentEvent]:
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

        options = self.build_options(
            spec,
            ctx,
            openrouter_base_url=(f"{proxy.base_url}/api" if proxy is not None else None),
            openrouter_client_token=(proxy.client_token if proxy is not None else None),
        )
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
        budget_interrupted = False
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

        # An OpenRouter turn's cost is the proxy's record of what OpenRouter charged, and
        # nothing else (see `_final_result`).
        is_openrouter = (spec.model or "").startswith(_OPENROUTER_PREFIX)

        def proxy_cost_usd() -> float | None:
            return proxy.exact_cost_usd if proxy is not None else None

        async def drain_proxy() -> AsyncIterator[AgentEvent]:
            """Let the proxy finish recording before the result quotes its total.

            The CLI sees the last response bytes before the proxy's handler has parsed
            them, so a result built immediately can miss the final request's charge.
            """
            if proxy is None:
                return
            if not await asyncio.to_thread(proxy.wait_until_idle, 5.0):
                yield AgentEvent(
                    kind="status",
                    summary="timed out waiting for OpenRouter response metadata",
                    raw={"event": "openrouter_metadata_timeout"},
                )
            for event in proxy.drain_events():
                yield event

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
                if proxy is not None:
                    for proxy_event in proxy.drain_events():
                        yield proxy_event
                translated = list(translator.translate(message))
                for event in translated:
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
                        async for proxy_event in drain_proxy():
                            yield proxy_event
                        if is_openrouter and proxy_cost_usd() is None:
                            yield _openrouter_cost_unknown(spec.model)
                        yield self._final_result(
                            event,
                            turns_total,
                            segment_summaries(event),
                            is_openrouter,
                            proxy_cost_usd(),
                            spec.max_budget_usd if is_openrouter else None,
                            proxy.budget_blocked if proxy is not None else False,
                        )
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
                if (
                    is_openrouter
                    and not budget_interrupted
                    and proxy_cost_usd() is not None
                    and proxy_cost_usd() >= spec.max_budget_usd
                ):
                    budget_interrupted = True
                    yield AgentEvent(
                        kind="status",
                        summary=(
                            "max_budget_usd reached after an OpenRouter response; "
                            "interrupting before another model request"
                        ),
                        raw={"event": "limit_interrupt", "limit": "budget"},
                    )
                    try:
                        await client.interrupt()
                    except Exception:  # noqa: BLE001 — the final result still records the cap
                        pass
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
                async for proxy_event in drain_proxy():
                    yield proxy_event
                if is_openrouter and proxy_cost_usd() is None:
                    yield _openrouter_cost_unknown(spec.model)
                yield self._final_result(
                    last,
                    turns_total,
                    segment_summaries(last),
                    is_openrouter,
                    proxy_cost_usd(),
                    spec.max_budget_usd if is_openrouter else None,
                    proxy.budget_blocked if proxy is not None else False,
                )
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
            async for proxy_event in drain_proxy():
                yield proxy_event
            if is_openrouter and proxy_cost_usd() is None:
                yield _openrouter_cost_unknown(spec.model)
            yield self._final_result(
                last,
                turns_total,
                segment_summaries(last),
                is_openrouter,
                proxy_cost_usd(),
                spec.max_budget_usd if is_openrouter else None,
                proxy.budget_blocked if proxy is not None else False,
            )
            if fin is not None:
                yield fin
        elif not finalized:
            fin = self._finalize(spec, ctx)
            if fin is not None:
                yield fin

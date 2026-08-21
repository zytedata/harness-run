"""``CodexHarness`` — the OpenAI Codex SDK binding (DESIGN.md §7).

Drives an ``openai_codex.AsyncCodex`` app-server against the per-run working directory
and translates each streamed notification into a generic :class:`AgentEvent`, so specs
that select ``harness="codex"`` run through the exact same Engine/Session/Run surface
(both runtimes) as the Claude binding. ``openai_codex`` is imported lazily inside
methods, so importing this module needs no third-party deps.

Spec translation (parity notes):

* ``model``            → thread model (e.g. ``gpt-5.6-luna``); auth is the per-invocation
                         ``OPENAI_API_KEY`` secret (or ambient env), routed to ``codex
                         login`` inside the per-job ``CODEX_HOME`` — the OpenAI API is
                         called directly (there is no Vertex path for OpenAI models).
* ``permission_mode``  → sandbox + approval policy (see ``_PERMISSION_MAP``).
* ``system_prompt``    → plain ``str`` replaces Codex's base instructions;
                         ``SystemPrompt.append`` becomes developer instructions.
* ``skills``           → staged by the runtimes into ``<cwd>/.agents/skills`` (Codex's
                         repo-level discovery path; see ``skills.skills_subdir``).
* ``mcp_servers``      → ``--config mcp_servers.*`` overrides; a ``github`` server's
                         token rides an env var (``bearer_token_env_var``), never argv.
* ``max_turns``        → enforced by the harness: each ``thread/tokenUsage/updated``
                         notification is one model call; the turn is interrupted at the
                         cap (Codex has no native turn cap).
* ``max_budget_usd``   → enforced by the harness from token usage × the model's price
                         (Codex reports no USD). Prices resolve via :mod:`pricing` —
                         LiteLLM's live dataset first, a baked fallback offline; a model
                         unknown to both runs uncapped with a status warning.
* ``output_schema``    → per-turn ``output_schema`` (the final message is the JSON).
* ``reasoning_effort`` → per-turn ``effort`` (thread-sticky server-side, and re-applied
                         on every turn, so resume keeps it). Codex has no ``max`` level;
                         it is mapped to ``xhigh`` with a status warning.
* ``allowed_tools`` / ``disallowed_tools`` → no Codex equivalent; ignored with a status
                         warning.
* ``max_buffer_size``  → inert: it caps one NDJSON message on the Claude Agent SDK's own
                         stdout transport, and the Codex app-server SDK frames its stream
                         itself with no equivalent knob.
* background tasks     → no Codex equivalent of Claude Code's task re-invocation
                         (``spec.background_task_timeout`` is inert here).
* checkpoint/resume    → the workspace snapshot is shared machinery; the conversation
                         itself is persisted by copying the thread's rollout file to the
                         BlobStore and restoring it into ``CODEX_HOME`` before
                         ``thread_resume`` (Codex's own session store is a local file).
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, TYPE_CHECKING

from ..events import AgentEvent
from ..spec import SystemPrompt
from . import pricing
from ._shared import (
    GITHUB_MCP_TOKEN_KEYS,
    INTERACTIVE_SUFFIX,
    finalize_checkpoint,
    runtime_env,
)

if TYPE_CHECKING:
    from ..spec import AgentSpec
    from .context import RunContext

# spec.permission_mode → (Sandbox enum name, ApprovalMode enum name). Claude's modes are
# the spec's vocabulary; this is the closest Codex semantics for each:
# bypassPermissions (unattended default) = no sandbox, never ask; acceptEdits = write the
# workspace freely but stay sandboxed; default = sandboxed with Codex's auto-reviewer
# resolving escalations (headless runs must never block on a human); plan = read-only.
_PERMISSION_MAP = {
    "bypassPermissions": ("full_access", "deny_all"),
    "acceptEdits": ("workspace_write", "deny_all"),
    "default": ("workspace_write", "auto_review"),
    "plan": ("read_only", "auto_review"),
}

# The env var name the github MCP bearer token rides (config references the NAME; the
# value goes into the codex process env — never onto the argv-visible --config flags).
_GITHUB_MCP_TOKEN_ENV = "RATK_GITHUB_MCP_TOKEN"
_GITHUB_MCP_URL = "https://api.githubcopilot.com/mcp/"

# Tool-result content kept in events is truncated: command output can be megabytes, and
# events ride Cloud Logging on gemini (per-entry size limits).
_CONTENT_CAP = 4000

# Blob-key prefix for persisted Codex conversations (the rollout file + thread id).
_THREADS_PREFIX = "codex-threads"


@dataclass
class _CodexOptions:
    """Everything ``run`` needs, resolved from ``(spec, ctx)`` by ``build_options``."""

    codex_config: Any  # openai_codex.CodexConfig
    thread_args: dict[str, Any]
    run_args: dict[str, Any]
    api_key: str | None
    codex_home: Path
    warnings: list[str] = field(default_factory=list)


class _RunAccounting:
    """Per-run cost/turn accounting from ``thread/tokenUsage/updated`` notifications.

    Each notification is one model call (verified live), so their count is the Codex
    equivalent of Claude's ``num_turns``. Per-run token sums come from the ``last``
    breakdown of each call — the ``total`` breakdown is *thread*-cumulative and would
    double-bill resumed sessions. ``price`` comes from :mod:`pricing` (LiteLLM live
    dataset, baked fallback); ``None`` means cost is unreportable and the budget cap
    unenforceable.
    """

    def __init__(self, model: str, price: pricing.ModelPrice | None) -> None:
        self.model = model
        self.price = price
        self.model_calls = 0
        self.input_tokens = 0
        self.cached_input_tokens = 0
        self.output_tokens = 0
        self.reasoning_output_tokens = 0

    def observe(self, last: Any) -> None:
        self.model_calls += 1
        self.input_tokens += int(last.input_tokens or 0)
        self.cached_input_tokens += int(last.cached_input_tokens or 0)
        self.output_tokens += int(last.output_tokens or 0)
        self.reasoning_output_tokens += int(last.reasoning_output_tokens or 0)

    @property
    def cost_usd(self) -> float | None:
        if self.price is None:
            return None
        return self.price.cost_usd(
            self.input_tokens, self.cached_input_tokens, self.output_tokens
        )

    def usage(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_output_tokens": self.reasoning_output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
        }


class CodexEventTranslator:
    """Codex notification → generic ``AgentEvent`` translation.

    Mapping (notification → ``AgentEvent.kind``):

    * ``turn/started``                       → ``status`` (raw ``subtype: init``)
    * ``item/started`` (tool-ish items)      → ``tool_use``   (command/fileChange/mcp/web)
    * ``item/completed`` (tool-ish items)    → ``tool_result`` (exit code, capped output)
    * ``item/completed`` ``agentMessage``    → ``message`` (also remembered: the last
      final-phase message is the turn's final response — ``turn.items`` arrives with
      empty ``text`` fields, so the result text must be collected from these)
    * ``item/completed`` ``reasoning``       → ``thinking``
    * ``error``                              → ``status`` (raw ``event: codex_error``)
    * everything else (deltas, diffs, token usage, ``turn/completed``) → nothing here;
      accounting and the terminal result are the run loop's job.
    """

    def __init__(self) -> None:
        self.final_text = ""
        self._final_is_final_phase = False

    def _remember_final(self, item: Any) -> None:
        phase = getattr(item, "phase", None)
        is_final = getattr(phase, "value", phase) == "final_answer"
        # Prefer the last explicit final-answer message; else the last plain message.
        if is_final or not self._final_is_final_phase:
            self.final_text = item.text or ""
            self._final_is_final_phase = is_final

    def translate(self, notification: Any) -> Iterator[AgentEvent]:
        method = notification.method
        payload = notification.payload
        if method == "turn/started":
            yield AgentEvent(
                kind="status",
                summary="codex turn started",
                raw={"subtype": "init"},
            )
            return
        if method == "error":
            err = payload.error
            yield AgentEvent(
                kind="status",
                summary=f"codex error: {err.message}"[:300],
                raw={
                    "event": "codex_error",
                    "message": str(err.message)[:2000],
                    "will_retry": bool(getattr(payload, "will_retry", False)),
                },
            )
            return
        if method not in ("item/started", "item/completed"):
            return
        item = payload.item.root
        kind = item.type
        started = method == "item/started"
        if kind == "agentMessage":
            if not started and (item.text or "").strip():
                self._remember_final(item)
                yield AgentEvent(kind="message", summary=item.text)
        elif kind == "reasoning":
            if not started:
                parts = list(getattr(item, "summary", None) or [])
                parts += list(getattr(item, "content", None) or [])
                text = "\n".join(str(p) for p in parts if p)
                if text.strip():
                    yield AgentEvent(kind="thinking", summary=text)
        elif kind == "commandExecution":
            if started:
                yield AgentEvent(
                    kind="tool_use",
                    summary=f"CommandExecution {' '.join((item.command or '').split())[:160]}",
                    raw={
                        "tool": "commandExecution",
                        "id": item.id,
                        "input": {"command": item.command, "cwd": item.cwd},
                    },
                )
            else:
                failed = getattr(item.status, "value", item.status) == "failed" or (
                    item.exit_code not in (0, None)
                )
                yield AgentEvent(
                    kind="tool_result",
                    summary=f"CommandExecution{' (error)' if failed else ''}",
                    raw={
                        "tool": "commandExecution",
                        "id": item.id,
                        "is_error": bool(failed),
                        "exit_code": item.exit_code,
                        "content": (item.aggregated_output or "")[-_CONTENT_CAP:],
                    },
                )
        elif kind == "fileChange":
            paths = [c.path for c in (item.changes or [])]
            if started:
                yield AgentEvent(
                    kind="tool_use",
                    summary=f"FileChange {' '.join(paths)[:160]}",
                    raw={"tool": "fileChange", "id": item.id, "input": {"paths": paths}},
                )
            else:
                failed = getattr(item.status, "value", item.status) == "failed"
                yield AgentEvent(
                    kind="tool_result",
                    summary=f"FileChange{' (error)' if failed else ''}",
                    raw={
                        "tool": "fileChange",
                        "id": item.id,
                        "is_error": bool(failed),
                        "content": paths,
                    },
                )
        elif kind == "mcpToolCall":
            name = f"mcp__{item.server}__{item.tool}"
            if started:
                yield AgentEvent(
                    kind="tool_use",
                    summary=name,
                    raw={"tool": name, "id": item.id, "input": item.arguments},
                )
            else:
                failed = getattr(item.status, "value", item.status) == "failed" or bool(item.error)
                content = item.error or item.result
                yield AgentEvent(
                    kind="tool_result",
                    summary=f"{name}{' (error)' if failed else ''}",
                    raw={
                        "tool": name,
                        "id": item.id,
                        "is_error": bool(failed),
                        "content": str(content)[:_CONTENT_CAP] if content is not None else None,
                    },
                )
        elif kind == "webSearch":
            if started:
                yield AgentEvent(
                    kind="tool_use",
                    summary=f"WebSearch {item.query or ''}"[:160],
                    raw={"tool": "webSearch", "id": item.id, "input": {"query": item.query}},
                )
            else:
                yield AgentEvent(
                    kind="tool_result",
                    summary="WebSearch",
                    raw={"tool": "webSearch", "id": item.id, "is_error": False,
                         "content": item.query},
                )
        elif kind == "userMessage":
            pass  # the echo of our own prompt
        elif not started:
            # Unmapped item kinds (plan updates, sub-agent activity, …): surfaced as a
            # status so nothing disappears silently from the stream.
            yield AgentEvent(
                kind="status",
                summary=f"codex item: {kind}",
                raw={"event": "codex_item", "item_type": kind, "id": getattr(item, "id", None)},
            )


class CodexHarness:
    """The OpenAI Codex SDK harness (implements ``harness.base.Harness``)."""

    # -- option building -------------------------------------------------------

    def _instructions(self, spec: AgentSpec, interactive: bool) -> dict[str, Any]:
        """Resolve ``spec.system_prompt`` (+ interactive suffix) to thread args.

        * plain ``str``    → ``base_instructions`` (replace Codex's prompt entirely).
        * ``SystemPrompt`` → ``developer_instructions`` (inherit + append).
        * ``None``         → Codex default (+ developer instructions if interactive).
        """
        suffix = INTERACTIVE_SUFFIX if interactive else ""
        sp = spec.system_prompt
        args: dict[str, Any] = {}
        if isinstance(sp, str):
            args["base_instructions"] = sp + suffix
        elif isinstance(sp, SystemPrompt):
            append = (sp.append or "") + suffix
            if append:
                args["developer_instructions"] = append
        elif suffix:
            args["developer_instructions"] = suffix
        return args

    def _mcp_overrides(
        self, spec: AgentSpec, ctx: RunContext
    ) -> tuple[list[str], dict[str, str]]:
        """Translate ``spec.mcp_servers`` into ``--config`` overrides (+ env for tokens).

        Codex has no programmatic MCP registration; servers are config entries. Values
        land on the spawned process's argv, so secrets never go here — a github server's
        bearer token is referenced by env var NAME and its value returned separately for
        the process env.
        """
        overrides: list[str] = []
        env: dict[str, str] = {}
        for srv in spec.mcp_servers:
            if srv.kind == "github":
                token = next(
                    (ctx.secrets[k] for k in GITHUB_MCP_TOKEN_KEYS if ctx.secrets.get(k)), None
                )
                if not token:
                    continue  # no token resolved — skip silently (never log the token)
                overrides.append(f'mcp_servers.github.url="{_GITHUB_MCP_URL}"')
                overrides.append(
                    f'mcp_servers.github.bearer_token_env_var="{_GITHUB_MCP_TOKEN_ENV}"'
                )
                env[_GITHUB_MCP_TOKEN_ENV] = token
            elif srv.kind == "remote":
                name = srv.name or "remote"
                overrides.append(f"mcp_servers.{name}.url={json.dumps(srv.url)}")
                if srv.headers:
                    table = ", ".join(
                        f"{json.dumps(k)} = {json.dumps(v)}" for k, v in srv.headers.items()
                    )
                    overrides.append(f"mcp_servers.{name}.http_headers={{{table}}}")
            elif srv.kind == "stdio":
                name = srv.name or "stdio"
                overrides.append(f"mcp_servers.{name}.command={json.dumps(srv.command)}")
                if srv.args:
                    arr = ", ".join(json.dumps(a) for a in srv.args)
                    overrides.append(f"mcp_servers.{name}.args=[{arr}]")
        return overrides, env

    def build_options(self, spec: AgentSpec, ctx: RunContext) -> _CodexOptions:
        """Build ``openai_codex`` config + thread/turn args from ``spec`` + runtime ``ctx``."""
        from openai_codex import ApprovalMode, CodexConfig, Sandbox

        # Beside, not inside, the workspace (bookkeeping level) — holds auth.json, the
        # session rollouts, logs. The codex CLI refuses a CODEX_HOME that doesn't exist.
        codex_home = ctx.job_dir / "codex_home"
        codex_home.mkdir(parents=True, exist_ok=True)
        sandbox_name, approval_name = _PERMISSION_MAP.get(
            spec.permission_mode, _PERMISSION_MAP["bypassPermissions"]
        )
        warnings: list[str] = []
        if spec.permission_mode not in _PERMISSION_MAP:
            warnings.append(
                f"permission_mode {spec.permission_mode!r} has no codex mapping; "
                "using bypassPermissions semantics (full access, no approvals)"
            )
        if spec.allowed_tools is not None or spec.disallowed_tools:
            warnings.append(
                "allowed_tools/disallowed_tools are Claude-specific and have no codex "
                "equivalent; ignored"
            )
        if spec.transcript and not spec.checkpoint:
            warnings.append(
                "transcript=True persists nothing on codex: it keeps its conversation as a "
                "rollout blob written under checkpoint, not in the transcript store, so "
                "transcripts() reads back empty"
            )
        if ctx.hooks:
            # Fail the turn rather than warn: hooks gate tool calls, and a caller that
            # only awaits the result would never see a streamed warning telling them the
            # gate is not in place.
            raise ValueError(
                "run(hooks=...) is Claude-specific and has no codex equivalent; the "
                "turn would run with the hooks never called. Drop them, or run this "
                "turn on the claude-code harness."
            )

        mcp_overrides, mcp_env = self._mcp_overrides(spec, ctx)
        # The agent's shell env: Codex filters *KEY*/*SECRET*/*TOKEN*-named vars from the
        # shell by default — the opposite of the toolkit's contract (the caller's own
        # secrets ARE for the agent). Lift the default excludes, but keep the two the
        # harness consumes itself out of the shell explicitly.
        overrides = [
            "shell_environment_policy.ignore_default_excludes=true",
            f'shell_environment_policy.exclude=["OPENAI_API_KEY", "{_GITHUB_MCP_TOKEN_ENV}"]',
            *mcp_overrides,
        ]
        env = runtime_env(spec, ctx)
        env["CODEX_HOME"] = str(codex_home)
        env.update(mcp_env)

        api_key = ctx.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")

        thread_args: dict[str, Any] = {
            "model": spec.model or None,
            "cwd": str(ctx.workspace),
            "sandbox": Sandbox[sandbox_name],
            "approval_mode": ApprovalMode[approval_name],
            **self._instructions(spec, ctx.interactive),
        }
        run_args: dict[str, Any] = {}
        if spec.reasoning_effort is not None:
            effort = spec.reasoning_effort
            if effort == "max":  # Claude-only level; xhigh is Codex's ceiling
                warnings.append("reasoning_effort 'max' has no codex level; using 'xhigh'")
                effort = "xhigh"
            # A plain str, not openai_codex.types.ReasoningEffort: the enum is a str
            # subclass whose validation accepts arbitrary strings, so unknown levels
            # pass through to the SDK/CLI to reject in one place.
            run_args["effort"] = effort
        if spec.output_schema is not None:
            from ..spec import _output_schema_to_dict

            schema = _output_schema_to_dict(spec.output_schema)
            if isinstance(schema, dict):
                run_args["output_schema"] = schema

        return _CodexOptions(
            codex_config=CodexConfig(env=env, config_overrides=tuple(overrides)),
            thread_args=thread_args,
            run_args=run_args,
            api_key=api_key,
            codex_home=codex_home,
            warnings=warnings,
        )

    # -- conversation persistence (checkpoint/resume) ---------------------------

    def _rollout_path(self, codex_home: Path, thread_id: str) -> Path | None:
        """Locate the thread's rollout file under ``CODEX_HOME/sessions``."""
        sessions = codex_home / "sessions"
        if not sessions.is_dir():
            return None
        matches = sorted(sessions.rglob(f"rollout-*-{thread_id}.jsonl"))
        return matches[-1] if matches else None

    def _persist_thread(self, ctx: RunContext, codex_home: Path, thread_id: str) -> None:
        """Copy the thread's rollout file to the BlobStore, keyed by OUR session id.

        Codex's session store is a local file — worthless across serverless workers.
        Best-effort (never fails the run); ``_restore_thread`` is the inverse.
        """
        path = self._rollout_path(codex_home, thread_id)
        if path is None:
            return
        rel = path.relative_to(codex_home)
        meta = {"thread_id": thread_id, "relpath": str(rel)}
        ctx.blobs.put_bytes(
            f"{_THREADS_PREFIX}/{ctx.session_id}/meta.json", json.dumps(meta).encode()
        )
        ctx.blobs.put_bytes(
            f"{_THREADS_PREFIX}/{ctx.session_id}/rollout.jsonl", path.read_bytes()
        )

    def _restore_thread(self, ctx: RunContext, codex_home: Path) -> str | None:
        """Materialize a persisted conversation into ``CODEX_HOME``; return the thread id."""
        try:
            meta = json.loads(
                ctx.blobs.get_bytes(f"{_THREADS_PREFIX}/{ctx.resume_sid}/meta.json").decode()
            )
            body = ctx.blobs.get_bytes(f"{_THREADS_PREFIX}/{ctx.resume_sid}/rollout.jsonl")
        except Exception:  # noqa: BLE001 — no/unreadable persisted thread: start fresh
            return None
        dest = codex_home / meta["relpath"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(body)
        return meta["thread_id"]

    def _finalize(self, spec: AgentSpec, ctx: RunContext, codex_home: Path,
                  thread_id: str | None) -> AgentEvent | None:
        """Inline checkpoint: workspace snapshot (shared) + the Codex conversation."""
        if thread_id and spec.checkpoint and ctx.blobs is not None:
            try:
                self._persist_thread(ctx, codex_home, thread_id)
            except Exception:  # noqa: BLE001 — checkpoint is best-effort
                pass
        return finalize_checkpoint(spec, ctx)

    # -- run loop ---------------------------------------------------------------

    def _result_event(
        self,
        *,
        turn: Any,
        translator: CodexEventTranslator,
        acct: _RunAccounting,
        ctx: RunContext,
        thread_id: str,
        limit: str | None,
    ) -> AgentEvent:
        status = getattr(turn.status, "value", turn.status)
        if limit == "max_turns":
            subtype, is_error = "error_max_turns", True
            text = translator.final_text or f"(interrupted at max_turns={acct.model_calls})"
        elif limit == "budget":
            subtype, is_error = "error_budget_exceeded", True
            text = translator.final_text or "(interrupted: budget exceeded)"
        elif status == "failed":
            subtype, is_error = "error", True
            text = (turn.error.message if turn.error else None) or "(run errored)"
        elif status == "interrupted":
            subtype, is_error = "interrupted", True
            text = translator.final_text or "(interrupted)"
        else:
            subtype, is_error = "success", False
            text = translator.final_text or "(no final text)"
        return AgentEvent(
            kind="result",
            summary=text,
            cost_usd=acct.cost_usd,
            usage=acct.usage(),
            raw={
                "subtype": subtype,
                "is_error": is_error,
                "num_turns": acct.model_calls,
                "duration_ms": turn.duration_ms,
                "session_id": ctx.session_id,
                "thread_id": thread_id,
                "model": acct.model,
                "price_source": acct.price.source if acct.price else None,
            },
        )

    async def run(self, spec: AgentSpec, ctx: RunContext) -> AsyncIterator[AgentEvent]:
        """Drive one turn over an ``AsyncCodex`` app-server, yielding ``AgentEvent``s.

        The stream is ``TurnHandle.stream()`` (terminates at ``turn/completed``); cost
        and model-call accounting come from token-usage notifications, and the harness
        interrupts the turn when ``spec.max_turns`` / ``spec.max_budget_usd`` is hit —
        Codex enforces neither natively. Checkpoint (workspace + conversation rollout)
        finalizes inline at the terminal result event, as in the Claude binding.
        """
        from openai_codex import AsyncCodex

        options = self.build_options(spec, ctx)
        translator = CodexEventTranslator()
        # Price via the LiteLLM live dataset (baked fallback) — one fetch per process,
        # off the loop; needed up front because budget enforcement runs mid-stream.
        price = await asyncio.to_thread(pricing.model_price, spec.model)
        acct = _RunAccounting(spec.model, price)
        limit: str | None = None  # which cap tripped, if any
        thread_id = ""

        async with AsyncCodex(config=options.codex_config) as codex:
            for w in options.warnings:
                yield AgentEvent(kind="status", summary=w, raw={"event": "spec_warning"})
            if price is None:
                yield AgentEvent(
                    kind="status",
                    summary=(
                        f"no price data for model {spec.model!r} (LiteLLM dataset "
                        "unreachable or model unknown): cost_usd will be unknown and "
                        "max_budget_usd cannot be enforced"
                    ),
                    raw={"event": "cost_unknown", "model": spec.model},
                )
            if options.api_key and not (options.codex_home / "auth.json").exists():
                await codex.login_api_key(options.api_key)
            elif not options.api_key and not (options.codex_home / "auth.json").exists():
                raise RuntimeError(
                    "codex harness has no OpenAI credentials: pass OPENAI_API_KEY in the "
                    "per-invocation secrets (or set it in the environment)"
                )

            restored_tid = self._restore_thread(ctx, options.codex_home) if (
                ctx.resume_sid and ctx.blobs is not None
            ) else None
            if restored_tid:
                resume_args = {
                    k: v for k, v in options.thread_args.items()
                    if k != "base_instructions"  # thread_resume keeps the original prompt shape
                }
                thread = await codex.thread_resume(restored_tid, **resume_args)
                yield AgentEvent(
                    kind="status",
                    summary=f"codex thread resumed ({restored_tid})",
                    raw={"event": "thread_resumed", "thread_id": restored_tid,
                         "session_id": ctx.session_id},
                )
            else:
                thread = await codex.thread_start(**options.thread_args)
            thread_id = thread.id

            handle = await thread.turn(ctx.prompt, **options.run_args)
            async for notification in handle.stream():
                if notification.method == "thread/tokenUsage/updated":
                    acct.observe(notification.payload.token_usage.last)
                    if limit is None:
                        cost = acct.cost_usd
                        if acct.model_calls >= spec.max_turns:
                            limit = "max_turns"
                        elif cost is not None and cost >= spec.max_budget_usd:
                            limit = "budget"
                        if limit is not None:
                            yield AgentEvent(
                                kind="status",
                                summary=(
                                    f"{limit} cap hit "
                                    f"(calls={acct.model_calls}, cost=${cost or 0.0:.4f}); "
                                    "interrupting the turn"
                                ),
                                raw={"event": "limit_interrupt", "limit": limit},
                            )
                            try:
                                await handle.interrupt()
                            except Exception:  # noqa: BLE001 — stream end still bounds the run
                                pass
                    continue
                if notification.method == "turn/completed":
                    fin = self._finalize(spec, ctx, options.codex_home, thread_id)
                    yield self._result_event(
                        turn=notification.payload.turn,
                        translator=translator,
                        acct=acct,
                        ctx=ctx,
                        thread_id=thread_id,
                        limit=limit,
                    )
                    if fin is not None:
                        yield fin
                    return
                for event in translator.translate(notification):
                    yield event

        # Stream ended without turn/completed (transport death — TransportClosedError from
        # the SDK carries the CLI's stderr tail and propagates to the runtimes' error
        # handling; a bare end lands here): surface it rather than ending silently.
        raise RuntimeError(
            "codex stream ended without a turn/completed event "
            f"(thread {thread_id or 'not started'})"
        )

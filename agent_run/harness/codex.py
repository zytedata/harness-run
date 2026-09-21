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
                         An ``openrouter/<vendor>/<model>`` id instead routes the turn
                         through OpenRouter (see "OpenRouter models" below).
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
* ``max_budget_usd``   → enforced by the harness. OpenRouter supplies its exact charge;
                         OpenAI models use token usage × :mod:`pricing`. A model with no
                         exact charge or known price runs uncapped with a status warning.
* collab subagents     → Codex reports no subagent token usage on its wire
                         (openai/codex#14642), so the harness recovers it post-hoc from
                         the subagent rollout files under ``CODEX_HOME/sessions``: the
                         terminal ``usage`` and (natively priced) ``cost_usd`` include
                         subagent spend — each thread priced at the model its rollout
                         says it ran on (a role can pin a different model), the
                         with an unpriced subagent model making ``cost_usd`` ``None``
                         (never a guess), flagged by a warning. A wire-seen thread
                         with no rollout found raises a
                         ``subagent_usage_missing`` status event (the recovery reads
                         non-public rollout details; ``make live-usage`` re-checks
                         them live). The mid-turn ``max_budget_usd``/``max_turns``
                         checks cannot see it while the turn runs; the budget is
                         re-checked once it is known (at turn end), so a turn that
                         subagent spend pushes over the cap ends
                         ``error_budget_exceeded``, never ``success``.
* ``output_schema``    → per-turn ``output_schema`` (the final message is the JSON).
* ``reasoning_effort`` → per-turn ``effort`` (thread-sticky server-side, and re-applied
                         on every turn, so resume keeps it). Codex has no ``max`` level;
                         it is mapped to ``xhigh`` with a status warning.
* ``allowed_tools`` / ``disallowed_tools`` → no Codex equivalent; rejected before launch.
* ``max_buffer_size``  → inert: it caps one NDJSON message on the Claude Agent SDK's own
                         stdout transport, and the Codex app-server SDK frames its stream
                         itself with no equivalent knob.
* background tasks     → no Codex equivalent of Claude Code's task re-invocation
                         (``spec.background_task_timeout`` is inert here).
* checkpoint/resume    → the workspace snapshot is shared machinery; the conversation
                         itself is persisted by copying the thread's rollout file to the
                         BlobStore and restoring it into ``CODEX_HOME`` before
                         ``thread_resume`` (Codex's own session store is a local file).
                         Save failures emit ``checkpoint_error`` before the terminal
                         result without failing completed work. A conversation failure
                         preserves the workspace's key when its archive was saved, even
                         on a stop before the first rollout exists. These are separate,
                         non-atomic writes. Absent metadata permits a fresh start;
                         unreadable or malformed metadata and missing/unreadable
                         rollouts fail resume explicitly, without echoing storage errors.
                         Unsafe restore destinations retain the warning/fresh-start policy.

OpenRouter models
-----------------

A model id prefixed ``openrouter/`` (e.g. ``openrouter/moonshotai/kimi-k3``) runs the
turn on OpenRouter instead of the OpenAI API, so non-OpenAI models — Kimi, GLM,
DeepSeek — reach the same Engine/Session/Run surface. The prefix is the whole API: no
new spec field, so the model stays a per-turn knob (``TurnConfig(model=...)``) and one
session can move between OpenAI and OpenRouter models.

Codex reaches a non-OpenAI provider through ``model_providers.*`` config overrides, the
same ``--config`` channel the MCP servers use. What the binding emits, and why each part
is needed (all four settings were established live against OpenRouter, 2026-08-20):

* ``base_url``/``env_key`` — the key is referenced by env var NAME; its value goes into
  the app-server's process env, never onto the argv-visible ``--config`` flags. Auth is
  the per-invocation ``OPENROUTER_API_KEY`` secret; ``codex login`` is NOT used (it
  writes OpenAI credentials, which this path never consults).
* ``wire_api="responses"`` — Codex 0.147 dropped ``"chat"`` ("no longer supported"), and
  OpenRouter serves a Responses endpoint, so this is the only wire left.
* ``web_search="disabled"`` — Codex otherwise sends its server-side web-search tool as
  ``{"type": "web_search", "external_web_access": true}``, and OpenRouter rejects that
  extra field with ``400 Server tool request failed``, which kills the turn before the
  first token. Web search is off for OpenRouter turns; agents get their normal shell and
  file tools.
* a non-``none`` reasoning effort — OpenRouter's Responses endpoint answers
  ``400 Reasoning is mandatory for this endpoint and cannot be disabled``, and Codex
  sends ``effort: "none"`` for a model whose metadata it does not know. When the spec
  leaves ``reasoning_effort`` unset the binding sends ``low`` rather than letting the
  turn fail.

``output_schema`` needs the same treatment: OpenRouter accepts Codex's json_schema
response format but does not enforce it, so the schema is ALSO stated in the developer
instructions (:func:`_shared.openrouter_schema_steer`). Without that, a model that
ignores the unenforced format returns prose and ``structured_output`` comes back ``None``.

Codex ships no catalog entry for these models, so it falls back to generic metadata and
warns that this "can degrade performance". :data:`pricing.OPENROUTER_CONTEXT_WINDOWS`
carries the context window for the models we vouch for, passed as ``model_context_window``
so the agent is not compacted at a guessed limit. Any other ``openrouter/*`` id still
runs with Codex's generic context metadata. OpenRouter's response metadata supplies its
exact cost and budget accounting.

Every OpenRouter turn goes through the toolkit's own proxy (:mod:`._openrouter_proxy`),
started and closed by :func:`_shared.run_with_openrouter_proxy`, which the Claude binding
uses too. Only how Codex is pointed at it — ``model_providers.*`` overrides rather than
environment variables — is this binding's own.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, TYPE_CHECKING

from ..control import ControlledStream, ControlMessage
from ..events import AgentEvent
from ..spec import SystemPrompt
from . import pricing
from ._usage import from_codex_counts, merge_usage
from ._shared import (
    GITHUB_MCP_TOKEN_KEYS,
    INTERACTIVE_SUFFIX,
    OPENROUTER_KEY_ENV,
    OPENROUTER_PREFIX,
    drain_proxy,
    finalize_checkpoint,
    harness_consumed_secret_names,
    mcp_header_secrets,
    missing_openrouter_key_error,
    openrouter_cost_unknown,
    openrouter_schema_steer,
    run_with_openrouter_proxy,
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
_GITHUB_MCP_TOKEN_ENV = "AGENT_RUN_GITHUB_MCP_TOKEN"
_GITHUB_MCP_URL = "https://api.githubcopilot.com/mcp/"

# OpenRouter routing (see "OpenRouter models" in the module docstring). The `openrouter/`
# prefix is stripped before the id reaches Codex: OpenRouter's own ids are
# `<vendor>/<model>`. The prefix, the key name and the proxy's own lifecycle are shared
# with the Claude binding (see :mod:`._shared`).
_OPENROUTER_PROVIDER = "openrouter"
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_OPENROUTER_PROXY_KEY_ENV = "AGENT_RUN_OPENROUTER_PROXY_TOKEN"
_OPENROUTER_WIRE_API = "responses"
# OpenRouter's Responses endpoint refuses a turn with reasoning disabled, and Codex sends
# effort "none" for a model it has no metadata for.
_OPENROUTER_MIN_EFFORT = "low"

# Tool-result content kept in events is truncated: command output can be megabytes, and
# events ride Cloud Logging on gemini (per-entry size limits).
_CONTENT_CAP = 4000

# Blob-key prefix for persisted Codex conversations (the rollout file + thread id).
_THREADS_PREFIX = "codex-threads"

logger = logging.getLogger(__name__)


def _rollout_destination(codex_home: Path, relpath: object) -> Path | None:
    """Where a persisted rollout may be written back, or ``None`` when ``relpath`` is unsafe.

    ``relpath`` comes from the checkpoint's ``meta.json``, an object the run's own token can
    write, so it is untrusted: it must be a relative ``sessions/...jsonl`` path with no ``..``,
    and the resolved destination (symlinks followed) must stay under ``CODEX_HOME/sessions``.
    The allowed root is built from the resolved ``CODEX_HOME`` plus a literal ``sessions``,
    never by resolving ``CODEX_HOME/sessions`` itself: if that directory were a symlink, a
    destination inside its target would otherwise pass as "inside sessions".
    """
    if not isinstance(relpath, str) or "\\" in relpath:
        return None
    path = Path(relpath)
    if (path.is_absolute() or ".." in path.parts or not path.parts
            or path.parts[0] != "sessions" or path.suffix != ".jsonl"):
        return None
    try:
        sessions = codex_home.resolve() / "sessions"
        dest = (codex_home / path).resolve()
    except OSError:
        return None
    if not dest.is_relative_to(sessions):
        return None
    return dest

# Bookkeeping file (in CODEX_HOME, beside sessions/) recording each subagent thread's
# already-billed cumulative totals, so a later turn in the same job dir bills deltas only.
_SUBAGENT_STATE_FILE = "subagent_usage_state.json"

# Subagent (collab-agent) statuses that mean the thread is still producing tokens; a turn
# ending while one is in these states may under-report that thread's usage.
_SUBAGENT_NONTERMINAL_STATUSES = frozenset({"pendingInit", "running"})

# The wire-convention counter names of Codex's TokenUsageBreakdown, as persisted in
# rollout token_count lines (cache_write_input_tokens is absent on older CLIs).
_CODEX_COUNTER_KEYS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
)


@dataclass
class _CodexOptions:
    """Everything ``run`` needs, resolved from ``(spec, ctx)`` by ``build_options``."""

    codex_config: Any  # openai_codex.CodexConfig
    thread_args: dict[str, Any]
    run_args: dict[str, Any]
    api_key: str | None
    codex_home: Path
    warnings: list[str] = field(default_factory=list)
    # True when the model routes through OpenRouter: auth is the provider's env_key, so
    # `run` must not call `codex login` (that path stores OpenAI credentials).
    openrouter: bool = False


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
        # cache_write_input_tokens is optional on the wire; None until any call reports
        # it, so an older CLI's silence stays "not reported" instead of a fake 0.
        self.cache_write_input_tokens: int | None = None

    def observe(self, last: Any) -> None:
        self.model_calls += 1
        self.input_tokens += int(last.input_tokens or 0)
        self.cached_input_tokens += int(last.cached_input_tokens or 0)
        self.output_tokens += int(last.output_tokens or 0)
        self.reasoning_output_tokens += int(last.reasoning_output_tokens or 0)
        cache_write = getattr(last, "cache_write_input_tokens", None)
        if cache_write is not None:
            self.cache_write_input_tokens = (self.cache_write_input_tokens or 0) + int(
                cache_write
            )

    @property
    def cost_usd(self) -> float | None:
        if self.price is None:
            return None
        return self.price.cost_usd(self.input_tokens, self.cached_input_tokens, self.output_tokens)

    def usage(self) -> dict[str, int | None]:
        # The wire's input_tokens INCLUDES the cached and cache-written parts;
        # from_codex_counts converts to the normalized disjoint buckets.
        return from_codex_counts(
            self.input_tokens,
            self.cached_input_tokens,
            self.cache_write_input_tokens,
            self.output_tokens,
            self.reasoning_output_tokens,
        )


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
    * ``item/completed`` ``collabAgentToolCall`` / ``subAgentActivity`` → ``status``; both
      also record the subagent thread ids (and last known statuses) in
      ``subagent_threads`` — the run loop reads those threads' usage from their rollout
      files, since the wire reports no subagent token usage (openai/codex#14642).
    * ``error``                              → ``status`` (raw ``event: codex_error``)
    * everything else (deltas, diffs, token usage, ``turn/completed``) → nothing here;
      accounting and the terminal result are the run loop's job.
    """

    def __init__(self) -> None:
        self.final_text = ""
        self._final_is_final_phase = False
        # subagent thread id → last observed CollabAgentStatus value (None when the
        # thread was only ever seen via subAgentActivity, which carries no status).
        self.subagent_threads: dict[str, str | None] = {}

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
                    raw={
                        "tool": "webSearch",
                        "id": item.id,
                        "is_error": False,
                        "content": item.query,
                    },
                )
        elif kind == "collabAgentToolCall":
            statuses = {
                tid: str(getattr(state.status, "value", state.status))
                for tid, state in (getattr(item, "agents_states", None) or {}).items()
            }
            self.subagent_threads.update(statuses)
            if not started:
                tool = getattr(item, "tool", None)
                tool = getattr(tool, "value", tool)
                yield AgentEvent(
                    kind="status",
                    summary=f"codex subagents ({tool}): "
                    + (", ".join(f"{t}={s}" for t, s in statuses.items()) or "none")[:200],
                    raw={
                        "event": "codex_subagents",
                        "id": getattr(item, "id", None),
                        "tool": tool,
                        "agent_statuses": statuses,
                    },
                )
        elif kind == "subAgentActivity":
            agent_tid = getattr(item, "agent_thread_id", None)
            if agent_tid:
                self.subagent_threads.setdefault(agent_tid, None)
            if not started:
                activity = getattr(item, "kind", None)
                activity = getattr(activity, "value", activity)
                yield AgentEvent(
                    kind="status",
                    summary=f"codex subagent activity: {activity}"[:160],
                    raw={
                        "event": "codex_subagent_activity",
                        "id": getattr(item, "id", None),
                        "agent_thread_id": agent_tid,
                        "activity": activity,
                    },
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


def _sdk() -> Any:
    """The ``openai_codex`` module — ALL lazy SDK imports go through this seam.

    Routing them here (not only ``build_options``) keeps the [local]-extra message on
    the real path: ``_run`` imports the SDK before it ever calls ``build_options``.
    """
    from ._local_sdk import import_local_sdk

    return import_local_sdk("openai_codex", "openai-codex", "Local Codex execution")


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

    def _mcp_overrides(self, spec: AgentSpec, ctx: RunContext) -> tuple[list[str], dict[str, str]]:
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
                secret_headers = mcp_header_secrets(srv, ctx)
                if secret_headers:
                    references = {}
                    for header, value in secret_headers.items():
                        env_name = f"AGENT_RUN_MCP_HEADER_{len(env)}"
                        env[env_name] = value
                        references[header] = env_name
                    table = ", ".join(f"{json.dumps(k)} = {json.dumps(v)}"
                                      for k, v in references.items())
                    overrides.append(f"mcp_servers.{name}.env_http_headers={{{table}}}")
            elif srv.kind == "stdio":
                name = srv.name or "stdio"
                overrides.append(f"mcp_servers.{name}.command={json.dumps(srv.command)}")
                if srv.args:
                    arr = ", ".join(json.dumps(a) for a in srv.args)
                    overrides.append(f"mcp_servers.{name}.args=[{arr}]")
        return overrides, env

    def _openrouter_overrides(
        self,
        model: str,
        base_url: str | None = None,
        key_env: str = OPENROUTER_KEY_ENV,
    ) -> list[str]:
        """``--config`` overrides that point Codex at OpenRouter for ``model``.

        Values are literals here; the API key is referenced by env var NAME only (its
        value goes into the process env), as with the github MCP token.
        """
        overrides = [
            f"model_provider={json.dumps(_OPENROUTER_PROVIDER)}",
            f'model_providers.{_OPENROUTER_PROVIDER}.name="OpenRouter"',
            f"model_providers.{_OPENROUTER_PROVIDER}.base_url="
            f"{json.dumps(base_url or _OPENROUTER_BASE_URL)}",
            f"model_providers.{_OPENROUTER_PROVIDER}.env_key={json.dumps(key_env)}",
            f"model_providers.{_OPENROUTER_PROVIDER}.wire_api={json.dumps(_OPENROUTER_WIRE_API)}",
            # Codex's server-side web-search tool carries a field OpenRouter rejects
            # outright (400 before the first token), so it is off for these turns.
            'web_search="disabled"',
        ]
        window = pricing.OPENROUTER_CONTEXT_WINDOWS.get(model)
        if window is not None:
            overrides.append(f"model_context_window={window}")
        return overrides

    async def _resolved_routing(self, thread: Any) -> dict[str, Any]:
        """Report the provider that the Codex app-server bound to this thread.

        ``thread.read()`` confirms that the provider override took effect. OpenRouter's
        selected upstream is reported separately by ``openrouter_request`` events.

        What comes back today is ``model_provider``; the SDK's ``Thread`` carries no
        settings block, so ``resolved_model`` is normally absent (the read is attempted
        anyway, so it starts working if the SDK gains it). Best-effort throughout:
        attribution must never fail a turn.
        """
        try:
            resp = await thread.read()
            thread_obj = getattr(resp, "thread", None)
            provider = getattr(thread_obj, "model_provider", None)
            out: dict[str, Any] = {}
            if provider is not None:
                out["resolved_model_provider"] = provider
            settings = getattr(thread_obj, "settings", None)
            model = getattr(settings, "model", None)
            if model is not None:
                out["resolved_model"] = model
            return out
        except Exception:  # noqa: BLE001 — reporting only; must not fail the turn
            return {}

    def build_options(
        self,
        spec: AgentSpec,
        ctx: RunContext,
        *,
        openrouter_base_url: str | None = None,
        openrouter_client_token: str | None = None,
    ) -> _CodexOptions:
        """Build ``openai_codex`` config + thread/turn args from ``spec`` + runtime ``ctx``."""
        # These are restrictions, not optional hints. Validate before SDK setup or
        # filesystem side effects; callers awaiting only a result may never see warnings.
        if spec.permission_mode not in _PERMISSION_MAP:
            raise ValueError("permission_mode has no codex mapping; choose a supported mode")
        if spec.allowed_tools is not None or spec.disallowed_tools:
            raise ValueError("allowed_tools/disallowed_tools have no codex equivalent")
        sdk = _sdk()
        ApprovalMode, CodexConfig, Sandbox = sdk.ApprovalMode, sdk.CodexConfig, sdk.Sandbox

        # Beside, not inside, the workspace (bookkeeping level) — holds auth.json, the
        # session rollouts, logs. The codex CLI refuses a CODEX_HOME that doesn't exist.
        codex_home = ctx.job_dir / "codex_home"
        codex_home.mkdir(parents=True, exist_ok=True)
        sandbox_name, approval_name = _PERMISSION_MAP[spec.permission_mode]
        warnings: list[str] = []
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
        model = spec.model or ""
        openrouter = model.startswith(OPENROUTER_PREFIX)
        # The agent's shell env: Codex filters *KEY*/*SECRET*/*TOKEN*-named vars from the
        # shell by default — the opposite of the toolkit's contract (the caller's own
        # secrets ARE for the agent). Lift the default excludes, but keep the ones the
        # harness consumes itself out of the shell explicitly.
        overrides = [
            # A login shell can source ~/.bashrc and restore credentials removed below.
            # The toolkit supplies PATH and caller-owned env explicitly.
            "allow_login_shell=false",
            "shell_environment_policy.ignore_default_excludes=true",
            "shell_environment_policy.exclude=" + json.dumps(sorted({
                "OPENAI_API_KEY", OPENROUTER_KEY_ENV, _OPENROUTER_PROXY_KEY_ENV,
                _GITHUB_MCP_TOKEN_ENV, *mcp_env, *harness_consumed_secret_names(spec),
            })),
            *(
                self._openrouter_overrides(
                    model,
                    openrouter_base_url,
                    _OPENROUTER_PROXY_KEY_ENV if openrouter_client_token else OPENROUTER_KEY_ENV,
                )
                if openrouter
                else []
            ),
            *mcp_overrides,
        ]
        env = runtime_env(spec, ctx)
        # The app-server inherits this process's environment before applying ``env``.
        # Clear unrelated ambient model credentials unless the caller explicitly included
        # them in the spec/runtime environment for the agent to use.
        for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
            if name not in env:
                env[name] = ""
        env["CODEX_HOME"] = str(codex_home)
        env.update(mcp_env)

        key_env = OPENROUTER_KEY_ENV if openrouter else "OPENAI_API_KEY"
        api_key = ctx.secrets.get(key_env) or os.environ.get(key_env)
        if openrouter and api_key:
            # The provider reads its key from the app-server's env (env_key), so unlike
            # the OpenAI path there is no login step — the value must be in the env.
            if openrouter_client_token:
                # Keep the real provider credential out of the CLI process. The random
                # token only authenticates this run to its localhost proxy.
                env[OPENROUTER_KEY_ENV] = ""
                env[_OPENROUTER_PROXY_KEY_ENV] = openrouter_client_token
            else:
                env[OPENROUTER_KEY_ENV] = api_key

        thread_args: dict[str, Any] = {
            "model": model[len(OPENROUTER_PREFIX) :] if openrouter else (model or None),
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
            if openrouter and effort == "none":
                # OpenRouter's Responses endpoint fails the turn outright with reasoning
                # disabled, so 'none' is not an option we can honor here.
                warnings.append(
                    "reasoning_effort 'none' is rejected by OpenRouter (reasoning is "
                    f"mandatory on its Responses endpoint); using {_OPENROUTER_MIN_EFFORT!r}"
                )
                effort = _OPENROUTER_MIN_EFFORT
            # A plain str, not openai_codex.types.ReasoningEffort: the enum is a str
            # subclass whose validation accepts arbitrary strings, so unknown levels
            # pass through to the SDK/CLI to reject in one place.
            run_args["effort"] = effort
        elif openrouter:
            # Codex sends effort "none" for a model it has no metadata for, which
            # OpenRouter refuses — send its lowest real level instead of failing.
            run_args["effort"] = _OPENROUTER_MIN_EFFORT
        if spec.output_schema is not None:
            from ..spec import _output_schema_to_dict

            schema = _output_schema_to_dict(spec.output_schema)
            if isinstance(schema, dict):
                run_args["output_schema"] = schema
                # Sent as well as asked for: if OpenRouter starts enforcing the format,
                # the request already carries it.
                steer = openrouter_schema_steer(spec)
                if steer:
                    existing = thread_args.get("developer_instructions", "")
                    # The steer carries its own leading blank line for appending; with
                    # nothing to append to it would just indent the instructions.
                    thread_args["developer_instructions"] = (
                        existing + steer if existing else steer.lstrip("\n")
                    )

        return _CodexOptions(
            codex_config=CodexConfig(env=env, config_overrides=tuple(overrides)),
            thread_args=thread_args,
            run_args=run_args,
            api_key=api_key,
            codex_home=codex_home,
            warnings=warnings,
            openrouter=openrouter,
        )

    # -- conversation persistence (checkpoint/resume) ---------------------------

    def _rollout_path(self, codex_home: Path, thread_id: str) -> Path | None:
        """Locate the thread's rollout file under ``CODEX_HOME/sessions``."""
        sessions = codex_home / "sessions"
        if not sessions.is_dir():
            return None
        matches = sorted(sessions.rglob(f"rollout-*-{thread_id}.jsonl"))
        return matches[-1] if matches else None

    def _rollout_usage_record(self, path: Path) -> tuple[dict, dict | None, str | None]:
        """One rollout's ``(session_meta payload, last cumulative token total, model)``.

        The first ``session_meta`` line identifies the thread (``id``) and, for a
        subagent thread, names its ``parent_thread_id``. ``token_count`` lines ride
        ``event_msg`` payloads whose ``info.total_token_usage`` is thread-cumulative,
        so only the LAST one matters; ``None`` when the file has no usage record yet.
        Each turn opens with a ``turn_context`` line naming the ``model`` that turn ran
        on — a subagent's role config can put it on a different model than its parent,
        so the LAST one is what its spend is priced with (a thread switching models
        mid-way would need per-turn attribution; not observed, not attempted).
        """
        meta: dict = {}
        total = None
        model = None
        try:
            with path.open(encoding="utf-8") as lines:
                for line in lines:
                    if (
                        '"session_meta"' not in line
                        and '"token_count"' not in line
                        and '"turn_context"' not in line
                    ):
                        continue
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    payload = record.get("payload") or {}
                    kind = record.get("type")
                    if kind == "session_meta":
                        meta = meta or payload
                    elif kind == "turn_context":
                        model = payload.get("model") or model
                    elif payload.get("type") == "token_count":
                        total = (payload.get("info") or {}).get("total_token_usage") or total
        except OSError:
            return {}, None, None
        return meta, total, model

    def _collect_subagent_usage(
        self, codex_home: Path, parent_thread_id: str
    ) -> tuple[dict[str, dict[str, Any]], set[str]]:
        """Wire-convention token deltas per subagent thread, read from their rollouts.

        Returns ``(deltas, found)``: ``found`` is every subagent thread id a rollout
        exists for — a superset of ``deltas`` (a thread fully billed on an earlier turn
        has a rollout but no new delta). The run loop compares ``found`` against the
        thread ids seen on the wire to detect a rollout that never appeared
        (``subagent_usage_missing``) — the tripwire for this recovery relying on
        non-public rollout details.

        Each delta also carries ``"model"``: the model the thread's rollout says it ran
        on (``None`` when the file has no ``turn_context`` yet), so its spend can be
        priced at its own rate rather than the parent's.

        Codex reports no subagent usage on the wire (openai/codex#14642), but every
        subagent thread persists its own rollout under ``CODEX_HOME/sessions``, and its
        ``session_meta`` line names a ``parent_thread_id`` — that marker, not the
        filename, is what identifies a subagent (a session's later turn starts a new
        parent thread, whose rollout must never be billed as a subagent). Rollout
        totals are thread-cumulative, so a state file beside ``sessions/`` records what
        earlier turns already billed and only the delta is returned — a session's next
        turn (same job dir) must not double-bill. Unrecognizable rollouts are skipped:
        under-reporting is the safe failure mode, never billing someone else's thread.
        """
        sessions = codex_home / "sessions"
        if not sessions.is_dir():
            return {}, set()
        state_path = codex_home / _SUBAGENT_STATE_FILE
        try:
            previous = json.loads(state_path.read_text())
        except (OSError, ValueError):
            previous = {}
        # Latest total per thread: a resumed thread gets a new file carrying the
        # cumulative totals forward, and the rglob sort puts it last.
        totals: dict[str, tuple[dict, str | None]] = {}
        for path in sorted(sessions.rglob("rollout-*.jsonl")):
            if path.name.endswith(f"-{parent_thread_id}.jsonl"):
                continue
            meta, total, model = self._rollout_usage_record(path)
            if not meta.get("parent_thread_id") or not total:
                continue  # not a subagent rollout (e.g. an earlier turn's parent thread)
            totals[meta.get("id") or path.stem[-36:]] = (total, model)
        deltas: dict[str, dict[str, Any]] = {}
        state = dict(previous)
        for thread_id, (total, model) in totals.items():
            billed = previous.get(thread_id) or {}
            delta: dict[str, Any] = {}
            for key in _CODEX_COUNTER_KEYS:
                value = total.get(key)
                delta[key] = (
                    None if value is None else max(int(value) - int(billed.get(key) or 0), 0)
                )
            state[thread_id] = total
            if any(delta.values()):
                delta["model"] = model
                deltas[thread_id] = delta
        try:
            state_path.write_text(json.dumps(state))
        except OSError:
            pass  # bookkeeping is best-effort; worst case a later turn re-bills
        return deltas, set(totals)

    def _persist_thread(self, ctx: RunContext, codex_home: Path, thread_id: str) -> None:
        """Copy the thread's rollout file to the BlobStore, keyed by OUR session id.

        Codex's session store is a local file — worthless across serverless workers.
        Best-effort (never fails the run); ``_restore_thread`` is the inverse.
        """
        path = self._rollout_path(codex_home, thread_id)
        if path is None:
            raise RuntimeError("Codex checkpoint rollout was not found")
        rel = path.relative_to(codex_home)
        meta = {"thread_id": thread_id, "relpath": str(rel)}
        ctx.blobs.put_bytes(
            f"{_THREADS_PREFIX}/{ctx.session_id}/meta.json", json.dumps(meta).encode()
        )
        ctx.blobs.put_bytes(f"{_THREADS_PREFIX}/{ctx.session_id}/rollout.jsonl", path.read_bytes())

    def _restore_thread(self, ctx: RunContext, codex_home: Path) -> str | None:
        """Materialize a persisted conversation into ``CODEX_HOME``; return the thread id."""
        try:
            raw = ctx.blobs.get_bytes(f"{_THREADS_PREFIX}/{ctx.resume_sid}/meta.json")
        except KeyError:  # a known absent checkpoint may legitimately start fresh
            return None
        except Exception:  # noqa: BLE001 — never echo credential-bearing storage errors
            raise RuntimeError("Codex checkpoint metadata could not be read") from None
        try:
            meta = json.loads(raw.decode())
            if (not isinstance(meta, dict) or not isinstance(meta.get("thread_id"), str)
                    or not meta["thread_id"]):
                raise ValueError("invalid thread metadata")
        except (ValueError, TypeError):
            raise RuntimeError("Codex checkpoint metadata is invalid") from None
        dest = _rollout_destination(codex_home, meta.get("relpath"))
        if dest is None:
            # The run's own token can write this object, so the previous turn's agent could
            # have edited it: never let ``relpath`` point outside CODEX_HOME/sessions.
            logger.warning("persisted Codex thread for %s has an invalid rollout path; "
                           "starting a fresh conversation", ctx.resume_sid)
            return None
        try:
            body = ctx.blobs.get_bytes(f"{_THREADS_PREFIX}/{ctx.resume_sid}/rollout.jsonl")
        except Exception:  # noqa: BLE001 — an existing checkpoint must not lose its history silently
            raise RuntimeError("Codex checkpoint rollout could not be read") from None
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(body)
        return meta["thread_id"]

    def _finalize(
        self, spec: AgentSpec, ctx: RunContext, codex_home: Path, thread_id: str | None
    ) -> AgentEvent | None:
        """Inline checkpoint: workspace snapshot (shared) + the Codex conversation."""
        conversation_error = False
        if thread_id and spec.checkpoint and ctx.blobs is not None:
            try:
                self._persist_thread(ctx, codex_home, thread_id)
            except Exception:  # noqa: BLE001 — checkpoint is best-effort
                conversation_error = True
        workspace_event = finalize_checkpoint(spec, ctx)
        if conversation_error:
            workspace_key = (workspace_event.raw or {}).get("workspace_key") if workspace_event else None
            return AgentEvent(
                kind="status", summary="checkpoint failed: Codex conversation was not saved",
                raw={"event": "checkpoint_error", "session_id": ctx.session_id,
                     "conversation_saved": False,
                     "workspace_saved": bool(workspace_key),
                     **({"workspace_key": workspace_key} if workspace_key else {})},
            )
        return workspace_event

    # -- run loop ---------------------------------------------------------------

    async def _subagent_prices(
        self, acct: _RunAccounting, subagent_usage: dict[str, dict[str, Any]]
    ) -> tuple[dict[str, pricing.ModelPrice | None], list[AgentEvent]]:
        """Resolve the price each subagent thread's spend is billed at.

        A thread whose rollout names the parent's model (or none) takes the parent's
        price. One on a different model is priced at that model's own rate; when that
        model is unknown to :mod:`pricing` its entry is ``None`` and a status event
        says so — the turn's ``cost_usd`` then becomes ``None`` (usage stays complete),
        because a cost is a real number or absent, never a guess (README contract).
        """
        prices: dict[str, pricing.ModelPrice | None] = {}
        events: list[AgentEvent] = []
        resolved: dict[str, pricing.ModelPrice | None] = {}
        for thread_id, delta in subagent_usage.items():
            model = delta.get("model")
            if not model or model == acct.model:
                prices[thread_id] = acct.price
                continue
            if model not in resolved:
                resolved[model] = await asyncio.to_thread(pricing.model_price, model)
            price = resolved[model]
            if price is None:
                events.append(
                    AgentEvent(
                        kind="status",
                        summary=(
                            f"subagent thread {thread_id} ran on {model}, which has no "
                            "known price; cost_usd is reported as unknown (its tokens "
                            "are still in usage and raw['subagent_usage'])"
                        ),
                        raw={
                            "event": "subagent_price_unknown",
                            "thread_id": thread_id,
                            "model": model,
                        },
                    )
                )
            prices[thread_id] = price
        return prices, events

    @staticmethod
    def _native_cost_usd(
        acct: _RunAccounting,
        subagent_usage: dict[str, dict[str, Any]] | None,
        subagent_prices: dict[str, pricing.ModelPrice | None] | None = None,
    ) -> float | None:
        """Token-priced cost of the whole turn: the parent's calls plus subagent deltas.

        Each subagent delta is priced with its thread's price from
        :meth:`_subagent_prices` (a thread absent from the mapping takes the parent's
        price); the wire deltas keep Codex's inclusive input convention, which is what
        ``ModelPrice.cost_usd`` expects. ``None`` when the parent model is unpriced —
        or when any thread's price is explicitly ``None`` (an unpriced subagent model):
        a partial total would be a guess, and the contract is a real number or absent.
        """
        cost_usd = acct.cost_usd
        if cost_usd is not None and subagent_usage:
            for thread_id, delta in subagent_usage.items():
                price = (subagent_prices or {}).get(thread_id, acct.price)
                if price is None:
                    return None
                cost_usd += price.cost_usd(
                    int(delta.get("input_tokens") or 0),
                    int(delta.get("cached_input_tokens") or 0),
                    int(delta.get("output_tokens") or 0),
                )
        return cost_usd

    def _result_event(
        self,
        *,
        turn: Any,
        translator: CodexEventTranslator,
        acct: _RunAccounting,
        ctx: RunContext,
        thread_id: str,
        limit: str | None,
        exact_cost_usd: float | None = None,
        subagent_usage: dict[str, dict[str, Any]] | None = None,
        subagent_prices: dict[str, pricing.ModelPrice | None] | None = None,
    ) -> AgentEvent:
        """Build the terminal ``result`` event.

        ``usage`` is the normalized record (see :mod:`._usage`) for the WHOLE turn:
        the parent thread's calls plus ``subagent_usage`` (per-thread wire-convention
        deltas from :meth:`_collect_subagent_usage`). ``cost_usd`` is OpenRouter's exact
        charge when there is one — already subagent-inclusive, since subagent requests
        ride the same proxy (verified live) — otherwise the token-priced cost, subagent
        tokens included, each thread at its own model's price (``subagent_prices``).
        The per-thread normalized detail, plus each thread's ``model``, rides
        ``raw["subagent_usage"]``.
        """
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
            # The operator's stop (Session.interrupt()): the turn ended through the normal
            # path with its accounting; a pause, not an error.
            subtype, is_error = "interrupted", False
            text = translator.final_text or "(interrupted)"
        else:
            subtype, is_error = "success", False
            text = translator.final_text or "(no final text)"
        per_thread = {
            tid: {
                **from_codex_counts(
                    int(delta.get("input_tokens") or 0),
                    int(delta.get("cached_input_tokens") or 0),
                    delta.get("cache_write_input_tokens"),
                    int(delta.get("output_tokens") or 0),
                    int(delta.get("reasoning_output_tokens") or 0),
                ),
                "model": delta.get("model"),
            }
            for tid, delta in (subagent_usage or {}).items()
        }
        cost_usd = exact_cost_usd
        if cost_usd is None:
            cost_usd = self._native_cost_usd(acct, subagent_usage, subagent_prices)
        raw = {
            "subtype": subtype,
            "is_error": is_error,
            "num_turns": acct.model_calls,
            "duration_ms": turn.duration_ms,
            "session_id": ctx.session_id,
            "thread_id": thread_id,
            "model": acct.model,
            "price_source": "openrouter"
            if exact_cost_usd is not None
            else (acct.price.source if acct.price else None),
        }
        if per_thread:
            raw["subagent_usage"] = per_thread
        return AgentEvent(
            kind="result",
            summary=text,
            cost_usd=cost_usd,
            usage=merge_usage(acct.usage(), *per_thread.values()),
            raw=raw,
        )

    async def run(self, spec: AgentSpec, ctx: RunContext) -> AsyncIterator[AgentEvent]:
        """Run Codex, with a local metadata proxy for OpenRouter turns."""
        async for event in run_with_openrouter_proxy(
            spec, ctx, lambda proxy: self._run(spec, ctx, proxy)
        ):
            yield event

    # After ``handle.interrupt()``: how long to wait for the ``turn/completed`` that
    # settles it (verified live: it arrives within milliseconds).
    _INTERRUPT_SETTLE_S = 5.0

    async def _run(
        self, spec: AgentSpec, ctx: RunContext, proxy: Any | None
    ) -> AsyncIterator[AgentEvent]:
        """Drive one turn over an ``AsyncCodex`` app-server, yielding ``AgentEvent``s.

        The stream is ``TurnHandle.stream()`` (terminates at ``turn/completed``); cost
        and model-call accounting come from token-usage notifications, and the harness
        interrupts the turn when ``spec.max_turns`` / ``spec.max_budget_usd`` is hit —
        Codex enforces neither natively. Checkpoint (workspace + conversation rollout)
        finalizes inline at the terminal result event, as in the Claude binding.

        Operator control (``ctx.control``) is read alongside the stream. A ``steer`` maps
        onto Codex's native ``turn/steer`` (``handle.steer()``: the model sees the message
        after its current step, the turn ends with one ``turn/completed`` — verified
        live). An ``interrupt`` calls ``handle.interrupt()``; Codex answers with
        ``turn/completed`` ``status=interrupted``, which ends the turn for a ``stop`` or,
        when a message follows, is a segment boundary: the message starts a new turn on
        the SAME thread (``thread.turn()``) and the run keeps streaming from that handle.
        Accounting spans the turns of the run. Each delivered message is surfaced as a
        ``user`` event.
        """
        AsyncCodex = _sdk().AsyncCodex

        options = self.build_options(
            spec,
            ctx,
            openrouter_base_url=(f"{proxy.base_url}/api/v1" if proxy is not None else None),
            openrouter_client_token=(proxy.client_token if proxy is not None else None),
        )
        translator = CodexEventTranslator()
        # Price via the LiteLLM live dataset (baked fallback) — one fetch per process,
        # off the loop; needed up front because budget enforcement runs mid-stream. An
        # OpenRouter turn is not priced here: it reports what OpenRouter charged, or
        # nothing at all (see :mod:`pricing`).
        price = (
            None
            if options.openrouter
            else await asyncio.to_thread(pricing.model_price, spec.model)
        )
        acct = _RunAccounting(spec.model, price)
        limit: str | None = None  # which cap tripped, if any
        thread_id = ""

        # The select's pending reads (the stream's next item, the control channel's next
        # message) are dropped however the turn ends.
        select: ControlledStream | None = None
        try:
            async with AsyncCodex(config=options.codex_config) as codex:
                for w in options.warnings:
                    yield AgentEvent(kind="status", summary=w, raw={"event": "spec_warning"})
                if price is None and proxy is None:
                    yield AgentEvent(
                        kind="status",
                        summary=(
                            f"no price data for model {spec.model!r} (LiteLLM dataset "
                            "unreachable or model unknown): cost_usd will be unknown and "
                            "max_budget_usd cannot be enforced"
                        ),
                        raw={"event": "cost_unknown", "model": spec.model},
                    )
                if options.openrouter:
                    # OpenRouter auth is the provider's env_key (already in the app-server's
                    # env), so there is nothing to log in with — `codex login` would store
                    # OpenAI credentials this path never reads.
                    if not options.api_key:
                        raise missing_openrouter_key_error("codex")
                elif options.api_key and not (options.codex_home / "auth.json").exists():
                    await codex.login_api_key(options.api_key)
                elif not options.api_key and not (options.codex_home / "auth.json").exists():
                    raise RuntimeError(
                        "codex harness has no OpenAI credentials: pass OPENAI_API_KEY in the "
                        "per-invocation secrets (or set it in the environment)"
                    )

                restored_tid = (
                    self._restore_thread(ctx, options.codex_home)
                    if (ctx.resume_sid and ctx.blobs is not None)
                    else None
                )
                if restored_tid:
                    resume_args = {
                        k: v
                        for k, v in options.thread_args.items()
                        if k != "base_instructions"  # thread_resume keeps the original prompt shape
                    }
                    thread = await codex.thread_resume(restored_tid, **resume_args)
                    yield AgentEvent(
                        kind="status",
                        summary=f"codex thread resumed ({restored_tid})",
                        raw={
                            "event": "thread_resumed",
                            "thread_id": restored_tid,
                            "session_id": ctx.session_id,
                        },
                    )
                else:
                    thread = await codex.thread_start(**options.thread_args)
                thread_id = thread.id

                routing = await self._resolved_routing(thread)
                if routing:
                    asked = _OPENROUTER_PROVIDER if options.openrouter else "openai"
                    got = routing.get("resolved_model_provider")
                    yield AgentEvent(
                        kind="status",
                        summary=(
                            f"model routing: provider={got or 'unreported'} "
                            f"model={routing.get('resolved_model') or options.thread_args.get('model')}"
                        ),
                        raw={
                            "event": "model_routing",
                            "asked_provider": asked,
                            **routing,
                            "matches_request": got == asked if got is not None else None,
                        },
                    )

                handle = await thread.turn(ctx.prompt, **options.run_args)
                select = ControlledStream(handle.stream(), ctx.control)
                interrupt_pending: ControlMessage | None = None
                interrupt_deadline: float | None = None

                async def continue_with(msg: ControlMessage) -> AgentEvent:
                    """Start the operator's message as a new turn on the same thread."""
                    nonlocal handle
                    handle = await thread.turn(msg.message or "", **options.run_args)
                    select.replace_stream(handle.stream())
                    return msg.user_event()

                while True:
                    timeout = None
                    if interrupt_pending is not None and interrupt_deadline is not None:
                        timeout = max(0.0, interrupt_deadline - asyncio.get_running_loop().time())
                    kind, item = await select.next(timeout)
                    if kind == "end":
                        break
                    if kind == "timeout":
                        # Codex answered the interrupt with no turn/completed within the
                        # settle window (nothing was running).
                        msg, interrupt_pending = interrupt_pending, None
                        interrupt_deadline = None
                        if msg.op == "interrupt":
                            yield await continue_with(msg)
                            continue
                        from types import SimpleNamespace

                        settled = SimpleNamespace(status="interrupted", duration_ms=None, error=None)
                        fin = self._finalize(spec, ctx, options.codex_home, thread_id)
                        if fin is not None:
                            yield fin
                        yield self._result_event(
                            turn=settled, translator=translator, acct=acct, ctx=ctx,
                            thread_id=thread_id, limit=limit,
                            exact_cost_usd=proxy.exact_cost_usd if proxy is not None else None,
                        )
                        return
                    if kind == "control":
                        msg = item
                        if msg.op == "steer":
                            try:
                                await handle.steer(msg.message or "")
                            except Exception as exc:  # noqa: BLE001 — report, keep the turn alive
                                yield AgentEvent(
                                    kind="status",
                                    summary=f"steer not accepted by codex: {str(exc)[:200]}",
                                    raw={"event": "steer_failed", "message_id": msg.message_id},
                                )
                                continue
                            yield msg.user_event()
                            continue
                        yield AgentEvent(
                            kind="status",
                            summary=(
                                "interrupt requested by the operator"
                                + ("; continuing with their message" if msg.op == "interrupt" else "")
                            ),
                            raw={"event": "interrupt_requested", "op": msg.op,
                                 "message_id": msg.message_id},
                        )
                        try:
                            await handle.interrupt()
                        except Exception:  # noqa: BLE001 — the settle wait bounds it either way
                            pass
                        interrupt_pending = msg
                        interrupt_deadline = (
                            asyncio.get_running_loop().time() + self._INTERRUPT_SETTLE_S
                        )
                        continue
                    notification = item
                    if proxy is not None:
                        for event in proxy.drain_events():
                            yield event
                    if notification.method == "thread/tokenUsage/updated":
                        acct.observe(notification.payload.token_usage.last)
                        if limit is None:
                            cost = proxy.exact_cost_usd if proxy is not None else acct.cost_usd
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
                        if interrupt_pending is not None:
                            msg, interrupt_pending = interrupt_pending, None
                            interrupt_deadline = None
                            if msg.op == "interrupt":
                                # The interrupted turn is a segment boundary; the operator's
                                # message continues the run as a new turn on the same thread.
                                yield AgentEvent(
                                    kind="status",
                                    summary="turn interrupted; continuing with the operator's message",
                                    raw={"event": "interrupted_segment",
                                         "message_id": msg.message_id},
                                )
                                yield await continue_with(msg)
                                continue
                        async for event in drain_proxy(proxy):
                            yield event
                        exact_cost = proxy.exact_cost_usd if proxy is not None else None
                        if (
                            limit is None
                            and exact_cost is not None
                            and exact_cost >= spec.max_budget_usd
                        ):
                            limit = "budget"
                        if proxy is not None and price is None and exact_cost is None:
                            # A model with no price and no proxy already reported this before
                            # the turn started; only the OpenRouter case is news here.
                            yield openrouter_cost_unknown(spec.model)
                        if proxy is not None and proxy.budget_blocked:
                            limit = "budget"
                        subagent_usage, subagent_found = self._collect_subagent_usage(
                            options.codex_home, thread_id
                        )
                        missing = sorted(set(translator.subagent_threads) - subagent_found)
                        if missing:
                            # A thread the wire announced but no rollout accounts for: the
                            # recovery relies on non-public rollout details, so say loudly
                            # that the numbers are undercounting rather than fail the turn.
                            # With an exact OpenRouter charge only token usage is affected —
                            # subagent requests ride the same proxy, so cost is complete.
                            affected = (
                                "usage (cost_usd is the provider's exact charge, which "
                                "already includes subagent requests)"
                                if exact_cost is not None
                                else "usage/cost"
                            )
                            yield AgentEvent(
                                kind="status",
                                summary=(
                                    "no rollout found for subagent thread(s) "
                                    f"{', '.join(missing)}; their tokens are NOT in this "
                                    f"result's {affected}"
                                ),
                                raw={
                                    "event": "subagent_usage_missing",
                                    "threads": missing,
                                    "cost_included": exact_cost is not None,
                                },
                            )
                        subagent_prices: dict[str, pricing.ModelPrice | None] = {}
                        if subagent_usage and exact_cost is None and acct.price is not None:
                            subagent_prices, price_events = await self._subagent_prices(
                                acct, subagent_usage
                            )
                            for event in price_events:
                                yield event
                        if limit is None and exact_cost is None and subagent_usage:
                            # Subagent spend only becomes visible here, after the mid-turn
                            # checks; a turn it pushes over the cap must not report success.
                            native_cost = self._native_cost_usd(
                                acct, subagent_usage, subagent_prices
                            )
                            if native_cost is not None and native_cost >= spec.max_budget_usd:
                                limit = "budget"
                                yield AgentEvent(
                                    kind="status",
                                    summary=(
                                        f"budget cap hit with subagent spend "
                                        f"(cost=${native_cost:.4f}); the turn had already ended"
                                    ),
                                    raw={"event": "limit_exceeded", "limit": "budget"},
                                )
                        pending = sorted(
                            tid
                            for tid, sub_status in translator.subagent_threads.items()
                            if sub_status in _SUBAGENT_NONTERMINAL_STATUSES
                        )
                        if pending:
                            yield AgentEvent(
                                kind="status",
                                summary=(
                                    "subagent usage may be partial: thread(s) "
                                    f"{', '.join(pending)} last reported a non-terminal status"
                                ),
                                raw={"event": "subagent_usage_partial", "threads": pending},
                            )
                        fin = self._finalize(spec, ctx, options.codex_home, thread_id)
                        if fin is not None:
                            yield fin
                        yield self._result_event(
                            turn=notification.payload.turn,
                            translator=translator,
                            acct=acct,
                            ctx=ctx,
                            thread_id=thread_id,
                            limit=limit,
                            exact_cost_usd=exact_cost,
                            subagent_usage=subagent_usage,
                            subagent_prices=subagent_prices,
                        )
                        return
                    for event in translator.translate(notification):
                        yield event

        finally:
            if select is not None:
                await select.close()

        # Stream ended without turn/completed (transport death — TransportClosedError from
        # the SDK carries the CLI's stderr tail and propagates to the runtimes' error
        # handling; a bare end lands here): surface it rather than ending silently.
        raise RuntimeError(
            "codex stream ended without a turn/completed event "
            f"(thread {thread_id or 'not started'})"
        )

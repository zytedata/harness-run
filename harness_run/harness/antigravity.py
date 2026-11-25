"""``AntigravityHarness`` — the Google Antigravity SDK binding (DESIGN.md §7).

Drives the ``google-antigravity`` SDK (a Python client over its bundled ``localharness``
binary) against the per-run working directory and translates each streamed ``Step`` into a
generic :class:`AgentEvent`, so specs that select ``harness="antigravity"`` run through the
same Engine/Session/Run surface as the other bindings. ``google.antigravity`` is imported
lazily, so importing this module needs no third-party deps.

Scope: the **local runtime only**, and the **Gemini 4 family** (``gemini-4-neon``,
``gemini-4-carbon``, ``gemini-4-hydrogen``, ``gemini-4-argon``) over the Gemini Developer
API. The sandbox image does not bake the SDK yet (``sandbox.deploy`` rejects the harness),
and there is no Vertex or OpenRouter route.

Spec translation (parity notes):

* ``model``            → the text model (``LocalAgentConfig(models=[ModelTarget])``); auth is
                         the per-invocation ``GEMINI_API_KEY`` secret (or ambient env),
                         handed to the SDK as config — never placed in the agent's shell env.
                         An ``openrouter/`` model is rejected (no OpenRouter route).
* ``permission_mode``  → only ``bypassPermissions``: every tool enabled, every call allowed
                         (``policy.allow_all()``). Antigravity's other modes ask a human,
                         which a headless run cannot answer; the rest are rejected.
* ``system_prompt``    → plain ``str`` replaces Antigravity's instructions entirely
                         (``CustomSystemInstructions``); ``SystemPrompt.append`` becomes an
                         appended section (``TemplatedSystemInstructions``).
* ``skills``           → staged by the runtimes into ``<cwd>/.agents/skills`` and passed to
                         the SDK as its ``skills_paths``.
* ``mcp_servers``      → ``McpStdioServer`` / ``McpStreamableHttpServer`` config entries; the
                         GitHub token and secret headers travel in that config (over the
                         SDK's local websocket), never in argv or the agent's env.
* ``max_turns``        → native ``BudgetConfig.max_model_calls`` scoped to this run; the turn
                         stops with ``MAX_MODEL_CALLS_EXCEEDED`` → ``error_max_turns``. The
                         harness's cap may count subagent calls too, so it can trip earlier
                         than ``num_turns`` (main-agent calls only) suggests.
* ``max_budget_usd``   → enforced by the harness: the SDK reports cumulative usage after each
                         model call (subagents included), priced via :mod:`pricing`; the turn
                         is cancelled at the cap. An unpriced model runs uncapped with a
                         ``cost_unknown`` status event.
* ``output_schema``    → ``response_schema``: the model ends the turn with a ``finish`` call
                         carrying the object, which becomes the result text (JSON).
* ``reasoning_effort`` → Gemini ``thinking_level`` (``xhigh``/``max`` → ``extra_high``;
                         ``none`` → ``minimal``, with a status warning).
* ``allowed_tools`` / ``disallowed_tools`` / ``hooks`` → rejected before launch (not mapped
                         yet), as on Codex.
* ``codex_config`` / ``max_buffer_size`` / ``background_task_timeout`` → inert here.
* subagents            → Antigravity's ``invoke_subagent`` runs subagents in the same
                         process. Their steps are surfaced with ``raw["subagent"]`` and their
                         trajectory id; their tokens are in ``usage`` and ``cost_usd`` and,
                         per trajectory, in ``raw["subagent_usage"]``.
* operator control     → ``interrupt``/``stop`` cancel the running turn (the conversation
                         survives; an interrupt's message continues it as a new turn). The
                         SDK has no way to add a message to a running turn, so a ``steer`` is
                         delivered when the current turn ends, as the next turn of the run.
* checkpoint/resume    → the workspace snapshot is shared machinery; the conversation (the
                         SDK's SQLite store plus its ``brain`` directory) is archived to the
                         BlobStore and restored before the SDK resumes it. The conversation
                         id is derived from OUR session id, so resume needs no metadata.

Usage: Gemini's ``prompt_token_count`` includes the cached part and its output price covers
thinking tokens, so ``usage`` reports ``output_tokens`` = candidates + thoughts (thoughts also
as ``reasoning_output_tokens``). Gemini 4 caching is implicit, so cache writes are not
reported (``cache_creation_input_tokens`` is ``None``).
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import tarfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, TYPE_CHECKING

from ..control import ControlledStream, ControlMessage
from ..events import AgentEvent
from ..spec import SystemPrompt
from . import pricing
from ._shared import (
    GITHUB_MCP_TOKEN_KEYS,
    INTERACTIVE_SUFFIX,
    OPENROUTER_PREFIX,
    finalize_checkpoint,
    harness_consumed_secret_names,
    mcp_header_secrets,
    runtime_env,
    unset_env_names,
)

if TYPE_CHECKING:
    from ..spec import AgentSpec
    from .context import RunContext


# The Gemini API key, as the caller passes it (GOOGLE_API_KEY is the google-genai alias).
GEMINI_KEY_NAMES = ("GEMINI_API_KEY", "GOOGLE_API_KEY")

# Model credentials kept out of the agent's shell. The SDK starts its binary with the parent
# process env plus ours, so a name can only be removed by blanking it. The caller's own
# copies (passed as secrets the agent may use) are kept, as on Codex.
_AMBIENT_MODEL_KEYS = (
    "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY",
    "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
)

_GITHUB_MCP_URL = "https://api.githubcopilot.com/mcp/"

# spec.reasoning_effort → Gemini thinking level (the SDK's ThinkingLevel values).
_EFFORT_MAP = {
    "minimal": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "extra_high",
}
_EFFORT_COERCED = {"none": "minimal", "max": "extra_high"}

# Tool-result content kept in events is truncated (command output can be megabytes).
_CONTENT_CAP = 4000

# Where the SDK keeps a conversation, below the job dir (bookkeeping level, not the agent's
# cwd): ``save`` holds ``<conversation id>.db``, ``app`` the ``brain/<trajectory id>`` dirs.
_STATE_DIR = "antigravity"
# Blob key of the archived conversation state, per session.
_STATE_PREFIX = "antigravity-conversations"

# After cancelling a turn: how long to wait for the SDK to settle it (immediate, live).
_CANCEL_SETTLE_S = 10.0


def conversation_id_for(session_id: str) -> str:
    """The Antigravity conversation id of a harness-run session (32 hex chars, stable)."""
    return uuid.uuid5(uuid.NAMESPACE_URL, f"harness-run-antigravity:{session_id}").hex


def _sdk() -> Any:
    """The ``google.antigravity`` package — ALL lazy SDK imports go through this seam."""
    from ._local_sdk import import_local_sdk

    return import_local_sdk("google.antigravity", "google-antigravity", "Local Antigravity execution")


def _value(x: Any) -> Any:
    return getattr(x, "value", x)


@dataclass(frozen=True)
class _TurnEnded:
    """Last item of :func:`_steps`: the turn was cancelled, or the SDK raised ``error``."""

    cancelled: bool = False
    error: str | None = None


async def _steps(conv: Any, cancelled_error: type[BaseException]) -> AsyncIterator[Any]:
    """``conv.receive_steps()`` with its terminal exceptions turned into a :class:`_TurnEnded`.

    The SDK ends a cancelled turn by raising ``AntigravityCancelledError``, a subclass of
    ``asyncio.CancelledError``. Raised inside the task that reads the stream, that would
    mark the task itself cancelled, indistinguishable from a cancelled run. Errors (e.g. an
    unknown model's 404) also arrive as exceptions from the stream.
    """
    try:
        async for step in conv.receive_steps():
            yield step
    except cancelled_error:
        yield _TurnEnded(cancelled=True)
    except Exception as exc:  # noqa: BLE001 — reported as the turn's error result
        yield _TurnEnded(error=f"antigravity error: {exc}"[:2000])


@dataclass
class _AntigravityOptions:
    """Everything ``run`` needs, resolved from ``(spec, ctx)`` by ``build_options``."""

    config_kwargs: dict[str, Any]
    conversation_id: str
    state_dir: Path
    resume: bool
    warnings: list[str] = field(default_factory=list)


class _Usage:
    """Turn usage from the SDK's cumulative counters (the session total minus its start)."""

    def __init__(self, start: Any) -> None:
        self._start = {k: int(getattr(start, k, None) or 0) for k in self._KEYS} if start else {}

    _KEYS = (
        "prompt_token_count",
        "cached_content_token_count",
        "candidates_token_count",
        "thoughts_token_count",
    )

    def delta(self, total: Any) -> dict[str, int]:
        return {
            k: max(int(getattr(total, k, None) or 0) - self._start.get(k, 0), 0)
            for k in self._KEYS
        }


def normalized_usage(counts: dict[str, int]) -> dict[str, int | None]:
    """Gemini counters (prompt includes cached; output excludes thoughts) → normalized."""
    prompt = counts.get("prompt_token_count", 0)
    cached = counts.get("cached_content_token_count", 0)
    thoughts = counts.get("thoughts_token_count", 0)
    return {
        "input_tokens": max(prompt - cached, 0),
        "cache_read_input_tokens": cached,
        "cache_creation_input_tokens": None,
        "output_tokens": counts.get("candidates_token_count", 0) + thoughts,
        "reasoning_output_tokens": thoughts,
    }


def cost_usd(price: pricing.ModelPrice | None, counts: dict[str, int]) -> float | None:
    if price is None:
        return None
    return price.cost_usd(
        counts.get("prompt_token_count", 0),
        counts.get("cached_content_token_count", 0),
        counts.get("candidates_token_count", 0) + counts.get("thoughts_token_count", 0),
    )


class AntigravityStepTranslator:
    """Antigravity ``Step`` → generic ``AgentEvent`` translation.

    The SDK re-sends a step on every status change (ACTIVE, ACTIVE, …, DONE), so a step is
    translated once, at its terminal status, except that a tool call is announced
    (``tool_use``) the first time it is seen. A finished tool call carries its result as
    extra arguments (``exit_code``, ``combined_output``, …), which become the
    ``tool_result``. Mapping (step type/source → ``AgentEvent.kind``):

    * ``TOOL_CALL`` (model)           → ``tool_use`` when first seen, ``tool_result`` when done
    * ``TEXT_RESPONSE`` (model)       → ``message``; the main agent's last one is the final text
    * ``THINKING`` / step thinking    → ``thinking``
    * ``FINISH``                      → nothing; its structured output is the final result
    * ``TEXT_RESPONSE`` (system)      → ``status`` (``antigravity_system_message``): subagent
      mail, cancelled-task notices
    * ``COMPACTION``                  → ``status`` (``antigravity_compaction``)
    * the echo of a user prompt       → nothing

    Subagent steps (a trajectory with a parent) carry ``raw["subagent"] = True`` and their
    ``trajectory_id``; they never become the final text.
    """

    _TERMINAL = frozenset({"DONE", "ERROR", "CANCELED"})

    def __init__(self, main_trajectory_id: str) -> None:
        self.main_trajectory_id = main_trajectory_id
        self.final_text = ""
        self.structured_output: Any = None
        self._announced: dict[tuple[str, int, int], dict] = {}
        self._finished: set[tuple[str, int]] = set()

    def _is_subagent(self, step: Any) -> bool:
        return bool(getattr(step, "parent_trajectory_id", "")) or (
            bool(step.trajectory_id) and step.trajectory_id != self.main_trajectory_id
        )

    def _tag(self, step: Any, raw: dict) -> dict:
        if self._is_subagent(step):
            raw["subagent"] = True
            raw["trajectory_id"] = step.trajectory_id
        return raw

    def translate(self, step: Any) -> Iterator[AgentEvent]:
        stype = _value(step.type)
        source = _value(step.source)
        status = _value(step.status)
        key = (step.trajectory_id, step.step_index)
        if source == "USER":
            return
        if stype == "TOOL_CALL" and source == "MODEL":
            for i, call in enumerate(step.tool_calls):
                ckey = (step.trajectory_id, step.step_index, i)
                if ckey not in self._announced:
                    self._announced[ckey] = {
                        **dict(call.args or {}), "__name__": str(_value(call.name)),
                    }
                    yield self._tool_use(step, call)
        if status not in self._TERMINAL or key in self._finished:
            return
        self._finished.add(key)
        if step.thinking and step.thinking.strip():
            yield AgentEvent(
                kind="thinking", summary=step.thinking, raw=self._tag(step, {"step": step.step_index})
            )
        if stype == "TOOL_CALL" and source == "MODEL":
            for i, call in enumerate(step.tool_calls):
                yield self._tool_result(step, call, self._announced.get((*key, i), {}))
        elif stype == "TEXT_RESPONSE" and source == "MODEL":
            text = step.content or ""
            if text.strip():
                if not self._is_subagent(step):
                    self.final_text = text
                yield AgentEvent(kind="message", summary=text, raw=self._tag(step, {}) or None)
        elif stype == "FINISH":
            if step.structured_output is not None and not self._is_subagent(step):
                self.structured_output = step.structured_output
        elif stype == "TEXT_RESPONSE":  # SYSTEM: subagent mail, task notices
            yield AgentEvent(
                kind="status",
                summary=f"antigravity system message: {(step.content or '')[:200]}",
                raw=self._tag(step, {
                    "event": "antigravity_system_message",
                    "content": (step.content or "")[:_CONTENT_CAP],
                }),
            )
        elif stype == "COMPACTION":
            yield AgentEvent(
                kind="status", summary="antigravity compacted the conversation",
                raw=self._tag(step, {"event": "antigravity_compaction"}),
            )
        if step.error:
            yield AgentEvent(
                kind="status",
                summary=f"antigravity step error: {step.error}"[:300],
                raw=self._tag(step, {
                    "event": "antigravity_step_error",
                    "step_type": stype,
                    "error": str(step.error)[:2000],
                }),
            )

    def unfinished(self) -> Iterator[AgentEvent]:
        """A ``tool_result`` for each announced call whose step never finished.

        A cancelled turn (operator interrupt, budget cap) leaves the call that was running
        without a result; closing it keeps every ``tool_use`` paired in the event log.
        """
        for (traj, index, i), args in list(self._announced.items()):
            if (traj, index) in self._finished:
                continue
            self._finished.add((traj, index))
            raw = {"tool": args.get("__name__", "tool"), "id": f"{index}", "is_error": True,
                   "exit_code": None, "content": "(cancelled before it finished)"}
            if traj and traj != self.main_trajectory_id:
                raw.update(subagent=True, trajectory_id=traj)
            yield AgentEvent(kind="tool_result", summary=f"{raw['tool']} (cancelled)", raw=raw)

    def _tool_use(self, step: Any, call: Any) -> AgentEvent:
        name = str(_value(call.name))
        args = dict(call.args or {})
        brief = next(
            (str(args[k]) for k in ("command_line", "file_path", "path", "TargetFile", "query",
                                    "url") if args.get(k)),
            "",
        )
        return AgentEvent(
            kind="tool_use",
            summary=f"{name} {' '.join(brief.split())}"[:160].strip(),
            raw=self._tag(step, {"tool": name, "id": call.id or f"{step.step_index}",
                                 "input": args}),
        )

    def _tool_result(self, step: Any, call: Any, announced: dict) -> AgentEvent:
        name = str(_value(call.name))
        args = dict(call.args or {})
        # What the finished call added to its arguments is the result.
        result = {k: v for k, v in args.items() if k not in announced and k != "__name__"}
        exit_code = result.get("exit_code")
        failed = (
            _value(step.status) in ("ERROR", "CANCELED")
            or bool(step.error)
            or exit_code not in (None, 0, "0")
        )
        content: Any = result.get("combined_output", result) if result else None
        if content is not None and not isinstance(content, str):
            content = json.dumps(content, default=str)
        return AgentEvent(
            kind="tool_result",
            summary=f"{name}{' (error)' if failed else ''}",
            raw=self._tag(step, {
                "tool": name,
                "id": call.id or f"{step.step_index}",
                "is_error": bool(failed),
                "exit_code": exit_code,
                "content": content[-_CONTENT_CAP:] if isinstance(content, str) else None,
            }),
        )


class AntigravityHarness:
    """The Google Antigravity SDK harness (implements ``harness.base.Harness``)."""

    # -- option building -------------------------------------------------------

    def _system_instructions(self, spec: AgentSpec, interactive: bool) -> Any:
        sdk_types = _sdk().types
        suffix = INTERACTIVE_SUFFIX if interactive else ""
        sp = spec.system_prompt
        if isinstance(sp, str):
            return sdk_types.CustomSystemInstructions(text=sp + suffix)
        append = ((sp.append or "") if isinstance(sp, SystemPrompt) else "") + suffix
        if append.strip():
            return sdk_types.TemplatedSystemInstructions(
                sections=[sdk_types.SystemInstructionSection(content=append.strip())]
            )
        return None

    def _mcp_servers(self, spec: AgentSpec, ctx: RunContext) -> list[Any]:
        sdk_types = _sdk().types
        servers: list[Any] = []
        for srv in spec.mcp_servers:
            if srv.kind == "github":
                token = next(
                    (ctx.secrets[k] for k in GITHUB_MCP_TOKEN_KEYS if ctx.secrets.get(k)), None
                )
                if not token:
                    continue  # no token resolved — skip silently (never log the token)
                servers.append(sdk_types.McpStreamableHttpServer(
                    name="github", url=_GITHUB_MCP_URL,
                    headers={"Authorization": f"Bearer {token}"},
                ))
            elif srv.kind == "remote":
                headers = {**(srv.headers or {}), **mcp_header_secrets(srv, ctx)}
                servers.append(sdk_types.McpStreamableHttpServer(
                    name=srv.name or "remote", url=srv.url, headers=headers or None,
                ))
            elif srv.kind == "stdio":
                servers.append(sdk_types.McpStdioServer(
                    name=srv.name or "stdio", command=srv.command, args=list(srv.args or ()),
                ))
        return servers

    def _agent_env(self, spec: AgentSpec, ctx: RunContext) -> dict[str, str]:
        """The env layered over the parent's for the agent's shell; blanks what must not leak."""
        env = runtime_env(spec, ctx)
        blank = set(_AMBIENT_MODEL_KEYS) - env.keys()
        blank |= harness_consumed_secret_names(spec) | unset_env_names(spec, ctx)
        for name in blank:
            env[name] = ""
        return env

    def build_options(self, spec: AgentSpec, ctx: RunContext) -> _AntigravityOptions:
        """Build the SDK's ``LocalAgentConfig`` arguments from ``spec`` + runtime ``ctx``."""
        # Restrictions, not hints: validate before SDK setup or filesystem side effects.
        model = spec.model or ""
        if not model:
            raise ValueError("the antigravity harness needs spec.model (e.g. 'gemini-4-neon')")
        if model.startswith(OPENROUTER_PREFIX):
            raise ValueError("the antigravity harness has no OpenRouter route; use a Gemini model")
        if spec.permission_mode != "bypassPermissions":
            raise ValueError(
                "the antigravity harness supports only permission_mode='bypassPermissions' "
                "(its other modes ask a human, which a headless run cannot answer)"
            )
        if spec.allowed_tools is not None or spec.disallowed_tools:
            raise ValueError("allowed_tools/disallowed_tools are not mapped on antigravity yet")
        if ctx.hooks:
            raise ValueError(
                "run(hooks=...) is Claude-specific and has no antigravity equivalent; the "
                "turn would run with the hooks never called. Drop them, or run this turn on "
                "the claude-code harness."
            )
        api_key = next(
            (ctx.secrets[k] for k in GEMINI_KEY_NAMES if ctx.secrets.get(k)), None
        ) or next((os.environ[k] for k in GEMINI_KEY_NAMES if os.environ.get(k)), None)
        if not api_key:
            raise RuntimeError(
                "antigravity harness has no Gemini credentials: pass GEMINI_API_KEY in the "
                "per-invocation secrets (or set it in the environment)"
            )
        sdk = _sdk()
        from google.antigravity import types as sdk_types
        from google.antigravity.hooks import policy

        warnings: list[str] = []
        if spec.codex_config:
            warnings.append("codex_config is Codex-specific and has no antigravity equivalent; ignored")
        options = None
        effort = spec.reasoning_effort
        if effort is not None:
            level = _EFFORT_MAP.get(effort) or _EFFORT_COERCED.get(effort)
            if level is None:
                raise ValueError(f"reasoning_effort {effort!r} has no antigravity thinking level")
            if effort in _EFFORT_COERCED:
                warnings.append(f"reasoning_effort {effort!r} has no Gemini level; using {level!r}")
            options = sdk.GeminiModelOptions(thinking_level=sdk.ThinkingLevel(level))
        endpoint = sdk.GeminiAPIEndpoint(api_key=api_key, options=options)
        target = sdk.ModelTarget(name=model, types=[sdk.ModelType.TEXT], endpoint=endpoint)

        state_dir = ctx.job_dir / _STATE_DIR
        resume = bool(ctx.resume_sid)
        conversation_id = conversation_id_for(ctx.resume_sid or ctx.session_id)

        skills_dir = ctx.workspace / ".agents" / "skills"
        kwargs: dict[str, Any] = {
            "models": [target],
            "capabilities": sdk.CapabilitiesConfig(),
            "policies": [policy.allow_all()],
            "workspaces": [str(ctx.workspace)],
            "env": self._agent_env(spec, ctx),
            "save_dir": str(state_dir / "save"),
            "app_data_dir": str(state_dir / "app"),
            "conversation_id": conversation_id,
            "session_continuation_mode": (
                sdk_types.SessionContinuationMode.CREATE_OR_RESUME
                if resume
                else sdk_types.SessionContinuationMode.CREATE_ONLY
            ),
            "budget_config": sdk_types.BudgetConfig(
                max_model_calls=spec.max_turns, scope=sdk_types.BudgetScope.FORWARD_LOOKING
            ),
            "mcp_servers": self._mcp_servers(spec, ctx) or None,
            "skills_paths": [str(skills_dir)] if skills_dir.is_dir() else None,
            "system_instructions": self._system_instructions(spec, ctx.interactive),
        }
        if spec.output_schema is not None:
            from ..spec import _output_schema_to_dict

            kwargs["response_schema"] = _output_schema_to_dict(spec.output_schema)
        return _AntigravityOptions(
            config_kwargs={k: v for k, v in kwargs.items() if v is not None},
            conversation_id=conversation_id,
            state_dir=state_dir,
            resume=resume,
            warnings=warnings,
        )

    # -- conversation persistence ---------------------------------------------

    def _persist_state(self, ctx: RunContext, state_dir: Path) -> None:
        """Archive the conversation state (SQLite store + brain dirs) to the BlobStore."""
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for sub in ("save", "app"):
                if (state_dir / sub).exists():
                    tar.add(state_dir / sub, arcname=sub)
        ctx.blobs.put_bytes(f"{_STATE_PREFIX}/{ctx.session_id}/state.tar.gz", buf.getvalue())

    def _restore_state(self, ctx: RunContext, opts: _AntigravityOptions) -> bool:
        """Materialize a persisted conversation; ``False`` when there is none to restore.

        On ``local`` the job dir survives between turns, so a conversation that is already
        on disk is used as is. The archive is written by the run's own token, so it is
        untrusted: extraction uses tarfile's ``data`` filter (no links out, no absolute or
        ``..`` paths, no devices).
        """
        if (opts.state_dir / "save" / f"{opts.conversation_id}.db").exists():
            return True
        if ctx.blobs is None:
            return False
        try:
            body = ctx.blobs.get_bytes(f"{_STATE_PREFIX}/{ctx.resume_sid}/state.tar.gz")
        except KeyError:  # a known absent checkpoint may legitimately start fresh
            return False
        except Exception:  # noqa: BLE001 — never echo credential-bearing storage errors
            raise RuntimeError("Antigravity checkpoint could not be read") from None
        opts.state_dir.mkdir(parents=True, exist_ok=True)
        try:
            with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as tar:
                tar.extractall(opts.state_dir, filter="data")
        except (tarfile.TarError, OSError, EOFError):
            raise RuntimeError("Antigravity checkpoint is invalid") from None
        return True

    def _finalize(self, spec: AgentSpec, ctx: RunContext, opts: _AntigravityOptions) -> AgentEvent | None:
        """Inline checkpoint: workspace snapshot (shared) + the Antigravity conversation."""
        conversation_error = False
        if spec.checkpoint and ctx.blobs is not None:
            try:
                self._persist_state(ctx, opts.state_dir)
            except Exception:  # noqa: BLE001 — checkpoint is best-effort
                conversation_error = True
        workspace_event = finalize_checkpoint(spec, ctx)
        if conversation_error:
            workspace_key = (workspace_event.raw or {}).get("workspace_key") if workspace_event else None
            return AgentEvent(
                kind="status", summary="checkpoint failed: Antigravity conversation was not saved",
                raw={"event": "checkpoint_error", "session_id": ctx.session_id,
                     "conversation_saved": False, "workspace_saved": bool(workspace_key),
                     **({"workspace_key": workspace_key} if workspace_key else {})},
            )
        return workspace_event

    # -- run loop ---------------------------------------------------------------

    def _result_event(
        self,
        *,
        translator: AntigravityStepTranslator,
        ctx: RunContext,
        spec: AgentSpec,
        opts: _AntigravityOptions,
        price: pricing.ModelPrice | None,
        counts: dict[str, int],
        subagent_counts: dict[str, dict[str, int]],
        model_calls: int,
        limit: str | None,
        error: str | None,
        stop_reason: str | None,
    ) -> AgentEvent:
        if error is not None:
            subtype, is_error, text = "error", True, error
        elif limit == "max_turns":
            subtype, is_error = "error_max_turns", True
            text = translator.final_text or f"(stopped at max_turns={spec.max_turns})"
        elif limit == "budget":
            subtype, is_error = "error_budget_exceeded", True
            text = translator.final_text or "(interrupted: budget exceeded)"
        elif limit is not None:  # quota / token caps
            subtype, is_error = "error", True
            text = translator.final_text or f"(stopped: {stop_reason})"
        elif stop_reason == "interrupted":
            subtype, is_error = "interrupted", False
            text = translator.final_text or "(interrupted)"
        else:
            subtype, is_error = "success", False
            if translator.structured_output is not None:
                text = json.dumps(translator.structured_output)
            else:
                text = translator.final_text or "(no final text)"
        raw: dict[str, Any] = {
            "subtype": subtype,
            "is_error": is_error,
            "num_turns": model_calls,
            "session_id": ctx.session_id,
            "conversation_id": opts.conversation_id,
            "model": spec.model,
            "price_source": price.source if price else None,
            "stop_reason": stop_reason,
        }
        if subagent_counts:
            raw["subagent_usage"] = {
                tid: {**normalized_usage(c), "model": spec.model}
                for tid, c in subagent_counts.items()
            }
        return AgentEvent(
            kind="result",
            summary=text,
            cost_usd=cost_usd(price, counts),
            usage=normalized_usage(counts),
            raw=raw,
        )

    async def run(self, spec: AgentSpec, ctx: RunContext) -> AsyncIterator[AgentEvent]:
        """Drive one turn over an Antigravity ``Conversation``, yielding ``AgentEvent``s.

        The stream is ``Conversation.receive_steps()`` (it ends when the agent goes idle).
        Cost comes from the SDK's cumulative usage counters, checked after every step
        against ``spec.max_budget_usd``; ``max_turns`` is the SDK's own model-call budget.
        Operator control (``ctx.control``) is read alongside the stream: an ``interrupt``
        or ``stop`` cancels the turn (``AntigravityCancelledError`` ends the stream); an
        interrupt's message, and any ``steer``, start the next turn on the same
        conversation. Checkpoint finalizes inline at the terminal result event.
        """
        options = self.build_options(spec, ctx)
        sdk = _sdk()
        sdk_types = sdk.types
        if not options.resume:
            # A fresh conversation: a previous run's state for this session would make the
            # SDK's CREATE_ONLY fail (local re-runs share the job dir).
            shutil.rmtree(options.state_dir, ignore_errors=True)
        elif not self._restore_state(ctx, options):
            options.config_kwargs["session_continuation_mode"] = (
                sdk_types.SessionContinuationMode.CREATE_ONLY
            )
        options.state_dir.mkdir(parents=True, exist_ok=True)

        price = await asyncio.to_thread(pricing.model_price, spec.model)
        translator = AntigravityStepTranslator(options.conversation_id)
        config = sdk.LocalAgentConfig(**options.config_kwargs)

        limit: str | None = None
        error: str | None = None
        stop_reason: str | None = None
        model_calls = 0
        steers: list[ControlMessage] = []
        stop_requested: ControlMessage | None = None
        interrupt_pending: ControlMessage | None = None
        select: ControlledStream | None = None
        counts: dict[str, int] = {}
        subagent_counts: dict[str, dict[str, int]] = {}

        async with sdk.Agent(config) as agent:
            conv = agent.conversation
            yield AgentEvent(
                kind="status",
                summary=f"antigravity conversation started ({spec.model})",
                raw={
                    "subtype": "init",
                    "data": {
                        "model": spec.model,
                        "conversation_id": options.conversation_id,
                        "session_id": ctx.session_id,
                        "resumed": options.resume,
                        "skills_paths": options.config_kwargs.get("skills_paths") or [],
                    },
                },
            )
            for w in options.warnings:
                yield AgentEvent(kind="status", summary=w, raw={"event": "spec_warning"})
            if price is None:
                yield AgentEvent(
                    kind="status",
                    summary=(
                        f"no price data for model {spec.model!r} (LiteLLM dataset unreachable "
                        "or model unknown): cost_usd will be unknown and max_budget_usd "
                        "cannot be enforced"
                    ),
                    raw={"event": "cost_unknown", "model": spec.model},
                )
            usage = _Usage(conv.total_usage)
            traj_start = {
                tid: _Usage(u) for tid, u in (getattr(conv, "trajectory_usages", None) or {}).items()
            }
            # The main agent's cumulative total; each change is one model call of the main
            # agent (the SDK sends a usage update per call; two calls between steps would
            # count as one, so num_turns is a lower bound).
            main_total = int(getattr(
                (getattr(conv, "trajectory_usages", None) or {}).get(options.conversation_id),
                "total_token_count", None,
            ) or 0)

            def observe() -> None:
                """Refresh turn usage + the main agent's model-call count from the SDK."""
                nonlocal counts, main_total, model_calls
                counts = usage.delta(conv.total_usage)
                for tid, u in (getattr(conv, "trajectory_usages", None) or {}).items():
                    if tid == options.conversation_id:
                        total = int(getattr(u, "total_token_count", None) or 0)
                        if total > main_total:
                            model_calls += 1
                            main_total = total
                    else:
                        c = traj_start.get(tid, _Usage(None)).delta(u)
                        if any(c.values()):
                            subagent_counts[tid] = c

            async def start_turn(message: str) -> None:
                nonlocal select
                await conv.send(message)
                stream = _steps(conv, sdk_types.AntigravityCancelledError)
                if select is None:
                    select = ControlledStream(stream, ctx.control)
                else:
                    select.replace_stream(stream)

            try:
                await start_turn(ctx.prompt)
                assert select is not None
                cancel_deadline: float | None = None
                while True:
                    timeout = None
                    if cancel_deadline is not None:
                        timeout = max(0.0, cancel_deadline - asyncio.get_running_loop().time())
                    kind, item = await select.next(timeout)
                    if kind == "message" and isinstance(item, _TurnEnded):
                        if item.error is not None and not item.cancelled:
                            observe()
                            error = item.error
                            break
                        kind = "end"
                    if kind == "message":
                        observe()
                        for event in translator.translate(item):
                            yield event
                        if limit is None and cancel_deadline is None:
                            cost = cost_usd(price, counts)
                            if cost is not None and cost >= spec.max_budget_usd:
                                limit = "budget"
                                yield AgentEvent(
                                    kind="status",
                                    summary=f"budget cap hit (cost=${cost:.4f}); cancelling the turn",
                                    raw={"event": "limit_interrupt", "limit": "budget"},
                                )
                                await conv.cancel()
                                cancel_deadline = asyncio.get_running_loop().time() + _CANCEL_SETTLE_S
                        continue
                    if kind == "control":
                        msg = item
                        if msg.op == "steer":
                            steers.append(msg)
                            yield AgentEvent(
                                kind="status",
                                summary="steer queued: antigravity takes it when the current turn ends",
                                raw={"event": "steer_queued", "message_id": msg.message_id},
                            )
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
                        if msg.op == "stop":
                            stop_requested = msg
                        else:
                            interrupt_pending = msg
                        await conv.cancel()
                        cancel_deadline = asyncio.get_running_loop().time() + _CANCEL_SETTLE_S
                        continue
                    # "end", "cancelled" or "timeout": the turn is over.
                    observe()
                    cancel_deadline = None
                    if limit is None:
                        cost = cost_usd(price, counts)
                        if cost is not None and cost >= spec.max_budget_usd:
                            limit = "budget"
                    if limit is None:
                        reason = _value(getattr(conv, "_last_turn_stop_reason", None))
                        if reason == "MAX_MODEL_CALLS_EXCEEDED":
                            limit, stop_reason = "max_turns", reason
                        elif reason not in (None, "UNSPECIFIED"):
                            limit, stop_reason = "sdk_budget", reason
                    if limit is not None:
                        break
                    if interrupt_pending is not None:
                        msg, interrupt_pending = interrupt_pending, None
                        yield AgentEvent(
                            kind="status",
                            summary="turn interrupted; continuing with the operator's message",
                            raw={"event": "interrupted_segment", "message_id": msg.message_id},
                        )
                        await start_turn(msg.message or "")
                        yield msg.user_event()
                        continue
                    if stop_requested is not None:
                        stop_reason = "interrupted"
                        break
                    if steers:
                        msg = steers.pop(0)
                        await start_turn(msg.message or "")
                        yield msg.user_event()
                        continue
                    break
            finally:
                if select is not None:
                    await select.close()

        for event in translator.unfinished():
            yield event
        fin = self._finalize(spec, ctx, options)
        if fin is not None:
            yield fin
        yield self._result_event(
            translator=translator, ctx=ctx, spec=spec, opts=options, price=price,
            counts=counts, subagent_counts=subagent_counts, model_calls=model_calls,
            limit=limit, error=error, stop_reason=stop_reason,
        )

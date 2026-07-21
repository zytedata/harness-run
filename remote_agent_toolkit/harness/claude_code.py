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

import os
from collections import deque
from pathlib import Path
from typing import Any, AsyncIterator, TYPE_CHECKING

from ..events import AgentEvent
from ..spec import SystemPrompt

if TYPE_CHECKING:
    from ..spec import AgentSpec
    from .context import RunContext

# Sensible coding-agent default when ``spec.allowed_tools`` is unset: the filesystem/exec
# tools skills depend on. The Skill tool is enabled separately via the ``skills`` option,
# not listed here (passing "Skill" in allowed_tools is deprecated in the SDK).
DEFAULT_ALLOWED_TOOLS = ("Read", "Write", "Edit", "Bash", "Glob", "Grep", "TodoWrite")

# Conventional GitHub token names a ``github`` MCP server pulls from the per-invocation secrets.
_GITHUB_MCP_TOKEN_KEYS = ("GH_TOKEN", "GITHUB_TOKEN", "GH_PAT")


def harness_consumed_secret_names(spec: AgentSpec) -> set[str]:
    """Secret names the *harness* consumes on the agent's behalf, so they stay OUT of its env.

    A repo's ``auth`` token is embedded into ``origin`` (git uses it), and a GitHub MCP's token
    goes into the server's headers — neither needs to be a plain environment variable the agent
    (or a prompt-injection) can read. Everything else the caller passes is the agent's own to use.
    """
    names: set[str] = {r.auth for r in spec.repos if getattr(r, "auth", None)}
    if any(m.kind == "github" for m in spec.mcp_servers):
        names.update(_GITHUB_MCP_TOKEN_KEYS)
    return names

# Appended to the system prompt in interactive (checkpoint) mode. Turns every "present X
# and wait for the user" step into "end the turn and await the next message", and steers
# the agent away from interactive prompt tools that can't work in a one-shot background
# job. The operator's reply arrives as the next turn (a resume). Generalized from the PoC
# (no Zyte/spider specifics).
_INTERACTIVE_SUFFIX = (
    "\n\nINTERACTIVE MODE: You are in a multi-turn conversation with a human operator who "
    "can reply between your turns. A step that would normally prompt the operator must be "
    "handled one of two ways:\n"
    "- Genuine decision (the operator's input actually changes what you do): present it "
    "clearly, then STOP — end your turn and wait. Do NOT call interactive prompt tools and "
    "do NOT block in-process; just finish your message with a specific question. The "
    "operator's reply arrives as your next message and you continue where you left off "
    "(workspace and conversation are preserved).\n"
    "- Trivial setup choice with an obvious default: pick the sensible default and proceed "
    "WITHOUT stopping.\n"
    "CRITICAL: do NOT end your turn at intermediate stage boundaries, and never stop merely "
    "to 'continue later'. Once the operator has given the go-ahead, run ALL remaining steps "
    "to completion in the SAME turn. End your turn ONLY when you genuinely need an operator "
    "decision, or when the entire task is complete — and then say so explicitly."
)


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


class ClaudeCodeHarness:
    """The Claude Agent SDK harness (implements ``harness.base.Harness``)."""

    # -- option building -------------------------------------------------------

    def _system_prompt(self, spec: AgentSpec, interactive: bool) -> Any:
        """Resolve ``spec.system_prompt`` (+ interactive suffix) to an SDK value.

        * ``SystemPrompt`` → inherit Claude Code's preset and append.
        * plain ``str``    → replace the prompt entirely (suffix appended in interactive).
        * ``None``         → harness default (a preset+suffix only if interactive).
        """
        suffix = _INTERACTIVE_SUFFIX if interactive else ""
        sp = spec.system_prompt
        if isinstance(sp, str):
            return sp + suffix if suffix else sp
        if isinstance(sp, SystemPrompt):
            append = (sp.append or "") + suffix
            preset: dict[str, Any] = {"type": "preset", "preset": "claude_code"}
            if append:
                preset["append"] = append
            return preset
        # None: only override the default when we must inject the interactive suffix.
        if suffix:
            return {"type": "preset", "preset": "claude_code", "append": suffix}
        return None

    def _runtime_env(self, spec: AgentSpec, ctx: RunContext) -> dict[str, str]:
        """Env forwarded to the agent subprocess: agent-visible secrets + spec/runtime env + uv PATH.

        Layering (later wins): per-invocation secrets → ``spec.env`` (caller's static config)
        → ``ctx.env`` (runtime-resolved, e.g. the local deploy-time packages venv). Only the
        caller's own secrets land here — those consumed by the harness (repo push tokens,
        GitHub MCP token) are routed to git/MCP and excluded, so they never appear as
        environment variables the agent can read.

        SECURITY: returns secret *values* — callers must never log this dict.
        """
        consumed = harness_consumed_secret_names(spec)
        env: dict[str, str] = {k: v for k, v in ctx.secrets.items() if k not in consumed}
        if spec.env:
            env.update(spec.env)
        if ctx.env:
            env.update(ctx.env)
        # Ensure `uv` is on the agent's Bash-tool PATH. uv is a Python dependency, but its
        # console script often isn't on PATH in serverless runtimes; the wheel exposes the
        # bundled binary's real location, so prepend that dir (harmless if uv is absent).
        path_dirs: list[str] = []
        try:
            from uv import find_uv_bin

            path_dirs.append(os.path.dirname(find_uv_bin()))
        except Exception:  # noqa: BLE001 — uv may be absent; harmless
            pass
        if path_dirs:
            # A PATH supplied by the caller/runtime (spec.env / ctx.env) is the BASE, not
            # discarded — prepending onto os.environ silently threw caller intent away
            # (eval-harness feedback issue 2). Ambient PATH is only the fallback base.
            base = env.get("PATH") or os.environ.get("PATH", "")
            env["PATH"] = ":".join(path_dirs) + (f":{base}" if base else "")
        return env

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
            env=self._runtime_env(spec, ctx),
            mcp_servers=self._mcp_servers(spec, ctx),
            system_prompt=self._system_prompt(spec, ctx.interactive),
            **extra,
        )

    # -- run loop --------------------------------------------------------------

    def _finalize(self, spec: AgentSpec, ctx: RunContext) -> AgentEvent | None:
        """Checkpoint the workspace inline at the terminal result event (DESIGN.md §6).

        Done as a direct side-effect (not post-loop): the async Agent Engine executor
        stops draining the generator after the final event, so post-loop work is dead
        code in-cloud. Best-effort — a checkpoint failure never fails the run.
        """
        if not (spec.checkpoint and ctx.blobs is not None and ctx.session_store is not None):
            return None
        from ..checkpoint.workspace import snapshot
        from ..integrations.git import scrub_repo_tokens

        try:
            # Never persist push tokens to the checkpoint: strip them from every .git/config
            # before archiving. resume re-embeds from the freshly supplied per-invocation secret.
            scrub_repo_tokens(str(ctx.workspace))
            key = snapshot(ctx.blobs, ctx.session_id, str(ctx.workspace))
            return AgentEvent(
                kind="status",
                summary=f"checkpoint saved ({ctx.session_id})",
                raw={"event": "checkpoint_saved", "workspace_key": key, "session_id": ctx.session_id},
            )
        except Exception as exc:  # noqa: BLE001 — checkpoint is best-effort
            return AgentEvent(
                kind="status",
                summary=f"checkpoint failed: {str(exc)[:120]}",
                raw={"event": "checkpoint_error", "session_id": ctx.session_id},
            )

    async def run(self, spec: AgentSpec, ctx: RunContext) -> AsyncIterator[AgentEvent]:
        """Drive the Agent SDK ``query()`` loop, yielding ``AgentEvent``s.

        Finalizes (checkpoint) inline at the terminal ``result`` event, before the
        trailing checkpoint status event, so the snapshot happens while a non-draining
        executor is still pulling.
        """
        from .translate import EventTranslator

        options = self.build_options(spec, ctx)
        # Capture the CLI's stderr to <job_dir>/stderr.log + an in-memory tail; without a
        # registered callback the SDK doesn't pipe it at all and CLI-exit failures are
        # undiagnosable by construction (the ProcessError text promises stderr details).
        stderr_log = _StderrCapture(ctx.job_dir / "stderr.log")
        options.stderr = stderr_log
        translator = EventTranslator()
        finalized = False

        from claude_agent_sdk import query

        try:
            async for message in query(prompt=ctx.prompt, options=options):
                for event in translator.translate(message):
                    if not finalized and event.kind == "result":
                        fin = self._finalize(spec, ctx)
                        finalized = True
                        yield event
                        if fin is not None:
                            yield fin
                    else:
                        yield event
        except Exception as exc:
            # Surface the captured stderr with the failure: as a status event (reaches the
            # stream / Cloud Logging on gemini) and embedded in the raised error, so a
            # non-zero CLI exit is classifiable post-mortem instead of "check stderr".
            tail = stderr_log.tail()
            if not tail:
                raise
            yield AgentEvent(
                kind="status",
                summary=f"claude stderr (tail): {tail}",
                raw={"event": "claude_stderr", "log": str(stderr_log.path)},
            )
            raise RuntimeError(
                f"{exc} [claude stderr tail: {tail[-500:]}] (full log: {stderr_log.path})"
            ) from exc

        # Fallback for runners that fully drain the generator, or if no terminal event
        # was observed (defensive — query() normally always ends with a ResultMessage).
        if not finalized:
            fin = self._finalize(spec, ctx)
            if fin is not None:
                yield fin

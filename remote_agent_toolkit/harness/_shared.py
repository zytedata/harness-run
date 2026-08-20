"""Helpers shared by the harness bindings (``claude_code``, ``codex``).

Everything here is harness-agnostic policy — how secrets are routed, how the agent
subprocess env is layered, the interactive-mode prompt suffix, the inline workspace
checkpoint — extracted from the Claude binding when the Codex binding arrived so the
two stay behaviorally identical where the spec doesn't distinguish them. Stdlib-only
at import time.
"""

from __future__ import annotations

import os
from typing import Any, TYPE_CHECKING

from ..events import AgentEvent

if TYPE_CHECKING:
    from ..spec import AgentSpec
    from .context import RunContext

# Conventional GitHub token names a ``github`` MCP server pulls from the per-invocation secrets.
GITHUB_MCP_TOKEN_KEYS = ("GH_TOKEN", "GITHUB_TOKEN", "GH_PAT")

# Model-auth secrets the Codex binding routes itself: OpenAI directly, or OpenRouter for an
# ``openrouter/``-prefixed model (see ``harness.codex``).
CODEX_MODEL_AUTH_KEYS = ("OPENAI_API_KEY", "OPENROUTER_API_KEY")


def harness_consumed_secret_names(spec: AgentSpec) -> set[str]:
    """Secret names the *harness* consumes on the agent's behalf, so they stay OUT of its env.

    A repo's ``auth`` token is embedded into ``origin`` (git uses it), and a GitHub MCP's token
    goes into the server's headers — neither needs to be a plain environment variable the agent
    (or a prompt-injection) can read. The Codex binding additionally consumes its model-auth
    keys: ``OPENAI_API_KEY`` (routed to ``codex login``) and ``OPENROUTER_API_KEY`` (routed to
    the OpenRouter provider config). The Claude binding consumes ``OPENROUTER_API_KEY`` too
    when the model is an ``openrouter/`` one (routed to ``ANTHROPIC_AUTH_TOKEN``). Both are listed whichever model the turn runs, so passing
    one the turn doesn't use still doesn't leak it into the agent's shell. Everything else the
    caller passes is the agent's own to use.
    """
    names: set[str] = {r.auth for r in spec.repos if getattr(r, "auth", None)}
    if any(m.kind == "github" for m in spec.mcp_servers):
        names.update(GITHUB_MCP_TOKEN_KEYS)
    if getattr(spec, "harness", "claude-code") == "codex":
        names.update(CODEX_MODEL_AUTH_KEYS)
    elif (getattr(spec, "model", "") or "").startswith("openrouter/"):
        # claude-code reaches OpenRouter through ANTHROPIC_AUTH_TOKEN, so the key is the
        # harness's to route here too.
        names.add("OPENROUTER_API_KEY")
    return names


# Appended to the system prompt in interactive (checkpoint) mode. Turns every "present X
# and wait for the user" step into "end the turn and await the next message", and steers
# the agent away from interactive prompt tools that can't work in a one-shot background
# job. The operator's reply arrives as the next turn (a resume). Generalized from the PoC
# (no Zyte/spider specifics).
INTERACTIVE_SUFFIX = (
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


def runtime_env(spec: AgentSpec, ctx: RunContext) -> dict[str, str]:
    """Env forwarded to the agent subprocess: agent-visible secrets + spec/runtime env + uv PATH.

    Layering (later wins): per-invocation secrets → ``spec.env`` (caller's static config)
    → ``ctx.env`` (runtime-resolved, e.g. the local deploy-time packages venv). Only the
    caller's own secrets land here — those consumed by the harness (repo push tokens,
    GitHub MCP token, Codex's model-auth keys) are routed to git/MCP/the model provider
    and excluded,
    so they never appear as environment variables the agent can read.

    SECURITY: returns secret *values* — callers must never log this dict.
    """
    consumed = harness_consumed_secret_names(spec)
    env: dict[str, str] = {k: v for k, v in ctx.secrets.items() if k not in consumed}
    if spec.env:
        env.update(spec.env)
    if ctx.env:
        env.update(ctx.env)
    # Ensure `uv` is on the agent's shell PATH. uv is a Python dependency, but its
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


def finalize_checkpoint(spec: AgentSpec, ctx: Any) -> AgentEvent | None:
    """Checkpoint the workspace inline at the terminal result event (DESIGN.md §6).

    Done as a direct side-effect (not post-loop): the async Agent Engine executor
    stops draining the generator after the final event, so post-loop work is dead
    code in-cloud. Best-effort — a checkpoint failure never fails the run.

    A caller-owned cwd (``ctx.workspace_dir``) is checkpointed as conversation only: the
    directory is already durable, so there is nothing to preserve, while snapshotting it
    would archive whatever else lives there, restoring it would roll files back over newer
    work, and the pre-archive scrub would rewrite the caller's own ``.git/config`` files.
    """
    if not (spec.checkpoint and ctx.blobs is not None and ctx.session_store is not None):
        return None
    if ctx.workspace_dir is not None:
        return AgentEvent(
            kind="status",
            summary=f"checkpoint saved, conversation only ({ctx.session_id})",
            raw={
                "event": "checkpoint_saved",
                "workspace_key": None,
                "session_id": ctx.session_id,
            },
        )
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

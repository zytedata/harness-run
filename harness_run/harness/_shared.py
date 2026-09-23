"""Helpers shared by the harness bindings (``claude_code``, ``codex``).

Everything here is harness-agnostic policy — how secrets are routed, how the agent
subprocess env is layered, the interactive-mode prompt suffix, the inline workspace
checkpoint, the OpenRouter proxy's lifecycle and events — extracted from the Claude
binding when the Codex binding arrived so the two stay behaviorally identical where the
spec doesn't distinguish them. Stdlib-only at import time.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, AsyncIterator, Callable, TYPE_CHECKING

from ..events import AgentEvent

if TYPE_CHECKING:
    from ..spec import AgentSpec
    from .context import RunContext

# Conventional GitHub token names a ``github`` MCP server pulls from the per-invocation secrets.
GITHUB_MCP_TOKEN_KEYS = ("GH_TOKEN", "GITHUB_TOKEN", "GH_PAT")

# The model-id prefix that routes a turn to OpenRouter, and the key that pays for it. The
# rest of the OpenRouter helpers are grouped further down.
OPENROUTER_PREFIX = "openrouter/"
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"

# Model-auth secrets the Codex binding routes itself: OpenAI directly, or OpenRouter for an
# ``openrouter/``-prefixed model (see ``harness.codex``).
CODEX_MODEL_AUTH_KEYS = ("OPENAI_API_KEY", OPENROUTER_KEY_ENV)


def harness_consumed_secret_names(spec: AgentSpec) -> set[str]:
    """Secret names the *harness* consumes on the agent's behalf, so they stay OUT of its env.

    A repo's ``auth`` token is embedded into ``origin`` (git uses it), and a GitHub MCP's token
    goes into the server's headers — neither needs to be a plain environment variable the agent
    (or a prompt-injection) can read. The Codex binding additionally consumes its model-auth
    keys: ``OPENAI_API_KEY`` (routed to ``codex login``) and ``OPENROUTER_API_KEY`` (routed to
    the OpenRouter provider config). Codex lists both whichever model the turn runs, so passing
    one the turn doesn't use still doesn't leak it into the agent's shell. The Claude binding
    consumes ``OPENROUTER_API_KEY`` only for an ``openrouter/`` model (routed to
    ``ANTHROPIC_AUTH_TOKEN``); on a native Anthropic turn it has no model-auth key to route,
    so one passed anyway stays the agent's own. Everything else the caller passes is the
    agent's own to use.
    """
    names: set[str] = {r.auth for r in spec.repos if getattr(r, "auth", None)}
    for server in spec.mcp_servers:
        names.update((server.header_secrets or {}).values())
    if any(m.kind == "github" for m in spec.mcp_servers):
        names.update(GITHUB_MCP_TOKEN_KEYS)
    if getattr(spec, "harness", "") == "codex":
        names.update(CODEX_MODEL_AUTH_KEYS)
    elif (getattr(spec, "model", "") or "").startswith(OPENROUTER_PREFIX):
        # claude-code reaches OpenRouter through ANTHROPIC_AUTH_TOKEN, so the key is the
        # harness's to route here too.
        names.add(OPENROUTER_KEY_ENV)
    return names


def mcp_header_secrets(server: Any, ctx: RunContext) -> dict[str, str]:
    """Resolve header references only at the harness boundary; never log this result."""
    server._validate_headers()
    headers = {}
    for header, name in (server.header_secrets or {}).items():
        value = ctx.secrets.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError("required MCP header secret is missing from run(secrets=...)")
        headers[header] = value
    return headers


# Appended to the system prompt in interactive (checkpoint) mode. Turns every "present X
# and wait for the user" step into "end the turn and await the next message", and steers
# the agent away from interactive prompt tools that can't work in a one-shot background
# job. The operator's reply arrives as the next turn (a resume). Generalized from the PoC
# (nothing project-specific).
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
    → ``ctx.env`` (runtime-resolved, e.g. the local deploy-time packages venv). A ``None``
    in ``spec.env`` drops the name from the layers below it; the harness must also remove
    it from the ambient environment (see :func:`unset_env_names`). Only the
    caller's own secrets land here — those consumed by the harness (repo push tokens,
    GitHub MCP token, the model-auth keys) are routed to git/MCP/the model provider and
    excluded, so they never appear as environment variables the agent can read.

    SECURITY: returns secret *values* — callers must never log this dict.
    """
    from ..spec import _assert_non_secret_env

    _assert_non_secret_env(spec.env)
    consumed = harness_consumed_secret_names(spec)
    env: dict[str, str] = {k: v for k, v in ctx.secrets.items() if k not in consumed}
    for name, value in (spec.env or {}).items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
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


def unset_env_names(spec: AgentSpec, ctx: RunContext) -> set[str]:
    """Names ``spec.env`` unsets, which the harness removes from the agent's shell."""
    return {k for k, v in (spec.env or {}).items() if v is None} - set(ctx.env or {})


# -- OpenRouter ---------------------------------------------------------------
#
# Both bindings run ``openrouter/`` models, and both do it through the same per-run proxy
# (:mod:`._openrouter_proxy`). What differs is only how each CLI is configured to reach it
# — Claude Code through environment variables, Codex through ``--config`` overrides — so
# that part stays in the bindings. Everything below is the same on either one.

# How long to wait for the proxy to record the last response's charge before a result
# quotes its total.
METADATA_DRAIN_TIMEOUT_S = 5.0

# OpenRouter passes each CLI's native structured-output format on to the selected model,
# but does NOT enforce it — compliance is the model's. Measured 2026-08-20: DeepSeek v4
# Flash/Pro and Kimi K3 returned clean JSON; GLM-5.3 returned prose 3/3 times, which parses
# to structured_output=None. So the schema is asked for in words as well. The toolkit's own
# parse already tolerates a fenced block, but not prose.
OPENROUTER_SCHEMA_INSTRUCTION = (
    "\n\nFINAL MESSAGE FORMAT: your last message of the turn must be ONLY a single JSON "
    "object conforming to this schema — no prose before or after it, no code fence, no "
    "explanation:\n{schema}"
)


def openrouter_api_key(ctx: RunContext) -> str | None:
    """The OpenRouter key for this turn: the caller's secret, else the ambient one."""
    return ctx.secrets.get(OPENROUTER_KEY_ENV) or os.environ.get(OPENROUTER_KEY_ENV)


def missing_openrouter_key_error(harness: str) -> RuntimeError:
    return RuntimeError(
        f"{harness} harness has no OpenRouter credentials: pass "
        f"{OPENROUTER_KEY_ENV} in the per-invocation secrets (or set it in the environment)"
    )


def openrouter_provider_routing(spec: AgentSpec) -> dict[str, Any] | None:
    """The OpenRouter ``provider`` object this spec asks for, or ``None`` for no preference.

    ``spec.openrouter_provider`` is the shorthand for pinning one provider, and it means
    exactly ``{"only": [slug], "allow_fallbacks": False}`` — ``AgentSpec`` says so in the
    error it raises when both fields are set. Resolving it here means everything below the
    spec carries one value instead of two, and never has to ask which of the two the caller
    used.
    """
    if spec.openrouter_routing is not None:
        return dict(spec.openrouter_routing)
    if spec.openrouter_provider is not None:
        return {"only": [spec.openrouter_provider], "allow_fallbacks": False}
    return None


def openrouter_schema_steer(spec: AgentSpec) -> str:
    """Prompt text asking for a bare JSON object, for OpenRouter models only."""
    if not (spec.model or "").startswith(OPENROUTER_PREFIX) or spec.output_schema is None:
        return ""
    from ..spec import _output_schema_to_dict

    schema = _output_schema_to_dict(spec.output_schema)
    if not isinstance(schema, dict):
        return ""
    return OPENROUTER_SCHEMA_INSTRUCTION.format(schema=json.dumps(schema, separators=(",", ":")))


def openrouter_cost_unknown(model: str | None) -> AgentEvent:
    return AgentEvent(
        kind="status",
        summary=(
            f"OpenRouter did not report a charge for model {model!r}; cost_usd is "
            "unknown and max_budget_usd could not be enforced"
        ),
        raw={"event": "cost_unknown", "model": model},
    )


async def drain_proxy(proxy: Any | None) -> AsyncIterator[AgentEvent]:
    """Let the proxy finish recording before the result quotes its total.

    The CLI sees the last response bytes before the proxy's handler has parsed them, so a
    result built immediately can miss the final request's charge. The recorded requests are
    yielded first, then the notice if the wait ran out — that notice qualifies the total the
    result is about to report.
    """
    if proxy is None:
        return
    complete = await asyncio.to_thread(proxy.wait_until_idle, METADATA_DRAIN_TIMEOUT_S)
    for event in proxy.drain_events():
        yield event
    if not complete:
        yield AgentEvent(
            kind="status",
            summary="timed out waiting for OpenRouter response metadata",
            raw={"event": "openrouter_metadata_timeout"},
        )


async def run_with_openrouter_proxy(
    spec: AgentSpec,
    ctx: RunContext,
    run_inner: Callable[[Any | None], AsyncIterator[AgentEvent]],
) -> AsyncIterator[AgentEvent]:
    """Run a turn behind a per-run OpenRouter proxy, when the turn is an OpenRouter one.

    ``run_inner`` receives the proxy, or ``None`` when there is nothing to proxy: a native
    model, or an ``openrouter/`` model with no key (the binding raises for that itself, with
    its own name in the message). The proxy is closed however the turn ends.
    """
    from ._openrouter_proxy import OpenRouterProxy

    proxy = None
    if (spec.model or "").startswith(OPENROUTER_PREFIX):
        api_key = openrouter_api_key(ctx)
        if api_key:
            proxy = OpenRouterProxy(
                api_key,
                spec.max_budget_usd,
                expected_model=(spec.model or "").removeprefix(OPENROUTER_PREFIX),
                routing=openrouter_provider_routing(spec),
            ).start()
    try:
        async for event in run_inner(proxy):
            yield event
    finally:
        if proxy is not None:
            proxy.close()


def finalize_checkpoint(spec: AgentSpec, ctx: Any) -> AgentEvent | None:
    """Checkpoint the workspace inline at the terminal result event (DESIGN.md §6).

    Done as a direct side-effect (not post-loop): the former Agent Engine executor
    stopped draining the generator after the final event, so post-loop work was dead
    code in-cloud — the sandbox worker drains fully, but inline stays the contract.
    Best-effort — a checkpoint failure never fails the run.

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

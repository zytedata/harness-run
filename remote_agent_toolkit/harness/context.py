"""``RunContext`` — the per-run inputs a runtime hands the harness (DESIGN.md §7).

The harness is intentionally ignorant of *where* it runs: a runtime (``local`` or
``gemini``) resolves the platform-specific bits — the working directory, resolved
secret values, the SessionStore, the BlobStore for checkpointing — and packs them into
a ``RunContext``. The harness turns ``(spec, ctx)`` into backend options and drives the
loop. This keeps the Claude binding free of GCP / ADK / Secret Manager knowledge.

Stdlib-only. ``BlobStore`` / SessionStore are duck-typed (``Any``) so importing this
module pulls in no third-party deps and no port adapters.

SECURITY: ``secrets`` holds resolved secret *values*. Never log it, never put it in an
``AgentEvent.raw`` payload, and never echo it back to the model except as the runtime
env / MCP headers the harness wires up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..spec import AgentSpec


@dataclass
class RunContext:
    """Everything the harness needs to run one turn against an :class:`AgentSpec`.

    Attributes:
        spec: The agent definition being run.
        prompt: The user message for this turn (resume/repo directives already stripped).
        job_dir: Isolated, writable per-session root for this run. The agent does NOT run
            here — it runs in the ``workspace`` leaf below — leaving this level free for
            session bookkeeping the agent shouldn't see (e.g. the captured ``stderr.log``).
        session_id: Stable Claude session id, pinned up front so checkpoint keying never
            depends on parsing it out of the message stream.
        secrets: Per-invocation secret name → value map (NEVER logged), supplied by the caller
            at ``run``/``send`` time — never baked into the spec or the deployed engine. The
            harness routes them: those consumed by a repo's ``auth`` or a GitHub MCP are used
            for git/MCP auth and kept OUT of the agent's env; the rest are forwarded to the
            agent subprocess env (the caller's own API keys the agent's code reads).
        env: Runtime-resolved extra environment for the agent subprocess, applied after
            ``spec.env`` (e.g. the local runtime's deploy-time packages venv: ``VIRTUAL_ENV``
            plus a composed ``PATH``). ``None`` when the runtime adds nothing.
        resume_sid: When resuming, the prior session id to continue (else ``None``).
        session_store: A Claude SDK ``SessionStore`` to mirror the transcript to (enables
            cross-worker/cross-cwd resume), or ``None`` to keep the conversation local.
        blobs: A ``BlobStore`` used to snapshot/restore the workspace at a checkpoint, or
            ``None`` when checkpointing is off.
        interactive: Append the multi-turn "stop and await the operator" guidance to the
            system prompt (set by the runtime when checkpointing/interactive is on).
        hooks: Claude Agent SDK hook callbacks (``{HookEvent: [HookMatcher, ...]}``) for
            this turn, supplied by the caller at ``run``/``send`` time, or ``None``. Live
            in-process callables, so they are neither part of the spec nor of a config
            overlay (both are serialized data) and only the ``local`` runtime can carry
            them. ``PreToolUse`` fires under every permission mode, including
            ``bypassPermissions``, where ``can_use_tool`` is shadowed.
    """

    spec: AgentSpec
    prompt: str
    job_dir: Path
    session_id: str
    secrets: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] | None = None
    resume_sid: str | None = None
    session_store: Any | None = None
    blobs: Any | None = None
    interactive: bool = False
    hooks: Any | None = None

    @property
    def workspace(self) -> Path:
        """The agent's working directory: ``job_dir/workspace``.

        The agent cwd is a leaf literally named ``workspace`` — a bare ``jobs/<uuid>``
        cwd reads as a disposable temp location, and models have been observed taking
        that hint (``cd /tmp`` as their first command) and building the deliverable
        outside the directory callers collect artifacts from. Everything agent-visible
        (skills, cloned repos, checkpoint snapshot/restore) targets this path; callers
        seed inputs into and collect outputs from it (``Session.workspace``).
        """
        return self.job_dir / "workspace"

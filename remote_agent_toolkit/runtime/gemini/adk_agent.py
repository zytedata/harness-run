"""The deployed ADK agent that wraps the (ADK-free) harness — P2.

Gemini Agent Runtime serves an ADK ``BaseAgent``; this is the gemini-side equivalent of
``LocalSession``'s run factory. Its ``_run_async_impl`` resolves the platform bits
(secrets from injected env, GCS checkpoint ports, the per-job ``/tmp`` cwd), provisions
the **baked** skills, then drives :class:`ClaudeCodeHarness` — surfacing each generic
``AgentEvent`` two ways: to the ``EventSink`` (Cloud Logging, the near-real-time channel)
and as an ADK ``Event`` (the durable per-job stream).

The agent is cloudpickled at deploy, so it carries the spec as a plain serialized ``dict``
(rebuilt at runtime) — never absolute paths or secret values (DESIGN.md §3.5). ``google.adk``
is imported at module top *on purpose*: this module is only ever imported lazily by
``backend.deploy`` (never at package import), so the zero-dep import guarantee holds.
"""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Any, AsyncGenerator

from google.adk.agents import BaseAgent

from ...events import AgentEvent

_RESUME_DIRECTIVE = re.compile(r"^\s*AGENT_RESUME=(\S+)[ \t]*\r?\n", re.IGNORECASE)
_AGENT_DESCRIPTION = "A remote-agent-toolkit agent: a Claude Code session driven by the toolkit harness."


def _extract_prompt(content: Any) -> str:
    """Flatten an ADK ``Content`` into the user's prompt text."""
    if content is None or not getattr(content, "parts", None):
        return ""
    return "\n".join(p.text for p in content.parts if getattr(p, "text", None)).strip()


def _split_resume_directive(prompt: str) -> tuple[str | None, str]:
    """Strip a leading ``AGENT_RESUME=<session_id>`` directive (resume an existing turn)."""
    m = _RESUME_DIRECTIVE.match(prompt)
    if m:
        return m.group(1), prompt[m.end():]
    return None, prompt


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    rest = uri[len("gs://"):] if uri.startswith("gs://") else uri
    bucket, _, prefix = rest.partition("/")
    return bucket, prefix.strip("/")


def _find_baked_skills() -> Path | None:
    """Locate the ``skills/`` dir baked into the engine image at deploy (else ``None``)."""
    import remote_agent_toolkit

    pkg = Path(remote_agent_toolkit.__file__).resolve().parent
    for cand in (pkg.parent / "skills", Path.cwd() / "skills", Path("/app/skills")):
        if cand.is_dir():
            return cand
    return None


def _resolve_secrets(spec: Any) -> dict[str, str]:
    """Resolve declared secret names from the injected env (platform secret_refs)."""
    from ...ports.secrets import EnvSecretResolver

    resolver = EnvSecretResolver()
    out: dict[str, str] = {}
    for name in spec.secrets:
        try:
            out[name] = resolver.resolve(name)
        except KeyError:
            continue  # absent — never surface the (missing) value
    return out


def _checkpoint_ports(spec: Any) -> tuple[Any | None, Any | None]:
    """Build (BlobStore, SessionStore) from ``AGENT_CHECKPOINT_GCS`` when checkpointing is on."""
    ckpt = os.environ.get("AGENT_CHECKPOINT_GCS")
    if not (spec.checkpoint and ckpt):
        return None, None
    from ...checkpoint.session_store import BlobSessionStore
    from ...ports.blobstore import GcsBlobStore

    bucket, prefix = _parse_gcs_uri(ckpt)
    blobs = GcsBlobStore(bucket, (prefix + "/") if prefix else "")
    return blobs, BlobSessionStore(blobs)


def _prewarm(spec: Any) -> None:
    """Best-effort warm-up an idle pool worker runs before blocking on a claim.

    Warms the Cloud Logging client (by emitting the readiness marker under the pool id — which
    also signals the control plane that this worker is warm) and the GCS channel when
    checkpointing. Never raises — a pre-warm failure must not take the worker down.
    """
    from ...events import AgentEvent
    from ...ports.eventsink import CloudLoggingSink
    from .pool import pool_log_id_from_subscription

    if spec.checkpoint and os.environ.get("AGENT_CHECKPOINT_GCS"):
        try:
            from google.cloud import storage

            storage.Client()  # establish auth/channel so the first checkpoint call is fast
        except Exception:  # noqa: BLE001
            pass
    try:
        pool_id = pool_log_id_from_subscription(os.environ.get("AGENT_POOL_SUBSCRIPTION", ""))
        CloudLoggingSink(session_id=pool_id).emit(
            AgentEvent(kind="status", summary="pool worker ready", raw={"event": "pool_ready"})
        )
    except Exception:  # noqa: BLE001
        pass


def _prepare_workspace(rc: Any) -> dict:
    """Restore a prior workspace (resume) or stage the baked skills into a fresh cwd. Sync."""
    from ...skills import provision
    from ...spec import SkillSource

    restored = False
    if rc.resume_sid and rc.blobs is not None:
        from ...checkpoint.workspace import restore

        try:
            restored = restore(rc.blobs, rc.resume_sid, str(rc.job_dir))
        except Exception:  # noqa: BLE001 — fall back to a fresh workspace
            restored = False
    names: list[str] = []
    if not restored:
        rc.job_dir.mkdir(parents=True, exist_ok=True)
        baked = _find_baked_skills()
        # Provision from the baked dir (a local source) rather than re-resolving spec.skills,
        # which could re-clone a git source at runtime (slow / no network in the engine).
        sources = (SkillSource.local(str(baked)),) if baked else rc.spec.skills
        if sources:
            names = provision(sources, str(rc.job_dir))
    return {
        "event": "workspace_ready",
        "restored": restored,
        "skills": names,
        "summary": f"workspace ready (restored={restored}, skills staged={len(names)})",
    }


class ToolkitAgent(BaseAgent):
    """ADK custom agent whose run method drives the toolkit harness (deployed on gemini)."""

    # Declared as a Pydantic field on BaseAgent so assignment in __init__ is allowed.
    spec_data: dict

    def __init__(self, name: str, spec_data: dict) -> None:
        super().__init__(name=name, description=_AGENT_DESCRIPTION, spec_data=spec_data)

    async def _run_async_impl(self, ctx: Any) -> AsyncGenerator[Any, None]:
        from ...spec import AgentSpec
        from .pool import POOL_WAIT_SENTINEL

        spec = AgentSpec.from_dict(self.spec_data)
        prompt = _extract_prompt(ctx.user_content)

        # Warm-pool worker: block for a dispatched turn, then process it (under the dispatched
        # session_id). Cold start was paid at pool-fill time, so pickup is fast.
        if prompt.strip() == POOL_WAIT_SENTINEL:
            async for event in self._pool_worker(spec):
                yield event
            return

        # The ADK session id is the stable token: it tags the Cloud Logging stream the client
        # tails, and pins the Claude session id for checkpoint keying.
        session_id = getattr(getattr(ctx, "session", None), "id", None) or ctx.invocation_id
        resume_sid, prompt = _split_resume_directive(prompt)
        async for event in self._run_turn(spec, session_id, prompt, resume_sid):
            yield event

    async def _run_turn(
        self, spec: Any, session_id: str, prompt: str, resume_sid: str | None
    ) -> AsyncGenerator[Any, None]:
        """Process one turn under ``session_id``: prep workspace, drive the harness, surface events.

        Shared by the cold path and a warm worker's claimed turn — both tag the Cloud Logging
        stream and checkpoint with ``session_id``, so the client tails identically either way.
        """
        from ...harness.claude_code import ClaudeCodeHarness
        from ...harness.context import RunContext
        from ...ports.eventsink import CloudLoggingSink
        from .translate import to_adk_event

        sink = CloudLoggingSink(session_id=session_id)
        blobs, session_store = _checkpoint_ports(spec)
        rc = RunContext(
            spec=spec,
            prompt=prompt,
            job_dir=Path(os.environ.get("AGENT_JOBS_ROOT", "/tmp/agent-jobs")) / session_id,
            session_id=session_id,
            secrets=_resolve_secrets(spec),
            resume_sid=resume_sid if session_store is not None else None,
            session_store=session_store,
            blobs=blobs,
            interactive=spec.checkpoint,
        )

        prep = await asyncio.to_thread(_prepare_workspace, rc)
        prep_ev = AgentEvent(kind="status", summary=prep["summary"], raw=prep)
        sink.emit(prep_ev)
        yield to_adk_event(prep_ev, self.name)

        async for event in ClaudeCodeHarness().run(spec, rc):
            sink.emit(event)  # near-real-time channel (Cloud Logging); finalize is inline in run()
            yield to_adk_event(event, self.name)

    async def _pool_worker(self, spec: Any) -> AsyncGenerator[Any, None]:
        """Block pulling the dispatch subscription, then process the claimed turn.

        Heartbeats while idle (the async executor finalizes a job that yields no events), then
        runs the dispatched turn under the *dispatched* ``session_id`` so the waiting client
        sees it on its own Cloud Logging tail.
        """
        import time

        from .pool import resolve_max_wait_s, worker_dispatch_from_env
        from .translate import to_adk_event

        dispatch = worker_dispatch_from_env()
        if dispatch is None:
            ev = AgentEvent(
                kind="status",
                summary="pool worker started without AGENT_POOL_SUBSCRIPTION; exiting",
                raw={"event": "pool_error"},
            )
            yield to_adk_event(ev, self.name)
            return

        # Pre-warm during the idle wait (the dominant post-claim cost in-cloud is the first
        # GCS/Logging channel + auth). Emitting the readiness marker doubles as warming the
        # Cloud Logging client AND the "pool is warm" signal the control plane waits on.
        await asyncio.to_thread(_prewarm, spec)

        deadline = time.monotonic() + resolve_max_wait_s()
        claimed: dict | None = None
        beat = 0
        while time.monotonic() < deadline:
            claimed = await asyncio.to_thread(dispatch.claim, 10.0)
            if claimed is not None:
                break
            beat += 1
            hb = AgentEvent(
                kind="status",
                summary=f"pool worker waiting for assignment (beat {beat})",
                raw={"event": "pool_waiting", "beat": beat},
            )
            yield to_adk_event(hb, self.name)

        if claimed is None:
            ev = AgentEvent(
                kind="status", summary="pool worker idle-expired (no assignment)",
                raw={"event": "pool_idle_expired"},
            )
            yield to_adk_event(ev, self.name)
            return

        session_id = claimed.get("session_id") or ""
        message = claimed.get("message", "")
        resume = bool(claimed.get("resume"))
        async for event in self._run_turn(spec, session_id, message, resume_sid=session_id if resume else None):
            yield event


def _safe_node_name(name: str) -> str:
    """ADK ``BaseAgent.name`` must be a valid Python identifier; ``spec.name`` may not be.

    The engine's *display name* keeps ``spec.name`` verbatim (hyphens fine); only this
    internal ADK node name is sanitized (e.g. ``"spider-builder"`` -> ``"spider_builder"``).
    """
    safe = re.sub(r"\W", "_", name)
    if not safe or safe[0].isdigit():
        safe = "a_" + safe
    return safe


def build_agent(spec: Any) -> ToolkitAgent:
    """Build the deployable ADK agent from ``spec`` (carries the spec as a serialized dict)."""
    return ToolkitAgent(name=_safe_node_name(spec.name), spec_data=spec.to_dict())

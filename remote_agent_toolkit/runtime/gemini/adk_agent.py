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
        from ...harness.claude_code import ClaudeCodeHarness
        from ...harness.context import RunContext
        from ...ports.eventsink import CloudLoggingSink
        from ...spec import AgentSpec
        from .translate import to_adk_event

        spec = AgentSpec.from_dict(self.spec_data)
        prompt = _extract_prompt(ctx.user_content)
        # The ADK session id is the stable token: it tags the Cloud Logging stream the client
        # tails, and pins the Claude session id for checkpoint keying. (P2b: __POOL_WAIT__.)
        session_id = getattr(getattr(ctx, "session", None), "id", None) or ctx.invocation_id
        resume_sid, prompt = _split_resume_directive(prompt)

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

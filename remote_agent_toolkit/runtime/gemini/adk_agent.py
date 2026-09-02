"""The deployed ADK agent that wraps the (ADK-free) harness.

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

from ...checkpoint.session_store import _claude_session_id
from ...events import AgentEvent

_RESUME_DIRECTIVE = re.compile(r"^\s*AGENT_RESUME=(\S+)[ \t]*\r?\n", re.IGNORECASE)
_SECRETS_DIRECTIVE = re.compile(r"^\s*AGENT_SECRETS_GCS=(\S+)[ \t]*\r?\n", re.IGNORECASE)
_SESSION_DIRECTIVE = re.compile(r"^\s*AGENT_SESSION=(\S+)[ \t]*\r?\n", re.IGNORECASE)
_SESSION_CONFIG_DIRECTIVE = re.compile(
    r"^\s*AGENT_SESSION_CONFIG_GCS=(\S+)[ \t]*\r?\n", re.IGNORECASE
)
_TURN_CONFIG_DIRECTIVE = re.compile(r"^\s*AGENT_TURN_CONFIG_GCS=(\S+)[ \t]*\r?\n", re.IGNORECASE)
_GCS_TOKEN_DIRECTIVE = re.compile(r"^\s*AGENT_GCS_TOKEN=(\S+)[ \t]*\r?\n", re.IGNORECASE)
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


def _split_session_directive(prompt: str) -> tuple[str | None, str]:
    """Strip a leading ``AGENT_SESSION=<session_id>`` directive (the toolkit's session id).

    Cold jobs omit ``session_id`` from the query input (the platform's job runner lost the
    managed session service for new engines, so a supplied id fails; omitted, the app
    auto-creates a throwaway ADK session) — the toolkit's own session id arrives here and
    keys the event stream, mirror, and checkpoints instead of the ADK ctx id.
    """
    m = _SESSION_DIRECTIVE.match(prompt)
    if m:
        return m.group(1), prompt[m.end():]
    return None, prompt


def _split_secrets_directive(prompt: str) -> tuple[str | None, str]:
    """Strip a leading ``AGENT_SECRETS_GCS=<gs://...>`` directive (cold-path secrets pointer).

    Returns ``(uri, remaining_prompt)``; ``(None, prompt)`` when absent. The directive is
    consumed here so the model never sees it. The pointer (not values — the platform persists
    the job input) targets the staged secrets object; see :func:`_fetch_secrets`.
    """
    m = _SECRETS_DIRECTIVE.match(prompt)
    if m:
        return m.group(1), prompt[m.end():]
    return None, prompt


def _split_config_directives(prompt: str) -> tuple[str | None, str | None, str]:
    """Strip leading ``AGENT_SESSION_CONFIG_GCS=`` / ``AGENT_TURN_CONFIG_GCS=`` directives.

    Returns ``(session_config_uri, turn_config_uri, remaining_prompt)``. The pointers
    target the session's persisted ``SessionConfig`` and this turn's ``TurnConfig`` —
    overlays the turn must run on top of the deploy-baked spec (see ``_run_turn``).
    Consumed here so the model never sees them.
    """
    session_uri = turn_uri = None
    m = _SESSION_CONFIG_DIRECTIVE.match(prompt)
    if m:
        session_uri, prompt = m.group(1), prompt[m.end():]
    m = _TURN_CONFIG_DIRECTIVE.match(prompt)
    if m:
        turn_uri, prompt = m.group(1), prompt[m.end():]
    return session_uri, turn_uri, prompt


def _split_gcs_token_directive(prompt: str) -> tuple[str | None, str]:
    """Strip a leading ``AGENT_GCS_TOKEN=<token>`` directive (the run-scoped GCS token).

    The token is the one credential-like directive: the worker authorizes every GCS
    access of the turn with it instead of the runtime identity (``scoped_gcs.py``). It is
    written LAST by the client so a pre-token worker, which strips the other directives
    in order and leaves this line as prompt text, still runs the turn correctly (on the
    runtime identity, as before) — the leftover is a token worth this run's own objects.
    Consumed here so the model never sees it on a current worker.
    """
    m = _GCS_TOKEN_DIRECTIVE.match(prompt)
    if m:
        return m.group(1), prompt[m.end():]
    return None, prompt


def _use_run_gcs_token(gcs_token: str | None, session_id: str) -> bool:
    """Point every GcsBlobStore of this turn at the run's scoped token (or back to ADC).

    Returns whether a token is in force. Called first thing in a turn, before any config,
    secret, mirror or checkpoint access.
    """
    from ...ports.blobstore import set_default_gcs_credentials

    if not gcs_token:
        set_default_gcs_credentials(None)
        return False
    from .scoped_gcs import output_bucket_from_env, worker_credentials

    creds = worker_credentials(
        gcs_token, output_bucket_from_env(os.environ.get("AGENT_EVENTS_GCS")), session_id
    )
    set_default_gcs_credentials(creds)
    return True


def _fetch_secrets(secrets_uri: str | None) -> tuple[dict, AgentEvent | None]:
    """Fetch the staged per-invocation secrets; never raises.

    The staging object is NOT deleted here: the platform re-runs a killed attempt with the
    same payload (same pointer), so the object must survive until the turn reaches its
    terminal result — ``_run_turn`` deletes it then (see the handoff module docstring).

    Returns ``(secrets, warning_event)``: on a missing/unreadable staging the run proceeds
    without secrets and the caller surfaces the (value-free) warning so the failure mode is
    debuggable instead of a mystery of absent credentials.
    """
    if not secrets_uri:
        return {}, None
    from .handoff import fetch_secrets

    secrets = fetch_secrets(secrets_uri)
    if secrets:
        return secrets, None
    return {}, AgentEvent(
        kind="status",
        summary="invocation secrets unavailable (staging object missing/unreadable); running without them",
        raw={"event": "secrets_unavailable"},
    )


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    from ...ports.blobstore import parse_gcs_uri

    return parse_gcs_uri(uri)


def _find_baked_skills() -> Path | None:
    """Locate the ``skills/`` dir baked into the engine image at deploy (else ``None``)."""
    import remote_agent_toolkit

    pkg = Path(remote_agent_toolkit.__file__).resolve().parent
    for cand in (pkg.parent / "skills", Path.cwd() / "skills", Path("/app/skills")):
        if cand.is_dir():
            return cand
    return None


def _checkpoint_ports(spec: Any) -> tuple[Any | None, Any | None]:
    """Build (BlobStore, SessionStore) from ``AGENT_CHECKPOINT_GCS`` when checkpointing is on.

    Also built for a transcript-only spec: the transcript is mirrored to the store, while
    the workspace snapshot stays behind ``spec.checkpoint`` (``_shared.finalize_checkpoint``).
    """
    ckpt = os.environ.get("AGENT_CHECKPOINT_GCS")
    if not ((spec.checkpoint or spec.transcript) and ckpt):
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

    if (spec.checkpoint or spec.transcript) and os.environ.get("AGENT_CHECKPOINT_GCS"):
        try:
            from google.cloud import storage

            storage.Client()  # establish auth/channel so the first checkpoint call is fast
        except Exception:  # noqa: BLE001
            pass
    try:
        pool_id = pool_log_id_from_subscription(os.environ.get("AGENT_POOL_SUBSCRIPTION", ""))
        ready = AgentEvent(kind="status", summary="pool worker ready", raw={"event": "pool_ready"})
        # GCS marker under events/<pool_id>/ — what wait_until_warm tails (quota-free);
        # the Cloud Logging emit stays for ops (and pre-stream clients).
        events_uri = os.environ.get("AGENT_EVENTS_GCS")
        if events_uri:
            import time as _time

            from .history import mirror_line, write_turn_mirror

            write_turn_mirror(
                events_uri, pool_id, [mirror_line(ready)], now_ms=int(_time.time() * 1000)
            )
        CloudLoggingSink(session_id=pool_id).emit(ready)
    except Exception:  # noqa: BLE001
        pass


def _prepare_workspace(rc: Any, prefer_baked_skills: bool = True) -> dict:
    """Restore a prior workspace (resume) or stage the baked skills into a fresh cwd. Sync.

    ``prefer_baked_skills=False`` forces resolving ``rc.spec.skills`` from their sources at
    run time instead of the deploy-baked dir — used when a run-scoped spec declares skills
    that differ from the baked ones (including declaring none).
    """
    from ...skills import provision, skills_subdir
    from ...spec import SkillSource

    restored = False
    if rc.resume_sid and rc.blobs is not None:
        from ...checkpoint.workspace import restore

        try:
            restored = restore(rc.blobs, rc.resume_sid, str(rc.workspace))
        except Exception:  # noqa: BLE001 — fall back to a fresh workspace
            restored = False
    names: list[str] = []
    repos: list[str] = []
    if restored:
        # The snapshot was credential-scrubbed; re-embed push tokens from this turn's secrets.
        if rc.spec.repos:
            from ...integrations.git import reauth_repos

            repos = reauth_repos(rc.workspace, rc.spec.repos, rc.secrets)
    else:
        rc.workspace.mkdir(parents=True, exist_ok=True)
        baked = _find_baked_skills() if prefer_baked_skills else None
        # Provision from the baked dir (a local source) rather than re-resolving spec.skills,
        # which could re-clone a git source at runtime (slow / no network in the engine).
        sources = (SkillSource.local(str(baked)),) if baked else rc.spec.skills
        if sources:
            names = provision(sources, str(rc.workspace), skills_subdir(rc.spec.harness))
        if rc.spec.repos:
            from ...integrations.git import provision_repos

            repos = provision_repos(rc.workspace, rc.spec.repos, rc.secrets)
    return {
        "event": "workspace_ready",
        "restored": restored,
        "skills": names,
        "repos": repos,
        "summary": (
            f"workspace ready (restored={restored}, skills staged={len(names)}, "
            f"repos cloned={len(repos)})"
        ),
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
        # Stamped on every surfaced ADK event: the platform session-append API rejects
        # events without it (see translate.to_adk_event).
        invocation_id = ctx.invocation_id

        # Warm-pool worker: block for a dispatched turn, then process it (under the dispatched
        # session_id). Cold start was paid at pool-fill time, so pickup is fast.
        if prompt.strip() == POOL_WAIT_SENTINEL:
            async for event in self._pool_worker(spec, invocation_id):
                yield event
            return

        # Strip leading control directives (never shown to the model): the toolkit session
        # id, secrets pointer, session/turn config pointers, resume marker.
        directive_sid, prompt = _split_session_directive(prompt)
        secrets_uri, prompt = _split_secrets_directive(prompt)
        session_config_uri, turn_config_uri, prompt = _split_config_directives(prompt)
        resume_sid, prompt = _split_resume_directive(prompt)
        gcs_token, prompt = _split_gcs_token_directive(prompt)
        # The toolkit session id is the stable token: it tags the Cloud Logging stream the
        # client tails, and pins the Claude session id for checkpoint keying. It rides the
        # AGENT_SESSION directive (cold jobs run under a throwaway auto-created ADK
        # session); pre-directive clients fall back to the ADK ctx id as before.
        session_id = (
            directive_sid
            or getattr(getattr(ctx, "session", None), "id", None)
            or ctx.invocation_id
        )
        async for event in self._run_turn(
            spec, session_id, prompt, resume_sid, secrets_uri, invocation_id,
            session_config_uri=session_config_uri, turn_config_uri=turn_config_uri,
            gcs_token=gcs_token,
        ):
            yield event

    async def _run_turn(
        self, spec: Any, session_id: str, prompt: str, resume_sid: str | None,
        secrets_uri: str | None = None, invocation_id: str = "",
        session_config_uri: str | None = None, turn_config_uri: str | None = None,
        gcs_token: str | None = None,
    ) -> AsyncGenerator[Any, None]:
        """Process one turn under ``session_id``: prep workspace, drive the harness, surface events.

        Shared by the cold path and a warm worker's claimed turn — both tag the Cloud Logging
        stream and checkpoint with ``session_id``, so the client tails identically either way.
        ``secrets_uri`` points at the staged secrets object (fetched here; deleted only once
        the turn's terminal result went out, so a platform retry of a killed attempt still
        finds it); values never ride the invocation payload.

        ``session_config_uri`` / ``turn_config_uri`` point at the session's persisted
        ``SessionConfig`` and this turn's ``TurnConfig``: sparse overlays merged over the
        deploy-baked spec, and the merged result is what this turn RUNS (run-scoped
        parameters — target repo, model, prompt — without an engine redeploy). The merged
        spec is echoed as an ``effective_spec`` event: the durable ground-truth record of
        what actually ran (config objects themselves are also kept — see ``handoff.py``).
        A missing/unreadable config, an unknown config field, or a harness the image
        doesn't bake FAILS the turn: silently running the baked spec instead of the
        requested configuration would be a wrong-configuration run.

        ``gcs_token`` is the run-scoped GCS token (``scoped_gcs.py``). When present, every
        GCS access of this turn (configs, secrets, the mirror, checkpoints) authorizes with
        it instead of the runtime identity; when absent (a pre-token client) the turn runs
        on the runtime identity as before.
        """
        from ...harness import resolve_harness
        from ...harness.context import RunContext
        from ...ports.eventsink import CloudLoggingSink
        from .tracing import TurnTracer
        from .translate import to_adk_event

        scoped_gcs = _use_run_gcs_token(gcs_token, session_id)
        # The RAW session id keys the Cloud Logging stream (what the client tails); the
        # Claude/checkpoint side needs a canonical UUID, mapped deterministically from it.
        sink = CloudLoggingSink(session_id=session_id)
        claude_sid = _claude_session_id(session_id)

        def _terminal_error(summary: str) -> AgentEvent:
            """Emit a terminal error result to every channel (pre-MirrorStream failures)."""
            from .history import mirror_line, write_turn_mirror

            ev = AgentEvent(
                kind="result",
                summary=summary,
                raw={"event": "config_error", "is_error": True, "subtype": "error",
                     "session_id": session_id},
            )
            sink.emit(ev)
            events_uri = os.environ.get("AGENT_EVENTS_GCS")
            if events_uri:
                import time as _time

                write_turn_mirror(events_uri, session_id, [mirror_line(ev)],
                                  now_ms=int(_time.time() * 1000))
            return ev

        # Resolve the effective spec BEFORE anything derives from ``spec`` (checkpoint
        # ports, RunContext, tracer): deploy-baked spec ← session config ← turn config.
        prefer_baked_skills = True
        configs_applied = False
        if session_config_uri or turn_config_uri:
            from ...config import (
                SessionConfig,
                TurnConfig,
                apply_session_config,
                apply_turn_config,
                validate_harness_choice,
            )
            from .handoff import fetch_config

            baked = spec
            try:
                session_cfg = turn_cfg = None
                if session_config_uri:
                    fetched = fetch_config(session_config_uri)
                    if fetched is None:
                        raise ValueError(
                            "session config unavailable (persisted object missing/unreadable)"
                        )
                    session_cfg = SessionConfig.from_dict(fetched)
                if turn_config_uri:
                    fetched = fetch_config(turn_config_uri)
                    if fetched is None:
                        raise ValueError(
                            "turn config unavailable (staged object missing/unreadable)"
                        )
                    turn_cfg = TurnConfig.from_dict(fetched)
                spec = apply_turn_config(apply_session_config(baked, session_cfg), turn_cfg)
                validate_harness_choice(baked, spec)
            except Exception as exc:  # noqa: BLE001 — wrong-config runs must fail loudly
                yield to_adk_event(
                    _terminal_error(f"{str(exc)[:300]}; failing the turn"),
                    self.name, invocation_id,
                )
                return
            configs_applied = True
            # Baked skills are a staging fast path, valid only while the effective skills
            # are the deploy-baked ones; a differing declaration (including "none")
            # resolves from the effective spec's own sources.
            prefer_baked_skills = spec.to_dict().get("skills") == self.spec_data.get("skills")

        secrets, secrets_warning = _fetch_secrets(secrets_uri)
        blobs, session_store = _checkpoint_ports(spec)
        rc = RunContext(
            spec=spec,
            prompt=prompt,
            job_dir=Path(os.environ.get("AGENT_JOBS_ROOT", "/tmp/agent-jobs")) / claude_sid,
            session_id=claude_sid,
            secrets=secrets,
            resume_sid=(
                _claude_session_id(resume_sid)
                if (resume_sid and spec.checkpoint and session_store is not None)
                else None
            ),
            session_store=session_store,
            blobs=blobs,
            interactive=spec.checkpoint if spec.interactive is None else spec.interactive,
        )

        # Every surfaced event also streams into the session-keyed GCS mirror as it happens
        # (batched ~0.5s): the mirror is BOTH the durable history Session.history() reads AND
        # the live channel the client tails (stream.tail_stream) — Cloud Logging stays
        # emit-only for ops/debugging. Events never carry secret values, so neither does the
        # mirror. A missing AGENT_EVENTS_GCS (pre-stream local tests) degrades to no mirror.
        from .stream import MirrorStream

        events_uri = os.environ.get("AGENT_EVENTS_GCS")
        stream = MirrorStream(events_uri, session_id) if events_uri else None
        # Cloud Trace spans rebuilt from the same stream (the console's Traces tab).
        # Best-effort by construction.
        tracer = TurnTracer(agent_name=self.name, model=spec.model, session_id=session_id)

        # CPU/RAM self-sampling (OOM forensics; see resources.py): periodic samples go to a
        # side log, a first memory-pressure crossing lands on the MAIN stream (emitted from
        # the sampler thread: straight to the sink + mirror — the generator can't be driven
        # from a thread, so it skips the ADK event surface), and the terminal result's raw
        # is stamped with the observed peak in surface() below.
        from .resources import start_sampler

        sampler = start_sampler(
            session_id,
            on_event=lambda ev: (sink.emit(ev), stream.append(ev) if stream else None),
        )

        def surface(event: AgentEvent) -> Any:
            if event.kind == "result" and sampler is not None:
                event.raw = event.raw or {}
                sampler.enrich_result(event.raw)
            sink.emit(event)  # ops/debug channel (Cloud Logging; clients no longer tail it)
            if stream is not None:
                stream.append(event)  # live channel + durable history (GCS mirror)
            tracer.observe(event)  # span open/close + annotations (never raises)
            return to_adk_event(event, self.name, invocation_id)

        # The ENTIRE turn is guarded: workspace prep (clone/skills can fail on bad auth) as
        # much as the harness itself. Any crash becomes a TERMINAL result event — without one
        # the client's tail hangs forever and the failure is invisible (how the
        # numeric-session-id bug hid). Error strings are truncated and repo tokens are already
        # redacted by the git layer; never put secrets in an event.
        saw_result = False
        try:
            if configs_applied:
                # The ground-truth record of what this turn runs: the merged effective
                # spec (baked ← session config ← turn config), durable in the mirror.
                # Specs carry no secret values by contract, so neither does this event.
                yield surface(AgentEvent(
                    kind="status",
                    summary=(
                        "effective spec resolved "
                        f"(session config: {bool(session_config_uri)}, "
                        f"turn config: {bool(turn_config_uri)})"
                    ),
                    raw={"event": "effective_spec", "spec": spec.to_dict(),
                         "session_config_gcs": session_config_uri,
                         "turn_config_gcs": turn_config_uri},
                ))
            if secrets_warning is not None:  # value-free: staged secrets were unavailable
                yield surface(secrets_warning)
            prep = await asyncio.to_thread(_prepare_workspace, rc, prefer_baked_skills)
            prep["scoped_gcs"] = scoped_gcs  # durable record: GCS access ran on the run token
            yield surface(AgentEvent(kind="status", summary=prep["summary"], raw=prep))

            async for event in resolve_harness(spec).run(spec, rc):
                saw_result = saw_result or event.kind == "result"
                yield surface(event)  # finalize (checkpoint) is inline in the harness
        except Exception as exc:  # noqa: BLE001 — a silent server-side death is undebuggable
            if saw_result:
                # The terminal result already went out — it is authoritative (the run's
                # text/cost/usage are real; the checkpoint was taken inline at that event).
                # A harness death during shutdown is a warning, NOT a second result: an
                # error result here would overwrite a finished run in the durable history.
                yield surface(AgentEvent(
                    kind="status",
                    summary=(
                        "harness exited abnormally after the terminal result "
                        f"(result kept): {str(exc)[:300]}"
                    ),
                    raw={"event": "late_harness_error", "session_id": session_id},
                ))
            else:
                yield surface(AgentEvent(
                    kind="result",
                    summary=f"agent run failed: {str(exc)[:300]}",
                    raw={"event": "harness_error", "is_error": True, "subtype": "error",
                         "session_id": session_id},
                ))
                saw_result = True  # the error result IS the turn's terminal result
        finally:
            if sampler is not None:
                sampler.stop()
            tracer.close()  # end the turn span (+ any tool span orphaned by a crash)
            # Drain the mirror writer — also on early generator close (a partially-consumed
            # turn still leaves a durable record). SYNC on purpose: this finally also runs
            # under GeneratorExit, where awaiting is illegal. Best-effort by design.
            if stream is not None:
                stream.close()
        # The terminal result went out, so no platform retry of this attempt can follow —
        # NOW the staged secrets can go. A killed attempt never reaches this line, leaving
        # the object for the platform's automatic re-run of the same payload (the client's
        # completion cleanup and the bucket lifecycle rule back this up; see handoff docs).
        # Deliberately after `finally`, not in it: `finally` also runs under GeneratorExit
        # (a half-consumed turn), where the run may still be retried/resumed elsewhere.
        # Config objects are NOT deleted — they are the post-mortem record (handoff.py).
        if saw_result and secrets_uri:
            from .handoff import delete_staged_secrets

            await asyncio.to_thread(delete_staged_secrets, secrets_uri)

    async def _pool_worker(self, spec: Any, invocation_id: str = "") -> AsyncGenerator[Any, None]:
        """Block pulling the dispatch subscription, then process the claimed turn.

        Heartbeats while idle (the async executor finalizes a job that yields no events), then
        runs the dispatched turn under the *dispatched* ``session_id`` so the waiting client
        sees it on its own Cloud Logging tail.
        """
        import time

        from ...ports.eventsink import CloudLoggingSink
        from .pool import pool_log_id_from_subscription, resolve_max_wait_s, worker_dispatch_from_env
        from .translate import to_adk_event

        pool_id = pool_log_id_from_subscription(os.environ.get("AGENT_POOL_SUBSCRIPTION", ""))

        dispatch = worker_dispatch_from_env()
        if dispatch is None:
            ev = AgentEvent(
                kind="status",
                summary="pool worker started without AGENT_POOL_SUBSCRIPTION; exiting",
                raw={"event": "pool_error"},
            )
            yield to_adk_event(ev, self.name, invocation_id)
            return

        # Pre-warm during the idle wait (the dominant post-claim cost in-cloud is the first
        # GCS/Logging channel + auth). Emitting the readiness marker doubles as warming the
        # Cloud Logging client AND the "pool is warm" signal the control plane waits on.
        await asyncio.to_thread(_prewarm, spec)

        deadline = time.monotonic() + resolve_max_wait_s()
        claimed: dict | None = None
        beat = 0
        while time.monotonic() < deadline:
            try:
                # A pull is a long-poll (returns the instant a turn is published), so this
                # timeout mainly bounds the heartbeat cadence, not pickup latency.
                claimed = await asyncio.to_thread(dispatch.claim, 5.0)
            except Exception as exc:  # noqa: BLE001 — a missing grant must not crash silently
                # e.g. the RE service agent lacking pubsub.subscriber on the dispatch sub.
                # Surface it to Cloud Logging (the only async channel) instead of dying quietly.
                err = AgentEvent(
                    kind="status", summary=f"pool claim failed: {str(exc)[:200]}",
                    raw={"event": "pool_error"},
                )
                try:
                    CloudLoggingSink(session_id=pool_id).emit(err)
                except Exception:  # noqa: BLE001
                    pass
                yield to_adk_event(err, self.name, invocation_id)
                return
            if claimed is not None:
                break
            beat += 1
            hb = AgentEvent(
                kind="status",
                summary=f"pool worker waiting for assignment (beat {beat})",
                raw={"event": "pool_waiting", "beat": beat},
            )
            yield to_adk_event(hb, self.name, invocation_id)

        if claimed is None:
            ev = AgentEvent(
                kind="status", summary="pool worker idle-expired (no assignment)",
                raw={"event": "pool_idle_expired"},
            )
            yield to_adk_event(ev, self.name, invocation_id)
            return

        session_id = claimed.get("session_id") or ""
        message = claimed.get("message", "")
        resume = bool(claimed.get("resume"))
        secrets_uri = claimed.get("secrets_gcs")  # pointer only; values are staged in GCS
        session_config_uri = claimed.get("session_config_gcs")  # config pointers (_run_turn)
        turn_config_uri = claimed.get("turn_config_gcs")
        gcs_token = claimed.get("gcs_token")  # the run-scoped GCS token (scoped_gcs.py)
        async for event in self._run_turn(
            spec, session_id, message,
            resume_sid=session_id if resume else None, secrets_uri=secrets_uri,
            invocation_id=invocation_id,
            session_config_uri=session_config_uri, turn_config_uri=turn_config_uri,
            gcs_token=gcs_token,
        ):
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


def in_memory_session_service() -> Any:
    """Session-service builder pinned at deploy: always in-memory, never managed.

    The toolkit has no use for managed ADK sessions — the session id rides the
    AGENT_SESSION directive and durable history lives in the GCS event mirror — and the
    2026-07-28 platform runner ships job workers whose managed-session wiring is broken
    (create/get fail with "ReasoningEngine does not exist" / "Session not found").
    Pinning in-memory makes the worker self-contained either way. Module-level (not a
    lambda) so the pickled AdkApp references it importably from the staged package.
    """
    from google.adk.sessions.in_memory_session_service import InMemorySessionService

    return InMemorySessionService()

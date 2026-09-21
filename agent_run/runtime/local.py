"""Local, in-process run plane (DESIGN.md §3.2, §3.4).

``local.deploy(spec)`` returns a ``LocalEngine`` exposing the *same* Engine/Session/Run
surface as ``sandbox``, backed by the in-process :class:`ClaudeCodeHarness` and
filesystem/in-memory port adapters (no GCP beyond model access). This is the first-class
dev loop: develop against ``local``, ship against ``sandbox``, with zero code change.

A ``Run`` is consumable three ways — ``await`` it (→ ``RunResult``), ``async for`` over it
(→ ``AgentEvent``s), or poll ``run.done`` / ``run.status``. A single background task drives
the harness generator; the queue feeds iterators while the accumulated result feeds
``await``/poll.

The Claude Agent SDK and port adapters are imported lazily, so importing this module
needs no third-party deps.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, TYPE_CHECKING

from ..control import ControlMessage, ControlUnavailable, ExecResult, LocalControlChannel, run_shell
from ..events import AgentEvent, RunResult, RunStatus, StopReason
from ._run import DrivenRun

if TYPE_CHECKING:
    from ..config import SessionConfig, TurnConfig
    from ..spec import AgentSpec

# Where a session's bound config is persisted inside the engine's blob root, so a
# process re-attaching by id (with the same workdir) recovers it — same rule as sandbox:
# the config is bound once at session start and cannot be substituted on re-attach.
_SESSION_CONFIG_KEY = "session-config/{sid}.json"


def _reject_repos_in_chosen_workspace(spec: AgentSpec, workspace: Path | None) -> None:
    """Refuse ``repos`` in a caller-chosen workspace: cloning there is not ours to do.

    Provisioning clones each repo to a fixed ``<cwd>/<repo name>``, so the first session
    takes the path and every later one fails on the existing checkout — and reusing what
    is already there would mean deciding what to do with a dirty tree or a stale ref.
    """
    if workspace is not None and spec.repos:
        raise ValueError(
            "workspace= cannot be combined with repos: sessions sharing one directory "
            f"would all clone into {workspace}, and only the first would succeed. Clone "
            "the repos into that directory yourself (the agent finds them there), or drop "
            "workspace= for a per-session cwd."
        )


class LocalSession:
    """An in-process session with the CMA-style lifecycle (DESIGN.md §4).

    The session's :class:`~agent_run.config.SessionConfig` (bound at
    ``engine.start_session(config=...)``) overlays the engine's spec for every turn of
    this session; a per-turn :class:`~agent_run.config.TurnConfig` overlays
    that for one turn. Same scope model as sandbox — see ``runtime.base``.
    """

    def __init__(
        self,
        engine: LocalEngine,
        session_id: str,
        config: SessionConfig | None = None,
        config_resolved: bool = True,
    ) -> None:
        from ..config import apply_session_config, validate_harness_choice

        self._engine = engine
        self._session_id = session_id
        if not config_resolved and config is None:
            config = self._load_persisted_config()
        elif config is not None and config.set_fields():
            self._persist_config(config)
        else:
            config = None
        self._config = config
        # The session's effective spec: the deployed spec + the bound overlay. Everything
        # below (checkpoint wiring, turns) derives from THIS, not engine.spec.
        spec = apply_session_config(engine.spec, config)
        validate_harness_choice(engine.spec, spec)  # fail at bind, not mid-turn
        # The effective spec, so a SessionConfig bringing its own repos is caught as well.
        _reject_repos_in_chosen_workspace(spec, engine._workspace)
        self._spec = spec
        self._job_dir = engine._jobs_root / session_id
        # Wire the blob/session-store adapters only when the spec opts in (parity with
        # sandbox). ``transcript`` wires them for reading the transcript back; the workspace
        # snapshot stays behind ``checkpoint`` alone (see ``_shared.finalize_checkpoint``).
        self._blobs: Any | None = None
        self._session_store: Any | None = None
        if spec.checkpoint or spec.transcript:
            from ..checkpoint.session_store import BlobSessionStore

            ckpt_gcs = os.environ.get("AGENT_CHECKPOINT_GCS")
            if ckpt_gcs:
                # Same env hook as the sandbox runtime: durable checkpoints even for the
                # in-process engine (a pod-local <workdir>/blobs dies with the pod).
                from ..ports.blobstore import GcsBlobStore, parse_gcs_uri

                bucket, prefix = parse_gcs_uri(ckpt_gcs)
                self._blobs = GcsBlobStore(bucket, (prefix + "/") if prefix else "")
            else:
                from ..ports.blobstore import LocalBlobStore

                self._blobs = LocalBlobStore(str(engine._blob_root))
            self._session_store = BlobSessionStore(self._blobs)
        self._status = RunStatus.PENDING
        self._stop_reason: StopReason | None = None
        self._last_result: RunResult | None = None
        self._current_run: DrivenRun | None = None
        # The running turn's control channel (steer / interrupt / stop); one per turn.
        self._control: LocalControlChannel | None = None

    # -- session-config persistence (cross-process re-attach, same workdir) -----

    def _config_store(self) -> Any:
        from ..ports.blobstore import LocalBlobStore

        return LocalBlobStore(str(self._engine._blob_root))

    def _persist_config(self, config: SessionConfig) -> None:
        import json

        self._config_store().put_bytes(
            _SESSION_CONFIG_KEY.format(sid=self._session_id),
            json.dumps(config.to_dict()).encode("utf-8"),
        )

    def _load_persisted_config(self) -> SessionConfig | None:
        import json

        from ..config import SessionConfig

        try:
            data = self._config_store().get_bytes(_SESSION_CONFIG_KEY.format(sid=self._session_id))
        except Exception:  # noqa: BLE001 — no persisted config: the session has none
            return None
        return SessionConfig.from_dict(json.loads(data.decode("utf-8")))

    # -- run plane -------------------------------------------------------------

    def run(
        self,
        message: str,
        *,
        secrets: dict[str, str] | None = None,
        config: TurnConfig | None = None,
        hooks: Any | None = None,
    ) -> DrivenRun:
        """Start a fresh turn from ``message``.

        ``secrets`` is a per-invocation name → value map (the agent's own API keys, any repo
        ``auth`` / GitHub MCP token). Values live only for this run; they are never baked into
        the spec and never logged. ``config`` is this turn's
        :class:`~agent_run.config.TurnConfig` overlay (invocation knobs only).
        *hooks* are Claude Agent SDK hook callbacks for this turn
        (``{HookEvent: [HookMatcher, ...]}``); see :meth:`~agent_run.runtime.base.Session.run`.

        Raises ``RuntimeError`` while a turn is running: a second concurrent turn under
        the same session id would corrupt its checkpoint and transcript. Use :meth:`send`
        to talk to the running turn.
        """
        if self.busy:
            raise RuntimeError(
                "this session is running a turn; run() cannot start another one. Use "
                "send() to deliver a message into the running turn, or interrupt() first."
            )
        return self._start(
            message, resume_sid=None, secrets=secrets, turn_config=config, hooks=hooks
        )

    def send(
        self,
        message: str,
        *,
        secrets: dict[str, str] | None = None,
        config: TurnConfig | None = None,
        hooks: Any | None = None,
        interrupt: bool = False,
        message_id: str | None = None,
    ) -> DrivenRun:
        """Send ``message`` to this session.

        On an idle session this resumes the conversation as a new turn. Conversation +
        workspace continuity requires ``spec.checkpoint=True``; without it this runs a fresh
        turn with no memory of the prior one. Pass ``secrets`` again (they are not persisted
        across turns) so repo push auth is re-embedded on resume, and *hooks* again for the
        same reason. ``config`` is a per-turn :class:`~agent_run.config.TurnConfig`;
        the SESSION config cannot change here (bound at ``start_session``).

        On a RUNNING session the message goes INTO the running turn and the same
        :class:`Run` is returned (see :meth:`~agent_run.runtime.base.Session.send`):
        with ``interrupt=False`` the model sees it at its next step; with ``interrupt=True``
        the model is interrupted first and continues from the message. ``message_id`` is
        stamped on the ``user`` event that acknowledges delivery. ``secrets`` / ``config`` /
        ``hooks`` cannot change mid-turn and are rejected then.
        """
        if self.busy:
            return self._send_into_running_turn(
                message, interrupt=interrupt, message_id=message_id,
                secrets=secrets, config=config, hooks=hooks,
            )
        return self._start(
            message,
            resume_sid=self._session_id,
            secrets=secrets,
            turn_config=config,
            hooks=hooks,
        )

    @property
    def busy(self) -> bool:
        """Whether a turn of this session is running (or started and not yet finished)."""
        return self._current_run is not None and not self._current_run.done

    @property
    def current_run(self) -> DrivenRun | None:
        """The running turn's run, or None (``runtime.base.Session``; one process on ``local``)."""
        run = self._current_run
        return run if run is not None and not run.done else None

    def _send_into_running_turn(
        self,
        message: str,
        *,
        interrupt: bool,
        message_id: str | None,
        secrets: dict[str, str] | None,
        config: TurnConfig | None,
        hooks: Any | None,
    ) -> DrivenRun:
        if secrets or config is not None or hooks:
            raise ValueError(
                "secrets, config and hooks are bound when a turn starts and cannot change "
                "while it runs; send() into a running turn takes only the message (and "
                "interrupt= / message_id=). Interrupt the session first to start a new turn."
            )
        assert self._current_run is not None and self._control is not None
        msg = ControlMessage(
            op="interrupt" if interrupt else "steer",
            message=message,
            **({"message_id": message_id} if message_id else {}),
        )
        self._control.send(msg)
        return self._current_run

    def _start(
        self,
        message: str,
        resume_sid: str | None,
        secrets: dict[str, str] | None = None,
        turn_config: TurnConfig | None = None,
        hooks: Any | None = None,
    ) -> DrivenRun:
        from ..config import apply_turn_config
        from ..harness.context import RunContext

        spec = apply_turn_config(self._spec, turn_config)
        # The session's bookkeeping dir (stderr.log, a codex home, and what
        # list_sessions() reads) exists for every run, including the ones whose agent cwd
        # lives elsewhere entirely.
        self._job_dir.mkdir(parents=True, exist_ok=True)
        ctx = RunContext(
            spec=spec,
            prompt=message,
            job_dir=self._job_dir,
            session_id=self._session_id,
            secrets=dict(secrets) if secrets else {},
            env=self._engine._agent_env,
            # Resume is checkpointing's, not the transcript's: a transcript-only spec is
            # purely observational, so ``send()`` stays the documented fresh turn.
            resume_sid=(
                resume_sid if (spec.checkpoint and self._session_store is not None) else None
            ),
            session_store=self._session_store,
            blobs=self._blobs,
            interactive=spec.checkpoint if spec.interactive is None else spec.interactive,
            hooks=hooks,
            workspace_dir=self._engine._workspace,
            control=LocalControlChannel(),
        )
        self._control = ctx.control
        engine = self._engine

        async def factory() -> AsyncIterator[AgentEvent]:
            prep = await asyncio.to_thread(engine._prepare_workspace, ctx, resume_sid is not None)
            yield AgentEvent(kind="status", summary=prep["summary"], raw=prep)
            async for event in engine._harness_for(spec).run(spec, ctx):
                yield event

        run = DrivenRun(factory, self._session_id, spec, on_complete=self._on_complete)
        self._current_run = run
        self._status = RunStatus.RUNNING
        self._stop_reason = None
        # Eager-start when an event loop is running so polling (run.done) makes progress
        # even if the caller never awaits/iterates.
        try:
            asyncio.get_running_loop()
            run.ensure_started()
        except RuntimeError:
            pass
        return run

    def _on_complete(self, result: RunResult, stop_reason: StopReason) -> None:
        self._last_result = result
        self._stop_reason = stop_reason
        self._status = RunStatus.IDLE

    async def exec(
        self, command: str, *, cwd: str | None = None, timeout: float | None = None
    ) -> ExecResult:
        """Run a shell command in the running turn's workspace (``Session.exec``).

        Runs on the host under ``/bin/bash -c`` in :attr:`workspace` (or ``cwd`` under it),
        off the event loop. Same rule as ``sandbox``: only while a turn runs — otherwise it
        raises :class:`~agent_run.control.ControlUnavailable`, so code written
        against ``local`` behaves the same way remotely (on the host you can always read
        :attr:`workspace` directly).
        """
        if not isinstance(command, str) or not command:
            raise ValueError("exec() needs a non-empty command string")
        if not self.busy:
            raise ControlUnavailable(
                "no turn is running on this session: exec() probes the running turn's workspace "
                "(read session.workspace directly between turns)"
            )
        target = self.workspace if cwd is None else self.workspace / cwd  # absolute cwd stays absolute
        return await asyncio.to_thread(run_shell, command, cwd=str(target), timeout=timeout)

    async def interrupt(self, *, timeout: float = 60.0) -> None:
        """Interrupt the running turn and leave the session idle and resumable.

        The harness is asked to stop (``stop`` on the turn's control channel) and runs its
        normal end-of-turn path: workspace snapshot, transcript, a terminal result with
        ``StopReason.INTERRUPTED`` that keeps the turn's accounting. A later :meth:`send`
        resumes from that checkpoint. No-op when nothing is running.

        If the harness has not ended the turn within ``timeout`` seconds the driver task is
        cancelled instead (the pre-existing behavior): the run then ends as an error result
        with no checkpoint, and ``RunResult.warning`` says so.
        """
        run = self._current_run
        if run is None or run.done:
            return
        run.ensure_started()
        assert run.task is not None
        if self._control is not None:
            self._control.send(ControlMessage(op="stop"))
            try:
                await asyncio.wait_for(asyncio.shield(run.task), timeout)
                return
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                return
        run.cancel(
            note=(
                f"the harness did not end the turn within {timeout:g}s of the interrupt; "
                "the run was cancelled without a checkpoint"
            )
        )
        try:
            await run.task
        except asyncio.CancelledError:
            pass

    @property
    def status(self) -> RunStatus:
        return self._status

    @property
    def stop_reason(self) -> StopReason | None:
        return self._stop_reason

    @property
    def last_result(self) -> RunResult | None:
        return self._last_result

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def workspace(self) -> Path:
        """The agent's working directory on the host: ``<workdir>/jobs/<sid>/workspace``.

        Created on first access, so callers can seed input files into it before ``run()``
        and collect artifacts from it after — without deriving the layout themselves.
        The agent runs in this ``workspace`` leaf (not the anonymous ``jobs/<uuid>``
        session dir above it) so the cwd's own name says "this is your workspace".

        An engine deployed with ``workspace=`` returns that directory instead, shared by
        every session of the engine.
        """
        from ..harness.context import _workspace_path

        ws = _workspace_path(self._job_dir, self._engine._workspace)
        ws.mkdir(parents=True, exist_ok=True)
        return ws

    async def transcripts(self) -> dict[str, list[dict]]:
        """This session's persisted harness transcripts (see ``runtime.base.Session``)."""
        if self._session_store is None:
            raise RuntimeError(
                "no transcript is persisted for this session — deploy the spec with "
                "AgentSpec(transcript=True)"
            )
        return await self._session_store.load_all(self._session_id)

    def history(self) -> list[AgentEvent]:
        raise NotImplementedError(
            "local runs don't persist an event log yet — stream the Run (async for) or use "
            "the sandbox runtime, whose sessions keep a durable GCS/Cloud Logging history."
        )

    def fork(self) -> LocalSession:
        raise NotImplementedError(
            "fork is not supported yet — it would copy this session's workspace snapshot + "
            "transcript under a new session id so the branch shares prior history."
        )


class LocalEngine:
    """An in-process engine (implements ``runtime.base.Engine``; DESIGN.md §4)."""

    def __init__(
        self, spec: AgentSpec, workdir: str | None = None, workspace: str | None = None
    ) -> None:
        self.spec = spec
        root = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="agent-run-"))
        root.mkdir(parents=True, exist_ok=True)
        self._root = root
        self._workspace = Path(workspace) if workspace else None
        _reject_repos_in_chosen_workspace(spec, self._workspace)
        self._jobs_root = root / "jobs"
        self._blob_root = root / "blobs"
        self._jobs_root.mkdir(parents=True, exist_ok=True)
        self._blob_root.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, LocalSession] = {}

        # spec.packages parity with sandbox's baked engine image: resolve the agent's declared
        # packages into a per-engine venv at deploy time (uv; warm installs take seconds) and
        # activate it in the agent's env. Its bin leads PATH, composing with any spec.env PATH.
        self._agent_env: dict[str, str] | None = None
        if spec.packages:
            from .venv import provision_venv, venv_agent_env

            venv = provision_venv(root, spec.packages)
            self._agent_env = venv_agent_env(venv, (spec.env or {}).get("PATH"))

        from ..harness import resolve_harness

        self._harness = resolve_harness(spec)

    # -- helpers used by sessions/runs ----------------------------------------

    def _harness_for(self, spec: AgentSpec) -> Any:
        """The harness binding for a turn's effective spec.

        The engine-default harness is resolved once at deploy (``self._harness``); a
        session that selected a DIFFERENT harness (``SessionConfig(harness=...)``,
        validated against ``spec.harnesses`` at bind) resolves its own here.
        """
        if spec.harness == self.spec.harness:
            return self._harness
        from ..harness import resolve_harness

        return resolve_harness(spec)

    def _prepare_workspace(self, ctx: Any, is_resume: bool) -> dict:
        """Restore a prior workspace (resume) or stage skills + clone repos into a fresh cwd.

        On resume the repos/skills come back in the restored workspace, so we only stage on a
        fresh cwd. A caller-chosen cwd is never restored into: it persists on its own, and
        untarring a snapshot over it would roll its files back to whatever this session last
        saw. Sync (runs off the event loop).
        """
        from ..checkpoint.workspace import RestoreResult, try_restore, workspace_ready_event

        restored, restore_error = RestoreResult(found=False), None
        if is_resume and ctx.workspace_dir is None and ctx.blobs is not None and ctx.resume_sid:
            # A failed restore leaves no half-extracted files behind (restore cleans up), so
            # the fresh path below can clone into the directory.
            restored, restore_error = try_restore(ctx.blobs, ctx.resume_sid, str(ctx.workspace))
        names: list[str] = []
        repos: list[str] = []
        if restored:
            # The snapshot was credential-scrubbed; re-embed push tokens from this turn's secrets.
            if ctx.spec.repos:
                from ..integrations.git import reauth_repos

                repos = reauth_repos(ctx.workspace, ctx.spec.repos, ctx.secrets)
        else:
            ctx.workspace.mkdir(parents=True, exist_ok=True)
            if ctx.spec.skills:
                from ..skills import provision, skills_subdir

                names = provision(
                    ctx.spec.skills, str(ctx.workspace), skills_subdir(ctx.spec.harness)
                )
            if ctx.spec.repos:
                from ..integrations.git import provision_repos

                repos = provision_repos(ctx.workspace, ctx.spec.repos, ctx.secrets)
        return workspace_ready_event(restored, restore_error, names, repos)

    # -- Engine protocol -------------------------------------------------------

    def start_session(self, config: SessionConfig | None = None) -> LocalSession:
        """Begin a new session; ``config`` binds its ``SessionConfig`` for good (see base).

        The config is persisted under the engine's workdir so a process re-attaching by
        id (same workdir) recovers it — parity with sandbox's GCS persistence.
        """
        # Canonical UUID (dashed), NOT uuid4().hex: this id is passed to the Claude Agent
        # SDK as session_id (checkpoint keying) / resume, and the SDK rejects a non-canonical
        # id at runtime with "Invalid session ID. Must be a valid UUID".
        session = LocalSession(self, str(uuid.uuid4()), config=config)
        self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> LocalSession:
        """Re-attach by id (e.g. to resume from a checkpoint written earlier).

        Takes NO config on purpose: the session runs under the config bound at
        ``start_session`` (recovered from the workdir), never a substituted one.
        """
        session = self._sessions.get(session_id)
        if session is None:
            session = LocalSession(self, session_id, config_resolved=False)
            self._sessions[session_id] = session
        return session

    def list_sessions(self) -> list[dict]:
        """Sessions known to this engine's workdir (each per-session job dir), newest last id.

        Local runs keep no durable event log — this lists the job dirs so a persistent
        ``workdir`` can be re-attached across processes; ``history()`` stays sandbox-only.
        """
        if not self._jobs_root.is_dir():
            return []
        return [
            {"session_id": p.name, "sources": ["jobs-dir"], "last_file": None}
            for p in sorted(self._jobs_root.iterdir())
            if p.is_dir()
        ]

    def versions(self) -> list[str]:
        return ["local"]

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def version(self) -> str:
        return "local"

    @property
    def resource(self) -> str:
        return f"local:{self.spec.name}"


def deploy(
    spec: AgentSpec, *, workdir: str | None = None, workspace: str | None = None, **_: Any
) -> LocalEngine:
    """Stand up an in-process engine for ``spec`` (mirrors ``sandbox.deploy``).

    ``workdir`` sets the root for per-session job dirs and the local blob store; omit it
    for a throwaway temp dir. The harness + local/in-memory adapters are wired here.

    ``workspace`` runs the agent in a directory you choose instead of the per-session
    ``<workdir>/jobs/<sid>/workspace`` — see the README on picking the agent's cwd. It
    rules out ``spec.repos`` (every session would clone into the same path) and it takes
    the workspace out of checkpointing (the directory is yours and already durable, so
    only the conversation is snapshotted).
    """
    return LocalEngine(spec, workdir=workdir, workspace=workspace)


def run(
    spec: AgentSpec,
    message: str,
    *,
    secrets: dict[str, str] | None = None,
    workdir: str | None = None,
    workspace: str | None = None,
    config: TurnConfig | None = None,
    hooks: Any | None = None,
    **_: Any,
) -> RunResult | None:
    """Convenience: ``deploy`` → ``start_session`` → ``await run(message)`` (sync).

    Runs its own event loop, so call it from sync code. ``secrets``, ``config`` and *hooks*
    go to the single turn (see :meth:`LocalSession.run`); ``workdir`` and ``workspace`` are
    :func:`deploy`'s. For streaming/polling, or from inside an event loop, use ``deploy``
    and drive the ``Session``/``Run`` directly.
    """
    async def _arun() -> RunResult | None:
        engine = deploy(spec, workdir=workdir, workspace=workspace)
        session = engine.start_session()
        return await session.run(message, secrets=secrets, config=config, hooks=hooks)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_arun())
    raise RuntimeError(
        "local.run() is a synchronous convenience; inside an event loop use "
        "'await local.deploy(spec).start_session().run(message)' instead."
    )

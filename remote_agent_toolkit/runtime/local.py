"""Local, in-process run plane (DESIGN.md §3.2, §3.4).

``local.deploy(spec)`` returns a ``LocalEngine`` exposing the *same* Engine/Session/Run
surface as ``gemini``, backed by the in-process :class:`ClaudeCodeHarness` and
filesystem/in-memory port adapters (no GCP beyond model access). This is the first-class
dev loop: develop against ``local``, ship against ``gemini``, with zero code change.

A ``Run`` is consumable three ways — ``await`` it (→ ``RunResult``), ``async for`` over it
(→ ``AgentEvent``s), or poll ``run.done`` / ``run.status``. A single background task drives
the harness generator; the queue feeds iterators while the accumulated result feeds
``await``/poll.

The Claude Agent SDK and port adapters are imported lazily, so importing this module
needs no third-party deps.
"""

from __future__ import annotations

import asyncio
import tempfile
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Callable, TYPE_CHECKING

from ..events import AgentEvent, RunResult, RunStatus, StopReason
from ..structured import parse_structured_output

if TYPE_CHECKING:
    from ..spec import AgentSpec

_SENTINEL = object()


def _stop_reason(result_raw: dict, checkpoint: bool) -> StopReason:
    """Map a result message's subtype/error onto a :class:`StopReason`."""
    subtype = str(result_raw.get("subtype") or "")
    if "max_turn" in subtype:
        return StopReason.MAX_TURNS
    if "budget" in subtype:
        return StopReason.BUDGET_EXCEEDED
    if result_raw.get("is_error"):
        return StopReason.ERROR
    # A clean turn end: in interactive/checkpoint mode the session is paused awaiting the
    # operator's next message; otherwise the turn is simply complete.
    return StopReason.NEEDS_INPUT if checkpoint else StopReason.END_TURN


def _build_result(result_ev: AgentEvent, session_id: str, spec: AgentSpec) -> tuple[RunResult, StopReason]:
    raw = result_ev.raw or {}
    structured = (
        parse_structured_output(result_ev.summary, spec.output_schema)
        if spec.output_schema is not None
        else None
    )
    result = RunResult(
        text=result_ev.summary,
        structured_output=structured,
        is_error=bool(raw.get("is_error")),
        num_turns=int(raw.get("num_turns") or 0),
        cost_usd=float(result_ev.cost_usd or 0.0),
        usage=result_ev.usage,
        session_id=raw.get("session_id") or session_id,
    )
    return result, _stop_reason(raw, spec.checkpoint)


class LocalRun:
    """A single turn, driven by a background task feeding a queue + a result slot."""

    def __init__(
        self,
        agen_factory: Callable[[], AsyncIterator[AgentEvent]],
        session_id: str,
        spec: AgentSpec,
        on_complete: Callable[[RunResult, StopReason], None],
    ) -> None:
        self._agen_factory = agen_factory
        self._session_id = session_id
        self._spec = spec
        self._on_complete = on_complete
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._status = RunStatus.PENDING
        self._result: RunResult | None = None
        self._stop_reason: StopReason | None = None

    # -- driving ---------------------------------------------------------------

    def _ensure_started(self) -> None:
        if self._task is None:
            self._task = asyncio.ensure_future(self._drive())

    async def _drive(self) -> None:
        self._status = RunStatus.RUNNING
        result_ev: AgentEvent | None = None
        try:
            async for event in self._agen_factory():
                if event.kind == "result" and result_ev is None:
                    result_ev = event
                await self._queue.put(event)
        except asyncio.CancelledError:
            self._finalize(result_ev, error="interrupted")
            await self._queue.put(_SENTINEL)
            raise
        except Exception as exc:  # noqa: BLE001 — surface as an error result, don't crash callers
            await self._queue.put(
                AgentEvent(kind="status", summary=f"run failed: {str(exc)[:200]}", raw={"event": "error"})
            )
            self._finalize(result_ev, error=str(exc))
            await self._queue.put(_SENTINEL)
            return
        self._finalize(result_ev, error=None)
        await self._queue.put(_SENTINEL)

    def _finalize(self, result_ev: AgentEvent | None, error: str | None) -> None:
        if error is not None:
            text = None if error == "interrupted" else f"run failed: {error}"
            self._result = RunResult(text=text, is_error=True, session_id=self._session_id)
            self._stop_reason = StopReason.ERROR
        elif result_ev is not None:
            self._result, self._stop_reason = _build_result(result_ev, self._session_id, self._spec)
        else:
            self._result = RunResult(
                text=None, is_error=True, session_id=self._session_id
            )
            self._stop_reason = StopReason.ERROR
        self._status = RunStatus.IDLE
        self._on_complete(self._result, self._stop_reason)

    # -- the three consumption modes ------------------------------------------

    def __await__(self):
        return self._wait().__await__()

    async def _wait(self) -> RunResult | None:
        self._ensure_started()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        return self._result

    async def __aiter__(self) -> AsyncIterator[AgentEvent]:
        self._ensure_started()
        while True:
            item = await self._queue.get()
            if item is _SENTINEL:
                break
            yield item

    @property
    def done(self) -> bool:
        return self._status in (RunStatus.IDLE, RunStatus.TERMINATED)

    @property
    def status(self) -> RunStatus:
        return self._status

    @property
    def result(self) -> RunResult | None:
        return self._result


class LocalSession:
    """An in-process session with the CMA-style lifecycle (DESIGN.md §4)."""

    def __init__(self, engine: LocalEngine, session_id: str) -> None:
        self._engine = engine
        self._session_id = session_id
        spec = engine.spec
        self._job_dir = engine._jobs_root / session_id
        # Wire checkpoint adapters only when the spec opts in (parity with gemini).
        self._blobs: Any | None = None
        self._session_store: Any | None = None
        if spec.checkpoint:
            from ..checkpoint.session_store import BlobSessionStore
            from ..ports.blobstore import LocalBlobStore

            self._blobs = LocalBlobStore(str(engine._blob_root))
            self._session_store = BlobSessionStore(self._blobs)
        self._status = RunStatus.PENDING
        self._stop_reason: StopReason | None = None
        self._last_result: RunResult | None = None
        self._current_run: LocalRun | None = None

    # -- run plane -------------------------------------------------------------

    def run(self, message: str) -> LocalRun:
        """Start a fresh turn from ``message``."""
        return self._start(message, resume_sid=None)

    def send(self, message: str) -> LocalRun:
        """Resume this session with ``message`` (continues the conversation).

        Conversation + workspace continuity requires ``spec.checkpoint=True``; without it
        this runs a fresh turn with no memory of the prior one.
        """
        return self._start(message, resume_sid=self._session_id)

    def _start(self, message: str, resume_sid: str | None) -> LocalRun:
        from ..harness.context import RunContext

        spec = self._engine.spec
        ctx = RunContext(
            spec=spec,
            prompt=message,
            job_dir=self._job_dir,
            session_id=self._session_id,
            secrets=self._engine._resolve_secrets(spec),
            resume_sid=resume_sid if self._session_store is not None else None,
            session_store=self._session_store,
            blobs=self._blobs,
            interactive=spec.checkpoint,
        )
        engine = self._engine

        async def factory() -> AsyncIterator[AgentEvent]:
            prep = await asyncio.to_thread(engine._prepare_workspace, ctx, resume_sid is not None)
            yield AgentEvent(kind="status", summary=prep["summary"], raw=prep)
            async for event in engine._harness.run(spec, ctx):
                yield event

        run = LocalRun(factory, self._session_id, spec, on_complete=self._on_complete)
        self._current_run = run
        self._status = RunStatus.RUNNING
        self._stop_reason = None
        # Eager-start when an event loop is running so polling (run.done) makes progress
        # even if the caller never awaits/iterates.
        try:
            asyncio.get_running_loop()
            run._ensure_started()
        except RuntimeError:
            pass
        return run

    def _on_complete(self, result: RunResult, stop_reason: StopReason) -> None:
        self._last_result = result
        self._stop_reason = stop_reason
        self._status = RunStatus.IDLE

    async def interrupt(self) -> None:
        """Interrupt the in-flight run (cancels the driver task)."""
        run = self._current_run
        if run is not None and run._task is not None and not run._task.done():
            run._task.cancel()
            try:
                await run._task
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

    def fork(self) -> LocalSession:
        raise NotImplementedError(
            "P2: fork copies this session's workspace snapshot + transcript under a new "
            "session id so the branch shares prior history."
        )


class LocalEngine:
    """An in-process engine (implements ``runtime.base.Engine``; DESIGN.md §4)."""

    def __init__(self, spec: AgentSpec, workdir: str | None = None) -> None:
        self.spec = spec
        root = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="ratk-"))
        root.mkdir(parents=True, exist_ok=True)
        self._root = root
        self._jobs_root = root / "jobs"
        self._blob_root = root / "blobs"
        self._jobs_root.mkdir(parents=True, exist_ok=True)
        self._blob_root.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, LocalSession] = {}

        from ..harness.claude_code import ClaudeCodeHarness
        from ..ports.secrets import EnvSecretResolver

        self._harness = ClaudeCodeHarness()
        self._secrets = EnvSecretResolver()

    # -- helpers used by sessions/runs ----------------------------------------

    def _resolve_secrets(self, spec: AgentSpec) -> dict[str, str]:
        """Resolve declared secret names to values (missing names skipped, never logged)."""
        out: dict[str, str] = {}
        for name in spec.secrets:
            try:
                out[name] = self._secrets.resolve(name)
            except KeyError:
                continue  # leave unset; never surface the (absent) value
        return out

    def _prepare_workspace(self, ctx: Any, is_resume: bool) -> dict:
        """Restore a prior workspace (resume) or stage skills into a fresh cwd. Sync."""
        restored = False
        if is_resume and ctx.blobs is not None and ctx.resume_sid:
            from ..checkpoint.workspace import restore

            try:
                restored = restore(ctx.blobs, ctx.resume_sid, str(ctx.job_dir))
            except Exception:  # noqa: BLE001 — fall back to a fresh workspace
                restored = False
        names: list[str] = []
        if not restored:
            ctx.job_dir.mkdir(parents=True, exist_ok=True)
            if ctx.spec.skills:
                from ..skills import provision

                names = provision(ctx.spec.skills, str(ctx.job_dir))
        return {
            "event": "workspace_ready",
            "restored": restored,
            "skills": names,
            "summary": f"workspace ready (restored={restored}, skills staged={len(names)})",
        }

    # -- Engine protocol -------------------------------------------------------

    def start_session(self) -> LocalSession:
        session = LocalSession(self, uuid.uuid4().hex)
        self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> LocalSession:
        session = self._sessions.get(session_id)
        if session is None:
            # Re-attach by id (e.g. to resume from a checkpoint written earlier).
            session = LocalSession(self, session_id)
            self._sessions[session_id] = session
        return session

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


def deploy(spec: AgentSpec, *, workdir: str | None = None, **_: Any) -> LocalEngine:
    """Stand up an in-process engine for ``spec`` (mirrors ``gemini.deploy``).

    ``workdir`` sets the root for per-session job dirs and the local blob store; omit it
    for a throwaway temp dir. The harness + local/in-memory adapters are wired here.
    """
    return LocalEngine(spec, workdir=workdir)


def run(spec: AgentSpec, message: str, *, workdir: str | None = None, **_: Any) -> RunResult | None:
    """Convenience: ``deploy`` → ``start_session`` → ``await run(message)`` (sync).

    Runs its own event loop, so call it from sync code. For streaming/polling, or from
    inside an event loop, use ``deploy`` and drive the ``Session``/``Run`` directly.
    """
    async def _arun() -> RunResult | None:
        engine = deploy(spec, workdir=workdir)
        session = engine.start_session()
        return await session.run(message)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_arun())
    raise RuntimeError(
        "local.run() is a synchronous convenience; inside an event loop use "
        "'await local.deploy(spec).start_session().run(message)' instead."
    )

"""Gemini Agent Runtime control plane: deploy / get_engine / list_engines + Engine/Session/Run.

Encodes the §6 platform contracts. ``deploy`` packages the agent (``deploy.py``), wraps it
in an ``AdkApp``, and creates a reasoningEngine; app code addresses engines by name via
``get_engine`` and never deploys (DESIGN.md §3.3).

The run plane mirrors ``local`` exactly via the shared :class:`DrivenRun`: a ``Session.run``
submits a ``run_query_job`` and the ``Run`` is driven by tailing the ``CloudLoggingSink`` for
that session — so **stream** = tail, **await** = wait for the terminal ``result`` log event,
**poll** = read the latest status. Same awaitable/iterable/pollable handle as local.

The Google SDK (``vertexai`` / ``google-cloud-aiplatform``) and the sink are imported lazily
inside the bodies, so importing this module needs no third-party deps.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, AsyncIterator, TYPE_CHECKING

from ...events import AgentEvent, RunResult, RunStatus, StopReason
from .._run import DrivenRun
from .pool import POOL_WAIT_SENTINEL, dispatch_payload, pool_paths

if TYPE_CHECKING:
    from ...spec import AgentSpec
    from ..base import Engine

# Must match CloudLoggingSink's default log_name (the deployed agent emits here, the client tails it).
_LOG_NAME = "remote_agent_toolkit_steps"
_USER_ID = "ratk"

# Cold-run watchdog: probe the job's state after this much stream silence, and end the run
# this long after the job is seen terminal with still no terminal event (ingestion-lag grace).
_WATCHDOG_QUIET_S = 60.0
_WATCHDOG_GRACE_S = 120.0


async def _watched_tail(
    tail_factory: Any,
    probe: Any,
    session_id: str,
    history_reader: Any = None,
    quiet_s: float = _WATCHDOG_QUIET_S,
    grace_s: float = _WATCHDOG_GRACE_S,
) -> AsyncIterator[AgentEvent]:
    """Yield the tail's events, ending the run if the job terminates without a terminal event.

    The client normally learns a run finished from the terminal ``result`` event on the log
    tail; a job that ends WITHOUT the tail seeing one (worker OOM, external cancel, engine
    deleted, logging outage — or, observed live, multi-minute Cloud Logging ingestion lag)
    would leave the tail waiting up to its 1 h cap. So: while events flow this wrapper is pure
    passthrough (zero probe calls); after ``quiet_s`` of silence it asks ``probe`` for the
    authoritative job state ("RUNNING"/"SUCCESS"/"FAILED"/None). A terminal state starts a
    ``grace_s`` window for the event to still arrive; after that:

    1. **Recover from the durable record first**: ``history_reader`` reads the GCS-mirrored
       events (written by the worker at turn end — no ingestion lag). If it holds the terminal
       result, the not-yet-seen events are delivered — the run ends with its REAL result (this
       is exactly the ingestion-lag case observed live).
    2. Only with no durable terminal record either, a synthetic error ``result`` explains what
       happened instead of hanging.

    A probe/reader failure is treated as RUNNING/absent — flakiness must never kill a healthy
    run; the tail's own cap remains the backstop.
    """
    queue: asyncio.Queue = asyncio.Queue()
    done = object()

    async def pump() -> None:
        try:
            async for ev in tail_factory():
                await queue.put(ev)
        finally:
            await queue.put(done)

    pump_task = asyncio.ensure_future(pump())
    job_terminal_state: str | None = None
    job_terminal_at: float | None = None
    yielded = 0
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=quiet_s)
            except (TimeoutError, asyncio.TimeoutError):
                state = await probe()
                if state not in ("SUCCESS", "FAILED"):
                    continue
                now = time.monotonic()
                if job_terminal_at is None:
                    job_terminal_state, job_terminal_at = state, now
                    continue
                if now - job_terminal_at < grace_s:
                    continue
                # Job over, grace elapsed, tail still silent. Try the durable record: the
                # mirror holds the same event stream in the same order, so skip what the
                # tail already delivered and finish with the real remainder if it's terminal.
                history: list[AgentEvent] = []
                if history_reader is not None:
                    try:
                        history = await history_reader() or []
                    except Exception:  # noqa: BLE001 — recovery is best-effort
                        history = []
                if any(ev.kind == "result" for ev in history):
                    for ev in history[yielded:]:
                        yield ev
                        if ev.kind == "result":
                            return
                    return
                yield AgentEvent(
                    kind="result",
                    summary=(
                        f"job finished (state={job_terminal_state}) but no terminal "
                        "event was observed — the worker likely died mid-run; see "
                        "session.history() / Cloud Logging for the last steps"
                    ),
                    raw={"event": "job_terminated_without_result", "is_error": True,
                         "subtype": "error", "job_state": job_terminal_state,
                         "session_id": session_id},
                )
                return
            if item is done:
                return
            yielded += 1
            yield item
            if item.kind == "result":
                return
    finally:
        pump_task.cancel()


def _run_blocking(make_coro, timeout: float) -> bool:
    """Run ``make_coro()`` to completion (≤ ``timeout``), from sync OR async context.

    ``make_coro`` is a zero-arg factory returning a fresh coroutine. With no running event loop
    we drive it directly; inside a running loop we run it on a dedicated thread with its own loop
    (so a blocking helper like ``wait_until_warm`` works either way). Returns ``False`` on timeout.
    """
    def _runner() -> bool:
        try:
            return asyncio.run(asyncio.wait_for(make_coro(), timeout))
        except (TimeoutError, asyncio.TimeoutError):
            return False

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _runner()  # no ambient loop — safe to drive our own
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_runner).result()


def _fallback_spec(name: str) -> AgentSpec:
    """A minimal spec for a looked-up engine when the caller didn't pass the real one.

    Result-building needs ``output_schema`` (structured parsing) and ``checkpoint`` (the
    idle stop-reason). Without the real spec, structured output is skipped and a clean turn
    reports ``END_TURN``. Pass ``spec=`` to ``get_engine`` to recover both.
    """
    from ...spec import AgentSpec

    return AgentSpec(name=name, model="")


# -- control plane -------------------------------------------------------------


def deploy(
    spec: AgentSpec,
    project: str,
    location: str,
    warm_pool: bool = False,
    *,
    staging_bucket: str | None = None,
    output_bucket: str | None = None,
    model: str | None = None,
    use_vertex: bool = True,
    min_instances: int = 0,
    max_instances: int = 1,
    resource_limits: dict[str, str] | None = None,
    pool_size: int = 2,
    credentials: Any | None = None,
    **_: Any,
) -> Engine:
    """Deploy ``spec`` to Gemini Agent Runtime, minting an engine (ops/CI action).

    Bakes the toolkit package + resolved skills, applies the §6 deploy contracts, and returns
    a :class:`GeminiEngine`. With ``warm_pool=True`` it also ensures the dispatch topic/sub and
    fills the pool with ``pool_size`` pre-warmed workers (each a job blocked on ``__POOL_WAIT__``).

    ``use_vertex`` (default) routes the model through Vertex, so the engine authenticates as its
    own GCP identity and **no LLM API key is ever in the agent's environment** (the recommended,
    prompt-injection-safe default). Set ``use_vertex=False`` only if the project can't use Vertex
    Claude; then supply ``ANTHROPIC_API_KEY`` as a per-invocation secret — but note the agent can
    then read that key (see the README security section). No secrets are ever baked into the engine.

    ``min_instances`` defaults to 0: the toolkit only uses the async ``run_query_job`` path,
    where every job provisions its own worker, so an idle engine costs (almost) nothing. A
    min-instances container would serve only the unused sync path while billing continuously.

    ``resource_limits`` sets the container CPU/memory, e.g. ``{"cpu": "4", "memory": "16Gi"}``
    (platform default ``4``/``4Gi``). Raise the memory for agents whose turns do memory-heavy
    work (dependency builds, whole-project imports) — under the default 4Gi the platform's
    job runner can OOM-kill a worker mid-turn, losing the attempt's work and spend even
    though the retry (see the handoff docs) picks the turn up from scratch.
    """
    import dataclasses
    import os

    import vertexai
    from vertexai._genai import types as gt
    from vertexai.preview.reasoning_engines import AdkApp

    from .adk_agent import build_agent
    from .deploy import build_engine_config, stage_agent, validate_resource_limits

    # Fail fast BEFORE any side effect (pub/sub ensure, staging, the ~4 min billable build).
    if resource_limits is not None:
        validate_resource_limits(resource_limits)
    # A model override applies to the harness too: the deployed agent reads spec.model, so
    # bake the override into the spec (not just the env) before serializing it.
    if model:
        spec = dataclasses.replace(spec, model=model)
    staging_bucket = staging_bucket or f"gs://{project}-agent-staging"
    output_bucket = output_bucket or f"gs://{project}-agent-output"

    # Warm pool: the workers pull a shared dispatch subscription; the engine env points at it.
    topic = subscription = None
    if warm_pool:
        from ...ports.dispatch import PubSubDispatch

        topic, subscription = pool_paths(project, spec.name)
        # Ensure the topic/sub FIRST — fail fast on missing pub/sub perms BEFORE the (~4 min,
        # billable) engine build, so a perms error never leaks a half-provisioned engine.
        PubSubDispatch(
            topic=topic, subscription=subscription, project=project, credentials=credentials
        ).ensure()

    stage_dir, extra_packages = stage_agent(spec)
    os.chdir(stage_dir)  # extra_packages are resolved relative to the cwd
    # session_service_builder pins the app to in-memory ADK sessions: the toolkit keys
    # everything by its own session id (AGENT_SESSION directive + GCS mirror), and the
    # 2026-07-28 platform runner's managed-session wiring is broken on new engines.
    from .adk_agent import in_memory_session_service

    app = AdkApp(
        agent=build_agent(spec),
        enable_tracing=True,
        session_service_builder=in_memory_session_service,
    )
    config_kwargs = build_engine_config(
        spec,
        project=project,
        location=location,
        staging_bucket=staging_bucket,
        extra_packages=extra_packages,
        model=model,
        output_bucket=output_bucket,
        use_vertex=use_vertex,
        warm_pool=warm_pool,
        pool_subscription=subscription,
        min_instances=min_instances,
        max_instances=max_instances,
        resource_limits=resource_limits,
    )
    client = vertexai.Client(project=project, location=location, credentials=credentials)
    engine = client.agent_engines.create(agent=app, config=gt.AgentEngineConfig(**config_kwargs))
    geng = GeminiEngine(
        resource=engine.api_resource.name,
        spec=spec,
        project=project,
        location=location,
        output_bucket=output_bucket,
        credentials=credentials,
        warm=warm_pool,
        topic=topic,
        subscription=subscription,
    )
    if warm_pool:
        try:
            geng.fill_pool(pool_size)  # block pool_size workers on the dispatch sub
        except Exception:  # don't leak the freshly-created engine if filling the pool fails
            try:
                client.agent_engines.delete(name=geng.resource, force=True)
            except Exception:  # noqa: BLE001
                pass
            raise
    return geng


def _resolve_resource(client: Any, name: str, version: str | None) -> str:
    if version is not None:
        raise NotImplementedError("version pinning is not supported yet; omit `version` for the latest engine.")
    if "/reasoningEngines/" in name or "/agentEngines/" in name:
        return name  # already a full resource path
    matches = [
        e for e in client.agent_engines.list()
        if getattr(e.api_resource, "display_name", None) == name
    ]
    if not matches:
        raise LookupError(f"no deployed engine named {name!r} in this project/location")
    return matches[-1].api_resource.name  # latest match


def get_engine(
    name: str,
    project: str | None = None,
    location: str | None = None,
    version: str | None = None,
    *,
    spec: AgentSpec | None = None,
    output_bucket: str | None = None,
    warm_pool: bool = False,
    credentials: Any | None = None,
) -> Engine:
    """Look up a deployed engine by ``name`` (the app-code hot path; never deploys).

    Pass ``spec=`` (the one you deployed) to enable structured-output parsing and the correct
    idle stop-reason; otherwise a minimal fallback spec is used. Pass ``warm_pool=True`` to
    address a warm-pool engine (turns are dispatched to its pool instead of cold-started).
    """
    import vertexai

    client = vertexai.Client(project=project, location=location, credentials=credentials)
    resource = _resolve_resource(client, name, version)
    topic = subscription = None
    if warm_pool:
        topic, subscription = pool_paths(project, name)
    return GeminiEngine(
        resource=resource,
        spec=spec or _fallback_spec(name),
        project=project,
        location=location,
        output_bucket=output_bucket or (f"gs://{project}-agent-output" if project else None),
        credentials=credentials,
        warm=warm_pool,
        topic=topic,
        subscription=subscription,
    )


def list_engines(project: str, location: str, *, credentials: Any | None = None) -> list[dict]:
    """Discover deployed engines in ``project``/``location`` (control plane)."""
    import vertexai

    client = vertexai.Client(project=project, location=location, credentials=credentials)
    out: list[dict] = []
    for engine in client.agent_engines.list():
        resource = engine.api_resource
        out.append({"name": getattr(resource, "display_name", None), "resource": resource.name})
    return out


# -- run plane -----------------------------------------------------------------


class GeminiSession:
    """A run-plane session over Agent Engine (CMA-style lifecycle; DESIGN.md §4)."""

    def __init__(self, engine: GeminiEngine, session_id: str) -> None:
        self._engine = engine
        self._session_id = session_id
        self._status = RunStatus.PENDING
        self._stop_reason: StopReason | None = None
        self._last_result: RunResult | None = None
        self._current_run: DrivenRun | None = None
        self._last_job: Any | None = None
        self._staged_secrets_uri: str | None = None  # single-use handoff object (cleanup on complete)

    def run(self, message: str, *, secrets: dict[str, str] | None = None) -> DrivenRun:
        """Start a fresh turn (submits a ``run_query_job``).

        ``secrets`` is a per-invocation name → value map (the agent's own keys, any repo
        ``auth`` / GitHub MCP token) — never baked into the engine, never logged. Values are
        staged at a single-use GCS object the worker fetches and deletes; only that pointer
        rides the invocation (the platform persists a job's input verbatim, and a Pub/Sub
        message is retained until acked, so values must never travel in either).
        """
        return self._submit(message, resume=False, secrets=secrets)

    def send(self, message: str, *, secrets: dict[str, str] | None = None) -> DrivenRun:
        """Resume this session with ``message``. Pass ``secrets`` again (not persisted)."""
        return self._submit(message, resume=True, secrets=secrets)

    def _stage_secrets(self, secrets: dict[str, str] | None) -> str | None:
        """Stage per-invocation secrets to a single-use GCS object; return its gs:// URI."""
        if not secrets:
            return None
        engine = self._engine
        if not engine._output_bucket:
            raise ValueError(
                "passing secrets requires the engine's output bucket (the values are staged "
                "there for the worker); construct the engine with output_bucket/project set."
            )
        from .handoff import stage_secrets

        return stage_secrets(engine._output_bucket, self._session_id, secrets)

    def _submit(
        self, message: str, resume: bool, secrets: dict[str, str] | None = None
    ) -> DrivenRun:
        from ...ports.eventsink import CloudLoggingSink

        engine = self._engine
        sid = self._session_id
        # Watermark the log tail just before kicking off (small slack for clock skew).
        since = time.time() - 5
        secrets_uri = self._stage_secrets(secrets)
        self._staged_secrets_uri = secrets_uri  # cleaned up on completion (worker deletes first)

        if engine._warm:
            # Warm path: dispatch the turn to the pool (a warm worker adopts our session_id),
            # then refill so the next turn stays warm. No cold run_query_job. The payload
            # carries only the secrets *pointer*, never values.
            engine._dispatch().publish(dispatch_payload(sid, message, resume, secrets_uri))
            try:
                engine.fill_pool(1)
            except Exception:  # noqa: BLE001 — refill is best-effort; the turn already dispatched
                pass
        else:
            # Cold path: the prompt is the only reliable channel to the agent, so the
            # session id, secrets POINTER, and resume marker ride leading directive lines
            # the agent strips. Never secret values: the platform persists the job input
            # to jobs/<sid>_input.jsonl.
            directives = f"AGENT_SESSION={sid}\n"
            if secrets_uri:
                directives += f"AGENT_SECRETS_GCS={secrets_uri}\n"
            if resume:
                directives += f"AGENT_RESUME={sid}\n"
            prompt = directives + message
            # Two platform-runner regressions of 2026-07-28 shape this payload (engines
            # created before still work the old way; this form works on both):
            # * class_method must be EXPLICIT — the runner no longer resolves a default
            #   method for newly created engines; a job without it completes SUCCESS
            #   having invoked nothing (empty output, no events, no logs).
            # * session_id must be OMITTED — new engines' job workers lose the managed
            #   session service (no GOOGLE_CLOUD_AGENT_ENGINE_ID), so a supplied id 498s
            #   against the in-memory fallback. Omitted, the app auto-creates a throwaway
            #   ADK session; OUR session id rides the AGENT_SESSION directive instead and
            #   keys the event stream/mirror/checkpoints exactly as before.
            payload = {
                "class_method": "async_stream_query",
                "input": {"user_id": _USER_ID, "message": prompt},
            }
            cfg: dict[str, Any] = {"query": json.dumps(payload)}
            if engine._output_bucket:
                cfg["output_gcs_uri"] = f"{engine._output_bucket}/jobs/{sid}.jsonl"
            self._last_job = engine._agent_engines().run_query_job(name=engine._resource, config=cfg)

        sink = CloudLoggingSink(
            log_name=_LOG_NAME, project=engine._project, credentials=engine._credentials
        )

        # Cold runs hold the job's operation name, so the tail gets a watchdog: if the job
        # terminates without ever writing a terminal event, the run ends with an explained
        # error instead of hanging. Warm turns have no per-turn job handle (the turn runs in
        # whichever pool worker claimed it) — plain tail, protected by the per-poll timeouts.
        job_name = getattr(self._last_job, "job_name", None)
        if job_name:
            async def probe() -> str | None:
                def _check() -> str | None:
                    result = engine._agent_engines().check_query_job(name=job_name)
                    return getattr(result, "status", None)

                try:
                    return await asyncio.wait_for(asyncio.to_thread(_check), timeout=30)
                except Exception:  # noqa: BLE001 — unreachable probe must not kill a healthy run
                    return None

            async def history_reader() -> list:
                from .history import read_history

                return await asyncio.wait_for(
                    asyncio.to_thread(
                        read_history, engine._output_bucket, sid,
                        project=engine._project, credentials=engine._credentials,
                    ),
                    timeout=60,
                )

            def factory() -> AsyncIterator:
                return _watched_tail(
                    lambda: sink.tail(sid, since=since), probe, sid, history_reader
                )
        else:
            def factory() -> AsyncIterator:
                return sink.tail(sid, since=since)

        run = DrivenRun(factory, sid, engine.spec, on_complete=self._on_complete)
        self._current_run = run
        self._status = RunStatus.RUNNING
        self._stop_reason = None
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
        # Backstop cleanup of the staged secrets object; the worker normally deleted it on
        # read, but a run that failed before the worker fetched would otherwise leave it.
        if self._staged_secrets_uri:
            from .handoff import delete_staged_secrets

            delete_staged_secrets(self._staged_secrets_uri)
            self._staged_secrets_uri = None

    async def interrupt(self) -> None:
        """Interrupt the in-flight run: stop tailing AND cancel the remote job.

        Cancelling the ``run_query_job`` stops the agent (and its billing). The warm path
        has no per-turn job handle (the turn runs in a pool worker), so there it only stops
        tailing — the worker finishes its turn.
        """
        run = self._current_run
        if run is not None and run.task is not None and not run.task.done():
            run.task.cancel()
            try:
                await run.task
            except asyncio.CancelledError:
                pass
        job_name = getattr(self._last_job, "job_name", None)
        if job_name:
            engine = self._engine
            try:
                await asyncio.to_thread(
                    lambda: engine._agent_engines().cancel_query_job(
                        name=engine._resource, config={"operation_name": job_name}
                    )
                )
            except Exception:  # noqa: BLE001 — best effort (job may already be done)
                pass

    def history(self) -> list:
        """All persisted events of this session, oldest first (see ``history.read_history``).

        Reads the durable record — mirrored ``events/`` files, else the platform job output,
        else Cloud Logging — so it works for a session re-attached from another process long
        after the run. Empty when nothing was persisted (or the log entries expired).
        """
        from .history import read_history

        engine = self._engine
        return read_history(
            engine._output_bucket,
            self._session_id,
            project=engine._project,
            credentials=engine._credentials,
        )

    def resource_samples(self) -> list[dict]:
        """This session's worker CPU/RAM samples, oldest first (OOM forensics).

        Each row: ``time`` (aware datetime) + ``memory_current_bytes`` /
        ``memory_limit_bytes`` / ``memory_peak_bytes`` / ``cpu_usec`` as available. Read
        from the ``remote_agent_toolkit_resources`` Cloud Logging side log (~30 day
        retention), so it works for re-attached sessions — and for a worker the platform
        killed mid-turn, whose last sample landed at most one sample interval before death.
        """
        from .resources import read_samples

        engine = self._engine
        return read_samples(
            self._session_id, project=engine._project, credentials=engine._credentials
        )

    @property
    def status(self) -> RunStatus:
        return self._status

    @property
    def stop_reason(self) -> StopReason | None:
        return self._stop_reason

    @property
    def last_result(self) -> RunResult | None:
        """The most recent run's result — reconstructed from history for re-attached sessions.

        When this process ran the turn, the completed run fills it directly. Otherwise (a
        session re-attached by id) accessing this property reads the persisted history and
        rebuilds the result from its terminal event — one storage/logging round-trip per
        access until a result exists, so poll ``run.done`` for in-flight runs, not this.
        """
        if self._last_result is None and self._current_run is None:
            events = self.history()
            result_ev = next((e for e in reversed(events) if e.kind == "result"), None)
            if result_ev is not None:
                from .._run import build_result

                self._last_result, self._stop_reason = build_result(
                    result_ev, self._session_id, self._engine.spec
                )
                self._status = RunStatus.IDLE
        return self._last_result

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def workspace(self):
        raise NotImplementedError(
            "the agent's workspace lives on the remote worker's filesystem "
            "($AGENT_JOBS_ROOT/<claude-sid>/workspace) — seed inputs via the prompt or "
            "spec.repos, and collect outputs via events/history or a repo push."
        )

    def fork(self) -> GeminiSession:
        raise NotImplementedError(
            "fork is not supported yet (would copy the workspace snapshot + transcript under a new id)."
        )


class GeminiEngine:
    """A deployed Agent Engine handle (implements ``runtime.base.Engine``; DESIGN.md §4)."""

    def __init__(
        self,
        resource: str,
        spec: AgentSpec,
        project: str | None,
        location: str | None,
        output_bucket: str | None = None,
        credentials: Any | None = None,
        warm: bool = False,
        topic: str | None = None,
        subscription: str | None = None,
    ) -> None:
        self._resource = resource
        self.spec = spec
        self._project = project
        self._location = location
        self._output_bucket = output_bucket
        self._credentials = credentials
        self._warm = warm
        self._topic = topic
        self._subscription = subscription
        self._created = time.time()  # readiness-tail watermark (ignore stale pool markers)
        self._sessions: dict[str, GeminiSession] = {}
        self._pool_jobs: list[str] = []  # tracked pool-worker job names (to cancel on delete)

    def _client(self) -> Any:
        import vertexai

        return vertexai.Client(
            project=self._project, location=self._location, credentials=self._credentials
        )

    def _agent_engines(self) -> Any:
        return self._client().agent_engines

    def _dispatch(self) -> Any:
        """The control-plane ``DispatchTransport`` (publish + ensure) for the warm pool."""
        from ...ports.dispatch import PubSubDispatch

        return PubSubDispatch(
            topic=self._topic, subscription=self._subscription,
            project=self._project, credentials=self._credentials,
        )

    def fill_pool(self, n: int) -> None:
        """Submit ``n`` pre-warmed workers (each a job blocked on ``__POOL_WAIT__``).

        Use this to top a warm pool back up — e.g. after reusing an engine via
        ``get_engine`` whose workers have idle-expired, or to grow the pool.
        """
        ae = self._agent_engines()
        query = json.dumps({"input": {"user_id": _USER_ID, "message": POOL_WAIT_SENTINEL}})
        bucket = self._output_bucket or f"gs://{self._project}-agent-output"
        for _ in range(n):
            # run_query_job requires output_gcs_uri; a worker's job output is never read (we
            # tail Cloud Logging), so a throwaway per-worker path is fine.
            cfg = {"query": query, "output_gcs_uri": f"{bucket}/pool/{uuid.uuid4().hex}.jsonl"}
            result = ae.run_query_job(name=self._resource, config=cfg)
            # Track the worker's job so delete() can cancel it (a waiting worker is a
            # long-running op that otherwise blocks engine deletion for up to its max-wait).
            job_name = getattr(result, "job_name", None)
            if job_name:
                self._pool_jobs.append(job_name)

    def start_session(self) -> GeminiSession:
        if self._warm:
            # Warm turns run in pool workers, not a per-session engine invocation, so the
            # session id is just a client-chosen token (used for log-tail + checkpoint keying).
            # Must be a canonical UUID (dashed), NOT uuid4().hex: it reaches the Claude Agent
            # SDK as session_id / resume, which rejects a non-canonical id at runtime with
            # "Invalid session ID. Must be a valid UUID".
            session_id = str(uuid.uuid4())
        else:
            created = self._agent_engines().sessions.create(name=self._resource, user_id=_USER_ID)
            session_id = created.response.name.rsplit("/", 1)[-1]
        session = GeminiSession(self, session_id)
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> GeminiSession:
        return self._sessions.get(session_id) or GeminiSession(self, session_id)

    def list_sessions(self) -> list[dict]:
        """Enumerate this engine's known past sessions, newest first.

        Merges the durable GCS records under the output bucket (mirrored ``events/`` files,
        platform ``jobs/`` outputs) with the engine's ADK sessions. Each entry is
        ``{"session_id", "sources", "last_file"}`` — feed the id to :meth:`get_session` and
        read its :meth:`GeminiSession.history`. NB: the GCS layers are bucket-wide, so with a
        shared output bucket, sessions of other engines appear too.
        """
        from .history import list_sessions as _list

        try:
            adk_sessions = self._agent_engines().sessions
        except Exception:  # noqa: BLE001 — no GCP client available; GCS layers still answer
            adk_sessions = None
        return _list(
            output_bucket=self._output_bucket,
            resource=self._resource,
            adk_sessions=adk_sessions,
        )

    def wait_until_warm(self, timeout: float = 600.0) -> bool:
        """Block until a pool worker reports ready (or ``timeout``); ``True`` if warm.

        Tails Cloud Logging for the pool readiness marker a worker emits once it has finished
        its cold start and pre-warmed. The worker's cold start varies (~2.5–5 min: image pull
        + scheduling), so the default timeout is generous. No-op (returns ``True``) for a
        non-warm engine.

        Blocking, and safe to call from **either** sync code (ops/CI, right after ``deploy``) or
        from inside a running event loop — in the latter case the wait runs on its own thread so
        it doesn't clash with the caller's loop. From async code you can also
        ``await asyncio.to_thread(engine.wait_until_warm)`` to avoid blocking the loop.
        """
        if not self._warm:
            return True
        from .pool import pool_log_id

        pool_id = pool_log_id(self.name)
        created = self._created

        async def _await() -> bool:
            from ...ports.eventsink import CloudLoggingSink

            sink = CloudLoggingSink(
                log_name=_LOG_NAME, project=self._project, credentials=self._credentials
            )
            async for _ in sink.tail(pool_id, since=created):
                return True  # first marker since deploy => a worker is warm
            return False

        return _run_blocking(_await, timeout)

    def delete(self, *, delete_pool_resources: bool = False, timeout: float = 600.0) -> None:
        """Tear down the engine (ops action), cancelling its pool workers first.

        A warm pool's idle workers are long-running query jobs that block engine deletion
        until they stop. We cancel this process's tracked jobs up front, then on each blocked
        delete attempt parse the blocking operation names *out of the FAILED_PRECONDITION
        error* and ``cancel_query_job`` them — the reliable way to find a reused engine's
        workers, since the engine's ``/operations`` collection lists only engine LROs, not
        query jobs. A cancelled worker takes a few minutes to actually stop, so deletion
        retries up to ``timeout``. ``delete_pool_resources`` also removes the topic + sub.
        """
        import re

        ae = self._agent_engines()
        for job_name in self._pool_jobs:  # fast path: jobs this process submitted
            try:
                ae.cancel_query_job(name=self._resource, config={"operation_name": job_name})
            except Exception:  # noqa: BLE001 — already done / cancelled / unknown
                pass

        op_pat = re.compile(r"projects/[^/\s]+/locations/[^/\s]+/operations/\d+")
        deadline = time.time() + timeout
        while True:
            try:
                ae.delete(name=self._resource, force=True)
                break
            except Exception as exc:  # noqa: BLE001 — retry past blocking/finalizing operations
                if time.time() >= deadline:
                    raise
                # The platform names the blocking query-job operations in the error; cancel each.
                for op in set(op_pat.findall(str(exc))):
                    try:
                        ae.cancel_query_job(name=self._resource, config={"operation_name": op})
                    except Exception:  # noqa: BLE001 — not a query job / already cancelled
                        pass
                time.sleep(15)

        if delete_pool_resources and self._topic and self._subscription:
            from google.cloud import pubsub_v1

            try:
                pubsub_v1.SubscriberClient(credentials=self._credentials).delete_subscription(
                    subscription=self._subscription
                )
            except Exception:  # noqa: BLE001 — best effort (NotFound / perms)
                pass
            try:
                pubsub_v1.PublisherClient(credentials=self._credentials).delete_topic(topic=self._topic)
            except Exception:  # noqa: BLE001 — best effort (NotFound / perms)
                pass

    def versions(self) -> list[str]:
        return ["latest"]

    @property
    def name(self) -> str:
        return self.spec.name if self.spec is not None else self._resource.rsplit("/", 1)[-1]

    @property
    def version(self) -> str:
        return "latest"

    @property
    def resource(self) -> str:
        return self._resource

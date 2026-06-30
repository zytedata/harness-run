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

from ...events import RunResult, RunStatus, StopReason
from .._run import DrivenRun
from .pool import POOL_WAIT_SENTINEL, dispatch_payload, pool_paths

if TYPE_CHECKING:
    from ...spec import AgentSpec
    from ..base import Engine

# Must match CloudLoggingSink's default log_name (the deployed agent emits here, the client tails it).
_LOG_NAME = "remote_agent_toolkit_steps"
_USER_ID = "ratk"


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
    min_instances: int = 1,
    max_instances: int = 1,
    pool_size: int = 2,
    credentials: Any | None = None,
    **_: Any,
) -> Engine:
    """Deploy ``spec`` to Gemini Agent Runtime, minting an engine (ops/CI action).

    Bakes the toolkit package + resolved skills, applies the §6 deploy contracts, and returns
    a :class:`GeminiEngine`. With ``warm_pool=True`` it also ensures the dispatch topic/sub and
    fills the pool with ``pool_size`` pre-warmed workers (each a job blocked on ``__POOL_WAIT__``).
    """
    import dataclasses
    import os

    import vertexai
    from vertexai._genai import types as gt
    from vertexai.preview.reasoning_engines import AdkApp

    from .adk_agent import build_agent
    from .deploy import build_engine_config, stage_agent

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
    app = AdkApp(agent=build_agent(spec), enable_tracing=True)
    config_kwargs = build_engine_config(
        spec,
        project=project,
        location=location,
        staging_bucket=staging_bucket,
        extra_packages=extra_packages,
        model=model,
        output_bucket=output_bucket,
        warm_pool=warm_pool,
        pool_subscription=subscription,
        min_instances=min_instances,
        max_instances=max_instances,
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
            geng._fill_pool(pool_size)  # block pool_size workers on the dispatch sub
        except Exception:  # don't leak the freshly-created engine if filling the pool fails
            try:
                client.agent_engines.delete(name=geng.resource, force=True)
            except Exception:  # noqa: BLE001
                pass
            raise
    return geng


def _resolve_resource(client: Any, name: str, version: str | None) -> str:
    if version is not None:
        raise NotImplementedError("version pinning is P2c; omit `version` to use the latest engine.")
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

    def run(self, message: str) -> DrivenRun:
        """Start a fresh turn (submits a ``run_query_job``)."""
        return self._submit(message, resume=False)

    def send(self, message: str) -> DrivenRun:
        """Resume this session with ``message`` (prefixes the ``AGENT_RESUME`` directive)."""
        return self._submit(message, resume=True)

    def _submit(self, message: str, resume: bool) -> DrivenRun:
        from ...ports.eventsink import CloudLoggingSink

        engine = self._engine
        sid = self._session_id
        # Watermark the log tail just before kicking off (small slack for clock skew).
        since = time.time() - 5

        if engine._warm:
            # Warm path: dispatch the turn to the pool (a warm worker adopts our session_id),
            # then refill so the next turn stays warm. No cold run_query_job.
            engine._dispatch().publish(dispatch_payload(sid, message, resume))
            try:
                engine._fill_pool(1)
            except Exception:  # noqa: BLE001 — refill is best-effort; the turn already dispatched
                pass
        else:
            prompt = f"AGENT_RESUME={sid}\n{message}" if resume else message
            payload = {"input": {"session_id": sid, "user_id": _USER_ID, "message": prompt}}
            cfg: dict[str, Any] = {"query": json.dumps(payload)}
            if engine._output_bucket:
                cfg["output_gcs_uri"] = f"{engine._output_bucket}/jobs/{sid}.jsonl"
            self._last_job = engine._agent_engines().run_query_job(name=engine._resource, config=cfg)

        sink = CloudLoggingSink(
            log_name=_LOG_NAME, project=engine._project, credentials=engine._credentials
        )

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

    async def interrupt(self) -> None:
        """Stop tailing the in-flight run. (Platform-side job cancellation is P2c.)"""
        run = self._current_run
        if run is not None and run.task is not None and not run.task.done():
            run.task.cancel()
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

    def fork(self) -> GeminiSession:
        raise NotImplementedError("fork is P2c (copy workspace snapshot + transcript under a new id).")


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

    def _fill_pool(self, n: int) -> None:
        """Submit ``n`` pre-warmed workers (each a job blocked on ``__POOL_WAIT__``)."""
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
            session_id = uuid.uuid4().hex
        else:
            created = self._agent_engines().sessions.create(name=self._resource, user_id=_USER_ID)
            session_id = created.response.name.rsplit("/", 1)[-1]
        session = GeminiSession(self, session_id)
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> GeminiSession:
        return self._sessions.get(session_id) or GeminiSession(self, session_id)

    def wait_until_warm(self, timeout: float = 300.0) -> bool:
        """Block until a pool worker reports ready (or ``timeout``); ``True`` if warm.

        Tails Cloud Logging for the pool readiness marker a worker emits once it has finished
        its cold start and pre-warmed. No-op (returns ``True``) for a non-warm engine. Sync —
        call it from sync code, after ``deploy``, before dispatching turns.
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

        try:
            return asyncio.run(asyncio.wait_for(_await(), timeout))
        except (TimeoutError, asyncio.TimeoutError):
            return False

    def delete(self, *, delete_pool_resources: bool = False, timeout: float = 300.0) -> None:
        """Tear down the engine (ops action). Cancels tracked pool workers first.

        A warm pool's idle workers are long-running jobs that block engine deletion until
        they expire, so cancel them before deleting. ``delete_pool_resources`` also removes
        the dispatch topic + subscription. (Workers submitted by *another* process aren't
        tracked here; deletion still retries past their finalize, just more slowly.)
        """
        ae = self._agent_engines()
        for job_name in self._pool_jobs:
            try:
                ae.cancel_query_job(name=self._resource, config={"operation_name": job_name})
            except Exception:  # noqa: BLE001 — already done / cancelled / unknown
                pass

        deadline = time.time() + timeout
        while True:
            try:
                ae.delete(name=self._resource, force=True)
                break
            except Exception:  # noqa: BLE001 — retry past a still-finalizing operation
                if time.time() >= deadline:
                    raise
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

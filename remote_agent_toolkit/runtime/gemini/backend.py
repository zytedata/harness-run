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
from typing import Any, AsyncIterator, TYPE_CHECKING

from ...events import RunResult, RunStatus, StopReason
from .._run import DrivenRun

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
    credentials: Any | None = None,
    **_: Any,
) -> Engine:
    """Deploy ``spec`` to Gemini Agent Runtime, minting an engine (ops/CI action).

    Bakes the toolkit package + resolved skills, applies the §6 deploy contracts, and
    returns a :class:`GeminiEngine`. ``warm_pool`` provisioning of pool workers is P2b; for
    now it only flows into the engine config.
    """
    import os

    import vertexai
    from vertexai._genai import types as gt
    from vertexai.preview.reasoning_engines import AdkApp

    from .adk_agent import build_agent
    from .deploy import build_engine_config, stage_agent

    staging_bucket = staging_bucket or f"gs://{project}-agent-staging"
    output_bucket = output_bucket or f"gs://{project}-agent-output"

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
        min_instances=min_instances,
        max_instances=max_instances,
    )
    client = vertexai.Client(project=project, location=location, credentials=credentials)
    engine = client.agent_engines.create(agent=app, config=gt.AgentEngineConfig(**config_kwargs))
    return GeminiEngine(
        resource=engine.api_resource.name,
        spec=spec,
        project=project,
        location=location,
        output_bucket=output_bucket,
        credentials=credentials,
    )


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
    credentials: Any | None = None,
) -> Engine:
    """Look up a deployed engine by ``name`` (the app-code hot path; never deploys).

    Pass ``spec=`` (the one you deployed) to enable structured-output parsing and the
    correct idle stop-reason; otherwise a minimal fallback spec is used.
    """
    import vertexai

    client = vertexai.Client(project=project, location=location, credentials=credentials)
    resource = _resolve_resource(client, name, version)
    return GeminiEngine(
        resource=resource,
        spec=spec or _fallback_spec(name),
        project=project,
        location=location,
        output_bucket=output_bucket or (f"gs://{project}-agent-output" if project else None),
        credentials=credentials,
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
        prompt = f"AGENT_RESUME={self._session_id}\n{message}" if resume else message
        payload = {"input": {"session_id": self._session_id, "user_id": _USER_ID, "message": prompt}}
        cfg: dict[str, Any] = {"query": json.dumps(payload)}
        if engine._output_bucket:
            cfg["output_gcs_uri"] = f"{engine._output_bucket}/jobs/{self._session_id}.jsonl"
        # Watermark the log tail just before submitting (small slack for clock skew).
        since = time.time() - 5
        self._last_job = engine._agent_engines().run_query_job(name=engine._resource, config=cfg)

        sink = CloudLoggingSink(
            log_name=_LOG_NAME, project=engine._project, credentials=engine._credentials
        )
        sid = self._session_id

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
    ) -> None:
        self._resource = resource
        self.spec = spec
        self._project = project
        self._location = location
        self._output_bucket = output_bucket
        self._credentials = credentials
        self._sessions: dict[str, GeminiSession] = {}

    def _client(self) -> Any:
        import vertexai

        return vertexai.Client(
            project=self._project, location=self._location, credentials=self._credentials
        )

    def _agent_engines(self) -> Any:
        return self._client().agent_engines

    def start_session(self) -> GeminiSession:
        created = self._agent_engines().sessions.create(name=self._resource, user_id=_USER_ID)
        session_id = created.response.name.rsplit("/", 1)[-1]
        session = GeminiSession(self, session_id)
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> GeminiSession:
        return self._sessions.get(session_id) or GeminiSession(self, session_id)

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

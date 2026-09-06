"""Gemini Agent Runtime control plane: deploy / get_engine / list_engines + Engine/Session/Run.

Encodes the §6 platform contracts. ``deploy`` packages the agent (``_deploy.py``), wraps it
in an ``AdkApp``, and creates a reasoningEngine; app code addresses engines by name via
``get_engine`` and never deploys (DESIGN.md §3.3).

The run plane mirrors ``local`` exactly via the shared :class:`DrivenRun`: a ``Session.run``
submits a ``run_query_job`` and the ``Run`` is driven by tailing the session's GCS event
mirror (``stream.tail_stream``) — so **stream** = tail, **await** = wait for the terminal
``result`` event, **poll** = read the latest status. Same awaitable/iterable/pollable handle
as local. Cloud Logging is written by every worker but never read here (ops/debug only).

The Google SDK (``agentplatform`` / ``google-cloud-aiplatform``) and the sink are imported
lazily inside the bodies, so importing this module needs no third-party deps. ``agentplatform``
is the renamed successor of the ``vertexai`` namespace (aiplatform 1.154+); the old one still
works but warns on every ``Client`` instantiation, and its config types are not interchangeable
with the new client's.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from typing import Any, AsyncIterator, TYPE_CHECKING

from ...events import AgentEvent, RunResult, RunStatus, StopReason
from .._run import DrivenRun
from .pool import (
    DEFAULT_MAX_WAIT_S,
    POOL_WAIT_SENTINEL,
    WORKER_SUBSCRIPTION_RETENTION_S,
    WORKER_SUBSCRIPTION_TTL_S,
    dispatch_payload,
    new_generation,
    new_worker_id,
    pickup_deadline_s,
    pool_paths,
    pool_topic,
    pool_wait_prompt,
    topic_for_subscription,
    worker_attributes,
    worker_filter,
    worker_subscription,
    worker_subscription_prefix,
)
from .stream import tail_stream

if TYPE_CHECKING:
    from ...config import SessionConfig, TurnConfig
    from ...spec import AgentSpec
    from ..base import Engine
    from .roster import WorkerEntry

_USER_ID = "ratk"

# Cold-run watchdog: probe the job's state after this much stream silence, and end the run
# this long after the job is seen terminal with still no terminal event (ingestion-lag grace).
_WATCHDOG_QUIET_S = 60.0
_WATCHDOG_GRACE_S = 120.0

# Warm pickup watchdog: how many OTHER workers a turn is re-dispatched to when the one it
# was addressed to never starts it (pool.pickup_deadline_s per attempt), before giving up.
_MAX_REDISPATCH = 2

logger = logging.getLogger(__name__)


def _event_fingerprint(event: AgentEvent) -> tuple:
    """A stable content key for reconciling Cloud Logging with durable history.

    Cloud Logging can ingest entries out of order, so the events already yielded by the
    live tail are not necessarily a prefix of the GCS mirror. Both channels carry the
    same logical :class:`AgentEvent`; canonical JSON makes its nested payload hashable
    while preserving duplicate counts during recovery.
    """
    return (
        event.kind,
        event.summary,
        json.dumps(event.raw, sort_keys=True, default=str),
        event.cost_usd,
        json.dumps(event.usage, sort_keys=True, default=str),
    )


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
    yielded: list[AgentEvent] = []
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
                # Job over, grace elapsed, tail still silent. Try the durable record. Cloud
                # Logging can ingest out of order, so its delivered events are NOT necessarily
                # a prefix of the mirror: subtract them by content (as a multiset) instead of
                # slicing by count, then deliver the genuinely missing events in mirror order.
                history: list[AgentEvent] = []
                if history_reader is not None:
                    try:
                        history = await history_reader() or []
                    except Exception:  # noqa: BLE001 — recovery is best-effort
                        history = []
                if any(ev.kind == "result" for ev in history):
                    seen: dict[tuple, int] = {}
                    for ev in yielded:
                        key = _event_fingerprint(ev)
                        seen[key] = seen.get(key, 0) + 1
                    for ev in history:
                        key = _event_fingerprint(ev)
                        if seen.get(key, 0):
                            seen[key] -= 1
                            continue
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
            yielded.append(item)
            yield item
            if item.kind == "result":
                return
    finally:
        pump_task.cancel()


async def _pickup_watched_tail(
    tail_factory: Any,
    deadline_s: float,
    redispatch: Any,
    session_id: str,
    *,
    on_pickup: Any = None,
    on_abandon: Any = None,
    max_redispatch: int = _MAX_REDISPATCH,
) -> AsyncIterator[AgentEvent]:
    """Yield the tail's events, re-dispatching the turn if its worker never starts it.

    Per-worker dispatch addresses a turn to ONE pool worker. A worker that died at boot,
    was killed by the platform, or idled out a moment before the roster said so would then
    hold the turn forever: nothing else pulls its channel. So until the first event of the
    turn arrives (the worker's ``turn_started``), this wrapper watches the clock:
    ``deadline_s`` (``pool.pickup_deadline_s`` — the rest of the worker's boot window plus
    a grace) elapsing calls ``redispatch(attempt)`` on a thread, which is expected to drop
    the dead worker, address the same payload to another one and return the new deadline.
    After ``max_redispatch`` silent workers the run ends with an explained error result.
    From the first event on the wrapper is pure passthrough (``on_pickup`` runs once, on a
    thread, best-effort — the client uses it to drop the worker's now-useless channel).
    ``on_abandon`` runs when giving up, to drop the last worker tried.
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
    attempts = 0
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=deadline_s)
            except (TimeoutError, asyncio.TimeoutError):
                if attempts >= max_redispatch:
                    if on_abandon is not None:
                        try:
                            await asyncio.to_thread(on_abandon)
                        except Exception:  # noqa: BLE001 — cleanup is best-effort
                            pass
                    yield AgentEvent(
                        kind="result",
                        summary=(
                            f"no pool worker started the turn: {attempts + 1} worker(s) were "
                            "addressed and none picked it up within its boot window — the "
                            "pool may be unable to boot workers (quota, image, identity); "
                            "see Cloud Logging for the workers' own errors"
                        ),
                        raw={"event": "pool_pickup_timeout", "is_error": True,
                             "subtype": "error", "session_id": session_id,
                             "workers_tried": attempts + 1},
                    )
                    return
                attempts += 1
                try:
                    deadline_s = await asyncio.to_thread(redispatch, attempts)
                except Exception as exc:  # noqa: BLE001 — surface, never hang
                    yield AgentEvent(
                        kind="result",
                        summary=(
                            "the pool worker addressed never started the turn and "
                            f"re-dispatching it failed: {type(exc).__name__}: {str(exc)[:300]}"
                        ),
                        raw={"event": "pool_redispatch_failed", "is_error": True,
                             "subtype": "error", "session_id": session_id},
                    )
                    return
                continue
            if item is done:
                return
            if on_pickup is not None:
                try:
                    await asyncio.to_thread(on_pickup)
                except Exception:  # noqa: BLE001 — housekeeping must not kill a healthy run
                    pass
            yield item
            if item.kind == "result":
                return
            break
        while True:
            item = await queue.get()
            if item is done:
                return
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
    """A minimal spec for a looked-up engine (``get_engine`` never knows the deployed one).

    Client-side result building needs ``output_schema`` (structured parsing) and
    ``checkpoint`` (the idle stop-reason). Runs execute the engine's deploy-baked spec —
    overlaid with any ``SessionConfig``/``TurnConfig`` the caller binds, which is also how
    the client recovers structured parsing and the right stop-reason: set
    ``output_schema``/``checkpoint`` on the config and both sides see them.
    """
    from ...spec import AgentSpec

    return AgentSpec(name=name, model="")


# -- control plane -------------------------------------------------------------


def build_adk_app(
    spec: AgentSpec, *, project: str, location: str, credentials: Any | None = None
) -> Any:
    """Construct the AdkApp to pickle, with the deploy target pinned into it.

    ``AdkApp.__init__`` snapshots the aiplatform global config's project/location into
    its (pickled) template attrs; the engine worker later force-feeds that project into
    ``GOOGLE_CLOUD_PROJECT`` and routes all OTel span/metric export to it. Left to
    resolve from the deployer's environment (gcloud/ADC default), an unrelated project
    gets baked into the engine and every telemetry export 403s from then on — that
    exact incident (a pickled ``other-project``) burned 2026-08-03→06. So: initialize the
    global config from the deploy args, and refuse to ship a pickle that captured
    anything else.
    """
    import google.cloud.aiplatform as aiplatform
    from agentplatform.agent_engines import AdkApp

    # session_service_builder pins the app to in-memory ADK sessions: the toolkit keys
    # everything by its own session id (AGENT_SESSION directive + GCS mirror), and the
    # 2026-07-28 platform runner's managed-session wiring is broken on new engines.
    from .adk_agent import build_agent, in_memory_session_service

    aiplatform.init(project=project, location=location, credentials=credentials)
    app = AdkApp(
        agent=build_agent(spec),
        enable_tracing=True,
        session_service_builder=in_memory_session_service,
    )
    baked = (app._tmpl_attrs.get("project"), app._tmpl_attrs.get("location"))
    if baked != (project, location):
        raise RuntimeError(
            f"AdkApp captured project/location {baked!r} instead of "
            f"{(project, location)!r}; the engine would export its telemetry to the "
            "wrong project (persistent 403s on every span/metric batch). Refusing to "
            "deploy."
        )
    return app


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
    service_account: str | None = None,
    pool_size: int = 2,
    pool_max_wait_s: float | None = None,
    new_engine: bool = False,
    credentials: Any | None = None,
    workspace: str | None = None,
    **_: Any,
) -> Engine:
    """Deploy ``spec`` to Gemini Agent Runtime, minting a **new revision** (ops/CI action).

    Bakes the toolkit package + resolved skills, applies the §6 deploy contracts, and returns
    a :class:`GeminiEngine`. With ``warm_pool=True`` it also ensures the dispatch topic and
    fills the pool with ``pool_size`` pre-warmed workers (each a job blocked on ``__POOL_WAIT__``
    on its own subscription — per-worker dispatch, ``pool.py``).

    **Engine identity is ``spec.name``** (the engine display name): if an engine of that name
    already exists, this *updates* it, and the platform mints a new **runtime revision** of it
    (``.../reasoningEngines/{id}/runtimeRevisions/{rev}``) — the deployed engine keeps its
    resource name, so app code's ``get_engine(spec.name)`` picks the new code up without
    re-pointing at anything. Only the first deploy of a name creates an engine. Pass
    ``new_engine=True`` to force a separate engine instead (e.g. a side-by-side experiment);
    two engines then share the display name and ``get_engine`` resolves the most recent.

    The new revision serves traffic immediately **unless** the engine's traffic was pinned
    with :meth:`GeminiEngine.set_traffic` — a pin is sticky across deploys, so this warns and
    leaves the pin in place rather than silently promoting the fresh revision.

    Updating a warm pool cuts over atomically (issue #38): every deploy mints a fresh,
    generation-scoped dispatch topic (each worker's own subscription hangs off it), so
    workers still blocked on the previous generation's channels can never claim a turn
    dispatched after this deploy — and once the new pool is filled, the previous
    generation is retired (its worker subscriptions, topic and roster deleted), making
    those idle workers fail their next claim poll and exit within seconds instead of
    billing (and silently serving stale-spec turns) for up to ``pool_max_wait_s``. A
    worker already mid-turn finishes that turn on its own revision; a dispatch still
    queued undelivered on an old channel at that moment (a turn racing the redeploy) is
    dropped with it. Exception: when the engine's traffic is pinned to a previous
    revision, the pool its workers belong to is kept and dispatched to, and nothing is
    retired (see :meth:`GeminiEngine.set_traffic`); the fresh topic stays too (the new
    revision's baked env names it, so a later promotion needs it) — pools of revisions
    never promoted are swept by ``delete(delete_pool_resources=True)``, which retires
    every revision's pool.

    **Warm workers idle-expire.** An idle worker waits ``pool_max_wait_s`` (default: a
    day) for an assignment, then exits — silently, and WITHOUT replacement: the only
    automatic refill is the one worker submitted after each dispatch. A pool that drained
    to empty does not strand turns: a turn that finds no idle worker spawns one for itself
    and waits for its ~2.5 min boot (cold latency), and the refill after it starts warming
    the pool again; :meth:`GeminiEngine.fill_pool` re-warms it ahead of time. Idle workers
    bill while they wait, so ``pool_max_wait_s`` is the idle-cost/latency dial: lower it
    for engines that are dispatched to constantly (expiry never fires), keep or raise it
    for pools that must stay warm across quiet gaps. Whatever you pass, the platform's
    **max job duration** (7 days at the time of writing — a platform contract that can
    move; DESIGN.md §6) is the effective ceiling: a worker that outlives it is killed like
    any job and, as above, not replaced.

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

    ``service_account`` sets the engine's runtime identity (a service account email you
    created; the deployer needs ``roles/iam.serviceAccountUser`` on it). Omitted → the
    platform default, the Agent Runtime service agent shared by every engine in the
    project. That default holds a Google-managed project role that reads every bucket in
    the project, and the agent's shell can use its token, so a custom service account with
    only the bindings the worker needs is the recommended setup (README "The runtime
    identity is reachable by the agent" lists the roles).
    """
    # Fail fast BEFORE any side effect (pub/sub ensure, staging, the ~4 min billable build).
    if workspace is not None:
        raise ValueError(
            "workspace= is local-only: a deployed engine's filesystem is the worker's own "
            "/tmp, one job at a time, so there is no host directory to point turns at and "
            "nothing for sessions to share. Seed the agent's cwd through the prompt or "
            "spec.repos instead."
        )
    if pool_max_wait_s is not None and not warm_pool:
        # Loud on purpose: a silently-ignored idle-life knob is exactly the operational
        # surprise this parameter exists to remove.
        raise ValueError("pool_max_wait_s only applies to warm-pool engines; pass warm_pool=True")
    if pool_max_wait_s is not None:
        import math

        # Value check HERE, not only in build_env(): build_env runs after the pub/sub
        # ensure, so a bad value rejected there would leave an orphaned topic/sub behind.
        if not (math.isfinite(pool_max_wait_s) and pool_max_wait_s > 0):
            raise ValueError(
                f"pool_max_wait_s must be a positive, finite number of seconds; "
                f"got {pool_max_wait_s!r}"
            )

    import dataclasses
    import os

    import agentplatform
    from agentplatform import types as gt

    from ._deploy import (
        build_engine_config,
        stage_agent,
        validate_resource_limits,
        verify_deploy_env,
    )

    if resource_limits is not None:
        validate_resource_limits(resource_limits)
    verify_deploy_env()  # pickle-coupled venv pins must match constraints.txt
    # A model override applies to the harness too: the deployed agent reads spec.model, so
    # bake the override into the spec (not just the env) before serializing it.
    if model:
        spec = dataclasses.replace(spec, model=model)
    staging_bucket = staging_bucket or f"gs://{project}-agent-staging"
    output_bucket = output_bucket or f"gs://{project}-agent-output"

    # Backstop reapers for handoff objects: staged secrets orphaned by a crash on both
    # sides (worker delete at turn end + client delete on completion are the main lines)
    # and the persisted session/turn configs (kept for debugging, aged out after 30 days).
    # Best-effort: the rules are backstops and the operator may lack
    # storage.buckets.update — warn, never fail the deploy.
    from .handoff import ensure_handoff_lifecycle

    if not ensure_handoff_lifecycle(output_bucket, credentials=credentials):
        import warnings

        warnings.warn(
            f"could not ensure the handoff lifecycle rules on {output_bucket}; staged "
            "secrets orphaned by crashed jobs and old config objects will not be "
            "auto-reaped (needs storage.buckets.update on the bucket)",
            stacklevel=2,
        )

    # Warm pool: this deploy's workers belong to a FRESH, generation-scoped dispatch topic
    # (baked into the engine env); each worker gets its own subscription on it at
    # fill_pool (per-worker dispatch, pool.py). Scoping the generation per deploy is the
    # #38 fix: workers of the previous revision keep pulling the old generation's channels,
    # so they can never claim (and run with a stale baked spec/skills) a turn dispatched
    # after this deploy.
    topic = None
    if warm_pool:
        from ...ports.dispatch import PubSubDispatch

        topic = pool_topic(project, spec.name, generation=new_generation())
        # Ensure the topic FIRST — fail fast on missing pub/sub perms BEFORE the (~4 min,
        # billable) engine build, so a perms error never leaks a half-provisioned engine.
        PubSubDispatch(topic=topic, project=project, credentials=credentials).ensure_topic()

    stage_dir, extra_packages = stage_agent(spec)
    os.chdir(stage_dir)  # extra_packages are resolved relative to the cwd
    app = build_adk_app(spec, project=project, location=location, credentials=credentials)
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
        pool_topic=topic,
        pool_max_wait_s=pool_max_wait_s,
        min_instances=min_instances,
        max_instances=max_instances,
        resource_limits=resource_limits,
        service_account=service_account,
    )
    client = agentplatform.Client(project=project, location=location, credentials=credentials)
    # Engine identity is the display name: update the existing engine (minting a revision)
    # rather than piling up look-alike engines. Only a first deploy creates one.
    existing = None if new_engine else _find_engine(client, spec.name)
    # The pool live workers currently belong to, read back from the deployed env of the
    # SERVING revision — under a pin that is the pinned revision's pool, NOT the
    # engine-level env, which reflects the latest revision: one that never served has no
    # workers, so dispatching there would strand turns (PR #39 review). Retired once the
    # new pool is filled, so its idle workers exit promptly instead of idling out.
    old: tuple[str, str | None] | None = None  # (topic, legacy shared subscription)
    if warm_pool and existing:
        try:
            old = _address_from_env(_serving_pool_env(client, existing))
        except Exception:  # noqa: BLE001 — cutover cleanup is best-effort, never blocks a deploy
            old = None
    config = gt.AgentEngineConfig(**config_kwargs)
    pinned = False
    if existing:
        engine = client.agent_engines.update(name=existing, agent=app, config=config)
        _warn_if_traffic_pinned(engine.api_resource)
        from . import revisions as rev

        pinned = bool(rev.traffic_targets(getattr(engine.api_resource, "traffic_config", None)))
    else:
        engine = client.agent_engines.create(agent=app, config=config)
    subscription = None  # only set to address a LEGACY shared-subscription pool
    if warm_pool and pinned and old and old[0] != topic:
        # Traffic stays pinned to a previous revision, so live workers — and any refill,
        # which always cold-starts on the SERVING revision — belong to the OLD pool.
        # Dispatch there and retire nothing. The fresh topic is deliberately KEPT even
        # though nothing pulls it yet: its name is baked into the new revision's env, so a
        # later promotion (set_traffic) makes it live — deleting it here would permanently
        # break that revision. Pools of revisions never promoted thus accumulate under a
        # pin; delete(delete_pool_resources=True) sweeps every revision's pool.
        topic, subscription = old
        old = None
    elif warm_pool and pinned and not old:
        import warnings

        # The pin means live workers belong to SOME previous pool, but we could not learn
        # which (env read failed, or an engine deployed before the env carried it). The
        # handle will dispatch to the fresh pool, which those workers do not pull — loud,
        # so a turn sitting unclaimed afterwards is traceable to this deploy.
        warnings.warn(
            "traffic is pinned but the serving revision's dispatch topic could not be "
            "read; dispatching to this deploy's fresh pool, which the pinned revision's "
            "workers do NOT pull — turns may sit unclaimed until traffic moves "
            "(set_traffic) or the pool is re-addressed via get_engine",
            stacklevel=2,
        )
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
        pool_max_wait_s=pool_max_wait_s,
    )
    if warm_pool:
        try:
            geng.fill_pool(pool_size)  # pool_size workers, each on its own subscription
        except Exception:
            # Don't leak the freshly-created engine if filling the pool fails — but only
            # ours: on the update path the engine predates this call (with its revision
            # history and, possibly, live sessions), so a failed refill must not delete it.
            if existing is None:
                try:
                    client.agent_engines.delete(name=geng.resource, force=True)
                except Exception:  # noqa: BLE001
                    pass
            raise
        # Retire the previous generation: deleting its worker subscriptions makes every
        # idle worker still blocked on one fail the next claim poll and exit within
        # seconds — instead of claiming post-redeploy turns and serving them with the old
        # baked spec/skills for up to pool_max_wait_s (#38).
        if old and old[0] != topic:
            _retire_pool(old[0], old[1], output_bucket, credentials)
    return geng


def _find_engine(client: Any, display_name: str) -> str | None:
    """The resource name of the engine displayed as ``display_name``, or ``None``.

    With ``deploy(new_engine=True)`` a name can map to several engines; the most recent wins
    (both here and in :func:`_resolve_resource`), so a side-by-side experiment doesn't strand
    ``get_engine`` on the older one.
    """
    matches = [
        e for e in client.agent_engines.list()
        if getattr(e.api_resource, "display_name", None) == display_name
    ]
    return matches[-1].api_resource.name if matches else None


def _resolve_resource(client: Any, name: str) -> str:
    if "/reasoningEngines/" in name or "/agentEngines/" in name:
        return name  # already a full resource path
    resource = _find_engine(client, name)
    if resource is None:
        raise LookupError(f"no deployed engine named {name!r} in this project/location")
    return resource


def _warn_if_traffic_pinned(api_resource: Any) -> None:
    """Warn when a freshly minted revision won't serve because traffic is pinned elsewhere."""
    from . import revisions as rev

    targets = rev.traffic_targets(getattr(api_resource, "traffic_config", None))
    if targets:
        import warnings

        pinned = ", ".join(f"{v} ({p}%)" for v, p in targets)
        warnings.warn(
            f"this engine's traffic is pinned to {pinned}, so the revision just deployed is "
            "NOT serving; call engine.set_traffic() to promote it (or set_traffic(version) to "
            "keep routing deliberately)",
            stacklevel=3,
        )


_POOL_ENV_VARS = ("AGENT_POOL_TOPIC", "AGENT_POOL_SUBSCRIPTION", "AGENT_POOL_MAX_WAIT_S")


def _pool_env(api_resource: Any) -> dict[str, str]:
    """The pool-related env vars of a deployed engine/revision resource.

    The env baked at deploy is the ground truth for which pool that code's workers belong
    to — it is exactly what a worker cold-started from that revision reads at runtime.
    """
    deployment = getattr(getattr(api_resource, "spec", None), "deployment_spec", None)
    env: dict[str, str] = {}
    for var in getattr(deployment, "env", None) or []:
        name = getattr(var, "name", None)
        value = getattr(var, "value", None)
        if name in _POOL_ENV_VARS and value:
            env[name] = value
    return env


def _address_from_env(env: dict[str, str]) -> tuple[str, str | None] | None:
    """``(topic, legacy_shared_subscription)`` the pool env addresses; ``None`` for no pool.

    A per-worker-dispatch engine bakes ``AGENT_POOL_TOPIC`` (worker subscriptions are per
    worker, never in the env). An engine deployed before that baked the one shared
    ``AGENT_POOL_SUBSCRIPTION`` all its workers pull, from which its topic derives; the
    handle then dispatches the legacy way.
    """
    if env.get("AGENT_POOL_TOPIC"):
        return env["AGENT_POOL_TOPIC"], None
    if env.get("AGENT_POOL_SUBSCRIPTION"):
        sub = env["AGENT_POOL_SUBSCRIPTION"]
        return topic_for_subscription(sub), sub
    return None


def _serving_pool_env(client: Any, resource: str) -> dict[str, str]:
    """The pool env of the revision pool workers actually run on.

    Workers always cold-start on the SERVING revision, so with traffic pinned the pool
    they belong to is the pinned revision's. The engine-level env reflects the *latest*
    revision instead — under a pin possibly one that never served, whose pool has no
    workers — so it is only the fallback (and the answer when traffic is unpinned or
    split). An env with no pool address means the code was deployed without a pool (or
    before the env carried one); a failed read raises, and each caller decides how to
    degrade.
    """
    from . import revisions as rev

    api = client.agent_engines.get(name=resource).api_resource
    targets = rev.traffic_targets(getattr(api, "traffic_config", None))
    if len(targets) == 1:
        pinned = rev.revision_resource(resource, targets[0][0])
        for revision in client.agent_engines.runtimes.revisions.list(name=resource):
            if getattr(revision.api_resource, "name", "") == pinned:
                env = _pool_env(revision.api_resource)
                if _address_from_env(env):
                    return env
                break
    return _pool_env(api)


def _discover_pool(
    client: Any, resource: str, project: str | None, name: str
) -> tuple[str, str | None, float | None]:
    """``(topic, legacy_subscription, pool_max_wait_s)`` of the pool live workers pull.

    The pool is generation-scoped (a fresh topic per deploy — the #38 cutover), so it
    cannot be derived from the engine name: read it back from the deployed env instead
    (:func:`_serving_pool_env`). ``legacy_subscription`` is set only for an engine deployed
    before per-worker dispatch (its workers share that one subscription); ``None`` means
    per-worker dispatch. The legacy fixed names are used only when the env was READ
    successfully and simply has no pool address (an engine deployed without ``warm_pool``,
    or from a toolkit older than the baked env var). A *failed* read raises so the caller
    can retry: silently falling back on, say, a network timeout handed out legacy names
    that were never created for a generation-scoped engine — a handle that looks fine and
    only fails at its first publish, with a ``NotFound`` nothing ties back to the read
    error (PR #39 review).
    """
    env = _serving_pool_env(client, resource)
    max_wait = float(env["AGENT_POOL_MAX_WAIT_S"]) if env.get("AGENT_POOL_MAX_WAIT_S") else None
    address = _address_from_env(env)
    if address:
        return address[0], address[1], max_wait
    if not project:
        raise ValueError(
            "warm_pool=True could not discover the engine's dispatch topic from its "
            "deployed env, and without project= the legacy names cannot be derived either; "
            "pass project= (or redeploy the engine with warm_pool=True)."
        )
    topic, sub = pool_paths(project, name)
    return topic, sub, max_wait


def _delete_pool_pair(topic: str | None, subscription: str | None, credentials: Any) -> None:
    """Best-effort delete of a LEGACY dispatch topic/shared-subscription pair."""
    from google.cloud import pubsub_v1

    if subscription:
        try:
            pubsub_v1.SubscriberClient(credentials=credentials).delete_subscription(
                subscription=subscription
            )
        except Exception:  # noqa: BLE001 — best effort (NotFound / perms)
            pass
    if topic:
        try:
            pubsub_v1.PublisherClient(credentials=credentials).delete_topic(topic=topic)
        except Exception:  # noqa: BLE001 — best effort (NotFound / perms)
            pass


def _retire_pool(
    topic: str, legacy_subscription: str | None, output_bucket: str | None, credentials: Any
) -> None:
    """Best-effort teardown of one pool generation's dispatch resources (cutover + teardown).

    A legacy pool is its topic plus the shared subscription. A per-worker pool is its
    topic, every worker subscription under it — listed by name prefix, plus whatever the
    roster still records in case listing is denied — and the roster itself. Deleting a
    worker's subscription is what makes an idle worker exit: its next pull fails NotFound.
    """
    if legacy_subscription:
        _delete_pool_pair(topic, legacy_subscription, credentials)
        return
    from ...ports.dispatch import PubSubDispatch
    from .roster import PoolRoster

    dispatch = PubSubDispatch(topic=topic, project=topic.split("/")[1], credentials=credentials)
    subscriptions: set[str] = set()
    if output_bucket:
        try:
            roster = PoolRoster(output_bucket, topic, credentials=credentials)
            subscriptions.update(entry.subscription for entry in roster.clear())
        except Exception:  # noqa: BLE001 — best effort
            pass
    try:
        subscriptions.update(dispatch.list_subscriptions(worker_subscription_prefix(topic)))
    except Exception:  # noqa: BLE001 — best effort (perms)
        pass
    for sub in sorted(subscriptions):
        try:
            dispatch.delete_subscription(sub)
        except Exception:  # noqa: BLE001 — best effort
            pass
    try:
        dispatch.delete_topic()
    except Exception:  # noqa: BLE001 — best effort
        pass


def _resolve_version(client: Any, resource: str, version: str) -> str:
    """Validate ``version`` against ``resource``'s revisions and return its id.

    Pinning is an assertion, not routing: the toolkit's run plane is ``asyncQuery``, which
    exists only on the engine (a revision has ``query``/``streamQuery`` but no ``asyncQuery``),
    so a run always lands on whichever revision serves. Pinning to a revision that is *not*
    serving would silently run something else — so that raises instead, pointing at
    :meth:`GeminiEngine.set_traffic`, the action that actually moves traffic.
    """
    from . import revisions as rev

    version = rev.revision_id(str(version))
    rows = rev.list_revisions(client, resource)
    known = [r["version"] for r in rows]
    if version not in known:
        raise LookupError(
            f"engine {resource} has no runtime revision {version!r}; "
            f"available (newest first): {known or '(none)'}"
        )
    engine = client.agent_engines.get(name=resource)
    serving = rev.serving_version(getattr(engine.api_resource, "traffic_config", None), rows)
    if serving != version:
        current = repr(serving) if serving else "a split across several revisions"
        raise ValueError(
            f"revision {version!r} is not the one serving traffic ({current}). Agent Runtime "
            "routes async queries per ENGINE, not per revision, so a pinned handle cannot run "
            "against a non-serving revision — move traffic first with "
            f"engine.set_traffic({version!r})."
        )
    return version


def get_engine(
    name: str,
    project: str | None = None,
    location: str | None = None,
    version: str | int | None = None,
    *,
    output_bucket: str | None = None,
    warm_pool: bool = False,
    credentials: Any | None = None,
    scoped_gcs: bool = True,
    **kwargs: Any,
) -> Engine:
    """Look up a deployed engine by ``name`` (the app-code hot path; never deploys).

    ``scoped_gcs`` (default on) mints a run-scoped GCS token per turn so the worker never
    touches the bucket with the runtime identity (``scoped_gcs.py``). Turn it off only to
    drive an engine deployed from a pre-token toolkit revision during a migration.

    Pure ADDRESSING: the handle identifies which engine turns run against; it carries no
    execution configuration. Runs execute the engine's deploy-baked spec, overlaid with
    the session's ``SessionConfig`` (``engine.start_session(config=...)``) and each turn's
    ``TurnConfig`` (``run(config=...)``) — see the README "Deploy / session / turn".
    Pass ``warm_pool=True`` to address a warm-pool engine (turns are dispatched to its
    pool instead of cold-started).

    ``version`` pins the handle to a **runtime revision** (the id, or a full revision resource
    name). It is an *assertion*: the lookup fails unless that revision exists and is the one
    serving traffic — so a deploy-then-pin CI flow catches a moved/rolled-back engine at lookup
    instead of running unknown code. It cannot route a run to a non-serving revision: Agent
    Runtime's async query path is engine-level (see ``revisions.py``); use
    :meth:`GeminiEngine.set_traffic` to move traffic first.
    """
    if "spec" in kwargs:
        raise TypeError(
            "get_engine() no longer accepts spec=: it used to be a client-side parsing "
            "hint only (runs always executed the deploy-baked spec). Run-time "
            "configuration now has its own types — bind a SessionConfig at "
            "engine.start_session(config=...) and per-turn overrides via "
            "run(config=TurnConfig(...)); both are honored by the worker. See "
            "https://github.com/zytedata/remote-agent-toolkit/pull/16"
        )
    if kwargs:
        raise TypeError(f"get_engine() got unexpected keyword argument(s) {sorted(kwargs)}")
    import agentplatform

    client = agentplatform.Client(project=project, location=location, credentials=credentials)
    resource = _resolve_resource(client, name)
    pinned = None if version is None else _resolve_version(client, resource, str(version))
    topic = subscription = None
    pool_max_wait_s = None
    if warm_pool:
        topic, subscription, pool_max_wait_s = _discover_pool(client, resource, project, name)
        if subscription:
            import warnings

            warnings.warn(
                f"engine {name!r} was deployed before per-worker dispatch: its warm pool "
                "shares one subscription across workers, so a shell in one worker can "
                "pull another run's turn (README, 'The runtime identity is reachable by "
                "the agent'). Redeploy it from this toolkit revision.",
                stacklevel=2,
            )
    return GeminiEngine(
        resource=resource,
        spec=_fallback_spec(name),
        project=project,
        location=location,
        output_bucket=output_bucket or (f"gs://{project}-agent-output" if project else None),
        credentials=credentials,
        warm=warm_pool,
        topic=topic,
        subscription=subscription,
        version=pinned,
        spec_known=False,
        scoped_gcs=scoped_gcs,
        pool_max_wait_s=pool_max_wait_s,
    )


def list_engines(project: str, location: str, *, credentials: Any | None = None) -> list[dict]:
    """Discover deployed engines in ``project``/``location`` (control plane)."""
    import agentplatform

    client = agentplatform.Client(project=project, location=location, credentials=credentials)
    out: list[dict] = []
    for engine in client.agent_engines.list():
        resource = engine.api_resource
        out.append({"name": getattr(resource, "display_name", None), "resource": resource.name})
    return out


# -- run plane -----------------------------------------------------------------


class GeminiSession:
    """A run-plane session over Agent Engine (CMA-style lifecycle; DESIGN.md §4).

    A session's :class:`~remote_agent_toolkit.config.SessionConfig` is bound ONCE, at
    ``engine.start_session(config=...)`` — it is persisted at a stable per-session GCS
    key, every turn's payload points at it, and a process re-attaching by id
    (``engine.get_session``) reads that same object back. A re-attacher can never pass a
    different config: the session's world (repos, skills, prompt) is created on its first
    turn and snapshot-restored on every later one, so substituting a config
    mid-conversation could not be honored — the API refuses to express it.
    """

    def __init__(
        self,
        engine: GeminiEngine,
        session_id: str,
        config: SessionConfig | None = None,
        config_resolved: bool = True,
    ) -> None:
        self._engine = engine
        self._session_id = session_id
        self._status = RunStatus.PENDING
        self._stop_reason: StopReason | None = None
        self._last_result: RunResult | None = None
        self._gcs_token_stop: threading.Event | None = None  # stops the token refresher
        self._submission_lock = threading.Lock()
        self._submission_may_have_dispatched = False
        self._submission_uncertain = False
        self._current_run: DrivenRun | None = None
        self._last_job: Any | None = None
        self._worker: WorkerEntry | None = None  # the pool worker the running turn is addressed to
        self._worker_retired = False  # its channel already dropped (at pickup)
        self._staged_secrets_uri: str | None = None  # staged handoff object (cleanup on complete)
        self._session_config = config
        self._session_config_uri: str | None = None
        # False for a re-attached session: the opener's persisted config (if any) is
        # loaded from GCS on first use — see _resolve_session_config.
        self._session_config_resolved = config_resolved
        # Last mirror object consumed by this session's stream tail; the NEXT turn's tail
        # starts strictly after it (the clock-free turn boundary — see stream.tail_stream).
        self._stream_watermark: dict = {"key": ""}

    def run(
        self,
        message: str,
        *,
        secrets: dict[str, str] | None = None,
        config: TurnConfig | None = None,
        hooks: Any | None = None,
    ) -> DrivenRun:
        """Start a fresh turn (submits a ``run_query_job``).

        ``secrets`` is a per-invocation name → value map (the agent's own keys, any repo
        ``auth`` / GitHub MCP token) — never baked into the engine, never logged. Values are
        staged at a per-invocation GCS object the worker fetches (and deletes once the turn
        completes — a platform retry of a killed attempt must still find it); only that pointer
        rides the invocation (the platform persists a job's input verbatim, and a Pub/Sub
        message is retained until acked, so values must never travel in either).

        ``config`` is this turn's :class:`~remote_agent_toolkit.config.TurnConfig` — a
        sparse overlay of the invocation knobs (model, budgets, tool policy, output
        schema) on top of the session's effective spec, for this turn only.

        *hooks* are rejected here: the turn runs in a remote worker, and a hook is a live
        callable in this process (see :meth:`~remote_agent_toolkit.runtime.base.Session.run`).
        """
        return self._submit(
            message, resume=False, secrets=secrets, turn_config=config, hooks=hooks
        )

    def send(
        self,
        message: str,
        *,
        secrets: dict[str, str] | None = None,
        config: TurnConfig | None = None,
        hooks: Any | None = None,
    ) -> DrivenRun:
        """Resume this session with ``message``. Pass ``secrets`` again (not persisted).

        ``config`` is a per-turn :class:`~remote_agent_toolkit.config.TurnConfig` (see
        :meth:`run`). There is deliberately no session config here: the session's world
        was bound at ``start_session`` and cannot change mid-conversation.
        """
        return self._submit(
            message, resume=True, secrets=secrets, turn_config=config, hooks=hooks
        )

    def _mint_gcs_token(self) -> str | None:
        """Mint this turn's run-scoped GCS token and start refreshing it while the run lives.

        Returns ``None`` only when the engine handle was built with ``scoped_gcs=False``.
        A minting failure raises: running the turn on the runtime identity instead would
        silently reopen the bucket to the agent (README "Secrets & security").
        """
        engine = self._engine
        if not getattr(engine, "_scoped_gcs", True):
            return None
        from .scoped_gcs import REFRESH_EVERY_S, mint_run_token, write_run_token

        sid = self._session_id
        try:
            token, expiry = mint_run_token(engine._credentials, engine._output_bucket, sid)
        except Exception as exc:  # noqa: BLE001 — re-raise with the fix spelled out
            raise RuntimeError(
                "could not mint the run-scoped GCS token for this turn "
                f"({type(exc).__name__}: {str(exc)[:200]}). The client identity needs "
                "object access on the engine's output bucket (README, IAM). To drive a "
                "pre-token engine during a migration pass scoped_gcs=False to get_engine()."
            ) from exc
        self._stop_gcs_token_refresh()
        stop = threading.Event()
        self._gcs_token_stop = stop

        def refresh_loop() -> None:
            while not stop.wait(REFRESH_EVERY_S):
                try:
                    fresh, fresh_expiry = mint_run_token(
                        engine._credentials, engine._output_bucket, sid
                    )
                    write_run_token(engine._output_bucket, sid, fresh, fresh_expiry,
                                    **engine._gcs_store_kwargs())
                except Exception:  # noqa: BLE001 — the worker keeps its current token
                    logger.warning("run-scoped GCS token refresh failed for %s", sid, exc_info=True)

        threading.Thread(target=refresh_loop, name=f"gcs-token-refresh-{sid}", daemon=True).start()
        return token

    def _stop_gcs_token_refresh(self) -> None:
        stop = self._gcs_token_stop
        if stop is not None:
            stop.set()
            self._gcs_token_stop = None
            engine = self._engine
            if engine._output_bucket:
                from .scoped_gcs import delete_run_token

                try:
                    delete_run_token(engine._output_bucket, self._session_id,
                                     **engine._gcs_store_kwargs())
                except Exception:  # noqa: BLE001 — the 1-day lifecycle rule backs this up
                    pass

    def _stage_secrets(self, secrets: dict[str, str] | None) -> str | None:
        """Stage per-invocation secrets to a nonce-keyed GCS object; return its gs:// URI."""
        if not secrets:
            return None
        engine = self._engine
        if not engine._output_bucket:
            raise ValueError(
                "passing secrets requires the engine's output bucket (the values are staged "
                "there for the worker); construct the engine with output_bucket/project set."
            )
        from .handoff import stage_secrets

        return stage_secrets(engine._output_bucket, self._session_id, secrets,
                             **engine._gcs_store_kwargs())

    def _bind_config(self, config: SessionConfig | None) -> None:
        """Persist the session's config at its stable key (called once, at session open)."""
        if config is None or not config.set_fields():
            self._session_config = None
            self._session_config_resolved = True
            return
        engine = self._engine
        if not engine._output_bucket:
            raise ValueError(
                "start_session(config=...) requires the engine's output bucket (the config "
                "is persisted there for the workers and for re-attach); construct the "
                "engine with output_bucket/project set."
            )
        from .handoff import persist_session_config

        self._session_config = config
        self._session_config_uri = persist_session_config(
            engine._output_bucket, self._session_id, config.to_dict(),
            **engine._gcs_store_kwargs(),
        )
        self._session_config_resolved = True

    def _resolve_session_config(self) -> None:
        """Re-attach path: load the opener's persisted config (pointer + dict) from GCS.

        One GCS read on the session's first use; a session with no persisted config
        resolves to none (runs execute the deploy-baked spec).
        """
        if self._session_config_resolved:
            return
        self._session_config_resolved = True
        engine = self._engine
        if not engine._output_bucket:
            return
        from ...config import SessionConfig
        from .handoff import load_session_config

        found = load_session_config(engine._output_bucket, self._session_id,
                                    **engine._gcs_store_kwargs())
        if found is not None:
            self._session_config_uri, config_dict = found
            self._session_config = SessionConfig.from_dict(config_dict)

    def _client_spec(self, turn_config: TurnConfig | None = None) -> AgentSpec:
        """The client-side effective spec: engine spec + session + turn overlays.

        Used for result building (structured-output parsing, the idle stop-reason). The
        worker computes the same overlay over its deploy-baked spec — and echoes the
        merged result as an ``effective_spec`` event, the ground-truth record.
        """
        from ...config import apply_session_config, apply_turn_config

        self._resolve_session_config()
        return apply_turn_config(
            apply_session_config(self._engine.spec, self._session_config), turn_config
        )

    def _submit(
        self,
        message: str,
        resume: bool,
        secrets: dict[str, str] | None = None,
        turn_config: TurnConfig | None = None,
        hooks: Any | None = None,
    ) -> DrivenRun:
        # A failed acquisition must never clean another in-flight turn's lease.
        # This is a local ownership guard, not a distributed session lock.
        if not self._submission_lock.acquire(blocking=False):
            raise RuntimeError("a submission is already in progress on this session handle")
        try:
            if self._submission_uncertain:
                raise RuntimeError("previous submission acceptance is unknown; reconcile it first")
            if self._current_run is not None and not self._current_run.done:
                raise RuntimeError("a turn is already active on this session handle")
            self._submission_may_have_dispatched = False
            self._last_job = None
            try:
                return self._submit_impl(message, resume, secrets, turn_config, hooks)
            except Exception:
                if self._submission_may_have_dispatched:
                    # A timeout can follow remote acceptance. Do not invalidate
                    # credentials or retire a worker that may be executing.
                    self._submission_uncertain = True
                else:
                    self._abort_submission()
                raise
        finally:
            self._submission_lock.release()

    def _abort_submission(self) -> None:
        """Best-effort rollback only when no dispatch could have been accepted."""
        try:
            self._stop_gcs_token_refresh()
        except Exception:  # noqa: BLE001 — preserve the original setup error
            logger.warning("could not clean up failed submission token lease")
        worker, self._worker = self._worker, None
        self._worker_retired = False
        if worker is not None:
            try:
                self._engine._retire_worker(worker, cancel_job=True)
            except Exception:  # noqa: BLE001 — pool TTL remains the backstop
                logger.warning("could not retire failed submission worker")
        uri, self._staged_secrets_uri = self._staged_secrets_uri, None
        if uri:
            from .handoff import delete_staged_secrets

            try:
                delete_staged_secrets(uri, **self._engine._gcs_store_kwargs(uri))
            except Exception:  # noqa: BLE001 — lifecycle remains the backstop
                logger.warning("could not clean up failed submission secrets")

    def _submit_impl(
        self,
        message: str,
        resume: bool,
        secrets: dict[str, str] | None = None,
        turn_config: TurnConfig | None = None,
        hooks: Any | None = None,
    ) -> DrivenRun:
        engine = self._engine
        sid = self._session_id
        if hooks:
            raise ValueError(
                "run(hooks=...) is local-only: a hook is a callable in this process, and "
                "this turn executes in a remote worker, so there is nothing to call it "
                "there. Observe the turn through its event stream (async for) or "
                "Session.history() instead."
            )
        if not engine._output_bucket:
            raise ValueError(
                "running a turn requires the engine's output bucket (events stream through "
                "it); construct the engine with output_bucket/project set."
            )
        # Clock floor for a re-attached session's first turn (small slack for clock skew);
        # subsequent turns use the exact key watermark instead (see tail_source below).
        since = time.time() - 5
        secrets_uri = self._stage_secrets(secrets)
        self._staged_secrets_uri = secrets_uri  # cleaned up on completion (worker deletes at turn end)
        gcs_token = self._mint_gcs_token()  # run-scoped GCS token (scoped_gcs.py), or None
        self._resolve_session_config()  # re-attach: recover the opener's persisted config
        turn_config_uri = None
        if turn_config is not None and turn_config.set_fields():
            from .handoff import stage_turn_config

            turn_config_uri = stage_turn_config(
                engine._output_bucket, sid, turn_config.to_dict(), **engine._gcs_store_kwargs()
            )
        # Resolve/validate the client overlay before crossing the dispatch boundary.
        client_spec = self._client_spec(turn_config)

        worker: WorkerEntry | None = None  # per-worker dispatch: the worker this turn went to
        payload: dict | None = None
        if engine._warm:
            # Warm path: dispatch the turn to the pool (a warm worker adopts our session_id),
            # then refill so the next turn stays warm. No cold run_query_job. The payload
            # carries only pointers, never values: the secrets object, the session's
            # persisted config, and this turn's config — the worker overlays the configs
            # on its deploy-baked spec and runs the result.
            payload = dispatch_payload(
                sid, message, resume, secrets_uri,
                session_config_gcs=self._session_config_uri,
                turn_config_gcs=turn_config_uri,
                gcs_token=gcs_token,
            )
            if engine._subscription:
                # Legacy shared-subscription pool (an engine deployed before per-worker
                # dispatch): every idle worker competes for the one subscription.
                dispatcher = engine._dispatch()
                self._submission_may_have_dispatched = True
                dispatcher.publish(payload)
            else:
                # Per-worker dispatch: take one idle worker off the roster (or spawn one
                # when the pool is empty — the turn then waits for its boot instead of
                # stranding) and address the message to that worker alone: its filtered
                # subscription is the only channel it can land on.
                worker = engine._claim_worker()
                self._worker, self._worker_retired = worker, False
                dispatcher = engine._dispatch()
                attributes = worker_attributes(worker.worker)
                self._submission_may_have_dispatched = True
                dispatcher.publish(payload, attributes=attributes)
            # Refill off the critical path: creating the new worker's subscription takes
            # seconds on Pub/Sub, and nothing about THIS turn waits for it. Best-effort.
            engine._in_background(engine.fill_pool, 1, name="pool-refill")
        else:
            # Cold path: the prompt is the only reliable channel to the agent, so the
            # session id, secrets POINTER, config pointers, and resume marker ride leading
            # directive lines the agent strips. Never secret values: the platform persists
            # the job input to jobs/<sid>_input.jsonl.
            directives = f"AGENT_SESSION={sid}\n"
            if secrets_uri:
                directives += f"AGENT_SECRETS_GCS={secrets_uri}\n"
            if self._session_config_uri:
                directives += f"AGENT_SESSION_CONFIG_GCS={self._session_config_uri}\n"
            if turn_config_uri:
                directives += f"AGENT_TURN_CONFIG_GCS={turn_config_uri}\n"
            if resume:
                directives += f"AGENT_RESUME={sid}\n"
            if gcs_token:
                # LAST on purpose: a pre-token worker strips the earlier lines in order and
                # leaves this one as prompt text, still running the turn on the runtime
                # identity; the leftover is a token worth this run's own objects only.
                directives += f"AGENT_GCS_TOKEN={gcs_token}\n"
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
            api = engine._agent_engines()
            self._submission_may_have_dispatched = True
            self._last_job = api.run_query_job(name=engine._resource, config=cfg)

        # Live channel: the GCS event mirror (quota-free, no ingestion lag). Cloud Logging
        # is still WRITTEN by every worker — it's the ops/debug channel, never tailed.
        # Engines deployed before streaming wrote the mirror only at end-of-turn: their
        # events all arrive with the terminal result — redeploy them for live streaming.
        events_uri = f"{engine._output_bucket}/events"
        watermark = self._stream_watermark

        def tail_source() -> AsyncIterator:
            # start_after (the previous turn's last consumed object) is the reliable
            # turn boundary; the `since` clock floor covers re-attached sessions only.
            return tail_stream(
                events_uri, sid, since=since,
                start_after=watermark["key"] or None, watermark=watermark,
                **engine._gcs_store_kwargs(),
            )

        async def history_reader() -> list:
            from .history import read_history

            return await asyncio.wait_for(
                asyncio.to_thread(
                    read_history, engine._output_bucket, sid,
                    project=engine._project, credentials=engine._credentials,
                ),
                timeout=60,
            )

        def job_probe(job_name_of: Any) -> Any:
            async def probe() -> str | None:
                job_name = job_name_of()
                if not job_name:
                    return None

                def _check() -> str | None:
                    result = engine._agent_engines().check_query_job(name=job_name)
                    return getattr(result, "status", None)

                try:
                    return await asyncio.wait_for(asyncio.to_thread(_check), timeout=30)
                except Exception:  # noqa: BLE001 — unreachable probe must not kill a healthy run
                    return None

            return probe

        # Runs that know their job get a watchdog: if the job terminates without ever
        # writing a terminal event, the run ends with an explained error instead of hanging.
        # Cold runs hold the job's operation name directly; a per-worker warm turn knows
        # its worker's job (the roster entry) and additionally gets the pickup watchdog,
        # which re-dispatches the turn if that worker never starts it. A legacy warm turn
        # has no per-turn job handle (whichever worker won the shared subscription runs
        # it) — plain tail, protected by the per-poll timeouts.
        job_name = getattr(self._last_job, "job_name", None)
        if job_name:
            def factory() -> AsyncIterator:
                return _watched_tail(tail_source, job_probe(lambda: job_name), sid, history_reader)
        elif worker is not None:
            picked_up = {"done": False}

            def current_job() -> str | None:
                # The job watchdog stays inert until the worker has started the turn:
                # before that a dead or slow worker is the pickup watchdog's call.
                current = self._worker
                return current.job_name if picked_up["done"] and current else None

            def redispatch(attempt: int) -> float:
                # The addressed worker never started the turn: drop its channel (and the
                # undelivered message with it), stop its job, address another worker, and
                # back-fill the pool for the worker just lost.
                dead = self._worker
                if dead is not None:
                    engine._retire_worker(dead, cancel_job=True)
                fresh = engine._claim_worker()
                self._worker, self._worker_retired = fresh, False
                engine._dispatch().publish(payload, attributes=worker_attributes(fresh.worker))
                engine._in_background(engine.fill_pool, 1, name="pool-refill")
                return pickup_deadline_s(fresh.submitted_at)

            def on_pickup() -> None:
                picked_up["done"] = True
                # The worker holds the turn; nothing is ever published to its channel
                # again, so drop it (a client crash leaves the TTL backstop). Off the
                # critical path: a subscription delete takes seconds on Pub/Sub, and the
                # turn's first event is waiting behind this callback.
                current = self._worker
                if current is not None and not self._worker_retired:
                    self._worker_retired = True
                    engine._in_background(engine._retire_worker, current, name="pool-channel-cleanup")

            def on_abandon() -> None:
                current = self._worker
                if current is not None:
                    engine._retire_worker(current, cancel_job=True)

            first_deadline = pickup_deadline_s(worker.submitted_at)

            def factory() -> AsyncIterator:
                return _pickup_watched_tail(
                    lambda: _watched_tail(tail_source, job_probe(current_job), sid, history_reader),
                    first_deadline, redispatch, sid,
                    on_pickup=on_pickup, on_abandon=on_abandon,
                )
        else:
            factory = tail_source

        run = DrivenRun(factory, sid, client_spec, on_complete=self._on_complete)
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
        self._stop_gcs_token_refresh()  # the run is over: no more tokens for it
        # Backstop for the worker's channel when the turn ended without a pickup (e.g. the
        # synthetic pickup-timeout result); normally it was dropped at pickup already.
        worker, self._worker = self._worker, None
        retired, self._worker_retired = self._worker_retired, False
        if worker is not None and not retired:
            self._engine._in_background(
                self._engine._retire_worker, worker, name="pool-channel-cleanup"
            )
        # Backstop cleanup of the staged SECRETS object; the worker normally deletes it at
        # turn end, but a run that failed before the worker fetched would otherwise leave
        # it for the lifecycle rule. Config objects are deliberately NOT deleted — they are
        # the post-mortem record of what the session/turn was asked to run (handoff.py).
        if self._staged_secrets_uri:
            from .handoff import delete_staged_secrets

            delete_staged_secrets(self._staged_secrets_uri,
                                 **self._engine._gcs_store_kwargs(self._staged_secrets_uri))
            self._staged_secrets_uri = None

    async def interrupt(self) -> None:
        """Interrupt the in-flight run: stop tailing AND cancel the remote job.

        Cancelling the ``run_query_job`` stops the agent (and its billing). A per-worker
        warm turn knows the job of the worker it was addressed to (dedicated to this turn),
        so it is cancelled the same way. A legacy shared-subscription warm turn has no
        per-turn job handle, so there this only stops tailing — the worker finishes its turn.
        """
        run = self._current_run
        # Local completion clears self._worker. Keep the exact remote target
        # before awaiting it, so warm cancellation cannot lose the job handle.
        worker = self._worker
        job_name = getattr(self._last_job, "job_name", None) or (worker.job_name if worker else None)
        if run is not None and run.task is not None and not run.task.done():
            run.task.cancel()
            try:
                await run.task
            except asyncio.CancelledError:
                pass
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

    async def transcripts(self) -> dict[str, list[dict]]:
        """This session's persisted harness transcripts (see ``runtime.base.Session``).

        Read straight from the engine's checkpoint prefix, so it works for a session
        re-attached from another process — the same objects the worker mirrored the
        transcript to during the run.
        """
        from ...checkpoint.session_store import BlobSessionStore, _claude_session_id
        from ...ports.blobstore import GcsBlobStore, parse_gcs_uri

        engine = self._engine
        if not engine._output_bucket:
            raise RuntimeError(
                "transcripts() reads the engine's checkpoint prefix under its output "
                "bucket; construct the engine with output_bucket/project set."
            )
        # Fail loudly on a spec that persists nothing, rather than returning the same ``{}``
        # a persisting session reads before its first turn. Only when the deployed spec is
        # known: a ``get_engine`` handle carries an addressing-only spec whose defaults say
        # nothing about what the engine bakes.
        spec = self._client_spec()
        if engine._spec_known and not (spec.checkpoint or spec.transcript):
            raise RuntimeError(
                "no transcript is persisted for this session — deploy the spec with "
                "AgentSpec(transcript=True)"
            )
        bucket, prefix = parse_gcs_uri(f"{engine._output_bucket}/checkpoints")
        blobs = GcsBlobStore(bucket, (prefix + "/") if prefix else "",
                             **({"credentials": engine._credentials}
                                if engine._credentials is not None else {}))
        # The worker keys the store by the SDK-canonical id, not the raw (numeric, on the
        # cold path) session id.
        return await BlobSessionStore(blobs).load_all(_claude_session_id(self._session_id))

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

                # The session-level effective spec (re-attach recovers the persisted
                # config, so structured parsing / stop-reason match what the opener saw).
                self._last_result, self._stop_reason = build_result(
                    result_ev, self._session_id, self._client_spec()
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
        version: str | None = None,
        spec_known: bool = True,
        scoped_gcs: bool = True,
        pool_max_wait_s: float | None = None,
    ) -> None:
        self._resource = resource
        self._scoped_gcs = scoped_gcs
        # The deploy handle carries the real deployed spec; a get_engine handle carries a
        # minimal fallback (addressing only), flagged by ``spec_known=False`` so nothing
        # client-side reads its defaults as the deployed truth. Either way runs execute the
        # engine's baked spec overlaid with the session/turn configs — the client uses this
        # spec only as the base of its own view for result parsing.
        self.spec = spec
        self._spec_known = spec_known
        self._project = project
        self._location = location
        self._output_bucket = output_bucket
        self._credentials = credentials
        self._warm = warm
        self._topic = topic
        # Set only for a LEGACY pool (deployed before per-worker dispatch): the one shared
        # subscription its workers compete for. None => per-worker dispatch (pool.py).
        self._subscription = subscription
        # The workers' idle life (the deploy's pool_max_wait_s, read back from the env by
        # get_engine): bounds how long a roster entry can be trusted (roster.py).
        self._pool_max_wait_s = float(pool_max_wait_s) if pool_max_wait_s else DEFAULT_MAX_WAIT_S
        self._version = version  # pinned revision id; None => resolve the serving one lazily
        self._created = time.time()  # readiness-tail watermark (ignore stale pool markers)
        self._sessions: dict[str, GeminiSession] = {}
        self._pool_jobs: list[str] = []  # tracked pool-worker job names (to cancel on delete)
        self._dispatch_cached: Any = None  # one Pub/Sub client set per handle (auth once)
        self._roster_cached: Any = None  # one GCS client per handle
        self._background: list[threading.Thread] = []  # pool housekeeping threads

    def _client(self) -> Any:
        import agentplatform

        return agentplatform.Client(
            project=self._project, location=self._location, credentials=self._credentials
        )

    def _agent_engines(self) -> Any:
        return self._client().agent_engines

    def _dispatch(self) -> Any:
        """The control-plane ``DispatchTransport`` for the warm pool (one client set per handle)."""
        if self._dispatch_cached is None:
            from ...ports.dispatch import PubSubDispatch

            self._dispatch_cached = PubSubDispatch(
                topic=self._topic, subscription=self._subscription,
                project=self._project, credentials=self._credentials,
            )
        return self._dispatch_cached

    def _in_background(self, fn: Any, *args: Any, name: str = "pool-housekeeping") -> None:
        """Run pool housekeeping (refill, channel cleanup) off a turn's critical path.

        Subscription admin calls take seconds each on Pub/Sub, and none of them has to
        finish before a turn's events can flow. Non-daemon, so a short-lived process still
        completes them before exiting: a refilled worker must reach the roster, or it idles
        unreachable (and billing) for a day. Failures are logged, never raised.
        """

        def run() -> None:
            try:
                fn(*args)
            except Exception:  # noqa: BLE001 — housekeeping never fails a run
                logger.warning("%s failed", name, exc_info=True)

        self._background = [t for t in self._background if t.is_alive()]
        thread = threading.Thread(target=run, name=name, daemon=False)
        thread.start()
        self._background.append(thread)

    def _join_background(self, timeout: float | None = None) -> None:
        """Wait for outstanding pool housekeeping (tests; teardown)."""
        for thread in list(self._background):
            thread.join(timeout)

    def fill_pool(self, n: int) -> None:
        """Submit ``n`` pre-warmed workers (each a job blocked on ``__POOL_WAIT__``).

        Use this to top a warm pool back up — e.g. after reusing an engine via
        ``get_engine`` whose workers have idle-expired, or to grow the pool. Each worker
        cold-boots (~2.5 min); ``wait_until_warm()`` blocks until one reports ready.

        Per-worker dispatch (``pool.py``): every worker gets its own filtered subscription
        on the pool's topic and an entry in the pool's roster (``roster.py``), the shared
        record every client process picks idle workers from. Workers that idle-expire
        (after ``pool_max_wait_s``) exit without replacement, but an empty roster does not
        strand a turn: the turn spawns a worker for itself and waits for its boot (cold
        latency), and the refill after it starts warming the pool again — this method just
        gets the pool warm ahead of the next turn.
        """
        if self._subscription:
            self._fill_legacy_pool(n)
            return
        for _ in range(n):
            self._spawn_worker(rostered=True)

    def _fill_legacy_pool(self, n: int) -> None:
        """``fill_pool`` for an engine deployed before per-worker dispatch (shared sub)."""
        ae = self._agent_engines()
        query = json.dumps({
            "class_method": "async_stream_query",
            "input": {"user_id": _USER_ID, "message": POOL_WAIT_SENTINEL},
        })
        bucket = self._output_bucket or f"gs://{self._project}-agent-output"
        for _ in range(n):
            cfg = {"query": query, "output_gcs_uri": f"{bucket}/pool/{uuid.uuid4().hex}.jsonl"}
            result = ae.run_query_job(name=self._resource, config=cfg)
            job_name = getattr(result, "job_name", None)
            if job_name:
                self._pool_jobs.append(job_name)

    def _roster(self) -> Any:
        """The pool's idle-worker roster (``roster.py``), in the output bucket."""
        if self._roster_cached is None:
            from .roster import PoolRoster

            if not self._output_bucket or not self._topic:
                raise ValueError(
                    "a warm pool's roster lives in the engine's output bucket, under its "
                    "dispatch topic; construct the engine with output_bucket/project set and "
                    "warm_pool=True."
                )
            self._roster_cached = PoolRoster(
                self._output_bucket, self._topic, credentials=self._credentials
            )
        return self._roster_cached

    def _spawn_worker(self, *, rostered: bool) -> WorkerEntry:
        """Submit one pool worker on its own subscription; record it as idle if ``rostered``.

        An un-rostered worker is spawned for a turn that found no idle worker: that turn
        is addressed to it right away, so no other client may ever pick it.
        """
        from .roster import WorkerEntry

        if not self._topic:
            raise ValueError(
                "this warm engine handle has no dispatch topic; re-address it via "
                "get_engine(..., warm_pool=True) or redeploy with warm_pool=True."
            )
        worker_id = new_worker_id()
        subscription = worker_subscription(self._topic, worker_id)
        dispatch = self._dispatch()
        # The filter is what makes the channel private: only a message the client
        # addressed to THIS worker id is ever delivered on it. TTL/retention are the
        # backstops for a client that dies before dropping the channel (pool.py).
        dispatch.create_subscription(
            subscription,
            filter=worker_filter(worker_id),
            ttl_s=WORKER_SUBSCRIPTION_TTL_S,
            retention_s=WORKER_SUBSCRIPTION_RETENTION_S,
        )
        # Pool workers are query jobs too; the class method must be named explicitly
        # (the runner silently invokes nothing otherwise). The job input carries the
        # worker's own subscription — never the engine env, which is shared.
        query = json.dumps({
            "class_method": "async_stream_query",
            "input": {"user_id": _USER_ID, "message": pool_wait_prompt(subscription)},
        })
        bucket = self._output_bucket or f"gs://{self._project}-agent-output"
        # run_query_job requires output_gcs_uri; a worker's job output is never read. It
        # goes under jobs/ because the platform writes it as the runtime identity, whose
        # output-bucket binding covers exactly that prefix (README IAM table).
        cfg = {"query": query, "output_gcs_uri": f"{bucket}/jobs/pool-{worker_id}.jsonl"}
        submitted_at = time.time()
        try:
            result = self._agent_engines().run_query_job(name=self._resource, config=cfg)
        except Exception:
            try:
                dispatch.delete_subscription(subscription)
            except Exception:  # noqa: BLE001 — the TTL backs this up
                pass
            raise
        job_name = getattr(result, "job_name", None)
        if job_name:
            # Track the worker's job so delete() can cancel it (a waiting worker is a
            # long-running op that otherwise blocks engine deletion for up to its max-wait).
            self._pool_jobs.append(job_name)
        entry = WorkerEntry(
            worker=worker_id, subscription=subscription, job_name=job_name,
            submitted_at=submitted_at, expires_at=submitted_at + self._pool_max_wait_s,
        )
        if rostered:
            try:
                self._roster().add(entry)
            except Exception:
                # A worker nobody can find is a worker nobody dispatches to: stop it rather
                # than let it bill idle for a day.
                self._retire_worker(entry, cancel_job=True)
                raise
        return entry

    def _claim_worker(self) -> WorkerEntry:
        """One idle worker for a turn, taken off the roster (oldest first) — or a fresh one.

        Expired entries (the worker idled out) are pruned and their channels dropped on
        the way. An empty roster spawns a worker for this turn alone: the turn then waits
        for its ~2.5 min boot — cold latency, never a stranded turn.
        """
        entry, expired = self._roster().claim()
        for stale in expired:  # their channels: housekeeping, off the critical path
            self._in_background(self._retire_worker, stale, name="pool-prune")
        if entry is None:
            entry = self._spawn_worker(rostered=False)
        return entry

    def _retire_worker(self, entry: WorkerEntry, *, cancel_job: bool = False) -> None:
        """Drop a worker's channel (best-effort, idempotent); ``cancel_job`` also stops its job."""
        try:
            self._dispatch().delete_subscription(entry.subscription)
        except Exception:  # noqa: BLE001 — the TTL backs this up
            pass
        if cancel_job and entry.job_name:
            try:
                self._agent_engines().cancel_query_job(
                    name=self._resource, config={"operation_name": entry.job_name}
                )
            except Exception:  # noqa: BLE001 — already done / cancelled / unknown
                pass
            if entry.job_name in self._pool_jobs:
                self._pool_jobs.remove(entry.job_name)

    def start_session(self, config: SessionConfig | None = None) -> GeminiSession:
        """Begin a new session; ``config`` binds its :class:`SessionConfig` for good.

        The config (a sparse overlay over the deploy-baked spec — repos, skills, prompt,
        model, ...) is persisted at a stable per-session GCS key here and cannot be
        changed afterwards: the session's world is created on its first turn and
        snapshot-restored on every later one. ``send()`` and re-attach take no config.
        """
        # Fail fast BEFORE the platform session is created: binding a config needs the
        # output bucket (it is persisted there for the workers and for re-attach).
        if config is not None and config.set_fields() and not self._output_bucket:
            raise ValueError(
                "start_session(config=...) requires the engine's output bucket (the config "
                "is persisted there for the workers and for re-attach); construct the "
                "engine with output_bucket/project set."
            )
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
        session._bind_config(config)
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> GeminiSession:
        """Re-attach to an existing session by id (poll / continue).

        Takes NO config on purpose: the session's :class:`SessionConfig` was bound at
        ``start_session`` and persisted next to the session's records — re-attach reads
        that back, so every process resuming the session runs the same world. Accepting a
        config here would let a re-attacher change repos/skills/prompt mid-conversation,
        which the workspace-snapshot model cannot honor.
        """
        return self._sessions.get(session_id) or GeminiSession(
            self, session_id, config_resolved=False
        )

    def _gcs_store_kwargs(self, uri: str | None = None) -> dict:
        """Inject this client's explicit identity without changing worker defaults.

        Handoff/stream helpers already support a store. Their keys include the bucket
        prefix, so the injected store deliberately has no prefix of its own.
        """
        from ...ports.blobstore import GcsBlobStore, parse_gcs_uri

        uri = uri or self._output_bucket
        if self._credentials is None or not uri:
            return {}
        bucket, _ = parse_gcs_uri(uri)
        return {"store": GcsBlobStore(bucket, credentials=self._credentials)}

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
            **self._gcs_store_kwargs(),
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
        from .pool import pool_log_id, pool_log_id_from_topic

        # Generation-scoped (derived from this pool's dispatch topic), so markers from a
        # previous deploy's workers never count as THIS pool being warm.
        pool_id = pool_log_id_from_topic(self._topic) if self._topic else pool_log_id(self.name)
        created = self._created

        async def _await() -> bool:
            # A worker writes its readiness marker to the GCS mirror (events/<pool_id>/).
            # NB engines deployed before event streaming only ever logged it — for those
            # this times out (False, a soft signal: dispatch works regardless); redeploy.
            markers = tail_stream(f"{self._output_bucket}/events", pool_id, since=created,
                                  **self._gcs_store_kwargs())
            async for _ in markers:
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
        retries up to ``timeout``. ``delete_pool_resources`` also retires the pool's
        Pub/Sub resources (topic, worker subscriptions, and the roster).
        """
        import re

        ae = self._agent_engines()
        # Collect every revision's pool BEFORE the engine — and with it the revision list —
        # is gone: deploys made while traffic was pinned each left their fresh pool idle
        # (see deploy()), so the handle's own pool is not necessarily the only one to
        # clean up. Best-effort: the handle's own pool is swept regardless.
        pools: dict[str, str | None] = {}  # topic -> legacy shared subscription
        if delete_pool_resources:
            if self._topic:
                pools[self._topic] = self._subscription
            try:
                revisions = self._client().agent_engines.runtimes.revisions.list(
                    name=self._resource
                )
                for revision in revisions:
                    address = _address_from_env(_pool_env(revision.api_resource))
                    if address:
                        pools.setdefault(address[0], address[1])
            except Exception:  # noqa: BLE001 — sweep is best-effort
                pass
        # Idle workers other client processes submitted are jobs too: the roster names
        # them, so cancel them up front as well (a reused engine's workers used to be
        # findable only through the blocked-delete error below).
        if self._topic and not self._subscription and self._output_bucket:
            try:
                for entry in self._roster().entries():
                    if entry.job_name and entry.job_name not in self._pool_jobs:
                        self._pool_jobs.append(entry.job_name)
            except Exception:  # noqa: BLE001 — best effort
                pass
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

        for topic, legacy_subscription in pools.items():
            _retire_pool(topic, legacy_subscription, self._output_bucket, self._credentials)

    def versions(self) -> list[str]:
        """This engine's runtime-revision ids, newest first (see :meth:`revisions`)."""
        return [r["version"] for r in self.revisions()]

    def revisions(self) -> list[dict]:
        """This engine's runtime revisions, newest first, with their serving status.

        Each row: ``version`` (the id used by ``get_engine(version=…)`` / :meth:`set_traffic`),
        ``resource`` (full revision path), ``create_time``, ``state``, and ``serving`` — the
        one revision every run currently lands on. ``serving`` is ``False`` on every row when
        traffic is split across several (no single revision a run is guaranteed to hit).
        """
        from . import revisions as rev

        client = self._client()
        rows = rev.list_revisions(client, self._resource)
        engine = client.agent_engines.get(name=self._resource)
        serving = rev.serving_version(getattr(engine.api_resource, "traffic_config", None), rows)
        for row in rows:
            row["serving"] = row["version"] == serving
        return rows

    def set_traffic(self, version: str | int | None = None) -> None:
        """Route **all** of this engine's traffic to ``version``; ``None`` = always-latest.

        The rollback / promote action (an ops action, like ``deploy``). This is what makes a
        revision reachable at all: the toolkit runs turns through the engine-level async query
        path, so the traffic config — not the caller — decides which revision executes them
        (see ``revisions.py``). Takes effect for turns started afterwards; an in-flight job and
        an already-cold-started warm-pool worker finish on the revision they began on.
        """
        from . import revisions as rev

        if version is None:
            traffic = rev.always_latest_traffic()
            resolved = None
        else:
            resolved = rev.revision_id(str(version))
            traffic = rev.pinned_traffic(rev.revision_resource(self._resource, resolved))
        # A dict config (the SDK validates it into an AgentEngineConfig) with no `agent`:
        # the update mask is just `traffic_config`, so no rebuild and no new revision.
        self._client().agent_engines.update(
            name=self._resource, config={"traffic_config": traffic}
        )
        self._version = resolved

    def delete_version(self, version: str | int) -> None:
        """Delete one runtime revision (revisions accumulate; the serving one can't go).

        Deleting the revision that serves traffic is rejected by the platform — promote
        another with :meth:`set_traffic` first.
        """
        from . import revisions as rev

        resource = rev.revision_resource(self._resource, rev.revision_id(str(version)))
        self._client().agent_engines.runtimes.revisions.delete(name=resource)

    @property
    def name(self) -> str:
        return self.spec.name if self.spec is not None else self._resource.rsplit("/", 1)[-1]

    @property
    def version(self) -> str:
        """The runtime revision this handle addresses: the pin, else the one serving traffic.

        Resolved from the platform on first access and cached (``get_engine(version=…)`` and
        :meth:`set_traffic` fill it in directly). Falls back to ``"latest"`` when there is no
        single such revision to name — an unreachable control plane (an identity property must
        not raise), an engine with no revisions, or traffic split across several.
        """
        if self._version is None:
            try:
                rows = self.revisions()
            except Exception:  # noqa: BLE001 — identity property; degrade, never raise
                return "latest"
            self._version = next((r["version"] for r in rows if r["serving"]), None)
        return self._version or "latest"

    @property
    def resource(self) -> str:
        return self._resource

"""Sandbox runtime control plane: deploy / get_engine / list_engines + Engine/Session/Run.

The ``gemini`` backend runs the harness inside an **Agent Sandbox custom container**
(DESIGN.md §13): ``deploy`` builds the agent's image, pushes it and creates an immutable
sandbox **template** (a template is a version); app code addresses engines by name via
``get_engine`` and never deploys. The **client is the whole control plane**: a turn claims a
ready sandbox off the roster (``roster.py``) or creates one, ``POST``\\s the turn to the
worker (``worker.py``) through the platform's authenticated proxy, long-polls the worker's
``/events`` for the live stream, and deletes the sandbox at the terminal event. The GCS
event mirror the worker writes is the durable record and the fallback stream.

The run plane mirrors ``local`` exactly via the shared :class:`DrivenRun`: **stream** =
iterate, **await** = wait for the terminal ``result`` event, **poll** = read the latest
status. The platform is reached only through the :class:`~.provider.SandboxProvider` seam;
the Google SDK is imported lazily inside the adapter, so importing this module needs no
third-party deps.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import re
import threading
import time
import uuid
from typing import Any, AsyncIterator, Callable, TYPE_CHECKING

from ...control import ControlMessage, ControlUnavailable
from ...events import AgentEvent, RunResult, RunStatus, StopReason
from .._run import DrivenRun
from .history import event_from_mirror
from .provider import SandboxError, SandboxGone, SandboxProvider, template_id
from .roster import SandboxEntry
from .stream import tail_stream

if TYPE_CHECKING:
    from ...config import SessionConfig, TurnConfig
    from ...spec import AgentSpec
    from ..base import Engine

logger = logging.getLogger(__name__)

# Default idle life of a ready-pool sandbox: how long it waits for a turn before its TTL
# deletes it. A day keeps a pool warm across working-hours gaps while bounding idle
# billing (a ready sandbox bills like an idle warm worker did). Deploy-time override:
# ``gemini.deploy(..., pool_max_wait_s=...)``.
DEFAULT_MAX_WAIT_S = 24 * 3600.0
# Longest turn a sandbox must survive: an on-demand sandbox's TTL, and the margin added to
# a pool sandbox's TTL past its idle life (a TTL cannot be extended, so a sandbox claimed at
# the end of its idle life still has this much life for the turn). The backstop for a
# client that dies mid-turn: the platform deletes the sandbox then, whatever else failed.
DEFAULT_MAX_TURN_S = 8 * 3600.0
DEFAULT_RESOURCE_LIMITS = {"cpu": "4", "memory": "8Gi"}

# After creating a sandbox, how long the proxy may take to route to it (measured 12–33 s,
# the odd 78 s while a template's pool is still provisioning).
READY_WAIT_S = 180.0
# The /events long-poll: the worker holds a call this long when nothing new arrived. The
# proxy allowed 120 s calls (measured); this stays well inside, and the client's per-call
# timeout leaves room for the proxy's own overhead.
EVENTS_WAIT_S = 20.0
EVENTS_CALL_TIMEOUT_S = EVENTS_WAIT_S + 40.0
# Consecutive direct-poll failures before the stream falls back to the mirror, and the pause
# between retries.
DIRECT_MAX_FAILURES = 3
DIRECT_RETRY_SLEEP_S = 1.0
# When the sandbox is gone, how long the mirror is given to show a terminal result the
# worker may have flushed just before dying.
GONE_GRACE_S = 30.0
# Attempts to hand a turn to a sandbox (each failure deletes that sandbox and claims another).
DISPATCH_ATTEMPTS = 3
# A pool entry expiring within this many seconds is not dispatched to.
CLAIM_MARGIN_S = 120.0
# Client-side refresh cadence for the run's tokens (source tokens last an hour).
TOKEN_REFRESH_S = 25 * 60


_SANDBOX_NAME_RE = re.compile(r"[0-9a-f]{8}")  # the suffix ``_create_sandbox`` appends


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9-]", "-", name.lower()).strip("-") or "agent"


def _fallback_spec(name: str) -> AgentSpec:
    """A minimal spec for a looked-up engine whose deploy record is unavailable.

    Client-side result building needs ``output_schema`` (structured parsing) and
    ``checkpoint`` (the idle stop-reason). Runs execute the image's baked spec — overlaid
    with any ``SessionConfig``/``TurnConfig`` the caller binds, which is also how the client
    recovers structured parsing and the right stop-reason without the record.
    """
    from ...spec import AgentSpec

    return AgentSpec(name=name, model="")


def _synthetic_result(summary: str, session_id: str, event: str) -> AgentEvent:
    return AgentEvent(
        kind="result", summary=summary,
        raw={"event": event, "is_error": True, "subtype": "error", "session_id": session_id},
    )


async def _stream_turn(
    provider: SandboxProvider,
    sandbox: str,
    turn_id: str,
    session_id: str,
    *,
    events_uri: str,
    store_kwargs: dict,
    mirror_max_wait_s: float = 3600.0,
) -> AsyncIterator[AgentEvent]:
    """The live event source of one turn: the worker's ``/events`` long-poll, then the mirror.

    Direct first: one call per event batch, held by the worker until something new exists
    (an event is seen within one proxy round trip of its emission). When the sandbox is
    gone (:class:`SandboxGone`) or the proxy fails :data:`DIRECT_MAX_FAILURES` times in a
    row, the stream switches to the GCS mirror tail — skipping the events already
    delivered — and ends with the mirror's terminal result. A sandbox that is gone and
    left no terminal record ends the run with an explained error result after
    :data:`GONE_GRACE_S`, never by hanging.
    """
    yielded = 0
    failures = 0
    gone = False
    while True:
        try:
            resp = await asyncio.to_thread(
                provider.call, sandbox, "/events",
                {"turn_id": turn_id, "since": yielded, "wait": EVENTS_WAIT_S},
                timeout_s=EVENTS_CALL_TIMEOUT_S,
            )
        except SandboxGone:
            gone = True
            break
        except Exception:  # noqa: BLE001 — proxy/platform blip: retry, then fall back
            failures += 1
            if failures >= DIRECT_MAX_FAILURES:
                break
            await asyncio.sleep(DIRECT_RETRY_SLEEP_S)
            continue
        failures = 0
        if not resp.get("ok"):
            break  # the worker does not know the turn (restarted?): the mirror decides
        for line in resp.get("events") or []:
            yielded += 1
            try:
                event = event_from_mirror(line)
            except (KeyError, TypeError):
                continue
            if (event.raw or {}).get("turn_id") not in (None, turn_id):
                continue
            yield event
            if event.kind == "result":
                return
        if resp.get("done"):
            error = str(resp.get("error") or "the worker's turn thread ended")[-600:]
            yield _synthetic_result(
                f"turn finished without a terminal result: {error}", session_id,
                "worker_ended_without_result",
            )
            return

    skip = yielded
    async for event in tail_stream(
        events_uri, session_id, turn_id=turn_id,
        max_wait_s=GONE_GRACE_S if gone else mirror_max_wait_s, **store_kwargs,
    ):
        if skip:
            skip -= 1
            continue
        yield event
        if event.kind == "result":
            return
    yield _synthetic_result(
        (
            "the sandbox is gone (deleted, TTL-expired or crashed — an OOM shows up this way) "
            "and its event mirror holds no terminal result"
            if gone else
            "the sandbox stopped answering and its event mirror never showed a terminal result"
        ),
        session_id, "sandbox_unreachable",
    )


# -- control plane -------------------------------------------------------------


def deploy(
    spec: AgentSpec,
    project: str,
    location: str,
    warm_pool: bool = False,
    *,
    output_bucket: str | None = None,
    image_repo: str | None = None,
    image: str | None = None,
    model: str | None = None,
    use_vertex: bool = True,
    model_service_account: str | None = None,
    vertex_region: str = "global",
    resource_limits: dict[str, str] | None = None,
    pool_size: int = 2,
    pool_max_wait_s: float | None = None,
    max_turn_s: float | None = None,
    internet_access: bool = True,
    credentials: Any | None = None,
    workspace: str | None = None,
    provider: SandboxProvider | None = None,
    log: Callable[[str], None] | None = None,
    **_: Any,
) -> Engine:
    """Deploy ``spec`` as a sandbox template, minting a **new version** (ops/CI action).

    Builds the agent's container image (the toolkit with the baked harness CLIs, the
    resolved skills, ``spec.packages`` and the spec itself — ``_image.py``), pushes it to
    ``image_repo`` (default: the ``ratk`` Artifact Registry repo of the project/location),
    and creates an immutable **sandbox template** displayed as ``spec.name`` with the image,
    ``resource_limits`` and internet egress. **A template is a version**: ``get_engine(name)``
    resolves the newest one, ``version=`` pins one. A deploy whose image and resources match
    the newest template is a no-op that returns the existing version. Pass ``image=`` to
    skip the build and use an image you pushed yourself.

    With ``warm_pool=True`` the deploy also fills a **ready pool** of ``pool_size`` sandboxes
    (``fill_pool``) so turns start on a sandbox that already answers (~1 s to the first
    event, measured) instead of creating one (~20 s); and once the new pool is filled, the
    previous versions' pools and templates are retired. Ready sandboxes idle for
    ``pool_max_wait_s`` (default a day) before the platform deletes them; the only automatic
    refill is the one sandbox created after each dispatch, so a pool that idled out costs
    each next turn the creation latency until it warms again.

    Model credentials: with ``use_vertex`` (default) every turn carries a one-hour Vertex
    token minted by impersonating ``model_service_account`` (default the project's
    ``ratk-model@``, a predict-only account ``ratk-gcp-setup`` creates; the deployer and
    every client need ``roles/iam.serviceAccountTokenCreator`` on it). The sandbox has no
    Google identity of its own. ``use_vertex=False`` is API-key mode: pass
    ``ANTHROPIC_API_KEY`` as a per-invocation secret.

    ``max_turn_s`` bounds a turn's life on a sandbox (default 8 h): an on-demand sandbox's
    TTL, and the margin a pool sandbox keeps past its idle life. The platform enforces it
    even when the client dies. ``resource_limits`` sizes the container (default
    ``{"cpu": "4", "memory": "8Gi"}``; at most 8 vCPUs).

    Prerequisites: the Docker CLI logged into the registry (``gcloud auth configure-docker
    <region>-docker.pkg.dev``), an Artifact Registry repo the Agent Sandbox service agent
    can read, and the client permissions in the README ("GCP setup & required permissions").
    """
    import dataclasses
    import json

    from ._image import (
        build_and_push,
        default_image_repo,
        image_exists,
        image_uri,
        stage_build_context,
        validate_resource_limits,
    )
    from .handoff import ensure_handoff_lifecycle, write_deploy_record
    from .model_token import default_model_service_account

    say = log or (lambda msg: None)
    if workspace is not None:
        raise ValueError(
            "workspace= is local-only: a sandbox's filesystem is its own, one turn at a "
            "time, so there is no host directory to point turns at. Seed the agent's cwd "
            "through the prompt or spec.repos instead."
        )
    if pool_max_wait_s is not None and not warm_pool:
        raise ValueError("pool_max_wait_s only applies to warm-pool engines; pass warm_pool=True")
    for label, value in (("pool_max_wait_s", pool_max_wait_s), ("max_turn_s", max_turn_s)):
        if value is not None and not (value > 0 and value != float("inf") and value == value):
            raise ValueError(f"{label} must be a positive, finite number of seconds; got {value!r}")
    limits = dict(resource_limits) if resource_limits else dict(DEFAULT_RESOURCE_LIMITS)
    validate_resource_limits(limits)
    if model:
        spec = dataclasses.replace(spec, model=model)
    output_bucket = output_bucket or f"gs://{project}-agent-output"
    model_sa = (
        (model_service_account or default_model_service_account(project)) if use_vertex else None
    )

    if not ensure_handoff_lifecycle(output_bucket, credentials=credentials):
        import warnings

        warnings.warn(
            f"could not ensure the lifecycle rules on {output_bucket}; token objects orphaned "
            "by crashed clients and old config records will not be auto-reaped (needs "
            "storage.buckets.update on the bucket)",
            stacklevel=2,
        )

    if image is None:
        context, digest = stage_build_context(spec)
        image = image_uri(image_repo or default_image_repo(project, location), spec.name, digest)
        if image_exists(image):
            say(f"image {image} already in the registry; skipping the build")
        else:
            build_and_push(context, image, log=say)

    provider = provider or _default_provider(project, location, credentials)
    existing = provider.list_templates(display_name=spec.name)
    newest = existing[0] if existing else None
    if (newest and newest.get("image_uri") == image and newest.get("state") in (None, "ACTIVE")
            and str(newest.get("cpu")) == limits["cpu"] and newest.get("memory") == limits["memory"]):
        template = newest["name"]
        say(f"template {template_id(template)} already serves this image; no new version")
    else:
        say(f"creating template {spec.name} from {image} ({limits['cpu']} CPU / {limits['memory']})")
        template = provider.create_template(
            display_name=spec.name, image_uri=image, cpu=limits["cpu"], memory=limits["memory"],
            internet_access=internet_access, log=say,
        )
        say(f"template {template_id(template)} is ACTIVE")
    record = {
        "name": spec.name,
        "spec": spec.to_dict(),
        "image": image,
        "resource_limits": limits,
        "use_vertex": use_vertex,
        "model_service_account": model_sa,
        "vertex_region": vertex_region,
        "warm_pool": warm_pool,
        "pool_size": pool_size,
        "pool_max_wait_s": pool_max_wait_s,
        "max_turn_s": max_turn_s,
        "toolkit_version": _toolkit_version(),
        "created_at": time.time(),
    }
    try:
        write_deploy_record(
            output_bucket, spec.name, template_id(template), json.loads(json.dumps(record)),
            **_store_kwargs(output_bucket, credentials),
        )
    except Exception as exc:  # noqa: BLE001 — the record is a convenience for get_engine
        import warnings

        warnings.warn(f"could not write the deploy record to {output_bucket}: {exc}", stacklevel=2)

    engine = GeminiEngine(
        template=template, spec=spec, project=project, location=location,
        output_bucket=output_bucket, credentials=credentials, provider=provider,
        warm=warm_pool, pool_max_wait_s=pool_max_wait_s, max_turn_s=max_turn_s,
        use_vertex=use_vertex, model_service_account=model_sa, vertex_region=vertex_region,
    )
    if warm_pool:
        say(f"filling the ready pool with {pool_size} sandbox(es)")
        engine.fill_pool(pool_size)
    previous = [t["name"] for t in existing if t["name"] != template]
    if previous:
        # Cutover: the new version serves; the old versions' idle sandboxes and templates
        # go, off the critical path (a template refuses deletion while sandboxes exist).
        engine._in_background(engine._retire_templates, previous, name="template-retire")
    return engine


def _toolkit_version() -> str | None:
    try:
        import importlib.metadata

        return importlib.metadata.version("remote-agent-toolkit")
    except Exception:  # noqa: BLE001
        return None


def _default_provider(project: str | None, location: str | None, credentials: Any) -> SandboxProvider:
    from .provider import AgentSandboxProvider

    if not project or not location:
        raise ValueError("project and location are required to address the sandbox platform")
    return AgentSandboxProvider(project, location, credentials)


def _store_kwargs(uri: str | None, credentials: Any) -> dict:
    """``{"store": GcsBlobStore}`` carrying the client's explicit identity, or ``{}`` for ADC."""
    from ...ports.blobstore import GcsBlobStore, parse_gcs_uri

    if credentials is None or not uri:
        return {}
    bucket, _ = parse_gcs_uri(uri)
    return {"store": GcsBlobStore(bucket, credentials=credentials)}


def get_engine(
    name: str,
    project: str | None = None,
    location: str | None = None,
    version: str | int | None = None,
    *,
    output_bucket: str | None = None,
    warm_pool: bool | None = None,
    credentials: Any | None = None,
    provider: SandboxProvider | None = None,
    **kwargs: Any,
) -> Engine:
    """Look up a deployed engine by ``name`` (the app-code hot path; never deploys).

    Pure ADDRESSING: the handle identifies which template turns run on; it carries no
    execution configuration. Runs execute the image's baked spec, overlaid with the
    session's ``SessionConfig`` (``engine.start_session(config=...)``) and each turn's
    ``TurnConfig`` (``run(config=...)``).

    ``version`` pins the handle to one **template** (its id, or the full resource name);
    the default is the newest template of that name. A pin is real routing: turns of a
    pinned handle run on that version even after a newer deploy.

    ``warm_pool`` addresses the version's ready pool (dispatch claims idle sandboxes off
    the roster and refills after each turn); ``None`` follows what the deploy recorded.
    The deploy record (written by ``deploy`` under the output bucket) also gives the
    handle the baked spec and the model-credential settings; without it the handle still
    addresses the engine but knows only the spec fields a config overlay tells it.
    """
    if "spec" in kwargs:
        raise TypeError(
            "get_engine() does not accept spec=: bind a SessionConfig at "
            "engine.start_session(config=...) and per-turn overrides via run(config=TurnConfig(...))"
        )
    for gone in ("scoped_gcs", "use_vertex", "model_service_account"):
        if gone in kwargs:
            raise TypeError(
                f"get_engine() got {gone}=: these are deploy-time settings now (the deploy "
                "record carries them); pass them to gemini.deploy()"
            )
    if kwargs:
        raise TypeError(f"get_engine() got unexpected keyword argument(s) {sorted(kwargs)}")
    from ...spec import AgentSpec
    from .handoff import load_deploy_record

    provider = provider or _default_provider(project, location, credentials)
    templates = provider.list_templates(display_name=name)
    if not templates:
        raise LookupError(f"no deployed engine named {name!r} in this project/location")
    if version is None:
        active = [t for t in templates if t.get("state") in (None, "ACTIVE")]
        if not active:
            raise LookupError(
                f"engine {name!r} has no ACTIVE version: "
                + ", ".join(f"{template_id(t['name'])} is {t.get('state')}" for t in templates)
                + ". A FAILED template is a deploy that did not come up; re-run gemini.deploy() "
                "(it creates a new version and retires the failed ones)."
            )
        chosen = active[0]
        if chosen is not templates[0]:
            import warnings

            skipped = [f"{template_id(t['name'])} ({t.get('state')})" for t in templates[: templates.index(chosen)]]
            warnings.warn(
                f"engine {name!r}: newer version(s) {', '.join(skipped)} are not ACTIVE; "
                f"resolving to {template_id(chosen['name'])}",
                stacklevel=2,
            )
    else:
        wanted = template_id(str(version))
        chosen = next((t for t in templates if template_id(t["name"]) == wanted), None)
        if chosen is None:
            raise LookupError(
                f"engine {name!r} has no version {wanted!r}; available (newest first): "
                f"{[template_id(t['name']) for t in templates]}"
            )
    output_bucket = output_bucket or (f"gs://{project}-agent-output" if project else None)
    record = (
        load_deploy_record(output_bucket, name, template_id(chosen["name"]),
                           **_store_kwargs(output_bucket, credentials))
        if output_bucket else None
    ) or {}
    if output_bucket and not record:
        import warnings

        warnings.warn(
            f"engine {name!r} version {template_id(chosen['name'])} has no deploy record under "
            f"{output_bucket}: this handle can only address the template — a turn routed "
            "through Vertex fails at dispatch for lack of a model service account. Usually the "
            "deploy that created this version was interrupted before it wrote the record; re-run "
            "gemini.deploy() for the engine (idempotent: the image and the template are reused, "
            "the record is written).",
            stacklevel=2,
        )
    spec = None
    if isinstance(record.get("spec"), dict):
        try:
            spec = AgentSpec.from_dict(record["spec"])
        except Exception:  # noqa: BLE001 — a record from another toolkit revision
            spec = None
    return GeminiEngine(
        template=chosen["name"],
        spec=spec or _fallback_spec(name),
        project=project,
        location=location,
        output_bucket=output_bucket,
        credentials=credentials,
        provider=provider,
        warm=bool(record.get("warm_pool")) if warm_pool is None else warm_pool,
        pool_max_wait_s=record.get("pool_max_wait_s"),
        max_turn_s=record.get("max_turn_s"),
        use_vertex=bool(record.get("use_vertex", True)),
        model_service_account=record.get("model_service_account"),
        vertex_region=record.get("vertex_region") or "global",
        spec_known=spec is not None,
        pinned=version is not None,
    )


def list_engines(
    project: str, location: str, *, credentials: Any | None = None,
    provider: SandboxProvider | None = None,
) -> list[dict]:
    """Discover deployed engines: one row per template display name, newest template first.

    Each row: ``{"name", "resource" (the newest template), "versions" (count)}``.
    """
    provider = provider or _default_provider(project, location, credentials)
    rows: dict[str, dict] = {}
    for tpl in provider.list_templates():
        row = rows.setdefault(tpl["display_name"], {"name": tpl["display_name"], "resource": tpl["name"], "versions": 0})
        row["versions"] += 1
    return list(rows.values())


# -- run plane -----------------------------------------------------------------


class GeminiSession:
    """A run-plane session over a sandbox engine (CMA-style lifecycle; DESIGN.md §4).

    A session's :class:`~remote_agent_toolkit.config.SessionConfig` is bound ONCE, at
    ``engine.start_session(config=...)`` — persisted at a stable per-session GCS key and
    read back by a process re-attaching by id (``engine.get_session``). A re-attacher can
    never pass a different config: the session's world (repos, skills, prompt) is created
    on its first turn and snapshot-restored on every later one.
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
        self._current_run: DrivenRun | None = None
        self._turn_id: str | None = None
        self._sandbox: str | None = None  # the sandbox the running turn was handed to
        self._refresh_stop: threading.Event | None = None  # stops the GCS token refresher
        self._gcs_token_expiry: str | None = None
        self._model_token_expiry: Any = None  # when the turn's model access ends
        self._session_config = config
        self._session_config_uri: str | None = None
        self._session_config_resolved = config_resolved
        # Set when the worker announced it reads /control (control_ready); reset per turn.
        self._control_ready = False
        # Messages sent into the turn before the worker was ready wait here and are posted,
        # in order, the moment control_ready arrives (``_flush_control``).
        self._control_lock = threading.Lock()
        self._pending_control: list[ControlMessage] = []
        self._flushing = False

    # -- public API ------------------------------------------------------------

    def run(
        self,
        message: str,
        *,
        secrets: dict[str, str] | None = None,
        config: TurnConfig | None = None,
        hooks: Any | None = None,
    ) -> DrivenRun:
        """Start a fresh turn on a sandbox.

        ``secrets`` is a per-invocation name → value map (the agent's own keys, any repo
        ``auth`` / GitHub MCP token) — never baked into the image, never logged; it travels
        in the turn's HTTP body to the worker and nowhere else (nothing is persisted).
        ``config`` is this turn's :class:`~remote_agent_toolkit.config.TurnConfig`. *hooks*
        are rejected: the turn runs in a remote worker and a hook is a callable here.

        Returns at once: claiming or creating the sandbox and handing it the turn happen
        as the first step of the run's driver (a creation takes ~20 s), and a dispatch
        failure ends the run with an explained error result. Raises ``RuntimeError`` while
        a turn is running — use :meth:`send` to talk to it.
        """
        if self.busy:
            raise RuntimeError(
                "this session is running a turn; run() cannot start another one. Use "
                "send() to deliver a message into the running turn, or interrupt() first."
            )
        return self._submit(message, resume=False, secrets=secrets, turn_config=config, hooks=hooks)

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

        On an idle session this resumes the conversation as a new turn (on a fresh sandbox,
        from the checkpoint). Pass ``secrets`` again (not persisted). ``config`` is a
        per-turn :class:`~remote_agent_toolkit.config.TurnConfig`.

        On a RUNNING session the message is posted to the worker's ``/control`` and the
        same :class:`Run` is returned: with ``interrupt=False`` the model sees it at its
        next step, with ``interrupt=True`` the model is interrupted first and continues
        from the message, in the same sandbox and workspace. The ``user`` event on the
        stream, carrying ``message_id``, acknowledges delivery.

        A message sent while the turn is still being dispatched (no sandbox yet, or the
        worker has not announced its control channel) is **queued in this session** and
        posted, in order, the moment the worker announces ``control_ready`` — the same run
        keeps flowing, and the ``user`` event acknowledges it like any other. An
        ``interrupt=True`` message queued that early interrupts the harness right after
        it starts, so the model's first step is the message. If the turn ends before the
        queue drained (dispatch failed, the sandbox died), the messages are dropped and
        ``RunResult.warning`` says how many. Raises
        :class:`~remote_agent_toolkit.control.ControlUnavailable` only when a ready
        worker did not take the message (proxy/platform failure, or the turn just ended).
        """
        if self.busy:
            return self._send_into_running_turn(
                message, interrupt=interrupt, message_id=message_id,
                secrets=secrets, config=config, hooks=hooks,
            )
        return self._submit(message, resume=True, secrets=secrets, turn_config=config, hooks=hooks)

    @property
    def busy(self) -> bool:
        """Whether a turn of this session is running (or started and not yet finished)."""
        return self._current_run is not None and not self._current_run.done

    def _send_into_running_turn(
        self, message: str, *, interrupt: bool, message_id: str | None,
        secrets: dict[str, str] | None, config: TurnConfig | None, hooks: Any | None,
    ) -> DrivenRun:
        if secrets or config is not None or hooks:
            raise ValueError(
                "secrets, config and hooks are bound when a turn starts and cannot change "
                "while it runs; send() into a running turn takes only the message (and "
                "interrupt= / message_id=). Interrupt the session first to start a new turn."
            )
        assert self._current_run is not None
        msg = ControlMessage(
            op="interrupt" if interrupt else "steer", message=message,
            **({"message_id": message_id} if message_id else {}),
        )
        self._post_control(msg)
        return self._current_run

    def _post_control(self, msg: ControlMessage) -> None:
        """Post ``msg`` to the running turn's worker, or queue it until the worker is ready.

        Queued while the turn has no sandbox yet, the worker has not announced
        ``control_ready``, or an earlier queue is still being flushed (so order holds).
        """
        with self._control_lock:
            if not self._control_ready or self._sandbox is None or self._flushing:
                self._pending_control.append(msg)
                return
        self._post_control_now(msg)

    def _post_control_now(self, msg: ControlMessage) -> None:
        sandbox, turn_id = self._sandbox, self._turn_id
        if sandbox is None or turn_id is None:
            raise ControlUnavailable("the turn has no reachable worker (it ended, or was never dispatched)")
        try:
            resp = self._engine._provider().call(
                sandbox, "/control", {"turn_id": turn_id, **msg.to_dict()}, timeout_s=30,
            )
        except Exception as exc:  # noqa: BLE001 — proxy/platform/worker failure alike
            raise ControlUnavailable(f"the turn's sandbox did not take the message: {exc}") from exc
        if not resp.get("ok"):
            raise ControlUnavailable(f"the worker refused the message: {resp.get('error')}")

    def _observe(self, event: AgentEvent) -> None:
        if event.kind == "status" and (event.raw or {}).get("event") == "control_ready":
            with self._control_lock:
                self._control_ready = True
                start = bool(self._pending_control) and not self._flushing
                if start:
                    self._flushing = True
            if start:
                self._engine._in_background(self._flush_control, name="control-flush")

    def _flush_control(self) -> None:
        """Post the queued control messages in order (off the event loop; one at a time)."""
        while True:
            with self._control_lock:
                if not self._pending_control:
                    self._flushing = False
                    return
                msg = self._pending_control[0]
            try:
                self._post_control_now(msg)
            except ControlUnavailable as exc:
                logger.warning("queued message %s for turn %s was not delivered: %s",
                               msg.message_id, self._turn_id, exc)
            with self._control_lock:
                self._pending_control.pop(0)

    def _drop_pending_control(self) -> int:
        """Forget the queued messages (the turn is over or being stopped); how many."""
        with self._control_lock:
            dropped = len(self._pending_control)
            self._pending_control.clear()
            self._flushing = False
        return dropped

    # -- credentials for the turn -------------------------------------------------

    def _mint_tokens(self, turn_id: str) -> tuple[str, dict[str, str]]:
        """Mint the run's GCS token and model token; start refreshing both while the run lives.

        Returns ``(gcs_token, model_env)``. Minting failures raise with the fix spelled out:
        running a turn without the GCS token would leave the worker no way to write its
        record, and without a model token no way to call the model.
        """
        engine = self._engine
        sid = self._session_id
        from .scoped_gcs import mint_run_token, write_run_token

        self._stop_refresh()  # a previous turn's refresher, if any
        try:
            token, expiry = mint_run_token(engine._credentials, engine._output_bucket, sid)
        except Exception as exc:  # noqa: BLE001 — re-raise with the fix spelled out
            raise RuntimeError(
                "could not mint the run-scoped GCS token for this turn "
                f"({type(exc).__name__}: {str(exc)[:200]}). STS rejected the exchange itself; "
                "a generic 'invalid_request' usually means the minted token would exceed "
                "STS's size cap (the caller's own token counts toward it)."
            ) from exc
        write_run_token(engine._output_bucket, sid, token, expiry, **engine._gcs_store_kwargs())
        self._gcs_token_expiry = expiry.isoformat() if expiry else None

        model_env: dict[str, str] = {}
        if engine._use_vertex:
            from .model_token import mint_turn_model_token, vertex_model_env

            if not engine._model_service_account or not engine._project:
                raise RuntimeError(
                    "this engine routes the model through Vertex but its handle has no model "
                    "service account / project (deploy record missing?); redeploy, or pass "
                    "ANTHROPIC_API_KEY as a secret to an engine deployed with use_vertex=False"
                )
            try:
                model_token, model_expiry, extended = mint_turn_model_token(
                    engine._credentials, engine._model_service_account, engine._max_turn_s
                )
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"could not mint the model token by impersonating {engine._model_service_account} "
                    f"({type(exc).__name__}: {str(exc)[:200]}); the caller needs "
                    "roles/iam.serviceAccountTokenCreator on that account (ratk-gcp-setup grants it)"
                ) from exc
            if not extended and not engine._short_token_warned:
                engine._short_token_warned = True
                logger.warning(
                    "model tokens for %s last an hour: a turn longer than that loses model access "
                    "(set constraints/iam.allowServiceAccountCredentialLifetimeExtension for the "
                    "account to mint tokens up to max_turn_s)", engine._model_service_account,
                )
            model_env = vertex_model_env(model_token, engine._project, engine._vertex_region)
            self._model_token_expiry = model_expiry

        stop = threading.Event()
        self._refresh_stop = stop

        def refresh_loop() -> None:
            while not stop.wait(TOKEN_REFRESH_S):
                try:
                    fresh, fresh_expiry = mint_run_token(engine._credentials, engine._output_bucket, sid)
                    write_run_token(engine._output_bucket, sid, fresh, fresh_expiry,
                                    **engine._gcs_store_kwargs())
                except Exception:  # noqa: BLE001 — the worker keeps its current token
                    logger.warning("run-scoped GCS token refresh failed for %s", sid, exc_info=True)

        threading.Thread(target=refresh_loop, name=f"token-refresh-{sid[:8]}", daemon=True).start()
        return token, model_env

    def _stop_refresh(self) -> None:
        self._gcs_token_expiry = None
        stop = self._refresh_stop
        if stop is not None:
            stop.set()
            self._refresh_stop = None
            engine = self._engine
            if engine._output_bucket:
                from .scoped_gcs import delete_run_token

                try:
                    delete_run_token(engine._output_bucket, self._session_id, **engine._gcs_store_kwargs())
                except Exception:  # noqa: BLE001 — the 1-day lifecycle rule backs this up
                    pass

    # -- session config -----------------------------------------------------------

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
                "is persisted there for re-attach); construct the engine with output_bucket/project set."
            )
        from .handoff import persist_session_config

        self._session_config = config
        self._session_config_uri = persist_session_config(
            engine._output_bucket, self._session_id, config.to_dict(), **engine._gcs_store_kwargs(),
        )
        self._session_config_resolved = True

    def _resolve_session_config(self) -> None:
        """Re-attach path: load the opener's persisted config (pointer + dict) from GCS."""
        if self._session_config_resolved:
            return
        engine = self._engine
        if not engine._output_bucket:
            self._session_config_resolved = True
            return
        from ...config import SessionConfig
        from .handoff import load_session_config

        found = load_session_config(engine._output_bucket, self._session_id, **engine._gcs_store_kwargs())
        if found is not None:
            uri, config_dict = found
            self._session_config_uri, self._session_config = uri, SessionConfig.from_dict(config_dict)
        self._session_config_resolved = True

    def _client_spec(self, turn_config: TurnConfig | None = None) -> AgentSpec:
        """The client-side effective spec: engine spec + session + turn overlays (result parsing)."""
        from ...config import apply_session_config, apply_turn_config

        self._resolve_session_config()
        return apply_turn_config(apply_session_config(self._engine.spec, self._session_config), turn_config)

    # -- dispatch -----------------------------------------------------------------

    def _submit(
        self, message: str, resume: bool, secrets: dict[str, str] | None = None,
        turn_config: TurnConfig | None = None, hooks: Any | None = None,
    ) -> DrivenRun:
        engine = self._engine
        sid = self._session_id
        if hooks:
            raise ValueError(
                "run(hooks=...) is local-only: a hook is a callable in this process, and "
                "this turn executes in a remote sandbox. Observe the turn through its event "
                "stream (async for) or Session.history() instead."
            )
        if not engine._output_bucket:
            raise ValueError(
                "running a turn requires the engine's output bucket (its record streams "
                "through it); construct the engine with output_bucket/project set."
            )
        turn_id = uuid.uuid4().hex
        self._resolve_session_config()
        client_spec = self._client_spec(turn_config)  # validates the overlay before anything else
        gcs_token, model_env = self._mint_tokens(turn_id)
        turn_config_dict = turn_config.to_dict() if turn_config is not None and turn_config.set_fields() else None
        turn_config_uri = None
        if turn_config_dict:
            from .handoff import stage_turn_config

            try:
                turn_config_uri = stage_turn_config(
                    engine._output_bucket, sid, turn_config_dict, **engine._gcs_store_kwargs()
                )
            except Exception:  # noqa: BLE001 — the record is best-effort; the body carries the config
                logger.warning("could not record the turn config for %s", sid, exc_info=True)
        body = {
            "turn_id": turn_id,
            "session_id": sid,
            "prompt": message,
            "resume": bool(resume),
            "session_config": self._session_config.to_dict() if self._session_config else None,
            "turn_config": turn_config_dict,
            "session_config_gcs": self._session_config_uri,
            "turn_config_gcs": turn_config_uri,
            "secrets": dict(secrets) if secrets else {},
            "gcs": {"token": gcs_token, "expiry": self._gcs_token_expiry,
                    "output_bucket": engine._output_bucket},
            "model_env": model_env,
        }
        self._turn_id = turn_id
        self._sandbox = None
        self._control_ready = False
        self._drop_pending_control()
        events_uri = f"{engine._output_bucket}/events"
        store_kwargs = engine._gcs_store_kwargs()

        async def source() -> AsyncIterator[AgentEvent]:
            try:
                sandbox = await asyncio.to_thread(self._dispatch, body)
            except Exception as exc:  # noqa: BLE001 — an explained error result, never a hang
                yield _synthetic_result(
                    f"could not hand the turn to a sandbox: {type(exc).__name__}: {str(exc)[:300]}",
                    sid, "dispatch_failed",
                )
                return
            async for event in _stream_turn(
                engine._provider(), sandbox, turn_id, sid,
                events_uri=events_uri, store_kwargs=store_kwargs,
            ):
                yield event

        run = DrivenRun(source, sid, client_spec, on_complete=self._on_complete, on_event=self._observe)
        self._current_run = run
        self._status = RunStatus.RUNNING
        self._stop_reason = None
        try:
            asyncio.get_running_loop()
            run.ensure_started()
        except RuntimeError:
            pass
        return run

    def _dispatch(self, body: dict) -> str:
        """Claim (or create) a sandbox and hand it the turn; return the sandbox name. Sync.

        Every ready-pool entry consumed on the way — the one that took the turn, and any
        that turned out dead (deleted under the roster, OOM, TTL) — is replaced by a refill
        off the critical path, so a batch of stale entries cannot drain the pool to empty
        on a turn that never used it.
        """
        engine = self._engine
        last: Exception | None = None
        consumed = 0  # pool entries taken off the roster by this dispatch
        try:
            for _attempt in range(DISPATCH_ATTEMPTS):
                sandbox, warm = engine._claim_sandbox()
                consumed += int(warm)
                body["sandbox"] = sandbox.rsplit("/", 1)[-1]
                body["warm"] = warm
                try:
                    resp = engine._provider().call(sandbox, "/turn", body, timeout_s=60)
                    if not resp.get("ok"):
                        raise SandboxError(f"the worker refused the turn: {resp.get('error')}")
                except Exception as exc:  # noqa: BLE001 — drop this sandbox, try another
                    last = exc
                    logger.warning("dispatch to %s failed: %s", sandbox, exc)
                    engine._release_sandbox(sandbox)
                    continue
                self._sandbox = sandbox
                return sandbox
            raise RuntimeError(f"no sandbox took the turn after {DISPATCH_ATTEMPTS} attempts: {last}")
        finally:
            if consumed and engine._warm:
                engine._in_background(engine.fill_pool, consumed, name="pool-refill")

    def _on_complete(self, result: RunResult, stop_reason: StopReason) -> None:
        self._last_result = result
        self._stop_reason = stop_reason
        self._status = RunStatus.IDLE
        self._stop_refresh()
        dropped = self._drop_pending_control()
        if dropped:
            note = (
                f"{dropped} message(s) sent into this turn were never delivered: the turn "
                "ended before its worker took control messages"
            )
            result.warning = f"{result.warning}; {note}" if result.warning else note
            logger.warning("session %s: %s", self._session_id, note)
        sandbox, self._sandbox = self._sandbox, None
        if sandbox is not None:
            # A sandbox bills while it exists; the turn is over, so it goes. Multi-turn
            # continuity rides the checkpoint tar (DESIGN.md §13.1).
            self._engine._in_background(self._engine._release_sandbox, sandbox, name="sandbox-release")

    async def interrupt(self, *, timeout: float = 120.0) -> None:
        """Interrupt the running turn and leave the session idle and resumable.

        Once the worker announced its control channel (``control_ready`` on the stream) a
        ``stop`` is posted to it: the worker interrupts the harness and runs its normal
        end-of-turn path — workspace snapshot, transcript, a terminal result with
        ``StopReason.INTERRUPTED`` that keeps the turn's accounting — and this returns once
        that result arrives. A later :meth:`send` resumes from that checkpoint. No-op when
        nothing is running.

        Fallback: when the worker is not reachable yet (the turn is still being dispatched)
        or does not end the turn within ``timeout`` seconds, the run is cancelled — it ends
        as an error result with no checkpoint, ``RunResult.warning`` says so — and the
        sandbox is deleted (which stops the agent and its billing).
        """
        run = self._current_run
        if run is None or run.done:
            return
        run.ensure_started()
        assert run.task is not None
        note: str | None
        if self._control_ready and self._sandbox is not None:
            self._drop_pending_control()  # a stop makes queued steers moot; it goes straight out
            try:
                await asyncio.to_thread(self._post_control_now, ControlMessage(op="stop"))
                await asyncio.wait_for(asyncio.shield(run.task), timeout)
                return
            except asyncio.TimeoutError:
                note = (
                    f"the worker did not end the turn within {timeout:g}s of the stop; "
                    "the run was cancelled without a checkpoint and the sandbox deleted"
                )
            except asyncio.CancelledError:
                return
            except ControlUnavailable as exc:
                note = f"the stop could not be delivered ({exc}); the run was cancelled without a checkpoint"
        else:
            note = (
                "the turn's worker had not announced its control channel yet (sandbox still "
                "being dispatched or starting); the run was cancelled without a checkpoint"
            )
        run.cancel(note=note)
        try:
            await run.task
        except asyncio.CancelledError:
            pass

    # -- records ------------------------------------------------------------------

    def history(self) -> list:
        """All persisted events of this session, oldest first (the GCS event mirror)."""
        from .history import read_history

        engine = self._engine
        return read_history(engine._output_bucket, self._session_id, credentials=engine._credentials)

    async def transcripts(self) -> dict[str, list[dict]]:
        """This session's persisted harness transcripts (see ``runtime.base.Session``)."""
        from ...checkpoint.session_store import BlobSessionStore, _claude_session_id
        from ...ports.blobstore import GcsBlobStore, parse_gcs_uri

        engine = self._engine
        if not engine._output_bucket:
            raise RuntimeError(
                "transcripts() reads the engine's checkpoint prefix under its output "
                "bucket; construct the engine with output_bucket/project set."
            )
        spec = self._client_spec()
        if engine._spec_known and not (spec.checkpoint or spec.transcript):
            raise RuntimeError(
                "no transcript is persisted for this session — deploy the spec with "
                "AgentSpec(transcript=True)"
            )
        bucket, prefix = parse_gcs_uri(f"{engine._output_bucket}/checkpoints")
        blobs = GcsBlobStore(bucket, (prefix + "/") if prefix else "",
                             **({"credentials": engine._credentials} if engine._credentials is not None else {}))
        return await BlobSessionStore(blobs).load_all(_claude_session_id(self._session_id))

    @property
    def status(self) -> RunStatus:
        return self._status

    @property
    def stop_reason(self) -> StopReason | None:
        return self._stop_reason

    @property
    def last_result(self) -> RunResult | None:
        """The most recent run's result — reconstructed from history for re-attached sessions."""
        if self._last_result is None and self._current_run is None:
            events = self.history()
            result_ev = next((e for e in reversed(events) if e.kind == "result"), None)
            if result_ev is not None:
                from .._run import build_result

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
            "the agent's workspace lives on the sandbox's filesystem "
            "(/workspace/jobs/<claude-sid>/workspace) — seed inputs via the prompt or "
            "spec.repos, and collect outputs via events/history or a repo push."
        )

    def fork(self) -> GeminiSession:
        raise NotImplementedError(
            "fork is not supported yet (would copy the workspace snapshot + transcript under a new id)."
        )


class GeminiEngine:
    """A deployed sandbox engine handle (implements ``runtime.base.Engine``; DESIGN.md §4).

    ``template`` is the version this handle's turns run on; ``spec`` is the deploy-baked
    spec when known (a deploy handle, or a looked-up one with its deploy record) or an
    addressing-only fallback flagged by ``spec_known=False``.
    """

    def __init__(
        self,
        template: str,
        spec: AgentSpec,
        project: str | None,
        location: str | None,
        output_bucket: str | None = None,
        credentials: Any | None = None,
        provider: SandboxProvider | None = None,
        warm: bool = False,
        pool_max_wait_s: float | None = None,
        max_turn_s: float | None = None,
        use_vertex: bool = True,
        model_service_account: str | None = None,
        vertex_region: str = "global",
        spec_known: bool = True,
        pinned: bool = False,
        roster_store: Any | None = None,
    ) -> None:
        self._template = template
        self.spec = spec
        self._spec_known = spec_known
        self._project = project
        self._location = location
        self._output_bucket = output_bucket
        self._credentials = credentials
        self._provider_obj = provider
        self._warm = warm
        self._pool_max_wait_s = float(pool_max_wait_s) if pool_max_wait_s else DEFAULT_MAX_WAIT_S
        self._max_turn_s = float(max_turn_s) if max_turn_s else DEFAULT_MAX_TURN_S
        self._use_vertex = use_vertex
        self._model_service_account = model_service_account
        self._vertex_region = vertex_region
        self._pinned = pinned
        self._roster_store = roster_store
        self._short_token_warned = False
        self._sessions: dict[str, GeminiSession] = {}
        self._roster_cached: Any = None
        self._background: list[threading.Thread] = []

    # -- plumbing -------------------------------------------------------------------

    def _provider(self) -> SandboxProvider:
        if self._provider_obj is None:
            self._provider_obj = _default_provider(self._project, self._location, self._credentials)
        return self._provider_obj

    def _gcs_store_kwargs(self, uri: str | None = None) -> dict:
        """Inject this client's explicit identity into the GCS record helpers (``{}`` for ADC)."""
        return _store_kwargs(uri or self._output_bucket, self._credentials)

    def _in_background(self, fn: Any, *args: Any, name: str = "housekeeping") -> None:
        """Run housekeeping (refill, release, retire) off a turn's critical path.

        Non-daemon, so a short-lived process still completes it before exiting: a released
        sandbox must actually be deleted or it bills until its TTL. Failures are logged.
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
        """Wait for outstanding housekeeping (tests; teardown)."""
        for thread in list(self._background):
            thread.join(timeout)

    def _roster(self) -> Any:
        if self._roster_cached is None:
            from .roster import PoolRoster

            if not self._output_bucket:
                raise ValueError(
                    "the ready pool's roster lives in the engine's output bucket; construct "
                    "the engine with output_bucket/project set."
                )
            self._roster_cached = PoolRoster(
                self._output_bucket, self._template, credentials=self._credentials,
                store=self._roster_store,
            )
        return self._roster_cached

    def _display_prefix(self) -> str:
        return f"ratk-{_slug(self.name)}-"

    # -- sandboxes ------------------------------------------------------------------

    def _create_sandbox(self, ttl_s: float) -> Any:
        """Create a sandbox from this version and wait until the worker answers ``/health``."""
        provider = self._provider()
        handle = provider.create(
            self._template, ttl_s=ttl_s, display_name=f"{self._display_prefix()}{uuid.uuid4().hex[:8]}",
        )
        deadline = time.monotonic() + READY_WAIT_S
        last: str = ""
        while time.monotonic() < deadline:
            try:
                if provider.call(handle.name, "/health", {}, timeout_s=30).get("ok"):
                    return handle
            except SandboxGone as exc:
                raise SandboxError(f"the new sandbox disappeared before answering: {exc}") from exc
            except Exception as exc:  # noqa: BLE001 — the proxy is not routing yet
                last = f"{type(exc).__name__}: {str(exc)[:120]}"
            time.sleep(1.0)
        self._release_sandbox(handle.name)
        raise SandboxError(f"sandbox {handle.id} never answered /health within {READY_WAIT_S:.0f}s: {last}")

    def _release_sandbox(self, sandbox: str) -> None:
        """Delete a sandbox (best-effort, idempotent): the cancel, and the end of every turn."""
        try:
            self._provider().delete(sandbox)
        except Exception:  # noqa: BLE001 — the TTL backs this up
            logger.warning("could not delete sandbox %s", sandbox, exc_info=True)

    def fill_pool(self, n: int) -> None:
        """Create ``n`` ready sandboxes for this version and record them in the roster.

        Each is created with a TTL of ``pool_max_wait_s`` (its idle life) plus ``max_turn_s``
        (so a turn claimed at the end of the idle life still completes), health-checked
        until the proxy routes to it (~12–33 s), then recorded as idle. Creation runs in
        parallel. Use it to top a pool back up — e.g. after reusing an engine via
        ``get_engine`` whose sandboxes idled out — or to grow the pool.
        """
        if n <= 0:
            return
        ttl = self._pool_max_wait_s + self._max_turn_s

        def one() -> None:
            handle = self._create_sandbox(ttl)
            entry = SandboxEntry(
                sandbox=handle.name, template=self._template,
                created_at=handle.created_at, expires_at=handle.created_at + self._pool_max_wait_s,
            )
            try:
                self._roster().add(entry)
            except Exception:
                self._release_sandbox(handle.name)  # a sandbox nobody can find bills for nothing
                raise

        if n == 1:
            one()
            return
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(n, 8)) as ex:
            futures = [ex.submit(one) for _ in range(n)]
            errors = [f.exception() for f in futures if f.exception() is not None]
        if errors:
            raise errors[0]

    def _claim_sandbox(self) -> tuple[str, bool]:
        """``(sandbox, from_pool)``: an idle ready sandbox off the roster, else a fresh one.

        Expired entries met on the way are pruned and their sandboxes deleted (off the
        critical path). An empty roster — or a non-pool engine — creates a sandbox for
        this turn alone (~20 s until it answers).
        """
        if self._warm and self._output_bucket:
            entry, expired = self._roster().claim(margin_s=CLAIM_MARGIN_S)
            for stale in expired:
                self._in_background(self._release_sandbox, stale.sandbox, name="pool-prune")
            if entry is not None:
                return entry.sandbox, True
        return self._create_sandbox(self._max_turn_s).name, False

    def wait_until_warm(self, timeout: float = 600.0) -> bool:
        """Block until the ready pool has a live sandbox (or ``timeout``); ``True`` if warm.

        The roster only ever records sandboxes whose worker answered, so one live entry
        means a turn dispatched now starts within a second. No-op (``True``) for a
        non-pool engine.
        """
        if not self._warm:
            return True
        deadline = time.monotonic() + timeout
        while True:
            try:
                live = [e for e in self._roster().entries() if e.expires_at > time.time() + CLAIM_MARGIN_S]
            except Exception:  # noqa: BLE001 — a listing blip is not "cold"
                live = []
            if live and self._prune_dead(live):
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(2.0)

    def _prune_dead(self, entries: list[SandboxEntry]) -> list[SandboxEntry]:
        """The entries whose worker still answers ``/health``; the others leave the roster.

        A rostered sandbox can be gone without the roster knowing (deleted by another
        action, OOM-killed, the platform reclaimed it): reporting such an entry as "ready"
        would be a lie, so the ones that do not answer are dropped and deleted best-effort.
        """
        alive: list[SandboxEntry] = []
        for entry in entries:
            try:
                ok = self._provider().call(entry.sandbox, "/health", {}, timeout_s=15).get("ok")
            except Exception as exc:  # noqa: BLE001 — gone, or the proxy cannot reach it
                logger.warning("rostered sandbox %s does not answer (%s); dropping it", entry.id, exc)
                ok = False
            if ok:
                alive.append(entry)
                continue
            try:
                self._roster().remove(entry.id)
            except Exception:  # noqa: BLE001
                logger.warning("could not remove %s from the roster", entry.id, exc_info=True)
            self._in_background(self._release_sandbox, entry.sandbox, name="pool-prune")
        return alive

    def _retire_templates(self, templates: list[str], timeout: float = 600.0) -> None:
        """Drop previous versions: their rostered sandboxes, then the templates (best-effort)."""
        from .roster import PoolRoster

        provider = self._provider()
        for tpl in templates:
            if self._output_bucket:
                try:
                    roster = PoolRoster(self._output_bucket, tpl, credentials=self._credentials,
                                        store=self._roster_store)
                    for entry in roster.clear():
                        self._release_sandbox(entry.sandbox)
                except Exception:  # noqa: BLE001
                    logger.warning("could not clear the roster of %s", tpl, exc_info=True)
        deadline = time.monotonic() + timeout
        pending = list(templates)
        while pending and time.monotonic() < deadline:
            for tpl in list(pending):
                try:
                    provider.delete_template(tpl)
                    pending.remove(tpl)
                except Exception:  # noqa: BLE001 — sandboxes still alive under it: retry
                    pass
            if pending:
                time.sleep(15.0)
        for tpl in pending:
            logger.warning("template %s still has sandboxes; not deleted (retry via delete_version)", tpl)

    # -- Engine protocol -------------------------------------------------------------

    def start_session(self, config: SessionConfig | None = None) -> GeminiSession:
        """Begin a new session; ``config`` binds its :class:`SessionConfig` for good.

        Session ids are client-minted canonical UUIDs (the Claude Agent SDK requires that
        shape for checkpoint keying / resume).
        """
        if config is not None and config.set_fields() and not self._output_bucket:
            raise ValueError(
                "start_session(config=...) requires the engine's output bucket (the config "
                "is persisted there for re-attach); construct the engine with output_bucket/project set."
            )
        session = GeminiSession(self, str(uuid.uuid4()))
        session._bind_config(config)
        self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> GeminiSession:
        """Re-attach to an existing session by id (poll / continue). Takes NO config on purpose."""
        return self._sessions.get(session_id) or GeminiSession(self, session_id, config_resolved=False)

    def list_sessions(self) -> list[dict]:
        """Sessions recorded under the output bucket (the event mirror), newest first.

        NB: the mirror is bucket-wide, so with a shared output bucket the sessions of
        other engines appear too.
        """
        from .history import list_sessions as _list

        return _list(output_bucket=self._output_bucket, **self._gcs_store_kwargs())

    def delete(self, *, timeout: float = 600.0) -> None:
        """Tear down the engine (ops action): its sandboxes, then every version's template.

        Deletes the rostered ready sandboxes of every version, sweeps any other sandbox
        created from this engine's templates under the host instance (orphans of crashed
        clients), then deletes the templates — retrying up to ``timeout`` while the
        platform still sees sandboxes under them.

        The sweep matches on the sandbox's **template**, not on its display name: the
        name of engine ``x`` is a prefix of the names of engine ``x-<revision>`` (the
        pattern the README recommends for side-by-side toolkit revisions), and tearing
        ``x`` down must leave ``x-<revision>``'s ready pool alone.
        """
        provider = self._provider()
        templates = [t["name"] for t in provider.list_templates(display_name=self.name)]
        if self._template not in templates:
            templates.append(self._template)
        owned = {template_id(t) for t in templates}
        try:
            for sb in provider.list(display_prefix=self._display_prefix()):
                if not self._owns_sandbox(sb, owned):
                    continue
                self._release_sandbox(sb["name"])
        except Exception:  # noqa: BLE001 — the roster sweep below still runs
            logger.warning("could not list sandboxes of %s", self.name, exc_info=True)
        self._retire_templates(templates, timeout=timeout)

    def _owns_sandbox(self, sandbox: dict, owned_templates: set[str]) -> bool:
        """Whether a listed sandbox (display name already under our prefix) is this engine's."""
        template = sandbox.get("template")
        if template:
            return template_id(template) in owned_templates
        # No template on the row: fall back to the exact shape of our names, ``<prefix><8 hex>``.
        display = sandbox.get("display_name") or ""
        return _SANDBOX_NAME_RE.fullmatch(display[len(self._display_prefix()):]) is not None

    def versions(self) -> list[str]:
        """This engine's template ids, newest first."""
        return [r["version"] for r in self.revisions()]

    def revisions(self) -> list[dict]:
        """This engine's templates, newest first: ``version``, ``resource``, ``image``,
        ``create_time``, ``state``, ``cpu``, ``memory`` and ``current`` (this handle's)."""
        rows = []
        for tpl in self._provider().list_templates(display_name=self.name):
            rows.append({
                "version": template_id(tpl["name"]),
                "resource": tpl["name"],
                "image": tpl.get("image_uri"),
                "create_time": tpl.get("create_time"),
                "state": tpl.get("state"),
                "cpu": tpl.get("cpu"),
                "memory": tpl.get("memory"),
                "current": tpl["name"] == self._template,
            })
        return rows

    def delete_version(self, version: str | int) -> None:
        """Delete one version (template) and its rostered sandboxes; the current one included
        only if you really mean it (the handle then addresses a deleted template)."""
        wanted = template_id(str(version))
        for tpl in self._provider().list_templates(display_name=self.name):
            if template_id(tpl["name"]) == wanted:
                self._retire_templates([tpl["name"]], timeout=300.0)
                return
        raise LookupError(f"engine {self.name!r} has no version {wanted!r}")

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def version(self) -> str:
        """The template id this handle's turns run on."""
        return template_id(self._template)

    @property
    def resource(self) -> str:
        """The template's full resource name."""
        return self._template

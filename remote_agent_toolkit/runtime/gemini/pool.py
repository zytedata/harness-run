"""Warm-pool helpers (DESIGN.md §6).

The way to get fast *and* long on Gemini Agent Runtime: pre-warmed async jobs blocked on an
inbound channel. A job submitted with the ``__POOL_WAIT__`` sentinel becomes a worker that
blocks pulling **its own** Pub/Sub subscription (via the ``DispatchTransport`` port). The
control plane picks one idle worker from the pool's roster (``roster.py``) and publishes
the turn payload addressed to it; that worker processes it as a normal turn — under the
**client-chosen ``session_id``**, so the client tails the event mirror by that id exactly
as on the cold path, and multi-turn continuity rides on checkpoint/resume across workers.
On claim the pool is refilled.

**Per-worker dispatch** (README "The runtime identity is reachable by the agent"): every
worker of a generation shares one topic, but each has its own subscription with a random,
unguessable name and a filter that only matches messages addressed to it. The runtime
identity's ``roles/pubsub.subscriber`` grants consume by name only (no list, no get), so a
shell in worker X can pull X's channel and nothing else — and X's channel only ever carries
the one turn the client dispatched to X. The earlier design (one shared subscription per
generation, competing consumers) let a rogue shell in one worker pull another run's turn,
pointers and run-scoped token included; that mode survives only as the legacy path for
engines deployed before this change (``AGENT_POOL_SUBSCRIPTION`` in their env).

This module is pure helpers (stdlib only at import). The worker run-loop lives in
``adk_agent`` (it must yield ADK events); the control-plane publish/provision lives in
``backend``; the idle-worker roster in ``roster``.
"""

from __future__ import annotations

import os
import re
import secrets
import time
import uuid
from typing import Any

# A worker submitted with this sentinel prompt waits for a real assignment via dispatch.
POOL_WAIT_SENTINEL = "__POOL_WAIT__"

# The job-input directive that hands a pool worker its own subscription (per-worker
# dispatch). It rides the worker's job input, not the engine env: the subscription is
# per worker, and its name is the capability that scopes what the worker can pull.
_POOL_SUBSCRIPTION_DIRECTIVE = re.compile(
    r"^\s*AGENT_POOL_SUBSCRIPTION=(\S+)[ \t]*$", re.IGNORECASE | re.MULTILINE
)


# Default idle life of a warm worker. A worker that idle-expires is NOT replaced (the only
# refill is claim-driven), so a short default silently drained pools that went quiet — a day
# keeps a pool warm across working-hours gaps while still bounding runaway idle billing.
# Query jobs support runs of up to 7 days (DESIGN.md §6), so a day is comfortably within
# platform limits. Deploy-time override: ``gemini.deploy(..., pool_max_wait_s=...)``.
DEFAULT_MAX_WAIT_S = 24 * 3600.0

# Per-worker subscription housekeeping. A subscription is deleted by the client once its
# worker has picked up its turn (or when the pool is retired); these are the backstops for
# a client that died in between. ``expiration_policy.ttl`` counts from the last subscriber
# activity (an idle worker pulls continuously, so a waiting worker never loses its
# channel); Pub/Sub's minimum is a day. Retention bounds how long an undelivered turn can
# sit on a dead worker's channel; Pub/Sub's minimum is 10 minutes, and a worker boots in
# ~2.5–5 min, so a turn dispatched to a still-booting worker is safely inside it.
WORKER_SUBSCRIPTION_TTL_S = 24 * 3600
WORKER_SUBSCRIPTION_RETENTION_S = 10 * 60

# Pickup watchdog (backend ``_pickup_watched_tail``): how long the client waits for the
# worker it dispatched to before treating it as dead and re-dispatching to another one.
# A worker's cold boot is ~2.5 min and has been seen at 5; a booted worker's pickup is a
# long-poll return plus the turn's first mirrored event, a few seconds.
WORKER_BOOT_S = 300.0
PICKUP_GRACE_S = 90.0


def resolve_max_wait_s() -> float:
    """How long a warm worker waits for an assignment before exiting (bounds idle job life).

    Read in the WORKER process from ``AGENT_POOL_MAX_WAIT_S``, which ``build_env()`` bakes
    into the engine when ``deploy(..., pool_max_wait_s=...)`` is set; otherwise the default.
    """
    return float(os.environ.get("AGENT_POOL_MAX_WAIT_S", str(DEFAULT_MAX_WAIT_S)))


def pickup_deadline_s(submitted_at: float, now: float | None = None) -> float:
    """Seconds to wait for a worker submitted at ``submitted_at`` to start the turn.

    A worker still inside its boot window gets the rest of that window plus the grace; a
    worker that should long be warm gets the grace alone.
    """
    now = time.time() if now is None else now
    return max(0.0, submitted_at + WORKER_BOOT_S - now) + PICKUP_GRACE_S


def _safe(name: str) -> str:
    """Sanitize an agent name into a Pub/Sub-resource-safe slug."""
    slug = re.sub(r"[^a-zA-Z0-9-]", "-", name).strip("-").lower()
    return slug or "agent"


def new_generation() -> str:
    """A fresh pool *generation* token, minted once per warm-pool deploy.

    Scoping the dispatch topic (and every worker subscription under it) by generation is
    what cuts a redeploy over atomically: workers cold-started under the previous revision
    keep pulling the OLD generation's channels (which the deploy then retires), so they can
    never claim a turn published for the new revision (issue #38).
    """
    return uuid.uuid4().hex[:8]


def new_worker_id() -> str:
    """A fresh, unguessable worker id: the capability that names a worker's subscription."""
    return secrets.token_hex(12)


def pool_paths(project: str, name: str, generation: str | None = None) -> tuple[str, str]:
    """``(topic, subscription)`` resource paths of a **shared-subscription** pool (legacy).

    With ``generation`` the pair is scoped to a deploy: ``ratk-<slug>-<generation>-dispatch``
    plus the shared ``...-dispatch-sub`` every worker of that generation pulled. Without it,
    the legacy fixed name shared by all revisions. Both survive only to address engines
    deployed before per-worker dispatch; new deploys use :func:`pool_topic` and one
    :func:`worker_subscription` per worker.
    """
    topic = pool_topic(project, name, generation)
    return topic, f"projects/{project}/subscriptions/{topic.rsplit('/', 1)[-1]}-sub"


def pool_topic(project: str, name: str, generation: str | None = None) -> str:
    """The dispatch topic of an engine's pool: ``ratk-<slug>[-<generation>]-dispatch``."""
    slug = _safe(name) if generation is None else f"{_safe(name)}-{generation}"
    return f"projects/{project}/topics/ratk-{slug}-dispatch"


def topic_for_subscription(subscription: str) -> str:
    """The dispatch topic path paired with a legacy shared ``subscription``.

    Lets the control plane reconstruct the pair from ``AGENT_POOL_SUBSCRIPTION`` alone —
    the one value baked into an engine deployed before per-worker dispatch.
    """
    project = subscription.split("/")[1]
    base = subscription.rsplit("/", 1)[-1]
    if base.endswith("-sub"):
        base = base[: -len("-sub")]
    return f"projects/{project}/topics/{base}"


def _pool_base(topic: str) -> str:
    """``ratk-<slug>-<generation>`` from a dispatch topic path."""
    base = topic.rsplit("/", 1)[-1]
    if base.endswith("-dispatch"):
        base = base[: -len("-dispatch")]
    return base


def worker_subscription_prefix(topic: str) -> str:
    """The resource-path prefix every worker subscription of ``topic``'s pool starts with."""
    project = topic.split("/")[1]
    return f"projects/{project}/subscriptions/{_pool_base(topic)}-w-"


def worker_subscription(topic: str, worker_id: str) -> str:
    """The per-worker subscription path: ``ratk-<slug>-<generation>-w-<worker_id>``.

    The random ``worker_id`` is the capability: ``roles/pubsub.subscriber`` can consume a
    subscription by name but neither list nor get them, so knowing one's own name (from
    the job input) grants nothing about any other worker's channel.
    """
    return f"{worker_subscription_prefix(topic)}{worker_id}"


def worker_id_from_subscription(subscription: str) -> str | None:
    """The worker id a per-worker subscription path was minted for (``None`` for a legacy one)."""
    base = subscription.rsplit("/", 1)[-1]
    _, sep, worker = base.rpartition("-w-")
    return worker if sep and worker else None


def worker_filter(worker_id: str) -> str:
    """The Pub/Sub subscription filter that admits only messages addressed to ``worker_id``."""
    return f'attributes.worker = "{worker_id}"'


def worker_attributes(worker_id: str) -> dict[str, str]:
    """The message attributes that address a dispatch to one worker (see :func:`worker_filter`)."""
    return {"worker": worker_id}


def roster_prefix(topic: str) -> str:
    """Bucket-relative key prefix of the pool's idle-worker roster (``roster.py``).

    SECURITY: this prefix must be writable by the client identity only — never by the
    runtime identity (its output-bucket bindings cover ``jobs/`` and ``events/ratk-``,
    not ``pool/``). The roster decides which worker a turn is dispatched to, so a worker
    that could add itself back to it could receive another run's turn.
    """
    return f"pool/{_pool_base(topic)}/idle/"


def pool_wait_prompt(subscription: str) -> str:
    """The job input of a pool worker: the sentinel plus its own subscription directive."""
    return f"{POOL_WAIT_SENTINEL}\nAGENT_POOL_SUBSCRIPTION={subscription}\n"


def parse_pool_wait(prompt: str) -> tuple[bool, str | None]:
    """``(is_pool_worker, subscription)`` from a job input.

    A bare sentinel (an old client's ``fill_pool``) is still recognised as a pool worker,
    with no subscription — the worker then reports the misconfiguration and exits.
    """
    if not prompt.strip().startswith(POOL_WAIT_SENTINEL):
        return False, None
    m = _POOL_SUBSCRIPTION_DIRECTIVE.search(prompt)
    return True, (m.group(1) if m else None)


def dispatch_payload(
    session_id: str,
    message: str,
    resume: bool,
    secrets_gcs: str | None = None,
    session_config_gcs: str | None = None,
    turn_config_gcs: str | None = None,
    gcs_token: str | None = None,
) -> dict:
    """The turn payload published to the pool: the worker adopts ``session_id`` for the turn.

    SECURITY: the payload never carries secret *values* — only ``secrets_gcs``, the gs://
    pointer to the staged object the worker fetches (and deletes once the turn completes —
    a redelivered dispatch must still find it; ``handoff.py``). A Pub/Sub message is
    retained until acked, so values in it would persist.

    ``session_config_gcs`` / ``turn_config_gcs`` point at the session's persisted
    ``SessionConfig`` and this turn's ``TurnConfig`` (``handoff.py``): the worker overlays
    them on its deploy-baked spec and runs the result. Same pointer-not-value shape — the
    configs are not secret, but the pointer keeps payloads small and matches the cold
    path. Workers older than these fields ignore them (they run the baked spec), which is
    why client and engine must deploy from the same toolkit revision.
    """
    payload = {"session_id": session_id, "message": message, "resume": bool(resume)}
    if secrets_gcs:
        payload["secrets_gcs"] = secrets_gcs
    if session_config_gcs:
        payload["session_config_gcs"] = session_config_gcs
    if turn_config_gcs:
        payload["turn_config_gcs"] = turn_config_gcs
    if gcs_token:
        # The run-scoped GCS token (scoped_gcs.py): a bearer token worth this run's own
        # objects for at most an hour. It is the one credential-like value in the payload,
        # and it is what lets the worker stop using the runtime identity for GCS. With
        # per-worker dispatch the message lands only on the channel of the one worker the
        # client picked for this turn; a shell in any other worker cannot pull it.
        payload["gcs_token"] = gcs_token
    return payload


def pool_log_id(name: str) -> str:
    """A stable Cloud Logging ``session_id`` an idle worker emits readiness under.

    Idle workers have no real session, so they tag a "pool ready" marker with this id; the
    control plane tails it to learn the pool is warm (see ``GeminiEngine.wait_until_warm``).
    """
    return f"ratk-{_safe(name)}-pool"


def pool_log_id_from_topic(topic: str) -> str:
    """The generation-scoped :func:`pool_log_id` derived from the pool's dispatch topic.

    The topic is ``.../ratk-<slug>-<generation>-dispatch``; strip to
    ``ratk-<slug>-<generation>-pool``, so client and worker (``AGENT_POOL_TOPIC``) agree
    on the readiness key without the worker needing the agent name, and a fresh deploy
    never counts a previous deploy's markers as its own.
    """
    return f"{_pool_base(topic)}-pool"


def pool_log_id_from_subscription(subscription: str) -> str:
    """:func:`pool_log_id_from_topic` for a legacy shared ``AGENT_POOL_SUBSCRIPTION``."""
    return pool_log_id_from_topic(topic_for_subscription(subscription))


def worker_dispatch(subscription: str, credentials: Any | None = None):
    """Build a claim-only ``DispatchTransport`` for a pool worker on its own subscription."""
    from ...ports.dispatch import PubSubDispatch

    return PubSubDispatch(subscription=subscription, credentials=credentials)

"""Warm-pool helpers (DESIGN.md §6).

The way to get fast *and* long on Gemini Agent Runtime: pre-warmed async jobs blocked on an
inbound channel. A job submitted with the ``__POOL_WAIT__`` sentinel becomes a worker that
blocks pulling a shared Pub/Sub subscription (competing-consumers = atomic claim, via the
``DispatchTransport`` port). The control plane publishes a turn payload; one warm worker
claims it and processes it as a normal turn — under the **client-chosen ``session_id``**, so
the client tails Cloud Logging by that id exactly as on the cold path, and multi-turn
continuity rides on checkpoint/resume across workers. On claim the pool is refilled.

This module is pure helpers (stdlib only at import). The worker run-loop lives in
``adk_agent`` (it must yield ADK events); the control-plane publish/provision lives in
``backend``.
"""

from __future__ import annotations

import os
import re
from typing import Any

# A worker submitted with this sentinel prompt waits for a real assignment via dispatch.
POOL_WAIT_SENTINEL = "__POOL_WAIT__"


def resolve_max_wait_s() -> float:
    """How long a warm worker waits for an assignment before exiting (bounds idle job life)."""
    return float(os.environ.get("AGENT_POOL_MAX_WAIT_S", "1800"))


def _safe(name: str) -> str:
    """Sanitize an agent name into a Pub/Sub-resource-safe slug."""
    slug = re.sub(r"[^a-zA-Z0-9-]", "-", name).strip("-").lower()
    return slug or "agent"


def pool_paths(project: str, name: str) -> tuple[str, str]:
    """``(topic, subscription)`` resource paths for an engine's warm-pool dispatch."""
    base = f"ratk-{_safe(name)}-dispatch"
    return (
        f"projects/{project}/topics/{base}",
        f"projects/{project}/subscriptions/{base}-sub",
    )


def dispatch_payload(session_id: str, message: str, resume: bool) -> dict:
    """The turn payload published to the pool: the worker adopts ``session_id`` for the turn."""
    return {"session_id": session_id, "message": message, "resume": bool(resume)}


def pool_log_id(name: str) -> str:
    """A stable Cloud Logging ``session_id`` an idle worker emits readiness under.

    Idle workers have no real session, so they tag a "pool ready" marker with this id; the
    control plane tails it to learn the pool is warm (see ``GeminiEngine.wait_until_warm``).
    """
    return f"ratk-{_safe(name)}-pool"


def pool_log_id_from_subscription(subscription: str) -> str:
    """Derive the same :func:`pool_log_id` from ``AGENT_POOL_SUBSCRIPTION`` (worker side).

    The subscription is ``.../ratk-<slug>-dispatch-sub``; strip to ``ratk-<slug>-pool`` so it
    matches ``pool_log_id(name)`` without the worker needing the agent name.
    """
    base = subscription.rsplit("/", 1)[-1]
    if base.endswith("-dispatch-sub"):
        base = base[: -len("-dispatch-sub")]
    return f"{base}-pool"


def worker_dispatch_from_env(credentials: Any | None = None):
    """Build a claim-only ``DispatchTransport`` for a pool worker, from the env.

    Returns ``None`` when ``AGENT_POOL_SUBSCRIPTION`` is unset (not a pool worker).
    """
    sub = os.environ.get("AGENT_POOL_SUBSCRIPTION")
    if not sub:
        return None
    from ...ports.dispatch import PubSubDispatch

    return PubSubDispatch(subscription=sub, credentials=credentials)

"""Runtime revisions — the platform's version axis for a deployed engine (DESIGN.md §5).

Agent Runtime versions a ``reasoningEngine`` by **runtime revision**
(``.../reasoningEngines/{id}/runtimeRevisions/{rev}``): every ``agent_engines.update`` that
changes the deployment mints a new immutable revision, and the engine's ``traffic_config``
decides which revision actually serves. This module maps the toolkit's ``version``
vocabulary (``Engine.versions()`` / ``Engine.version`` / ``get_engine(version=…)``) onto
exactly that, so ``deploy`` mints versions instead of piling up look-alike engines.

**The platform contract that shapes the API here:** only the *engine* exposes
``asyncQuery`` — a revision exposes ``query`` / ``streamQuery`` but **not** ``asyncQuery``
(v1beta1 discovery, checked 2026-08). The toolkit's entire run plane is ``run_query_job``
(= ``asyncQuery``), so a run can only ever land on whichever revision the engine's traffic
config points at. Two consequences, both deliberate:

* ``get_engine(name, version=…)`` is an **assertion**, not a routing choice: it verifies the
  revision exists *and* is the one serving, then labels the handle with it. Two callers
  cannot address two different revisions of one engine concurrently.
* Moving traffic is an explicit ops action (``GeminiEngine.set_traffic``) — a rollback or a
  promote, engine-wide.

Stdlib-only at import time; the Google SDK is only ever touched through the ``client``
object callers pass in.
"""

from __future__ import annotations

from typing import Any

_SEGMENT = "/runtimeRevisions/"


def revision_id(name: str) -> str:
    """The bare revision id from a full revision resource name (id in → id out)."""
    return name.rsplit("/", 1)[-1] if _SEGMENT in name else name


def revision_resource(engine_resource: str, version: str) -> str:
    """Full revision resource path for ``version`` (an id, or an already-full path)."""
    return version if _SEGMENT in version else f"{engine_resource}{_SEGMENT}{version}"


def list_revisions(client: Any, engine_resource: str) -> list[dict]:
    """List ``engine_resource``'s runtime revisions, newest first.

    Each row is ``{"version", "resource", "create_time", "state"}``. Ordering is by
    ``create_time`` descending; revisions the API returned without one keep their API order
    at the end (sorting is stable), so a platform that stops populating the field degrades
    to "as listed" rather than to a wrong order.
    """
    revisions = client.agent_engines.runtimes.revisions.list(name=engine_resource)
    rows = []
    for revision in revisions:
        resource = revision.api_resource
        name = getattr(resource, "name", "") or ""
        state = getattr(resource, "state", None)
        rows.append(
            {
                "version": revision_id(name),
                "resource": name,
                "create_time": getattr(resource, "create_time", None),
                # State is an enum on the wire; expose its plain name.
                "state": getattr(state, "name", state),
            }
        )
    rows.sort(key=lambda r: r["create_time"].timestamp() if r["create_time"] else 0.0, reverse=True)
    return rows


def traffic_targets(traffic_config: Any) -> list[tuple[str, int]]:
    """The engine's manual traffic split as ``[(version, percent), …]``.

    Empty means "no manual split" — i.e. the always-latest default, where the newest
    revision serves everything. Accepts the SDK model, a plain dict, or ``None``.
    """
    if traffic_config is None:
        return []
    manual = _attr(traffic_config, "traffic_split_manual")
    if manual is None:
        return []
    targets = _attr(manual, "targets") or []
    out: list[tuple[str, int]] = []
    for target in targets:
        name = _attr(target, "runtime_revision_name")
        if name:
            out.append((revision_id(str(name)), int(_attr(target, "percent") or 0)))
    return out


def serving_version(traffic_config: Any, versions: list[dict]) -> str | None:
    """Which revision serves this engine's traffic, or ``None`` if that's not a single one.

    ``versions`` is the newest-first list from :func:`list_revisions`. With no manual split
    the platform routes to the latest revision; with a manual split of one target, to that
    target. ``None`` means either "no revisions" or "traffic is split across several" — in
    both cases there is no single revision a run is guaranteed to hit.
    """
    targets = traffic_targets(traffic_config)
    if not targets:
        return versions[0]["version"] if versions else None
    if len(targets) == 1:
        return targets[0][0]
    return None


def always_latest_traffic() -> dict:
    """Traffic config sending everything to the newest revision (the platform default)."""
    return {"traffic_split_always_latest": {}}


def pinned_traffic(revision: str) -> dict:
    """Traffic config sending 100% to ``revision`` (a full revision resource name)."""
    targets = [{"percent": 100, "runtime_revision_name": revision}]
    return {"traffic_split_manual": {"targets": targets}}


def _attr(obj: Any, name: str) -> Any:
    """Read ``name`` off an SDK model or a plain dict (the SDK hands back either)."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)

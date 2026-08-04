"""Offline tests for runtime revisions — the version axis of a deployed engine (no GCP).

Covers the pure mapping in ``gemini/revisions.py`` (revision names, traffic configs, which
revision serves) and the control-plane decisions built on it: engine-vs-revision resolution,
``get_engine(version=…)`` as an assertion, and the engine handle's version surface. The
GCP-touching parts (the actual ``create``/``update`` round-trip) are validated live; here the
client is a fake.
"""

from __future__ import annotations

import datetime

import pytest

from remote_agent_toolkit import AgentSpec
from remote_agent_toolkit.runtime.gemini import backend
from remote_agent_toolkit.runtime.gemini import revisions as rev

_ENGINE = "projects/p/locations/l/reasoningEngines/123"


def _dt(day: int) -> datetime.datetime:
    return datetime.datetime(2026, 8, day, tzinfo=datetime.timezone.utc)


class _Resource:
    """Stand-in for an SDK api_resource (attribute access over a kwargs bag)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Wrapped:
    """The SDK wraps every resource in an object carrying ``api_resource``."""

    def __init__(self, resource):
        self.api_resource = resource


class FakeRevisions:
    def __init__(self, rows):
        self._rows = rows
        self.deleted: list[str] = []

    def list(self, *, name):
        assert name == _ENGINE
        return iter(_Wrapped(r) for r in self._rows)

    def delete(self, *, name):
        self.deleted.append(name)


class FakeAgentEngines:
    def __init__(self, rows=(), traffic_config=None, engines=()):
        self.runtimes = _Resource(revisions=FakeRevisions(list(rows)))
        self._traffic_config = traffic_config
        self._engines = list(engines)
        self.updates: list[dict] = []

    def get(self, *, name):
        return _Wrapped(_Resource(name=name, traffic_config=self._traffic_config))

    def list(self):
        return iter(self._engines)

    def update(self, *, name, config, agent=None):
        self.updates.append({"name": name, "config": config, "agent": agent})
        return _Wrapped(_Resource(name=name, traffic_config=self._traffic_config))


class FakeClient:
    def __init__(self, agent_engines):
        self.agent_engines = agent_engines


def _rows(*specs):
    """Revision api_resources from ``(id, day)`` pairs (day ``None`` => no create_time)."""
    return [
        _Resource(
            name=f"{_ENGINE}/runtimeRevisions/{rid}",
            create_time=_dt(day) if day else None,
            state=_Resource(name="ACTIVE"),
        )
        for rid, day in specs
    ]


def _engine(client, **kw):
    """A GeminiEngine whose control-plane client is ``client``."""
    engine = backend.GeminiEngine(
        resource=_ENGINE, spec=AgentSpec(name="agent", model="m"),
        project="p", location="l", **kw,
    )
    engine._client = lambda: client
    return engine


# -- pure mapping --------------------------------------------------------------


def test_revision_name_round_trip():
    full = f"{_ENGINE}/runtimeRevisions/7"
    assert rev.revision_id(full) == "7"
    assert rev.revision_id("7") == "7"  # id in => id out
    assert rev.revision_resource(_ENGINE, "7") == full
    assert rev.revision_resource(_ENGINE, full) == full  # already fully qualified


def test_list_revisions_is_newest_first():
    client = FakeClient(FakeAgentEngines(rows=_rows(("1", 1), ("3", 3), ("2", 2))))
    rows = rev.list_revisions(client, _ENGINE)

    assert [r["version"] for r in rows] == ["3", "2", "1"]
    assert rows[0]["resource"] == f"{_ENGINE}/runtimeRevisions/3"
    assert rows[0]["state"] == "ACTIVE"  # the enum is flattened to its name


def test_list_revisions_keeps_api_order_without_create_times():
    # A platform that stops populating create_time must degrade to "as listed", not to a
    # wrong order (the sort is stable and undated rows all share one key).
    client = FakeClient(FakeAgentEngines(rows=_rows(("a", None), ("b", None))))
    assert [r["version"] for r in rev.list_revisions(client, _ENGINE)] == ["a", "b"]


def test_traffic_targets_and_serving_version():
    newest_first = [{"version": "3"}, {"version": "2"}, {"version": "1"}]

    # No config / always-latest: the newest revision serves.
    assert rev.traffic_targets(None) == []
    assert rev.serving_version(None, newest_first) == "3"
    assert rev.serving_version(rev.always_latest_traffic(), newest_first) == "3"
    assert rev.serving_version(None, []) is None

    # A manual pin: that revision serves, regardless of age.
    pinned = rev.pinned_traffic(rev.revision_resource(_ENGINE, "1"))
    assert rev.traffic_targets(pinned) == [("1", 100)]
    assert rev.serving_version(pinned, newest_first) == "1"

    # A fractional split has no single revision a run is guaranteed to hit.
    split = {"traffic_split_manual": {"targets": [
        {"percent": 50, "runtime_revision_name": f"{_ENGINE}/runtimeRevisions/3"},
        {"percent": 50, "runtime_revision_name": f"{_ENGINE}/runtimeRevisions/2"},
    ]}}
    assert rev.serving_version(split, newest_first) is None


# -- control plane -------------------------------------------------------------


def test_resolve_resource_reuses_engine_by_display_name():
    engines = [
        _Wrapped(_Resource(display_name="other", name=f"{_ENGINE}0")),
        _Wrapped(_Resource(display_name="agent", name=_ENGINE)),
    ]
    client = FakeClient(FakeAgentEngines(engines=engines))

    assert backend._find_engine(client, "agent") == _ENGINE
    assert backend._find_engine(client, "nope") is None
    assert backend._resolve_resource(client, "agent") == _ENGINE
    assert backend._resolve_resource(client, _ENGINE) == _ENGINE  # already a resource path
    with pytest.raises(LookupError, match="no deployed engine"):
        backend._resolve_resource(client, "nope")


def test_deploy_warns_when_the_new_revision_will_not_serve():
    # A traffic pin is sticky across deploys: promoting must stay an explicit decision, but
    # silently deploying code that nothing runs would be worse.
    pinned = _Resource(traffic_config=rev.pinned_traffic(f"{_ENGINE}/runtimeRevisions/1"))
    with pytest.warns(UserWarning, match="pinned to 1 .100%."):
        backend._warn_if_traffic_pinned(pinned)


def test_resolve_version_accepts_the_serving_revision():
    client = FakeClient(FakeAgentEngines(rows=_rows(("2", 2), ("1", 1))))
    # Always-latest => the newest revision is the serving one.
    assert backend._resolve_version(client, _ENGINE, "2") == "2"
    # A full revision resource name resolves to the same pin.
    assert backend._resolve_version(client, _ENGINE, f"{_ENGINE}/runtimeRevisions/2") == "2"


def test_resolve_version_rejects_unknown_and_non_serving_revisions():
    client = FakeClient(FakeAgentEngines(rows=_rows(("2", 2), ("1", 1))))

    with pytest.raises(LookupError, match="no runtime revision '9'"):
        backend._resolve_version(client, _ENGINE, "9")

    # Runs go through the ENGINE-level async query path, so a handle cannot address a
    # revision that isn't serving — fail at lookup instead of running unknown code.
    with pytest.raises(ValueError, match="not the one serving traffic"):
        backend._resolve_version(client, _ENGINE, "1")


def test_engine_versions_and_serving_flag():
    ae = FakeAgentEngines(
        rows=_rows(("3", 3), ("2", 2), ("1", 1)),
        traffic_config=rev.pinned_traffic(f"{_ENGINE}/runtimeRevisions/2"),
    )
    engine = _engine(FakeClient(ae))

    assert engine.versions() == ["3", "2", "1"]
    assert [r["serving"] for r in engine.revisions()] == [False, True, False]
    # The handle reports the revision its runs actually execute on, not the newest.
    assert engine.version == "2"


def test_engine_version_falls_back_when_the_control_plane_is_unreachable():
    class Boom(FakeAgentEngines):
        def get(self, *, name):
            raise RuntimeError("no credentials")

    engine = _engine(FakeClient(Boom(rows=_rows(("1", 1)))))
    assert engine.version == "latest"  # identity property degrades, never raises


def test_engine_version_is_the_pin_without_a_round_trip():
    class Boom(FakeAgentEngines):
        def get(self, *, name):  # pragma: no cover - must never be reached
            raise AssertionError("a pinned handle must not re-resolve its version")

    assert _engine(FakeClient(Boom()), version="7").version == "7"


def test_set_traffic_pins_and_unpins():
    ae = FakeAgentEngines(rows=_rows(("2", 2), ("1", 1)))
    engine = _engine(FakeClient(ae))

    engine.set_traffic("1")  # rollback
    assert ae.updates[-1]["name"] == _ENGINE
    assert ae.updates[-1]["agent"] is None  # traffic-only: no rebuild, no new revision
    assert ae.updates[-1]["config"]["traffic_config"]["traffic_split_manual"]["targets"] == [
        {"percent": 100, "runtime_revision_name": f"{_ENGINE}/runtimeRevisions/1"}
    ]
    assert engine.version == "1"

    engine.set_traffic()  # back to always-latest
    assert "traffic_split_always_latest" in ae.updates[-1]["config"]["traffic_config"]
    ae._traffic_config = None
    assert engine.version == "2"  # re-resolved: the newest revision serves again


def test_delete_version_addresses_the_full_revision_path():
    ae = FakeAgentEngines(rows=_rows(("1", 1)))
    _engine(FakeClient(ae)).delete_version(1)  # ints are accepted (DESIGN's version=3 form)
    assert ae.runtimes.revisions.deleted == [f"{_ENGINE}/runtimeRevisions/1"]

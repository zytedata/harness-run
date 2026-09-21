"""Session/turn config transport: bind at session start, overlay per turn, worker honors.

The deploy/session/turn scope model (README). Under test here:

* client: ``start_session(config=)`` persists the SessionConfig at the session's STABLE
  GCS key (the re-attach record) and every turn's body carries the config itself plus
  that pointer; a turn that sets a TurnConfig carries it (and records it, nonce-keyed);
  with no configs nothing is recorded and the body's config fields are null;
* client: ``get_session`` re-attach recovers the opener's persisted config — it cannot
  substitute a different one (the method takes no config);
* client: ``get_engine(spec=)`` is gone, with an error that points at the replacement;
* worker: the turn runs baked ← session ← turn merged (``test_sandbox_runtime`` covers the
  echo and the fail-closed cases);
* local: the same session/turn config surface works in-process (dev/prod parity).
"""

from __future__ import annotations

import asyncio

import pytest
from sandbox_fakes import FakeSandboxProvider, make_engine

from agent_run import AgentSpec, SessionConfig, SkillSource, TurnConfig
from agent_run.ports.blobstore import LocalBlobStore
from agent_run.runtime import local
from agent_run.runtime.sandbox import backend, handoff, history, worker as worker_mod


def _patched_store(tmp_path, monkeypatch):
    """One local store for the output bucket: session records (handoff) and the event mirror
    (history — a re-attached session reads it once for a turn running under another process)."""
    store = LocalBlobStore(str(tmp_path / "blobs"))
    monkeypatch.setattr(handoff, "GcsBlobStore", lambda bucket, *a, **kw: store)
    monkeypatch.setattr(history, "GcsBlobStore", lambda bucket, *a, **kw: store)
    return store


def _turn_bodies(provider):
    return [c[2] for c in provider.calls if c[1] == "/turn"]


async def _await(run):
    return await run


# ---------------------------------------------------------------- client side


def test_session_config_is_persisted_once_and_every_turn_carries_it(tmp_path, monkeypatch):
    store = _patched_store(tmp_path, monkeypatch)
    provider = FakeSandboxProvider()
    engine = make_engine(provider, output_bucket="gs://bkt")
    cfg = SessionConfig(model="m-session", system_prompt="SESSION-PROMPT")
    session = engine.start_session(config=cfg)
    sid = session.session_id

    uri = handoff.session_config_uri("gs://bkt", sid)
    assert handoff.load_session_config("gs://bkt", sid, store=store) == (uri, cfg.to_dict())

    asyncio.run(_await(session.run("go")))
    asyncio.run(_await(session.send("more")))
    bodies = _turn_bodies(provider)
    assert [b["session_config"] for b in bodies] == [cfg.to_dict(), cfg.to_dict()]
    assert [b["session_config_gcs"] for b in bodies] == [uri, uri]
    assert store.list("session-config/") == [f"session-config/{sid}.json"]  # written once, kept


def test_turn_config_rides_the_body_and_is_recorded_per_turn(tmp_path, monkeypatch):
    store = _patched_store(tmp_path, monkeypatch)
    provider = FakeSandboxProvider()
    engine = make_engine(provider, output_bucket="gs://bkt")
    session = engine.start_session(config=SessionConfig(model="m-session"))
    asyncio.run(_await(session.run("go", config=TurnConfig(model="m-turn", max_turns=3))))
    asyncio.run(_await(session.send("more")))
    asyncio.run(_await(session.send("again", config=TurnConfig())))  # all-INHERIT: nothing
    bodies = _turn_bodies(provider)
    assert bodies[0]["turn_config"] == {"model": "m-turn", "max_turns": 3}
    assert bodies[0]["turn_config_gcs"].startswith(f"gs://bkt/turn-config/{session.session_id}-")
    assert bodies[1]["turn_config"] is None and bodies[2]["turn_config"] is None
    assert len(store.list("turn-config/")) == 1


def test_no_config_records_nothing_and_no_config_rides(tmp_path, monkeypatch):
    store = _patched_store(tmp_path, monkeypatch)
    provider = FakeSandboxProvider()
    session = make_engine(provider, output_bucket="gs://bkt").start_session()
    asyncio.run(_await(session.run("go")))
    body = _turn_bodies(provider)[0]
    assert body["session_config"] is None and body["turn_config"] is None
    assert body["session_config_gcs"] is None and body["turn_config_gcs"] is None
    assert store.list("") == []


def test_reattach_recovers_the_openers_config_and_cannot_take_a_new_one(tmp_path, monkeypatch):
    _patched_store(tmp_path, monkeypatch)
    provider = FakeSandboxProvider()
    opener = make_engine(provider, output_bucket="gs://bkt")
    sid = opener.start_session(config=SessionConfig(model="m-session")).session_id

    fresh = make_engine(provider, template=opener.resource, output_bucket="gs://bkt")
    reattached = fresh.get_session(sid)
    with pytest.raises(TypeError):
        fresh.get_session(sid, config=SessionConfig(model="hijack"))  # type: ignore[call-arg]
    asyncio.run(_await(reattached.run("continue")))
    body = _turn_bodies(provider)[-1]
    assert body["session_config"] == {"model": "m-session"}
    assert body["session_config_gcs"] == handoff.session_config_uri("gs://bkt", sid)
    assert reattached._client_spec().model == "m-session"


def test_start_session_with_config_requires_output_bucket():
    engine = make_engine(FakeSandboxProvider(), output_bucket=None)
    with pytest.raises(ValueError, match="output bucket"):
        engine.start_session(config=SessionConfig(model="m"))


def test_get_engine_spec_kwarg_is_a_clear_error():
    with pytest.raises(TypeError, match="SessionConfig"):
        backend.get_engine("name", "p", "l", spec=AgentSpec(name="n", model="m"))


# ---------------------------------------------------------------- worker side


def test_baked_skills_are_skipped_when_the_effective_skills_differ(monkeypatch, tmp_path):
    staged = {}

    def fake_provision(sources, workspace, subdir):
        staged["sources"] = tuple(sources)
        return []

    import agent_run.skills as skills_mod
    monkeypatch.setattr(skills_mod, "provision", fake_provision)

    class RC:
        resume_sid = None
        blobs = None
        workspace = tmp_path / "ws"
        spec = AgentSpec(name="w", model="m", skills=(SkillSource.git("https://h/run-skills.git"),))
        secrets = {}

    worker_mod._prepare_workspace(RC(), prefer_baked_skills=False, baked_dir=tmp_path / "baked")
    assert staged["sources"] == RC.spec.skills  # the effective spec's own sources
    worker_mod._prepare_workspace(RC(), prefer_baked_skills=True, baked_dir=tmp_path / "baked")
    assert staged["sources"][0].kind == "local" and staged["sources"][0].path == str(tmp_path / "baked")


# ---------------------------------------------------------------- local parity


def test_local_session_and_turn_configs_overlay_the_spec(tmp_path):
    engine = local.deploy(AgentSpec(name="d", model="m-baked"), workdir=str(tmp_path))
    session = engine.start_session(config=SessionConfig(model="m-session"))
    assert session._spec.model == "m-session"
    # Re-attach by id from a fresh handle recovers the persisted config.
    again = local.deploy(AgentSpec(name="d", model="m-baked"), workdir=str(tmp_path)).get_session(session.session_id)
    assert again._spec.model == "m-session"

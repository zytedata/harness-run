"""Session/turn config transport: bind at session start, overlay per turn, worker honors.

The deploy/session/turn scope model (README). Under test here:

* client: ``start_session(config=)`` persists the SessionConfig at the session's STABLE
  GCS key; every turn's payload carries that pointer (+ a nonce-keyed TurnConfig pointer
  when a turn sets one); with no configs, nothing is staged and no config keys ride;
* client: ``get_session`` re-attach recovers the opener's persisted config — it cannot
  substitute a different one (the method takes no config);
* client: ``get_engine(spec=)`` is gone, with an error that points at the replacement;
* worker: the turn runs baked ← session ← turn merged; the merged result is echoed as an
  ``effective_spec`` event (ground truth for post-mortem/replay); config objects are KEPT
  at turn end (unlike secrets);
* worker: a missing config, an unknown config field, or a harness the image doesn't bake
  FAILS the turn (running the baked spec instead would be a silent wrong-configuration run);
* local: the same session/turn config surface works in-process (dev/prod parity).
"""

from __future__ import annotations

import asyncio

import pytest

from remote_agent_toolkit import AgentSpec, SessionConfig, SkillSource, TurnConfig
from remote_agent_toolkit.events import AgentEvent
from remote_agent_toolkit.ports.blobstore import LocalBlobStore
from remote_agent_toolkit.ports.eventsink import InMemorySink
from remote_agent_toolkit.runtime import local
from remote_agent_toolkit.runtime.gemini import adk_agent, backend, handoff


def _patched_store(tmp_path, monkeypatch):
    store = LocalBlobStore(str(tmp_path / "blobs"))
    monkeypatch.setattr(handoff, "GcsBlobStore", lambda bucket, *a, **kw: store)
    return store


def _seed_tail(monkeypatch, sid):
    seed = InMemorySink(session_id=sid)
    seed.emit(AgentEvent(kind="result", summary="done",
                         raw={"subtype": "success", "is_error": False, "session_id": sid}))
    monkeypatch.setattr(backend, "tail_stream", lambda uri, s, **kw: seed.tail(s))
    return seed


def _warm_engine(monkeypatch, published, spec=None):
    engine = backend.GeminiEngine(
        resource="r/reasoningEngines/1", spec=spec or backend._fallback_spec("w"),
        project=None, location=None, output_bucket="gs://bkt", warm=True,
        topic="t", subscription="s",
    )
    monkeypatch.setattr(engine, "_dispatch",
                        lambda: type("D", (), {"publish": lambda _s, m: published.append(m)})())
    monkeypatch.setattr(engine, "fill_pool", lambda n: None)
    return engine


async def _await(run):
    return await run


# ---------------------------------------------------------------- client side


def test_session_config_is_persisted_once_and_every_turn_points_at_it(tmp_path, monkeypatch):
    store = _patched_store(tmp_path, monkeypatch)
    published = []
    engine = _warm_engine(monkeypatch, published)
    _seed_tail(monkeypatch, "unused")

    cfg = SessionConfig(model="m-session", system_prompt="SESSION-PROMPT")
    session = engine.start_session(config=cfg)
    sid = session.session_id

    # Bound at open: persisted at the session's stable key, before any turn ran.
    uri = handoff.session_config_uri("gs://bkt", sid)
    assert handoff.fetch_config(uri, store=store) == cfg.to_dict()

    asyncio.run(_await(session.run("go")))
    asyncio.run(_await(session.send("more")))

    # BOTH turns point at the same persisted object; nothing is re-staged per turn.
    assert [m["session_config_gcs"] for m in published] == [uri, uri]
    assert store.list("session-config/") == [f"session-config/{sid}.json"]
    # The config is the post-mortem record: still there after the turns completed.
    assert handoff.fetch_config(uri, store=store) == cfg.to_dict()


def test_turn_config_is_staged_per_turn_and_layers_on_the_session(tmp_path, monkeypatch):
    store = _patched_store(tmp_path, monkeypatch)
    published = []
    engine = _warm_engine(monkeypatch, published)
    _seed_tail(monkeypatch, "unused")

    session = engine.start_session(config=SessionConfig(model="m-session"))
    asyncio.run(_await(session.run("go", config=TurnConfig(model="m-turn", max_turns=3))))
    asyncio.run(_await(session.send("more")))  # no turn config on the second turn

    turn_uri = published[0]["turn_config_gcs"]
    assert turn_uri.startswith(f"gs://bkt/turn-config/{session.session_id}-")
    assert handoff.fetch_config(turn_uri, store=store) == {"model": "m-turn", "max_turns": 3}
    assert "turn_config_gcs" not in published[1]
    # An all-INHERIT turn config stages nothing either.
    asyncio.run(_await(session.send("again", config=TurnConfig())))
    assert "turn_config_gcs" not in published[2]


def test_no_config_stages_nothing_and_no_config_keys_ride(tmp_path, monkeypatch):
    # A plain session (no config) keeps the pre-config behavior exactly: zero staging,
    # zero new payload keys, the turn runs the deploy-baked spec.
    store = _patched_store(tmp_path, monkeypatch)
    published = []
    engine = _warm_engine(monkeypatch, published)
    _seed_tail(monkeypatch, "unused")

    session = engine.start_session()
    asyncio.run(_await(session.run("go")))

    assert "session_config_gcs" not in published[0]
    assert "turn_config_gcs" not in published[0]
    assert store.list("") == []


def test_reattach_recovers_the_openers_config_and_cannot_take_a_new_one(tmp_path, monkeypatch):
    _patched_store(tmp_path, monkeypatch)
    published = []
    opener_engine = _warm_engine(monkeypatch, published)
    _seed_tail(monkeypatch, "unused")

    cfg = SessionConfig(model="m-session")
    sid = opener_engine.start_session(config=cfg).session_id

    # A FRESH process (new engine handle) re-attaches by id: no config parameter exists —
    # the persisted one is recovered, so its turns run the same world the opener bound.
    fresh_engine = _warm_engine(monkeypatch, published)
    reattached = fresh_engine.get_session(sid)
    with pytest.raises(TypeError):
        fresh_engine.get_session(sid, config=SessionConfig(model="hijack"))  # type: ignore[call-arg]

    asyncio.run(_await(reattached.run("continue")))
    assert published[-1]["session_config_gcs"] == handoff.session_config_uri("gs://bkt", sid)
    # The recovered config also drives client-side parsing state (the effective view).
    assert reattached._client_spec().model == "m-session"


def test_start_session_with_config_requires_output_bucket(monkeypatch):
    engine = backend.GeminiEngine(
        resource="r/reasoningEngines/1", spec=backend._fallback_spec("w"),
        project=None, location=None, output_bucket=None,
    )
    with pytest.raises(ValueError, match="output bucket"):
        engine.start_session(config=SessionConfig(model="m"))


def test_get_engine_spec_kwarg_is_a_clear_error():
    # Removed outright (no deprecation shim): the old parameter silently did NOT configure
    # execution, so the error points at the types that do — and at the PR for context.
    with pytest.raises(TypeError, match=r"SessionConfig|pull/16"):
        backend.get_engine("name", spec=AgentSpec(name="n", model="m"))


# ---------------------------------------------------------------- worker side


class _RecordingHarness:
    seen: list = []

    async def run(self, spec, rc):
        _RecordingHarness.seen.append((spec, rc))
        yield AgentEvent(kind="result", summary="done",
                         raw={"subtype": "success", "is_error": False})


def _worker_env(tmp_path, monkeypatch, sid):
    monkeypatch.setenv("AGENT_JOBS_ROOT", str(tmp_path / "jobs"))
    monkeypatch.chdir(tmp_path)  # no baked skills dir on the lookup paths
    store = _patched_store(tmp_path, monkeypatch)
    _RecordingHarness.seen = []
    import remote_agent_toolkit.harness.claude_code as harness_mod
    import remote_agent_toolkit.ports.eventsink as eventsink_mod
    monkeypatch.setattr(harness_mod, "ClaudeCodeHarness", _RecordingHarness)
    monkeypatch.setattr(eventsink_mod, "CloudLoggingSink",
                        lambda **kw: InMemorySink(session_id=sid))
    return store


def _drive_turn(agent, baked_spec, sid, session_config_uri=None, turn_config_uri=None):
    async def drive():
        return [ev async for ev in agent._run_turn(
            baked_spec, sid, "go", None,
            session_config_uri=session_config_uri, turn_config_uri=turn_config_uri)]

    return asyncio.run(drive())


def test_worker_runs_the_merged_spec_and_echoes_it(tmp_path, monkeypatch):
    store = _worker_env(tmp_path, monkeypatch, "77")
    baked = AgentSpec(name="w", model="m-baked", max_budget_usd=5.0)
    s_uri = handoff.persist_session_config(
        "gs://bkt", "77",
        SessionConfig(model="m-session", system_prompt="SESSION").to_dict(), store=store)
    t_uri = handoff.stage_turn_config(
        "gs://bkt", "77", TurnConfig(model="m-turn", max_budget_usd=1.0).to_dict(), store=store)

    agent = adk_agent.build_agent(baked)
    events = _drive_turn(agent, baked, "77", session_config_uri=s_uri, turn_config_uri=t_uri)

    # The harness ran the MERGED spec: turn over session over baked.
    spec, _rc = _RecordingHarness.seen[0]
    assert (spec.model, spec.system_prompt, spec.max_budget_usd) == ("m-turn", "SESSION", 1.0)
    assert spec.name == "w"  # deploy-only fields untouched

    # The first surfaced event is the effective-spec echo — the durable ground-truth
    # record of what actually ran, with the config pointers for provenance.
    echo = events[0].custom_metadata
    assert echo["kind"] == "status" and echo["raw"]["event"] == "effective_spec"
    assert echo["raw"]["spec"]["model"] == "m-turn"
    assert echo["raw"]["session_config_gcs"] == s_uri
    assert echo["raw"]["turn_config_gcs"] == t_uri
    assert events[-1].custom_metadata["kind"] == "result"

    # Config objects are KEPT at turn end (post-mortem record) — only secrets are reaped.
    assert handoff.fetch_config(s_uri, store=store) is not None
    assert handoff.fetch_config(t_uri, store=store) is not None


def test_worker_fails_the_turn_when_a_config_is_missing(tmp_path, monkeypatch):
    # NEVER fall back to the baked spec: the caller bound a specific configuration; running
    # anything else would be a silent wrong-run (the bug class this transport prevents).
    _worker_env(tmp_path, monkeypatch, "78")
    baked = AgentSpec(name="w", model="m-baked")
    agent = adk_agent.build_agent(baked)
    events = _drive_turn(agent, baked, "78",
                         session_config_uri="gs://bkt/session-config/78.json")

    assert len(events) == 1  # the terminal error is the only event
    assert events[0].custom_metadata["kind"] == "result"
    assert events[0].custom_metadata["raw"]["is_error"] is True
    assert "session config unavailable" in events[0].error_message
    assert _RecordingHarness.seen == []


def test_worker_fails_the_turn_on_an_unknown_config_field(tmp_path, monkeypatch):
    # A newer client staged a field this engine revision can't honor: fail closed (the
    # client/engine same-revision rule), never silently drop it.
    store = _worker_env(tmp_path, monkeypatch, "79")
    baked = AgentSpec(name="w", model="m-baked")
    s_uri = handoff.persist_session_config("gs://bkt", "79", {"warp_drive": True}, store=store)
    agent = adk_agent.build_agent(baked)
    events = _drive_turn(agent, baked, "79", session_config_uri=s_uri)

    assert len(events) == 1
    assert events[0].custom_metadata["raw"]["is_error"] is True
    assert "same toolkit revision" in events[0].error_message
    assert _RecordingHarness.seen == []


def test_worker_fails_the_turn_on_a_harness_the_image_does_not_bake(tmp_path, monkeypatch):
    store = _worker_env(tmp_path, monkeypatch, "80")
    baked = AgentSpec(name="w", model="m-baked")  # claude-code only
    s_uri = handoff.persist_session_config(
        "gs://bkt", "80", SessionConfig(harness="codex").to_dict(), store=store)
    agent = adk_agent.build_agent(baked)
    events = _drive_turn(agent, baked, "80", session_config_uri=s_uri)

    assert len(events) == 1
    assert events[0].custom_metadata["raw"]["is_error"] is True
    assert "not baked into this engine" in events[0].error_message
    assert _RecordingHarness.seen == []


def test_cold_directives_are_stripped_and_resolve_the_configs(tmp_path, monkeypatch):
    # The cold path rides the pointers as leading directives; the model must never see
    # them, and the turn must run the merged configuration.
    store = _worker_env(tmp_path, monkeypatch, "81")

    both = "AGENT_SESSION_CONFIG_GCS=gs://b/s.json\nAGENT_TURN_CONFIG_GCS=gs://b/t.json\nreal"
    assert adk_agent._split_config_directives(both) == ("gs://b/s.json", "gs://b/t.json", "real")
    assert adk_agent._split_config_directives("no directive") == (None, None, "no directive")

    baked = AgentSpec(name="w", model="m-baked")
    s_uri = handoff.persist_session_config(
        "gs://bkt", "81", SessionConfig(model="m-run").to_dict(), store=store)

    class Content:
        parts = [type("P", (), {"text": f"AGENT_SESSION=81\nAGENT_SESSION_CONFIG_GCS={s_uri}\ngo"})()]

    class Ctx:
        user_content = Content()
        invocation_id = "e-1"
        session = None

    agent = adk_agent.build_agent(baked)

    async def drive():
        return [ev async for ev in agent._run_async_impl(Ctx())]

    events = asyncio.run(drive())
    assert events[-1].custom_metadata["kind"] == "result"
    spec, rc = _RecordingHarness.seen[0]
    assert (spec.model, rc.prompt) == ("m-run", "go")  # directives consumed, config applied


def test_baked_skills_are_skipped_when_the_effective_skills_differ(monkeypatch, tmp_path):
    # The deploy-baked skills dir is only a staging shortcut for the baked declaration; an
    # effective spec with different skills (including none) must resolve its own.
    staged = {}

    def fake_provision(sources, workspace, subdir):
        staged["sources"] = tuple(sources)
        return []

    import remote_agent_toolkit.skills as skills_mod
    monkeypatch.setattr(skills_mod, "provision", fake_provision)
    monkeypatch.setattr(adk_agent, "_find_baked_skills", lambda: tmp_path / "baked")

    class RC:
        resume_sid = None
        blobs = None
        workspace = tmp_path / "ws"
        spec = AgentSpec(name="w", model="m",
                         skills=(SkillSource.git("https://h/run-skills.git"),))
        secrets = {}

    adk_agent._prepare_workspace(RC(), prefer_baked_skills=False)
    assert staged["sources"] == RC.spec.skills  # the effective spec's own sources

    adk_agent._prepare_workspace(RC(), prefer_baked_skills=True)
    src = staged["sources"][0]
    assert src.kind == "local" and str(tmp_path / "baked") in (src.path or "")


# ---------------------------------------------------------------- local parity


class _FakeLocalHarness:
    def __init__(self):
        self.seen = []

    async def run(self, spec, ctx):
        self.seen.append(spec)
        yield AgentEvent(kind="result", summary="done",
                         raw={"subtype": "success", "is_error": False})


def test_local_session_and_turn_configs_apply(tmp_path):
    engine = local.deploy(AgentSpec(name="l", model="m-deploy"), workdir=str(tmp_path))
    fake = _FakeLocalHarness()
    engine._harness = fake

    session = engine.start_session(config=SessionConfig(model="m-session"))
    asyncio.run(_await(session.run("go")))
    asyncio.run(_await(session.send("more", config=TurnConfig(model="m-turn"))))

    assert [s.model for s in fake.seen] == ["m-session", "m-turn"]


def test_local_reattach_recovers_the_config_across_processes(tmp_path):
    spec = AgentSpec(name="l", model="m-deploy")
    opener = local.deploy(spec, workdir=str(tmp_path))
    sid = opener.start_session(config=SessionConfig(model="m-session")).session_id

    # A fresh engine over the SAME workdir (≈ a new process) re-attaches by id and runs
    # under the opener's config — get_session takes no config, same rule as gemini.
    fresh = local.deploy(spec, workdir=str(tmp_path))
    reattached = fresh.get_session(sid)
    assert reattached._spec.model == "m-session"
    with pytest.raises(TypeError):
        fresh.get_session(sid, config=SessionConfig(model="hijack"))  # type: ignore[call-arg]


def test_local_harness_choice_is_validated_at_bind(tmp_path):
    engine = local.deploy(AgentSpec(name="l", model="m"), workdir=str(tmp_path))
    with pytest.raises(ValueError, match="not baked into this engine"):
        engine.start_session(config=SessionConfig(harness="codex"))

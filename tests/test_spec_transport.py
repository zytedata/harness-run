"""Run-scoped spec transport: the spec passed to ``get_engine(spec=)`` is authoritative.

The engine bakes a spec at deploy, but run-scoped parameters (target repo/ref, model,
prompt) must not require an engine redeploy: the client stages the handle's spec per
invocation (``handoff.stage_spec``) and the worker executes THAT spec. Under test here:

* client: the staged pointer rides the warm dispatch payload / cold directive; a
  fallback-spec handle (no ``spec=``) stages nothing (baked behavior unchanged);
* client: a spec whose ``RepoSource.url`` embeds credentials is refused before staging
  (the staged spec is referenced from persisted payloads and must be credential-free);
* worker: the payload spec wins over the baked one; a missing staging FAILS the turn
  (running the baked spec instead of the requested one would be a silent wrong-run);
* worker: baked skills are only a fast path while the run's skills match the baked ones.
"""

from __future__ import annotations

import asyncio

from remote_agent_toolkit import AgentSpec, RepoSource, SkillSource
from remote_agent_toolkit.events import AgentEvent
from remote_agent_toolkit.ports.blobstore import LocalBlobStore
from remote_agent_toolkit.ports.eventsink import InMemorySink
from remote_agent_toolkit.runtime.gemini import adk_agent, backend, handoff


def _patched_store(tmp_path, monkeypatch):
    store = LocalBlobStore(str(tmp_path / "blobs"))
    monkeypatch.setattr(handoff, "GcsBlobStore", lambda bucket, *a, **kw: store)
    return store


def _seed_result_sink(monkeypatch, sid):
    seed = InMemorySink(session_id=sid)
    seed.emit(AgentEvent(kind="result", summary="done",
                         raw={"subtype": "success", "is_error": False, "session_id": sid}))
    import remote_agent_toolkit.ports.eventsink as eventsink_mod
    monkeypatch.setattr(eventsink_mod, "CloudLoggingSink", lambda **kw: seed)
    return seed


async def _await(run):
    return await run


# ---------------------------------------------------------------- client side


def test_warm_dispatch_carries_the_staged_spec_pointer(tmp_path, monkeypatch):
    store = _patched_store(tmp_path, monkeypatch)
    spec = AgentSpec(name="w", model="m-run")
    engine = backend.GeminiEngine(
        resource="r/reasoningEngines/1", spec=spec, project=None, location=None,
        output_bucket="gs://bkt", warm=True, topic="t", subscription="s",
    )
    published = []
    monkeypatch.setattr(engine, "_dispatch",
                        lambda: type("D", (), {"publish": lambda _s, m: published.append(m)})())
    monkeypatch.setattr(engine, "fill_pool", lambda n: None)
    _seed_result_sink(monkeypatch, "sid-w")

    # Spy on the (real) staging so the content is observable even after cleanup.
    staged = {}
    real_stage = handoff.stage_spec

    def spy_stage(bucket, sid, spec_dict, store_=None):
        staged["spec"] = spec_dict
        return real_stage(bucket, sid, spec_dict, store=store)

    import remote_agent_toolkit.runtime.gemini.handoff as handoff_mod
    monkeypatch.setattr(handoff_mod, "stage_spec", spy_stage)

    asyncio.run(_await(backend.GeminiSession(engine, "sid-w").run("go")))

    uri = published[0]["spec_gcs"]
    assert uri.startswith("gs://bkt/invocation-spec/sid-w-")
    # What was staged is THIS handle's spec — what the worker will run.
    assert staged["spec"] == spec.to_dict()
    # With no worker in this offline run, the client's completion backstop reaped the
    # staged object (same layered cleanup as secrets).
    assert handoff.fetch_spec(uri, store=store) is None


def test_fallback_spec_handle_stages_nothing(tmp_path, monkeypatch):
    # get_engine WITHOUT spec= → fallback spec → no transport: runs execute the baked spec
    # exactly as before this feature (and no stray staging objects are written).
    store = _patched_store(tmp_path, monkeypatch)
    engine = backend.GeminiEngine(
        resource="r/reasoningEngines/1", spec=backend._fallback_spec("w"),
        project=None, location=None, output_bucket="gs://bkt", warm=True,
        topic="t", subscription="s", spec_transport=False,
    )
    published = []
    monkeypatch.setattr(engine, "_dispatch",
                        lambda: type("D", (), {"publish": lambda _s, m: published.append(m)})())
    monkeypatch.setattr(engine, "fill_pool", lambda n: None)
    _seed_result_sink(monkeypatch, "sid-f")

    asyncio.run(_await(backend.GeminiSession(engine, "sid-f").run("go")))

    assert "spec_gcs" not in published[0]
    assert store.list("") == []  # nothing staged


def test_credentialed_repo_url_is_refused_before_staging(tmp_path, monkeypatch):
    # The staged spec is referenced from persisted payloads (job input / Pub/Sub), so it
    # must never smuggle a pre-authenticated clone URL. Name the token via auth=/auth_user=
    # and pass the value in run(secrets=...) instead.
    import pytest

    store = _patched_store(tmp_path, monkeypatch)
    spec = AgentSpec(
        name="w", model="m",
        repos=(RepoSource.git("https://x-token-auth:SECRET@bitbucket.org/o/r.git"),),
    )
    engine = backend.GeminiEngine(
        resource="r/reasoningEngines/1", spec=spec, project=None, location=None,
        output_bucket="gs://bkt", warm=True, topic="t", subscription="s",
    )
    with pytest.raises(ValueError, match="auth_user"):
        backend.GeminiSession(engine, "sid-g").run("go")
    assert store.list("") == []  # refused BEFORE anything was written


# ---------------------------------------------------------------- worker side


def _drive_turn(agent, baked_spec, sid, monkeypatch, spec_uri=None):
    async def drive():
        return [ev async for ev in agent._run_turn(
            baked_spec, sid, "go", None, spec_uri=spec_uri)]

    return asyncio.run(drive())


def test_worker_runs_the_payload_spec_not_the_baked_one(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_JOBS_ROOT", str(tmp_path / "jobs"))
    monkeypatch.chdir(tmp_path)  # no baked skills dir on the lookup paths
    store = _patched_store(tmp_path, monkeypatch)

    baked = AgentSpec(name="w", model="m-baked")
    run_scoped = AgentSpec(name="w", model="m-run", system_prompt="RUN-SCOPED")
    uri = handoff.stage_spec("gs://bkt", "77", run_scoped.to_dict(), store=store)
    key = uri[len("gs://bkt/"):]

    seen = {}

    class RecordingHarness:
        async def run(self, spec, rc):
            seen["model"] = spec.model
            seen["system_prompt"] = spec.system_prompt
            yield AgentEvent(kind="result", summary="done",
                             raw={"subtype": "success", "is_error": False})

    import remote_agent_toolkit.harness.claude_code as harness_mod
    import remote_agent_toolkit.ports.eventsink as eventsink_mod
    monkeypatch.setattr(harness_mod, "ClaudeCodeHarness", RecordingHarness)
    monkeypatch.setattr(eventsink_mod, "CloudLoggingSink",
                        lambda **kw: InMemorySink(session_id="77"))

    agent = adk_agent.build_agent(baked)
    events = _drive_turn(agent, baked, "77", monkeypatch, spec_uri=uri)

    assert events[-1].custom_metadata["kind"] == "result"
    assert seen == {"model": "m-run", "system_prompt": "RUN-SCOPED"}  # payload spec won
    assert not store.exists(key)  # staged spec reaped at turn end (same as secrets)


def test_worker_fails_the_turn_when_the_staged_spec_is_missing(tmp_path, monkeypatch):
    # NEVER fall back to the baked spec: the caller staged a specific configuration; running
    # anything else would be a silent wrong-run (the bug class this transport prevents).
    monkeypatch.setenv("AGENT_JOBS_ROOT", str(tmp_path / "jobs"))
    monkeypatch.chdir(tmp_path)
    _patched_store(tmp_path, monkeypatch)

    ran = {"harness": False}

    class NeverHarness:
        async def run(self, spec, rc):
            ran["harness"] = True
            yield AgentEvent(kind="result", summary="done", raw={})

    import remote_agent_toolkit.harness.claude_code as harness_mod
    import remote_agent_toolkit.ports.eventsink as eventsink_mod
    monkeypatch.setattr(harness_mod, "ClaudeCodeHarness", NeverHarness)
    sink = InMemorySink(session_id="78")
    monkeypatch.setattr(eventsink_mod, "CloudLoggingSink", lambda **kw: sink)

    baked = AgentSpec(name="w", model="m-baked")
    agent = adk_agent.build_agent(baked)
    events = _drive_turn(agent, baked, "78", monkeypatch,
                         spec_uri="gs://bkt/invocation-spec/78-gone.json")

    assert len(events) == 1  # the terminal error is the only event
    assert events[0].custom_metadata["kind"] == "result"
    assert events[0].custom_metadata["raw"]["is_error"] is True
    assert "run-scoped spec unavailable" in events[0].error_message
    assert ran["harness"] is False


def test_cold_directive_is_stripped_and_resolves_the_spec(tmp_path, monkeypatch):
    # The cold path rides the pointer as a leading AGENT_SPEC_GCS directive; the model must
    # never see it, and the turn must run the staged spec.
    monkeypatch.setenv("AGENT_JOBS_ROOT", str(tmp_path / "jobs"))
    monkeypatch.chdir(tmp_path)
    store = _patched_store(tmp_path, monkeypatch)

    uri, rest = adk_agent._split_spec_directive("AGENT_SPEC_GCS=gs://b/x.json\nreal prompt")
    assert (uri, rest) == ("gs://b/x.json", "real prompt")
    assert adk_agent._split_spec_directive("no directive") == (None, "no directive")

    baked = AgentSpec(name="w", model="m-baked")
    run_scoped = AgentSpec(name="w", model="m-run")
    staged = handoff.stage_spec("gs://bkt", "79", run_scoped.to_dict(), store=store)

    seen = {}

    class RecordingHarness:
        async def run(self, spec, rc):
            seen["model"] = spec.model
            seen["prompt"] = rc.prompt
            yield AgentEvent(kind="result", summary="done",
                             raw={"subtype": "success", "is_error": False})

    import remote_agent_toolkit.harness.claude_code as harness_mod
    import remote_agent_toolkit.ports.eventsink as eventsink_mod
    monkeypatch.setattr(harness_mod, "ClaudeCodeHarness", RecordingHarness)
    monkeypatch.setattr(eventsink_mod, "CloudLoggingSink",
                        lambda **kw: InMemorySink(session_id="79"))

    class Content:
        parts = [type("P", (), {"text": f"AGENT_SESSION=79\nAGENT_SPEC_GCS={staged}\ngo"})()]

    class Ctx:
        user_content = Content()
        invocation_id = "e-1"
        session = None

    agent = adk_agent.build_agent(baked)

    async def drive():
        return [ev async for ev in agent._run_async_impl(Ctx())]

    events = asyncio.run(drive())
    assert events[-1].custom_metadata["kind"] == "result"
    assert seen == {"model": "m-run", "prompt": "go"}  # directive consumed, spec resolved


# ---------------------------------------------------------------- skills fast path


def test_baked_skills_are_skipped_when_the_run_spec_declares_different_ones(monkeypatch, tmp_path):
    # The deploy-baked skills dir is only a staging shortcut for the baked declaration; a
    # run-scoped spec with different skills (including none) must resolve its own.
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
    assert staged["sources"] == RC.spec.skills  # run spec's own sources, not the baked dir

    adk_agent._prepare_workspace(RC(), prefer_baked_skills=True)
    src = staged["sources"][0]
    assert src.kind == "local" and str(tmp_path / "baked") in (src.path or "")

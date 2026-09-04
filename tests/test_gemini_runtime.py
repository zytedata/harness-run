"""Gemini run-plane offline tests: AgentEvent->ADK translation + Session/Run wiring.

The GCP-touching control plane (deploy / run_query_job / sessions.create) is validated
live; here we test the pure-Python glue with fakes — the AgentEvent->ADK Event mapping and
the GeminiSession -> DrivenRun -> EventSink.tail mechanism that drives a run.
"""

from __future__ import annotations

import asyncio

import pytest

from remote_agent_toolkit import AgentSpec
from remote_agent_toolkit.events import AgentEvent, RunStatus, StopReason
from remote_agent_toolkit.ports.eventsink import InMemorySink
from remote_agent_toolkit.runtime.gemini import adk_agent, backend
from remote_agent_toolkit.runtime.gemini.translate import to_adk_event


async def _await(run):
    return await run


async def _drain(agen):
    return [ev async for ev in agen]


def test_to_adk_event_kinds():
    msg = to_adk_event(AgentEvent(kind="message", summary="hi"), "agent")
    assert msg.partial is True and (msg.turn_complete in (None, False))
    assert msg.custom_metadata["kind"] == "message"

    think = to_adk_event(AgentEvent(kind="thinking", summary="hmm"), "agent")
    assert think.content.parts[0].thought is True

    result = to_adk_event(
        AgentEvent(kind="result", summary="done", cost_usd=0.3, usage={"t": 1},
                   raw={"is_error": False, "num_turns": 2}),
        "agent",
    )
    assert result.partial is False and result.turn_complete is True
    assert result.custom_metadata["cost_usd"] == 0.3 and result.error_message is None

    err = to_adk_event(AgentEvent(kind="result", summary="boom", raw={"is_error": True}), "agent")
    assert err.error_message == "boom"


def test_to_adk_event_stamps_invocation_id():
    # Regression: the platform's session-append API 400s on events without invocation_id
    # ("event.invocation_id: Required field is not set") — ADK defaults it to '' and the
    # runner appends every non-partial event (our terminal result) to the session service.
    ev = to_adk_event(AgentEvent(kind="result", summary="done"), "agent", "e-inv-123")
    assert ev.invocation_id == "e-inv-123"


def test_build_agent_sanitizes_node_name():
    # ADK BaseAgent.name must be a valid Python identifier; spec.name may have hyphens.
    agent = adk_agent.build_agent(AgentSpec(name="hello-coder", model="m"))
    assert agent.name == "hello_coder"
    assert agent.spec_data["name"] == "hello-coder"  # the real name is preserved in the spec
    assert adk_agent._safe_node_name("9to5") == "a_9to5"


def test_split_resume_directive_and_parse_gcs():
    sid, rest = adk_agent._split_resume_directive("AGENT_RESUME=abc123\nplease continue")
    assert sid == "abc123" and rest == "please continue"
    assert adk_agent._split_resume_directive("just a prompt") == (None, "just a prompt")
    assert adk_agent._parse_gcs_uri("gs://bkt/some/prefix") == ("bkt", "some/prefix")


def test_split_secrets_directive_strips_pointer():
    uri = "gs://bkt/invocation-secrets/s-1.json"
    got, rest = adk_agent._split_secrets_directive(f"AGENT_SECRETS_GCS={uri}\ndo the task")
    assert got == uri and rest == "do the task"
    assert adk_agent._split_secrets_directive("no directive") == (None, "no directive")


def test_split_session_directive():
    sid, rest = adk_agent._split_session_directive("AGENT_SESSION=sid-9\ndo the task")
    assert sid == "sid-9" and rest == "do the task"
    assert adk_agent._split_session_directive("no directive") == (None, "no directive")
    # Stacked directives strip in write order: session, then secrets, then resume.
    prompt = "AGENT_SESSION=s1\nAGENT_SECRETS_GCS=gs://b/k.json\nAGENT_RESUME=s1\ngo"
    sid, prompt = adk_agent._split_session_directive(prompt)
    uri, prompt = adk_agent._split_secrets_directive(prompt)
    resume, prompt = adk_agent._split_resume_directive(prompt)
    assert (sid, uri, resume, prompt) == ("s1", "gs://b/k.json", "s1", "go")


def test_cold_submit_stages_secrets_and_query_carries_only_pointer(monkeypatch):
    # SECURITY regression: the platform persists a job's input verbatim to GCS
    # (jobs/<sid>_input.jsonl), so the query must NEVER contain secret values — only the
    # single-use staged pointer.
    spec = AgentSpec(name="g", model="m")
    engine = backend.GeminiEngine(
        resource="r/reasoningEngines/1", spec=spec,
        project="p", location="l", output_bucket="gs://out",
    )
    captured = {}

    class FakeAE:
        def run_query_job(self, name, config):
            captured["config"] = config
            return {}

    monkeypatch.setattr(engine, "_agent_engines", lambda: FakeAE())
    staged = {}

    def fake_stage(bucket, sid, secrets):
        staged.update(bucket=bucket, sid=sid, secrets=secrets)
        return f"gs://out/invocation-secrets/{sid}-abc.json"

    import remote_agent_toolkit.runtime.gemini.handoff as handoff_mod
    monkeypatch.setattr(handoff_mod, "stage_secrets", fake_stage)

    seed = _seed_sink("sid-3", [
        AgentEvent(kind="result", summary="ok", raw={"subtype": "success", "is_error": False,
                                                     "num_turns": 1, "session_id": "sid-3"}),
    ])
    # _submit tails the GCS event stream — patch the backend seam with our seeded sink.
    monkeypatch.setattr(backend, "tail_stream", lambda uri, sid, **kw: seed.tail(sid))

    session = backend.GeminiSession(engine, "sid-3")
    asyncio.run(_await(session.run("go", secrets={"SH_APIKEY": "SUPERSECRET"})))

    query = captured["config"]["query"]
    assert "SUPERSECRET" not in query                      # never the value
    assert "AGENT_SECRETS_GCS=gs://out/invocation-secrets/sid-3-abc.json" in query
    assert staged["secrets"] == {"SH_APIKEY": "SUPERSECRET"}  # staged out-of-band
    # No session/turn config was bound, so nothing config-related rides the query — the
    # turn runs the deploy-baked spec, with zero extra staging (the pre-config behavior).
    assert "AGENT_SESSION_CONFIG_GCS" not in query
    assert "AGENT_TURN_CONFIG_GCS" not in query
    # Completion cleans up the staged object (backstop; the worker deletes at turn end).
    assert session._staged_secrets_uri is None
    # 2026-07-28 platform-runner regressions: the invoked method must be named
    # explicitly (an unnamed one silently runs nothing on new engines), and the ADK
    # session_id must be omitted (new engines' job workers lost the managed session
    # service; the toolkit sid rides the AGENT_SESSION directive instead).
    import json
    parsed = json.loads(query)
    assert parsed["class_method"] == "async_stream_query"
    assert "session_id" not in parsed["input"]
    assert query.count("AGENT_SESSION=sid-3") == 1


def test_submit_rejects_hooks():
    # A hook is a callable in the caller's process; the turn runs in a remote worker.
    spec = AgentSpec(name="g", model="m")
    engine = backend.GeminiEngine(resource="r/reasoningEngines/1", spec=spec,
                                  project=None, location=None, output_bucket=None)
    session = backend.GeminiSession(engine, "sid")
    import pytest
    with pytest.raises(ValueError, match="local-only"):
        session.run("go", hooks={"PreToolUse": []})


def test_submit_with_secrets_requires_output_bucket():
    spec = AgentSpec(name="g", model="m")
    engine = backend.GeminiEngine(resource="r/reasoningEngines/1", spec=spec,
                                  project=None, location=None, output_bucket=None)
    session = backend.GeminiSession(engine, "sid")
    import pytest
    with pytest.raises(ValueError, match="output bucket"):
        session.run("go", secrets={"K": "v"})


def test_find_baked_skills(tmp_path, monkeypatch):
    # No baked skills dir on the candidate paths -> None.
    monkeypatch.chdir(tmp_path)
    assert adk_agent._find_baked_skills() is None
    # A skills/ dir beside cwd is discovered.
    (tmp_path / "skills").mkdir()
    assert adk_agent._find_baked_skills() == tmp_path / "skills"


def _seed_sink(session_id, events):
    sink = InMemorySink(session_id=session_id)
    for ev in events:
        sink.emit(ev)
    return sink


def test_gemini_session_run_drives_from_sink(monkeypatch):
    spec = AgentSpec(name="g", model="m")
    engine = backend.GeminiEngine(
        resource="projects/p/locations/l/reasoningEngines/123",
        spec=spec, project=None, location=None, output_bucket="gs://out",
    )

    captured = {}

    class FakeAE:
        def run_query_job(self, name, config):
            captured["name"] = name
            captured["config"] = config
            return {"job": "fake"}

    monkeypatch.setattr(engine, "_agent_engines", lambda: FakeAE())

    seed = _seed_sink("sid-1", [
        AgentEvent(kind="message", summary="working"),
        AgentEvent(kind="result", summary="done", cost_usd=0.2, usage={"t": 1},
                   raw={"subtype": "success", "is_error": False, "num_turns": 4, "session_id": "sid-1"}),
    ])
    # _submit tails the GCS event stream — patch the backend seam with our seeded sink.
    monkeypatch.setattr(backend, "tail_stream", lambda uri, sid, **kw: seed.tail(sid))

    session = backend.GeminiSession(engine, "sid-1")
    result = asyncio.run(_await(session.run("go")))

    assert result.text == "done" and result.num_turns == 4 and result.cost_usd == 0.2
    assert result.is_error is False
    assert session.status == RunStatus.IDLE and session.stop_reason == StopReason.END_TURN
    # the run was actually submitted, to the right engine, with the message in the payload
    assert captured["name"] == engine.resource
    assert "go" in captured["config"]["query"]
    assert captured["config"]["output_gcs_uri"] == "gs://out/jobs/sid-1.jsonl"


def test_gemini_interrupt_cancels_remote_job_when_the_worker_has_no_control_channel(monkeypatch):
    # The fallback: the worker never announced the control inbox (a cold job still
    # starting, or an engine deployed before the inbox existed), so interrupt() cannot
    # ask it to stop cleanly — it cancels the tail and the run_query_job, and the run's
    # error result says why there is no checkpoint.
    import types

    spec = AgentSpec(name="g", model="m")
    engine = backend.GeminiEngine(resource="r/reasoningEngines/1", spec=spec,
                                  project=None, location=None, output_bucket="gs://out")
    cancelled = {}

    class FakeAE:
        def run_query_job(self, name, config):
            return types.SimpleNamespace(job_name="jobX")

        def cancel_query_job(self, name, config):
            cancelled["name"] = name
            cancelled["op"] = config["operation_name"]

    monkeypatch.setattr(engine, "_agent_engines", lambda: FakeAE())

    async def silent_tail(*a, **kw):  # the worker never writes anything
        await asyncio.Event().wait()
        yield  # pragma: no cover

    monkeypatch.setattr(backend, "tail_stream", silent_tail)
    monkeypatch.setattr(backend, "_watched_tail", lambda tail, *a, **kw: tail())
    session = backend.GeminiSession(engine, "sid")

    async def go():
        run = session.run("go")
        await asyncio.sleep(0.01)
        await session.interrupt()
        return run

    run = asyncio.run(go())
    assert cancelled == {"name": engine.resource, "op": "jobX"}
    assert run.done and run.result.is_error and "without a checkpoint" in run.result.warning
    assert session.stop_reason == StopReason.ERROR


def test_gemini_session_send_prefixes_resume(monkeypatch):
    spec = AgentSpec(name="g", model="m", checkpoint=True)
    engine = backend.GeminiEngine(resource="r/reasoningEngines/9", spec=spec,
                                  project=None, location=None, output_bucket="gs://out")
    captured = {}

    class FakeAE:
        def run_query_job(self, name, config):
            captured["config"] = config
            return {}

    monkeypatch.setattr(engine, "_agent_engines", lambda: FakeAE())
    seed = _seed_sink("sid-2", [
        AgentEvent(kind="result", summary="resumed", raw={"subtype": "success", "is_error": False,
                                                          "num_turns": 1, "session_id": "sid-2"}),
    ])
    # _submit tails the GCS event stream — patch the backend seam with our seeded sink.
    monkeypatch.setattr(backend, "tail_stream", lambda uri, sid, **kw: seed.tail(sid))

    session = backend.GeminiSession(engine, "sid-2")
    asyncio.run(_await(session.send("answer")))
    assert "AGENT_RESUME=sid-2" in captured["config"]["query"]
    # checkpoint spec => a clean turn end is NEEDS_INPUT (awaiting the next operator message)
    assert session.stop_reason == StopReason.NEEDS_INPUT


def test_claude_session_id_maps_numeric_adk_ids_to_stable_uuid():
    import uuid

    # Cold-path ADK session ids are numeric; the Claude SDK requires a canonical UUID.
    adk_sid = "1966652674296250368"  # the exact shape that crashed the CLI live
    mapped = adk_agent._claude_session_id(adk_sid)
    assert mapped != adk_sid
    assert str(uuid.UUID(mapped)) == mapped                        # canonical UUID
    assert adk_agent._claude_session_id(adk_sid) == mapped         # deterministic (resume-stable)
    # A sid that is already a canonical UUID (warm path) passes through unchanged.
    warm = str(uuid.uuid4())
    assert adk_agent._claude_session_id(warm) == warm


def test_run_turn_maps_session_id_and_surfaces_harness_crash(tmp_path, monkeypatch):
    """Regression for the two silent-death bugs found live (engine 8189…7648):

    (1) the numeric ADK session id must NOT reach the harness verbatim (the claude CLI
        exits 1 on a non-UUID --session-id), and
    (2) a harness crash must emit a TERMINAL error result event — without one the client's
        Cloud Logging tail hangs forever and the failure is invisible.
    """
    monkeypatch.setenv("AGENT_JOBS_ROOT", str(tmp_path / "jobs"))
    monkeypatch.chdir(tmp_path)  # no baked skills dir on the lookup paths

    seen = {}

    class CrashingHarness:
        async def run(self, spec, rc):
            seen["session_id"] = rc.session_id
            raise RuntimeError("Invalid session ID. Must be a valid UUID.")
            yield  # pragma: no cover — makes this an async generator

    import remote_agent_toolkit.harness.claude_code as harness_mod
    import remote_agent_toolkit.ports.eventsink as eventsink_mod
    monkeypatch.setattr(harness_mod, "ClaudeCodeHarness", CrashingHarness)
    sink = InMemorySink(session_id="1966652674296250368")
    monkeypatch.setattr(eventsink_mod, "CloudLoggingSink", lambda **kw: sink)

    spec = AgentSpec(name="w", model="m")
    agent = adk_agent.build_agent(spec)

    async def drive():
        return [ev async for ev in agent._run_turn(
            spec, "1966652674296250368", "go", None, invocation_id="e-inv-9")]

    events = asyncio.run(drive())

    # (1) the harness received the mapped canonical UUID, not the raw numeric id.
    import uuid
    assert str(uuid.UUID(seen["session_id"])) == seen["session_id"]
    # Every surfaced ADK event carries the invocation id — the platform session-append API
    # rejects events without one (400) and the runner appends each non-partial event.
    assert all(ev.invocation_id == "e-inv-9" for ev in events)
    # (2) the stream ends with a terminal error result (workspace_ready + harness_error).
    terminal = events[-1]
    assert terminal.custom_metadata["kind"] == "result"
    assert terminal.custom_metadata["raw"]["is_error"] is True
    assert "agent run failed" in terminal.error_message
    # ...and the same terminal event reached the sink, so a tailing client unblocks too.
    kinds = [e.kind for e in sink._events["1966652674296250368"]]
    assert kinds[-1] == "result"


def test_run_turn_late_crash_keeps_terminal_result(tmp_path, monkeypatch):
    # Same late-death hardening as local: if the harness dies AFTER the terminal result
    # was surfaced (e.g. the claude CLI exiting non-zero during shutdown), the finished
    # result is authoritative — a second, error result here would overwrite a finished
    # run (deliverable + spend) in the tailing client and the durable history.
    monkeypatch.setenv("AGENT_JOBS_ROOT", str(tmp_path / "jobs"))
    monkeypatch.chdir(tmp_path)  # no baked skills dir on the lookup paths

    class DyingHarness:
        async def run(self, spec, rc):
            yield AgentEvent(
                kind="result", summary="all done", cost_usd=9.8,
                raw={"subtype": "success", "is_error": False, "num_turns": 40,
                     "session_id": "claude-sid"},
            )
            raise RuntimeError("Command failed with exit code 1")

    import remote_agent_toolkit.harness.claude_code as harness_mod
    import remote_agent_toolkit.ports.eventsink as eventsink_mod
    monkeypatch.setattr(harness_mod, "ClaudeCodeHarness", DyingHarness)
    sink = InMemorySink(session_id="77")
    monkeypatch.setattr(eventsink_mod, "CloudLoggingSink", lambda **kw: sink)

    spec = AgentSpec(name="w", model="m")
    agent = adk_agent.build_agent(spec)

    async def drive():
        return [ev async for ev in agent._run_turn(spec, "77", "go", None)]

    events = asyncio.run(drive())
    kinds = [ev.custom_metadata["kind"] for ev in events]
    assert kinds.count("result") == 1  # the finished result, never a second error result
    assert events[-1].custom_metadata["kind"] == "status"
    assert "result kept" in events[-1].content.parts[0].text
    # The sink stream (what the client tails / history records) agrees.
    sunk = sink._events["77"]
    assert [e.kind for e in sunk].count("result") == 1
    assert next(e for e in sunk if e.kind == "result").cost_usd == 9.8


def test_run_turn_surfaces_workspace_prep_crash(monkeypatch):
    # Workspace prep (repo clone / skills staging) runs BEFORE the first emitted event; a
    # failure there (e.g. a bad repo token) must also yield a terminal error, not silence.
    def boom(rc, prefer_baked_skills=True):
        raise RuntimeError("git clone failed: fatal: Authentication failed")

    monkeypatch.setattr(adk_agent, "_prepare_workspace", boom)
    import remote_agent_toolkit.ports.eventsink as eventsink_mod
    sink = InMemorySink(session_id="42")
    monkeypatch.setattr(eventsink_mod, "CloudLoggingSink", lambda **kw: sink)

    spec = AgentSpec(name="w", model="m")
    agent = adk_agent.build_agent(spec)

    async def drive():
        return [ev async for ev in agent._run_turn(spec, "42", "go", None)]

    events = asyncio.run(drive())
    # turn_started, then — no prep event was possible — the error is the terminal event.
    assert [e.custom_metadata["kind"] for e in events] == ["status", "result"]
    assert events[0].custom_metadata["raw"]["event"] == "turn_started"
    assert events[1].custom_metadata["raw"]["is_error"] is True
    assert "git clone failed" in events[1].error_message


def _patched_local_handoff_store(tmp_path, monkeypatch):
    """Route the worker-side handoff (fetch/delete, normally GCS) to a LocalBlobStore."""
    from remote_agent_toolkit.ports.blobstore import LocalBlobStore
    from remote_agent_toolkit.runtime.gemini import handoff

    store = LocalBlobStore(str(tmp_path / "blobs"))
    monkeypatch.setattr(handoff, "GcsBlobStore", lambda bucket, *a, **kw: store)
    return store, handoff


def test_run_turn_deletes_staged_secrets_only_at_terminal_result(tmp_path, monkeypatch):
    """Regression for the 2026-07-28/29 live failures (engines 9871…6672, 8563…8240).

    The platform's job runner re-runs a killed attempt with the SAME payload, so the staged
    secrets must survive the read and go only once the turn's terminal result is out — the
    old fetch-and-delete-on-read stranded every retry credential-less (its repo clone died
    with "could not read Username for 'https://bitbucket.org'").
    """
    monkeypatch.setenv("AGENT_JOBS_ROOT", str(tmp_path / "jobs"))
    monkeypatch.chdir(tmp_path)  # no baked skills dir on the lookup paths
    store, handoff = _patched_local_handoff_store(tmp_path, monkeypatch)
    uri = handoff.stage_secrets("gs://bkt", "77", {"REPO_TOKEN": "tok"}, store=store)
    key = uri[len("gs://bkt/"):]

    seen = {}

    class DoneHarness:
        async def run(self, spec, rc):
            seen["secrets"] = dict(rc.secrets)
            seen["staged_mid_run"] = store.exists(key)
            yield AgentEvent(kind="result", summary="done",
                             raw={"subtype": "success", "is_error": False})

    import remote_agent_toolkit.harness.claude_code as harness_mod
    import remote_agent_toolkit.ports.eventsink as eventsink_mod
    monkeypatch.setattr(harness_mod, "ClaudeCodeHarness", DoneHarness)
    monkeypatch.setattr(eventsink_mod, "CloudLoggingSink",
                        lambda **kw: InMemorySink(session_id="77"))

    spec = AgentSpec(name="w", model="m")
    agent = adk_agent.build_agent(spec)

    async def drive():
        return [ev async for ev in agent._run_turn(spec, "77", "go", None, secrets_uri=uri)]

    events = asyncio.run(drive())
    assert events[-1].custom_metadata["kind"] == "result"
    assert seen["secrets"] == {"REPO_TOKEN": "tok"}  # values reached the harness
    assert seen["staged_mid_run"] is True  # fetch did NOT consume (a retry must find it)
    assert not store.exists(key)  # ...but the completed turn reaped it


def test_run_turn_killed_attempt_leaves_secrets_for_the_retry(tmp_path, monkeypatch):
    # Attempt 1 dies mid-turn (generator closed without a terminal result — the in-process
    # analogue of the platform killing the worker): the staged object must remain, so the
    # platform's automatic re-run of the same payload (attempt 2) still gets credentials.
    monkeypatch.setenv("AGENT_JOBS_ROOT", str(tmp_path / "jobs"))
    monkeypatch.chdir(tmp_path)  # no baked skills dir on the lookup paths
    store, handoff = _patched_local_handoff_store(tmp_path, monkeypatch)
    uri = handoff.stage_secrets("gs://bkt", "77", {"REPO_TOKEN": "tok"}, store=store)
    key = uri[len("gs://bkt/"):]

    fetched = []

    class TwoEventHarness:
        async def run(self, spec, rc):
            fetched.append(dict(rc.secrets))
            yield AgentEvent(kind="status", summary="working")
            yield AgentEvent(kind="result", summary="done",
                             raw={"subtype": "success", "is_error": False})

    import remote_agent_toolkit.harness.claude_code as harness_mod
    import remote_agent_toolkit.ports.eventsink as eventsink_mod
    monkeypatch.setattr(harness_mod, "ClaudeCodeHarness", TwoEventHarness)
    monkeypatch.setattr(eventsink_mod, "CloudLoggingSink",
                        lambda **kw: InMemorySink(session_id="77"))

    spec = AgentSpec(name="w", model="m")
    agent = adk_agent.build_agent(spec)

    async def attempt_1_killed():
        gen = agent._run_turn(spec, "77", "go", None, secrets_uri=uri)
        async for ev in gen:
            if ev.custom_metadata["kind"] == "status" and "working" in ev.content.parts[0].text:
                break  # die mid-turn
        await gen.aclose()

    asyncio.run(attempt_1_killed())
    assert store.exists(key)  # no terminal result -> the object survives for the retry

    async def attempt_2():
        return [ev async for ev in agent._run_turn(spec, "77", "go", None, secrets_uri=uri)]

    events = asyncio.run(attempt_2())
    assert fetched[-1] == {"REPO_TOKEN": "tok"}  # the retry got the credentials
    assert events[-1].custom_metadata["kind"] == "result"
    assert not store.exists(key)  # the completed retry reaped it


def test_run_turn_reaps_staged_secrets_on_application_failure_too(tmp_path, monkeypatch):
    # An application-level failure still emits a terminal result and exits cleanly — the
    # platform never retries it, so the staged object is reaped just like on success.
    monkeypatch.setenv("AGENT_JOBS_ROOT", str(tmp_path / "jobs"))
    monkeypatch.chdir(tmp_path)  # no baked skills dir on the lookup paths
    store, handoff = _patched_local_handoff_store(tmp_path, monkeypatch)
    uri = handoff.stage_secrets("gs://bkt", "77", {"K": "v"}, store=store)
    key = uri[len("gs://bkt/"):]

    class CrashingHarness:
        async def run(self, spec, rc):
            raise RuntimeError("boom")
            yield  # pragma: no cover — makes this an async generator

    import remote_agent_toolkit.harness.claude_code as harness_mod
    import remote_agent_toolkit.ports.eventsink as eventsink_mod
    monkeypatch.setattr(harness_mod, "ClaudeCodeHarness", CrashingHarness)
    monkeypatch.setattr(eventsink_mod, "CloudLoggingSink",
                        lambda **kw: InMemorySink(session_id="77"))

    spec = AgentSpec(name="w", model="m")
    agent = adk_agent.build_agent(spec)

    async def drive():
        return [ev async for ev in agent._run_turn(spec, "77", "go", None, secrets_uri=uri)]

    events = asyncio.run(drive())
    assert events[-1].custom_metadata["kind"] == "result"
    assert events[-1].custom_metadata["raw"]["is_error"] is True
    assert not store.exists(key)


def test_warm_start_session_mints_canonical_uuid():
    # Warm-pool sessions get a client-chosen id that reaches the Claude Agent SDK as
    # session_id / resume; it must be a CANONICAL UUID (dashed), not uuid4().hex, or the SDK
    # rejects it at runtime ("Invalid session ID. Must be a valid UUID"). No GCP calls here.
    import uuid

    engine = backend.GeminiEngine(resource="r/reasoningEngines/1",
                                  spec=AgentSpec(name="s", model="m"),
                                  project=None, location=None, warm=True)
    sid = engine.start_session().session_id
    assert "-" in sid
    assert str(uuid.UUID(sid)) == sid


def _transcript_engine(spec, monkeypatch, tmp_path):
    """A gemini engine whose checkpoint prefix is a local blob store (no GCP)."""
    import remote_agent_toolkit.ports.blobstore as bs

    blobs = bs.LocalBlobStore(str(tmp_path))
    monkeypatch.setattr(bs, "GcsBlobStore", lambda bucket, prefix: blobs)
    engine = backend.GeminiEngine(resource="r/reasoningEngines/1", spec=spec,
                                  project=None, location=None, output_bucket="gs://out")
    return engine, blobs


def _attached(engine, session_id):
    session = backend.GeminiSession(engine, session_id)
    session._session_config_resolved = True  # skip the (absent) persisted config's GCS read
    return session


def test_transcripts_read_the_id_the_worker_wrote_under(tmp_path, monkeypatch):
    # Cold sessions carry a NUMERIC ADK id, and the worker keys the store by the canonical
    # uuid5 the Claude SDK requires — so reading by the raw id finds nothing, which is
    # indistinguishable from "the session persisted nothing".
    from remote_agent_toolkit.checkpoint.session_store import (
        BlobSessionStore,
        _claude_session_id,
    )

    spec = AgentSpec(name="g", model="m", transcript=True)
    engine, blobs = _transcript_engine(spec, monkeypatch, tmp_path)

    raw_sid = "1966652674296250368"  # the cold-path shape
    entries = [{"uuid": "u1", "type": "user"}, {"uuid": "u2", "type": "assistant"}]
    store = BlobSessionStore(blobs)
    asyncio.run(store.append({"session_id": _claude_session_id(raw_sid)}, entries))
    asyncio.run(store.append(
        {"session_id": _claude_session_id(raw_sid), "subpath": "subagents/agent-X"},
        [{"uuid": "u3", "type": "user"}],
    ))

    assert asyncio.run(_attached(engine, raw_sid).transcripts()) == {
        "main": entries,
        "subagents/agent-X": [{"uuid": "u3", "type": "user"}],
    }


def test_transcripts_raise_only_when_the_spec_is_known_to_persist_nothing(tmp_path, monkeypatch):
    import pytest

    spec = AgentSpec(name="g", model="m")  # neither checkpoint nor transcript
    engine, _ = _transcript_engine(spec, monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match="transcript=True"):
        asyncio.run(_attached(engine, "sid").transcripts())

    # A get_engine handle's spec is addressing-only, so its defaults prove nothing about
    # what the engine bakes: read and report the truth on the wire instead of guessing.
    engine._spec_known = False
    assert asyncio.run(_attached(engine, "sid").transcripts()) == {}


def test_transcript_only_spec_does_not_resume_the_conversation(tmp_path, monkeypatch):
    # transcript=True wires the session store, but resume stays checkpointing's: on gemini
    # every turn starts on a fresh worker with no workspace snapshot to restore, so a
    # resumed conversation would remember files that no longer exist.
    monkeypatch.setenv("AGENT_JOBS_ROOT", str(tmp_path / "jobs"))
    monkeypatch.setenv("AGENT_CHECKPOINT_GCS", "gs://out/checkpoints")
    monkeypatch.chdir(tmp_path)  # no baked skills dir on the lookup paths
    import remote_agent_toolkit.ports.blobstore as bs

    monkeypatch.setattr(bs, "GcsBlobStore", lambda bucket, prefix: bs.LocalBlobStore(str(tmp_path)))
    seen = {}

    class RecordingHarness:
        async def run(self, spec, rc):
            seen["resume_sid"] = rc.resume_sid
            seen["session_store"] = rc.session_store
            yield AgentEvent(kind="result", summary="done",
                             raw={"subtype": "success", "is_error": False, "session_id": "s"})

    import remote_agent_toolkit.harness.claude_code as harness_mod
    import remote_agent_toolkit.ports.eventsink as eventsink_mod
    monkeypatch.setattr(harness_mod, "ClaudeCodeHarness", RecordingHarness)
    monkeypatch.setattr(eventsink_mod, "CloudLoggingSink",
                        lambda **kw: InMemorySink(session_id="sid-t"))

    spec = AgentSpec(name="w", model="m", transcript=True)
    agent = adk_agent.build_agent(spec)
    asyncio.run(_drain(agent._run_turn(spec, "sid-t", "go", "sid-t")))
    assert seen["session_store"] is not None  # the transcript is still mirrored
    assert seen["resume_sid"] is None


def test_deploy_rejects_the_local_only_workspace_argument():
    # A deployed engine has no host directory to run turns in, so the knob cannot mean
    # anything there — say so instead of dropping it, which is how someone graduating a
    # local script to gemini loses the shared cwd without noticing. Fails before any GCP
    # import or billable side effect.
    with pytest.raises(ValueError, match="local-only"):
        backend.deploy(AgentSpec(name="w", model="m"), project="p", location="l",
                       workspace="/some/dir")

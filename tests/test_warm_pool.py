"""Warm-pool offline tests: helpers, the control-plane warm path, and the worker claim.

Exercised with fakes (InMemoryDispatch + InMemorySink) — the live ~5s warm pickup is
validated separately on real infra.
"""

from __future__ import annotations

import asyncio
import json

from remote_agent_toolkit import AgentSpec
from remote_agent_toolkit.events import AgentEvent, RunStatus, StopReason
from remote_agent_toolkit.ports.dispatch import InMemoryDispatch
from remote_agent_toolkit.ports.eventsink import InMemorySink
from remote_agent_toolkit.runtime.gemini import adk_agent, backend, pool


async def _await(run):
    return await run


def test_pool_helpers(monkeypatch):
    topic, sub = pool.pool_paths("proj", "Spider-Builder")
    assert topic == "projects/proj/topics/ratk-spider-builder-dispatch"
    assert sub == "projects/proj/subscriptions/ratk-spider-builder-dispatch-sub"

    assert pool.dispatch_payload("s1", "go", True) == {
        "session_id": "s1", "message": "go", "resume": True,
    }
    # SECURITY: the payload carries only the staged-secrets POINTER, never values (a Pub/Sub
    # message is retained until acked; the platform persists a cold job's input verbatim).
    with_secrets = pool.dispatch_payload("s1", "go", False, "gs://bkt/invocation-secrets/s1-x.json")
    assert with_secrets["secrets_gcs"] == "gs://bkt/invocation-secrets/s1-x.json"
    assert "secrets" not in with_secrets

    monkeypatch.delenv("AGENT_POOL_SUBSCRIPTION", raising=False)
    assert pool.worker_dispatch_from_env() is None  # not a pool worker without the env

    # Readiness pool-id is consistent: derived from the name (control plane) == from the
    # subscription (worker side), so both tail/emit the same Cloud Logging key.
    assert pool.pool_log_id("Spider-Builder") == "ratk-spider-builder-pool"
    assert pool.pool_log_id_from_subscription(sub) == pool.pool_log_id("Spider-Builder")


def test_warm_session_dispatches_and_tails(monkeypatch):
    spec = AgentSpec(name="w", model="m")
    engine = backend.GeminiEngine(
        resource="r/reasoningEngines/1", spec=spec, project=None, location=None,
        output_bucket=None, warm=True, topic="t", subscription="s",
    )

    published = []
    monkeypatch.setattr(engine, "_dispatch", lambda: type("D", (), {"publish": lambda _self, m: published.append(m)})())
    refilled = []
    monkeypatch.setattr(engine, "fill_pool", lambda n: refilled.append(n))

    # The worker would emit to the session_id; pre-seed a sink that the client tails.
    seed = InMemorySink(session_id="warm-sid")
    seed.emit(AgentEvent(kind="message", summary="working"))
    seed.emit(AgentEvent(kind="result", summary="done", cost_usd=0.1,
                         raw={"subtype": "success", "is_error": False, "num_turns": 3, "session_id": "warm-sid"}))
    import remote_agent_toolkit.ports.eventsink as eventsink_mod
    monkeypatch.setattr(eventsink_mod, "CloudLoggingSink", lambda **kw: seed)

    session = backend.GeminiSession(engine, "warm-sid")
    result = asyncio.run(_await(session.run("go")))

    # The turn was dispatched to the pool (not cold-started) and the pool was refilled.
    assert published == [{"session_id": "warm-sid", "message": "go", "resume": False}]
    assert refilled == [1]
    assert result.text == "done" and result.num_turns == 3
    assert session.status == RunStatus.IDLE and session.stop_reason == StopReason.END_TURN


def test_fill_pool_names_query_job_method(monkeypatch):
    captured = []

    class FakeAE:
        def run_query_job(self, name, config):
            captured.append((name, config))
            return {}

    engine = backend.GeminiEngine(
        resource="r/reasoningEngines/1",
        spec=AgentSpec(name="w", model="m"),
        project="p",
        location="l",
        output_bucket="gs://out",
        warm=True,
    )
    monkeypatch.setattr(engine, "_agent_engines", lambda: FakeAE())

    engine.fill_pool(1)

    payload = json.loads(captured[0][1]["query"])
    assert payload["class_method"] == "async_stream_query"
    assert payload["input"] == {"user_id": "ratk", "message": pool.POOL_WAIT_SENTINEL}


def test_run_blocking_works_from_sync_and_async_contexts():
    # wait_until_warm is a blocking helper that must work whether or not the caller is inside a
    # running event loop (regression: it used asyncio.run() and crashed from async app code).
    from remote_agent_toolkit.runtime.gemini.backend import _run_blocking

    async def ok():
        return True

    assert _run_blocking(ok, 5) is True                       # sync context: no ambient loop

    async def driver():
        return _run_blocking(ok, 5)                           # called from inside a running loop

    assert asyncio.run(driver()) is True

    async def slow():
        await asyncio.sleep(1)
        return True

    assert _run_blocking(slow, 0.01) is False                 # timeout -> False


def test_engine_delete_cancels_pool_jobs_then_deletes(monkeypatch):
    spec = AgentSpec(name="w", model="m")
    engine = backend.GeminiEngine(
        resource="r/reasoningEngines/1", spec=spec, project=None, location=None,
        warm=True, topic="t", subscription="s",
    )
    engine._pool_jobs = ["jobA", "jobB"]
    calls = {"cancel": [], "delete": 0}

    class FakeAE:
        def cancel_query_job(self, name, config):
            calls["cancel"].append(config["operation_name"])

        def delete(self, name, force):
            calls["delete"] += 1

    monkeypatch.setattr(engine, "_agent_engines", lambda: FakeAE())
    engine.delete()  # delete_pool_resources defaults False -> no pubsub calls
    # Both tracked workers cancelled before the engine is deleted.
    assert calls["cancel"] == ["jobA", "jobB"]
    assert calls["delete"] == 1


def test_engine_delete_cancels_blocking_ops_from_error(monkeypatch):
    # A reused engine has no tracked jobs; the running worker is named in the delete error.
    spec = AgentSpec(name="w", model="m")
    engine = backend.GeminiEngine(resource="projects/p/locations/l/reasoningEngines/1",
                                  spec=spec, project="p", location="l")
    cancelled = []
    op = "projects/p/locations/l/operations/999"

    class FakeAE:
        def __init__(self):
            self.attempts = 0

        def cancel_query_job(self, name, config):
            cancelled.append(config["operation_name"])

        def delete(self, name, force):
            self.attempts += 1
            if self.attempts == 1:  # first attempt blocked, naming the running op
                raise RuntimeError(f"400 FAILED_PRECONDITION ... Operation(s) are: {op}.")
            # second attempt succeeds (after the cancel)

    fake = FakeAE()
    monkeypatch.setattr(engine, "_agent_engines", lambda: fake)
    monkeypatch.setattr(backend.time, "sleep", lambda _s: None)  # no real wait in the test
    engine.delete()
    assert cancelled == [op]  # parsed out of the error and cancelled
    assert fake.attempts == 2  # retried after cancelling, then succeeded


def test_pool_worker_claims_and_runs(monkeypatch):
    spec = AgentSpec(name="w", model="m")
    agent = adk_agent.build_agent(spec)

    # A dispatch pre-seeded with one turn assignment (the worker should claim it), carrying
    # the staged-secrets pointer that must flow through to the turn (values never ride here).
    disp = InMemoryDispatch()
    disp.publish({"session_id": "dispatched-sid", "message": "do it", "resume": False,
                  "secrets_gcs": "gs://bkt/invocation-secrets/dispatched-sid-x.json",
                  "spec_gcs": "gs://bkt/invocation-spec/dispatched-sid-x.json"})
    monkeypatch.setattr(pool, "worker_dispatch_from_env", lambda *a, **k: disp)

    # _prewarm emits a readiness marker via a real CloudLoggingSink; off-GCP that stalls for
    # ~55s discovering ambient credentials. Fake the sink (same pattern as the warm-path test
    # above) so this test exercises the claim/run logic without a live GCP round-trip.
    import remote_agent_toolkit.ports.eventsink as eventsink_mod
    monkeypatch.setattr(eventsink_mod, "CloudLoggingSink", lambda **kw: InMemorySink(**kw))

    # Replace the heavy turn (real harness/model) with a recorder.
    seen = {}

    async def fake_run_turn(spec_, session_id, prompt, resume_sid, secrets_uri=None,
                            invocation_id="", spec_uri=None):
        seen.update(session_id=session_id, prompt=prompt, resume_sid=resume_sid,
                    secrets_uri=secrets_uri, invocation_id=invocation_id, spec_uri=spec_uri)
        yield "turn-event"

    monkeypatch.setattr(agent, "_run_turn", fake_run_turn)

    async def drive():
        return [ev async for ev in agent._pool_worker(spec, "e-inv-77")]

    events = asyncio.run(drive())
    # Claimed immediately (no heartbeat), then handed the dispatched turn (+pointer) to _run_turn.
    # invocation_id must flow through: the warm path missed it at first and every session
    # append kept 400ing on live engines while the cold path was fixed.
    assert events == ["turn-event"]
    assert seen == {"session_id": "dispatched-sid", "prompt": "do it", "resume_sid": None,
                    "secrets_uri": "gs://bkt/invocation-secrets/dispatched-sid-x.json",
                    "spec_uri": "gs://bkt/invocation-spec/dispatched-sid-x.json",
                    "invocation_id": "e-inv-77"}

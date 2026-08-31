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
    # Legacy (no generation): the fixed names shared by every revision — kept only as the
    # get_engine discovery fallback.
    topic, sub = pool.pool_paths("proj", "Spider-Builder")
    assert topic == "projects/proj/topics/ratk-spider-builder-dispatch"
    assert sub == "projects/proj/subscriptions/ratk-spider-builder-dispatch-sub"

    # Generation-scoped (every deploy since #38): a fresh pair per deploy, so workers of a
    # previous revision can never claim a turn dispatched after a redeploy.
    gtopic, gsub = pool.pool_paths("proj", "Spider-Builder", generation="abc12345")
    assert gtopic == "projects/proj/topics/ratk-spider-builder-abc12345-dispatch"
    assert gsub == "projects/proj/subscriptions/ratk-spider-builder-abc12345-dispatch-sub"
    for pair_sub, pair_topic in ((sub, topic), (gsub, gtopic)):
        assert pool.topic_for_subscription(pair_sub) == pair_topic  # env value → full pair

    gen1, gen2 = pool.new_generation(), pool.new_generation()
    assert gen1 != gen2 and gen1.isalnum()

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

    # Idle life: the default is a day (an expired worker is never replaced, so a short
    # default silently drained pools over any quiet gap); deploy(pool_max_wait_s=...)
    # overrides it via the env var build_env() bakes into the engine.
    monkeypatch.delenv("AGENT_POOL_MAX_WAIT_S", raising=False)
    assert pool.resolve_max_wait_s() == 24 * 3600
    monkeypatch.setenv("AGENT_POOL_MAX_WAIT_S", "7200")
    assert pool.resolve_max_wait_s() == 7200.0

    # Readiness pool-id is consistent: derived from the name (control plane) == from the
    # subscription (worker side), so both tail/emit the same Cloud Logging key. With a
    # generation it is scoped to the deploy, so a fresh pool never counts a previous
    # deploy's readiness markers as its own.
    assert pool.pool_log_id("Spider-Builder") == "ratk-spider-builder-pool"
    assert pool.pool_log_id_from_subscription(sub) == pool.pool_log_id("Spider-Builder")
    assert pool.pool_log_id_from_subscription(gsub) == "ratk-spider-builder-abc12345-pool"


def test_deploy_rejects_pool_max_wait_without_pool():
    """The knob must fail loudly when it cannot take effect (deploy()'s **_ catch-all
    would otherwise swallow it silently — the exact surprise the parameter removes)."""
    import pytest

    with pytest.raises(ValueError, match="warm_pool"):
        backend.deploy(AgentSpec(name="x", model="m"), "proj", "loc", pool_max_wait_s=3600.0)


def test_deploy_rejects_invalid_pool_max_wait_values():
    """Value constraints hold at the deploy() boundary, BEFORE any side effect: rejected
    only in build_env() (which runs after the pub/sub ensure), an invalid value left an
    orphaned topic/sub behind. NaN is the nastiest of the four: it passes a bare `<= 0`
    check, and a NaN deadline makes every worker idle-expire instantly — silently
    recreating the permanently-cold pool this knob exists to prevent."""
    import pytest

    for bad in (0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive, finite"):
            backend.deploy(AgentSpec(name="x", model="m"), "proj", "loc",
                           warm_pool=True, pool_max_wait_s=bad)


def test_warm_session_dispatches_and_tails(monkeypatch):
    spec = AgentSpec(name="w", model="m")
    engine = backend.GeminiEngine(
        resource="r/reasoningEngines/1", spec=spec, project=None, location=None,
        output_bucket="gs://out", warm=True, topic="t", subscription="s",
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
    # _submit tails the GCS event stream — patch the backend seam with our seeded sink.
    monkeypatch.setattr(backend, "tail_stream", lambda uri, sid, **kw: seed.tail(sid))

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
                  "session_config_gcs": "gs://bkt/session-config/dispatched-sid.json",
                  "turn_config_gcs": "gs://bkt/turn-config/dispatched-sid-x.json"})
    monkeypatch.setattr(pool, "worker_dispatch_from_env", lambda *a, **k: disp)

    # _prewarm emits a readiness marker via a real CloudLoggingSink; off-GCP that stalls for
    # ~55s discovering ambient credentials. Fake the sink (same pattern as the warm-path test
    # above) so this test exercises the claim/run logic without a live GCP round-trip.
    import remote_agent_toolkit.ports.eventsink as eventsink_mod
    monkeypatch.setattr(eventsink_mod, "CloudLoggingSink", lambda **kw: InMemorySink(**kw))

    # Replace the heavy turn (real harness/model) with a recorder.
    seen = {}

    async def fake_run_turn(spec_, session_id, prompt, resume_sid, secrets_uri=None,
                            invocation_id="", session_config_uri=None, turn_config_uri=None):
        seen.update(session_id=session_id, prompt=prompt, resume_sid=resume_sid,
                    secrets_uri=secrets_uri, invocation_id=invocation_id,
                    session_config_uri=session_config_uri, turn_config_uri=turn_config_uri)
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
                    "session_config_uri": "gs://bkt/session-config/dispatched-sid.json",
                    "turn_config_uri": "gs://bkt/turn-config/dispatched-sid-x.json",
                    "invocation_id": "e-inv-77"}


def test_pool_worker_idle_expires_at_the_deadline(monkeypatch):
    """Drive the wait loop to expiry offline: an empty dispatch and a tiny
    ``AGENT_POOL_MAX_WAIT_S`` must end in ``pool_idle_expired`` — no turn, no crash. Pins
    that the env knob actually bounds the wait (the deadline arithmetic the live probe
    exercised) with only heartbeats before the expiry marker."""
    spec = AgentSpec(name="w", model="m")
    agent = adk_agent.build_agent(spec)

    class EmptyDispatch:  # instant no-message long-poll; never a real 5s block in the test
        def claim(self, timeout):
            return None

    monkeypatch.setattr(pool, "worker_dispatch_from_env", lambda *a, **k: EmptyDispatch())
    monkeypatch.setenv("AGENT_POOL_MAX_WAIT_S", "0.05")
    # Same fake sink as the claim test: _prewarm's readiness emit must not touch GCP.
    import remote_agent_toolkit.ports.eventsink as eventsink_mod
    monkeypatch.setattr(eventsink_mod, "CloudLoggingSink", lambda **kw: InMemorySink(**kw))

    async def drive():
        return [ev async for ev in agent._pool_worker(spec, "e-inv-88")]

    events = asyncio.run(drive())
    raws = [ev.custom_metadata["raw"] for ev in events]
    assert raws[-1]["event"] == "pool_idle_expired"
    assert raws[:-1] and all(r["event"] == "pool_waiting" for r in raws[:-1])


def _engine_resource(env: dict[str, str], name: str = "", traffic_targets=None):
    """A minimal fake of an engine/revision api_resource carrying a deployment env."""
    from types import SimpleNamespace as NS

    traffic_config = None
    if traffic_targets is not None:
        traffic_config = NS(traffic_split_manual=NS(targets=[
            NS(runtime_revision_name=rev_name, percent=percent)
            for rev_name, percent in traffic_targets
        ]))
    return NS(
        name=name,
        traffic_config=traffic_config,
        spec=NS(deployment_spec=NS(env=[NS(name=k, value=v) for k, v in env.items()])),
    )


class _FakeClient:
    """agent_engines.get / runtimes.revisions.list, over fake api_resources."""

    def __init__(self, engine_api, revision_apis=()):
        class Wrapped:
            def __init__(self, api):
                self.api_resource = api

        class Revisions:
            def list(self, name):
                return [Wrapped(api) for api in revision_apis]

        class Runtimes:
            revisions = Revisions()

        class AgentEngines:
            runtimes = Runtimes()

            def get(self, name):
                return Wrapped(engine_api)

        self.agent_engines = AgentEngines()


def test_get_engine_discovers_generation_scoped_pair_from_env():
    """The #38 cutover makes the dispatch pair per-deploy, so get_engine can no longer
    derive it from the engine name: it must read AGENT_POOL_SUBSCRIPTION back from the
    deployed env (the same value the workers themselves read at runtime)."""
    sub = "projects/proj/subscriptions/ratk-w-abc12345-dispatch-sub"
    client = _FakeClient(_engine_resource({"AGENT_POOL_SUBSCRIPTION": sub}))

    topic, found = backend._discover_pool_paths(client, "r/reasoningEngines/1", "proj", "w")
    assert found == sub
    assert topic == "projects/proj/topics/ratk-w-abc12345-dispatch"


def test_get_engine_pool_discovery_prefers_the_pinned_revision():
    """Workers always cold-start on the SERVING revision, so with traffic pinned the pair
    they pull is the pinned revision's — not the latest deploy's."""
    resource = "r/reasoningEngines/1"
    old_sub = "projects/proj/subscriptions/ratk-w-old00000-dispatch-sub"
    new_sub = "projects/proj/subscriptions/ratk-w-new11111-dispatch-sub"
    pinned_rev = f"{resource}/runtimeRevisions/5"
    client = _FakeClient(
        _engine_resource({"AGENT_POOL_SUBSCRIPTION": new_sub},
                         traffic_targets=[(pinned_rev, 100)]),
        revision_apis=[
            _engine_resource({"AGENT_POOL_SUBSCRIPTION": old_sub}, name=pinned_rev),
            _engine_resource({"AGENT_POOL_SUBSCRIPTION": new_sub},
                             name=f"{resource}/runtimeRevisions/6"),
        ],
    )

    topic, found = backend._discover_pool_paths(client, resource, "proj", "w")
    assert found == old_sub
    assert topic == "projects/proj/topics/ratk-w-old00000-dispatch"


def test_get_engine_pool_discovery_falls_back_to_legacy_names():
    """No AGENT_POOL_SUBSCRIPTION in the deployed env (an engine deployed without
    warm_pool) → the pre-#38 fixed names, so old addressing keeps working."""
    client = _FakeClient(_engine_resource({}))
    assert backend._discover_pool_paths(client, "r/reasoningEngines/1", "proj", "w") == (
        "projects/proj/topics/ratk-w-dispatch",
        "projects/proj/subscriptions/ratk-w-dispatch-sub",
    )

    import pytest

    with pytest.raises(ValueError, match="dispatch subscription"):
        backend._discover_pool_paths(client, "r/reasoningEngines/1", None, "w")


def test_wait_until_warm_tails_the_generation_scoped_pool_id(monkeypatch):
    """The readiness marker key must be the one THIS pool's workers emit under — derived
    from the generation-scoped subscription, not the bare engine name."""
    engine = backend.GeminiEngine(
        resource="r/reasoningEngines/1", spec=AgentSpec(name="w", model="m"),
        project=None, location=None, output_bucket="gs://out", warm=True,
        topic="projects/p/topics/ratk-w-abc12345-dispatch",
        subscription="projects/p/subscriptions/ratk-w-abc12345-dispatch-sub",
    )
    tailed = []

    async def fake_tail(uri, sid, **kw):
        tailed.append(sid)
        yield AgentEvent(kind="status", summary="pool worker ready")

    monkeypatch.setattr(backend, "tail_stream", fake_tail)
    assert engine.wait_until_warm(timeout=5) is True
    assert tailed == ["ratk-w-abc12345-pool"]

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
from remote_agent_toolkit.runtime.gemini import roster as roster_mod


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
    assert pool.pool_topic("proj", "Spider-Builder", generation="abc12345") == gtopic
    for pair_sub, pair_topic in ((sub, topic), (gsub, gtopic)):
        assert pool.topic_for_subscription(pair_sub) == pair_topic  # legacy env value → pair

    gen1, gen2 = pool.new_generation(), pool.new_generation()
    assert gen1 != gen2 and gen1.isalnum()

    # Per-worker dispatch: one random, unguessable subscription per worker under the
    # generation's topic, filtered to messages addressed to that worker. The id is the
    # capability (pubsub.subscriber can consume by name but not list), so it must be long
    # and fresh every time.
    wid = pool.new_worker_id()
    assert len(wid) == 24 and wid != pool.new_worker_id()
    wsub = pool.worker_subscription(gtopic, wid)
    assert wsub == f"projects/proj/subscriptions/ratk-spider-builder-abc12345-w-{wid}"
    assert wsub.startswith(pool.worker_subscription_prefix(gtopic))
    assert pool.worker_id_from_subscription(wsub) == wid
    assert pool.worker_id_from_subscription(gsub) is None  # a legacy shared sub has no worker
    assert pool.worker_filter(wid) == f'attributes.worker = "{wid}"'
    assert pool.worker_attributes(wid) == {"worker": wid}
    assert pool.roster_prefix(gtopic) == "pool/ratk-spider-builder-abc12345/idle/"

    # A worker learns its own subscription from its job input; a bare sentinel (an older
    # client's fill_pool) is still recognised as a pool worker, with no channel.
    assert pool.parse_pool_wait(pool.pool_wait_prompt(wsub)) == (True, wsub)
    assert pool.parse_pool_wait(pool.POOL_WAIT_SENTINEL) == (True, None)
    assert pool.parse_pool_wait("AGENT_SESSION=x\nhello") == (False, None)

    # Pickup deadline: the rest of the boot window plus the grace; grace alone once booted.
    assert pool.pickup_deadline_s(1000.0, now=1000.0) == pool.WORKER_BOOT_S + pool.PICKUP_GRACE_S
    assert pool.pickup_deadline_s(0.0, now=10_000.0) == pool.PICKUP_GRACE_S

    assert pool.dispatch_payload("s1", "go", True) == {
        "session_id": "s1", "message": "go", "resume": True,
    }
    # SECURITY: the payload carries only the staged-secrets POINTER, never values (a Pub/Sub
    # message is retained until acked; the platform persists a cold job's input verbatim).
    with_secrets = pool.dispatch_payload("s1", "go", False, "gs://bkt/invocation-secrets/s1-x.json")
    assert with_secrets["secrets_gcs"] == "gs://bkt/invocation-secrets/s1-x.json"
    assert "secrets" not in with_secrets

    # Idle life: the default is a day (an expired worker is never replaced, so a short
    # default silently drained pools over any quiet gap); deploy(pool_max_wait_s=...)
    # overrides it via the env var build_env() bakes into the engine.
    monkeypatch.delenv("AGENT_POOL_MAX_WAIT_S", raising=False)
    assert pool.resolve_max_wait_s() == 24 * 3600
    monkeypatch.setenv("AGENT_POOL_MAX_WAIT_S", "7200")
    assert pool.resolve_max_wait_s() == 7200.0

    # Readiness pool-id is consistent: derived from the name (control plane) == from the
    # topic (worker side, AGENT_POOL_TOPIC) == from a legacy shared subscription, so all
    # tail/emit the same key. With a generation it is scoped to the deploy, so a fresh
    # pool never counts a previous deploy's readiness markers as its own.
    assert pool.pool_log_id("Spider-Builder") == "ratk-spider-builder-pool"
    assert pool.pool_log_id_from_topic(topic) == pool.pool_log_id("Spider-Builder")
    assert pool.pool_log_id_from_topic(gtopic) == "ratk-spider-builder-abc12345-pool"
    assert pool.pool_log_id_from_subscription(gsub) == pool.pool_log_id_from_topic(gtopic)


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


def test_legacy_warm_session_dispatches_to_the_shared_subscription(monkeypatch):
    """An engine deployed before per-worker dispatch (a shared subscription in its env)
    is still driven the old way: unaddressed publish, no roster, no pickup watchdog."""
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
    engine._join_background()  # the refill runs off the turn's critical path

    # The turn was dispatched to the pool (not cold-started) and the pool was refilled.
    # The payload carries the run-scoped GCS token (a fake here: conftest stubs minting).
    assert published == [{"session_id": "warm-sid", "message": "go", "resume": False,
                          "gcs_token": "fake-run-token"}]
    assert refilled == [1]
    assert result.text == "done" and result.num_turns == 3
    assert session.status == RunStatus.IDLE and session.stop_reason == StopReason.END_TURN


def test_legacy_fill_pool_names_query_job_method(monkeypatch):
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
        topic="t",
        subscription="s",  # legacy shared-subscription pool
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

    # A turn addressed to THIS worker (per-worker dispatch: the message carries the worker
    # attribute and lands on its own channel only), carrying the staged-secrets pointer
    # that must flow through to the turn (values never ride here).
    subscription = "projects/p/subscriptions/ratk-w-gen00001-w-abc123abc123abc123abc123"
    disp = InMemoryDispatch()
    disp.publish({"session_id": "dispatched-sid", "message": "do it", "resume": False,
                  "secrets_gcs": "gs://bkt/invocation-secrets/dispatched-sid-x.json",
                  "session_config_gcs": "gs://bkt/session-config/dispatched-sid.json",
                  "turn_config_gcs": "gs://bkt/turn-config/dispatched-sid-x.json"},
                 attributes={"worker": "abc123abc123abc123abc123"})
    opened = []

    def fake_worker_dispatch(sub, credentials=None):
        opened.append(sub)
        return disp.for_worker(pool.worker_id_from_subscription(sub))

    monkeypatch.setattr(pool, "worker_dispatch", fake_worker_dispatch)

    # _prewarm emits a readiness marker via a real CloudLoggingSink; off-GCP that stalls for
    # ~55s discovering ambient credentials. Fake the sink (same pattern as the warm-path test
    # above) so this test exercises the claim/run logic without a live GCP round-trip.
    import remote_agent_toolkit.ports.eventsink as eventsink_mod
    monkeypatch.setattr(eventsink_mod, "CloudLoggingSink", lambda **kw: InMemorySink(**kw))

    # Replace the heavy turn (real harness/model) with a recorder.
    seen = {}

    async def fake_run_turn(spec_, session_id, prompt, resume_sid, secrets_uri=None,
                            invocation_id="", session_config_uri=None, turn_config_uri=None,
                            gcs_token=None, worker=None):
        seen.update(session_id=session_id, prompt=prompt, resume_sid=resume_sid,
                    secrets_uri=secrets_uri, invocation_id=invocation_id,
                    session_config_uri=session_config_uri, turn_config_uri=turn_config_uri,
                    gcs_token=gcs_token, worker=worker)
        yield "turn-event"

    monkeypatch.setattr(agent, "_run_turn", fake_run_turn)

    async def drive():
        return [ev async for ev in agent._pool_worker(spec, "e-inv-77", subscription)]

    events = asyncio.run(drive())
    # Pulled its OWN channel (the subscription from the job input), claimed immediately (no
    # heartbeat), then handed the dispatched turn (+pointer) to _run_turn with its worker
    # id. invocation_id must flow through: the warm path missed it at first and every
    # session append kept 400ing on live engines while the cold path was fixed.
    assert opened == [subscription]
    assert events == ["turn-event"]
    assert seen == {"session_id": "dispatched-sid", "prompt": "do it", "resume_sid": None,
                    "secrets_uri": "gs://bkt/invocation-secrets/dispatched-sid-x.json",
                    "session_config_uri": "gs://bkt/session-config/dispatched-sid.json",
                    "turn_config_uri": "gs://bkt/turn-config/dispatched-sid-x.json",
                    "invocation_id": "e-inv-77",
                    "gcs_token": None,
                    "worker": "abc123abc123abc123abc123"}


def test_pool_worker_without_a_subscription_directive_exits_loudly(monkeypatch):
    """A bare sentinel (a fill_pool from a client older than per-worker dispatch) has no
    channel to pull: report it and exit instead of idling on nothing for a day."""
    spec = AgentSpec(name="w", model="m")
    agent = adk_agent.build_agent(spec)

    async def drive():
        return [ev async for ev in agent._pool_worker(spec, "e-inv-99", None)]

    events = asyncio.run(drive())
    raws = [ev.custom_metadata["raw"] for ev in events]
    assert [r["event"] for r in raws] == ["pool_error"]
    assert "AGENT_POOL_SUBSCRIPTION" in events[0].content.parts[0].text


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

    monkeypatch.setattr(pool, "worker_dispatch", lambda *a, **k: EmptyDispatch())
    monkeypatch.setenv("AGENT_POOL_MAX_WAIT_S", "0.05")
    # Same fake sink as the claim test: _prewarm's readiness emit must not touch GCP.
    import remote_agent_toolkit.ports.eventsink as eventsink_mod
    monkeypatch.setattr(eventsink_mod, "CloudLoggingSink", lambda **kw: InMemorySink(**kw))

    async def drive():
        sub = "projects/p/subscriptions/ratk-w-gen00001-w-abc"
        return [ev async for ev in agent._pool_worker(spec, "e-inv-88", sub)]

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


def test_get_engine_discovers_the_generation_scoped_pool_from_env():
    """The #38 cutover makes the pool per-deploy, so get_engine can no longer derive it
    from the engine name: it reads AGENT_POOL_TOPIC back from the deployed env (the same
    value the workers themselves read at runtime), plus the workers' idle life. No
    subscription: those are per worker (per-worker dispatch)."""
    topic = "projects/proj/topics/ratk-w-abc12345-dispatch"
    client = _FakeClient(_engine_resource({"AGENT_POOL_TOPIC": topic,
                                           "AGENT_POOL_MAX_WAIT_S": "7200"}))
    assert backend._discover_pool(client, "r/reasoningEngines/1", "proj", "w") == (
        topic, None, 7200.0
    )

    # An engine deployed BEFORE per-worker dispatch baked the one shared subscription all
    # its workers pull: discovered as the legacy address (the handle then dispatches the
    # old way, and get_engine warns).
    sub = "projects/proj/subscriptions/ratk-w-abc12345-dispatch-sub"
    client = _FakeClient(_engine_resource({"AGENT_POOL_SUBSCRIPTION": sub}))
    assert backend._discover_pool(client, "r/reasoningEngines/1", "proj", "w") == (
        topic, sub, None
    )


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

    topic, found, _ = backend._discover_pool(client, resource, "proj", "w")
    assert found == old_sub
    assert topic == "projects/proj/topics/ratk-w-old00000-dispatch"


def test_get_engine_pool_discovery_falls_back_to_legacy_names():
    """No AGENT_POOL_SUBSCRIPTION in the deployed env (an engine deployed without
    warm_pool) → the pre-#38 fixed names, so old addressing keeps working."""
    client = _FakeClient(_engine_resource({}))
    assert backend._discover_pool(client, "r/reasoningEngines/1", "proj", "w") == (
        "projects/proj/topics/ratk-w-dispatch",
        "projects/proj/subscriptions/ratk-w-dispatch-sub",
        None,
    )

    import pytest

    with pytest.raises(ValueError, match="dispatch topic"):
        backend._discover_pool(client, "r/reasoningEngines/1", None, "w")


def test_wait_until_warm_tails_the_generation_scoped_pool_id(monkeypatch):
    """The readiness marker key must be the one THIS pool's workers emit under — derived
    from the generation-scoped topic, not the bare engine name."""
    engine = backend.GeminiEngine(
        resource="r/reasoningEngines/1", spec=AgentSpec(name="w", model="m"),
        project=None, location=None, output_bucket="gs://out", warm=True,
        topic="projects/p/topics/ratk-w-abc12345-dispatch",
    )
    tailed = []

    async def fake_tail(uri, sid, **kw):
        tailed.append(sid)
        yield AgentEvent(kind="status", summary="pool worker ready")

    monkeypatch.setattr(backend, "tail_stream", fake_tail)
    assert engine.wait_until_warm(timeout=5) is True
    assert tailed == ["ratk-w-abc12345-pool"]


def _deploy_with_fakes(monkeypatch, tmp_path, engine_api, revision_apis):
    """Run the real backend.deploy() over an in-memory control plane (PR #39 review repro).

    Everything expensive/external is faked at its seam: the agentplatform client, the
    pub/sub ensure, packaging, the AdkApp build, the lifecycle backstop, and fill_pool.
    Returns (engine_handle, ensured_pairs, retired_pairs).
    """
    import sys
    import types as _types
    from types import SimpleNamespace as NS

    from remote_agent_toolkit.ports import dispatch as dispatch_mod
    from remote_agent_toolkit.runtime.gemini import _deploy, handoff

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

        def list(self):
            return [Wrapped(engine_api)]

        def get(self, name):
            return Wrapped(engine_api)

        def update(self, name, agent, config):
            return Wrapped(engine_api)

        def create(self, agent, config):  # pragma: no cover - the update path must win
            raise AssertionError("deploy() must update the existing engine, not create one")

    client = NS(agent_engines=AgentEngines())
    mod = _types.ModuleType("agentplatform")
    mod.Client = lambda **kw: client
    mod.types = NS(AgentEngineConfig=lambda **kw: NS(kw=kw))
    monkeypatch.setitem(sys.modules, "agentplatform", mod)

    monkeypatch.setattr(_deploy, "verify_deploy_env", lambda: None)
    # The runtime-SA existence check is one real IAM read; the fake control plane has none.
    monkeypatch.setattr(_deploy, "check_runtime_service_account_exists", lambda *a, **kw: None)
    monkeypatch.setattr(_deploy, "stage_agent", lambda spec: (str(tmp_path), []))
    monkeypatch.setattr(_deploy, "build_engine_config", lambda spec, **kw: {})
    monkeypatch.setattr(backend, "build_adk_app", lambda spec, **kw: object())
    monkeypatch.setattr(handoff, "ensure_handoff_lifecycle", lambda bucket, **kw: True)
    monkeypatch.chdir(tmp_path)

    ensured = []

    class FakeDispatch:
        def __init__(self, **kw):
            self.kw = kw

        def ensure_topic(self):
            ensured.append(self.kw["topic"])

    monkeypatch.setattr(dispatch_mod, "PubSubDispatch", FakeDispatch)
    monkeypatch.setattr(backend.GeminiEngine, "fill_pool", lambda self, n: None)
    retired = []
    monkeypatch.setattr(
        backend, "_retire_pool",
        lambda topic, legacy_sub, bucket, creds: retired.append((topic, legacy_sub)),
    )

    geng = backend.deploy(AgentSpec(name="w", model="m"), "proj", "loc", warm_pool=True)
    return geng, ensured, retired


def test_deploy_under_pin_dispatches_to_the_pinned_revisions_pair(monkeypatch, tmp_path):
    """PR #39 review: pin v1, then deploy twice. The second pinned deploy used to resolve
    the "old" pair from the ENGINE-level env — the latest revision's (v2's), a queue no
    worker ever pulled because v2 never served — so its turns sat unclaimed forever. The
    old pair must come from the PINNED revision's env (what live workers actually pull)."""
    import warnings

    resource = "projects/p/locations/l/reasoningEngines/9"
    q1 = "projects/proj/subscriptions/ratk-w-gen00001-dispatch-sub"
    q2 = "projects/proj/subscriptions/ratk-w-gen00002-dispatch-sub"
    engine_api = _engine_resource(
        {"AGENT_POOL_SUBSCRIPTION": q2},  # engine env == latest revision's (v2, never served)
        name=resource,
        traffic_targets=[(f"{resource}/runtimeRevisions/1", 100)],
    )
    engine_api.display_name = "w"
    revisions = [
        _engine_resource({"AGENT_POOL_SUBSCRIPTION": q1}, name=f"{resource}/runtimeRevisions/1"),
        _engine_resource({"AGENT_POOL_SUBSCRIPTION": q2}, name=f"{resource}/runtimeRevisions/2"),
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the deploy warns that the pin keeps v1 serving
        geng, ensured, retired = _deploy_with_fakes(
            monkeypatch, tmp_path, engine_api, revisions
        )

    assert geng._subscription == q1  # the pinned revision's (legacy) pool, NOT v2's
    assert geng._topic == pool.topic_for_subscription(q1)
    assert retired == []  # nothing retired while the pin holds
    # The fresh topic was still provisioned (it is baked into the new revision's env and
    # becomes live if that revision is promoted) — kept, not deleted.
    assert len(ensured) == 1
    assert ensured[0] not in (pool.topic_for_subscription(q1), pool.topic_for_subscription(q2))


def test_deploy_unpinned_retires_the_previous_revisions_pool(monkeypatch, tmp_path):
    """No pin: the fresh per-worker pool goes live and the previous revision's pool is
    retired once the new pool is filled (the #38 cutover), resolved from the same
    serving-revision read — whether the old pool was a legacy shared subscription or a
    per-worker one."""
    import warnings

    resource = "projects/p/locations/l/reasoningEngines/9"
    q2 = "projects/proj/subscriptions/ratk-w-gen00002-dispatch-sub"
    engine_api = _engine_resource({"AGENT_POOL_SUBSCRIPTION": q2}, name=resource)
    engine_api.display_name = "w"
    revisions = [
        _engine_resource({"AGENT_POOL_SUBSCRIPTION": q2}, name=f"{resource}/runtimeRevisions/2"),
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        geng, ensured, retired = _deploy_with_fakes(
            monkeypatch, tmp_path, engine_api, revisions
        )

    assert ensured == [geng._topic]  # fresh topic is live
    assert geng._subscription is None  # per-worker dispatch: no shared subscription
    assert retired == [(pool.topic_for_subscription(q2), q2)]  # old legacy pair retired

    # Previous revision already on per-worker dispatch: its topic is what gets retired
    # (worker subscriptions + roster swept with it, in _retire_pool).
    t2 = "projects/proj/topics/ratk-w-gen00002-dispatch"
    engine_api = _engine_resource({"AGENT_POOL_TOPIC": t2}, name=resource)
    engine_api.display_name = "w"
    revisions = [_engine_resource({"AGENT_POOL_TOPIC": t2}, name=f"{resource}/runtimeRevisions/2")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        geng, ensured, retired = _deploy_with_fakes(
            monkeypatch, tmp_path, engine_api, revisions
        )
    assert geng._topic != t2 and geng._subscription is None
    assert retired == [(t2, None)]


def test_delete_pool_resources_sweeps_every_revisions_pool(monkeypatch):
    """PR #39 review: deploys made while traffic was pinned leave their fresh pool idle,
    so teardown must sweep the pools of ALL revisions, not just the handle's (dedup'd) —
    legacy shared-subscription pools and per-worker ones alike."""

    resource = "projects/p/locations/l/reasoningEngines/9"
    q1 = "projects/proj/subscriptions/ratk-w-gen00001-dispatch-sub"
    q2 = "projects/proj/subscriptions/ratk-w-gen00002-dispatch-sub"
    t3 = "projects/proj/topics/ratk-w-gen00003-dispatch"
    revisions = [
        _engine_resource({"AGENT_POOL_SUBSCRIPTION": q1}, name=f"{resource}/runtimeRevisions/1"),
        _engine_resource({"AGENT_POOL_SUBSCRIPTION": q2}, name=f"{resource}/runtimeRevisions/2"),
        _engine_resource({"AGENT_POOL_TOPIC": t3}, name=f"{resource}/runtimeRevisions/3"),
        _engine_resource({}, name=f"{resource}/runtimeRevisions/4"),  # deployed without a pool
    ]
    client = _FakeClient(_engine_resource({}), revision_apis=revisions)
    client.agent_engines.delete = lambda name, force: None

    engine = backend.GeminiEngine(
        resource=resource, spec=AgentSpec(name="w", model="m"), project="p", location="l",
        warm=True, topic=pool.topic_for_subscription(q2), subscription=q2,  # handle == rev 2
    )
    monkeypatch.setattr(engine, "_client", lambda: client)
    deleted = []
    monkeypatch.setattr(
        backend, "_retire_pool",
        lambda topic, legacy_sub, bucket, creds: deleted.append((topic, legacy_sub)),
    )

    engine.delete(delete_pool_resources=True)
    assert sorted(deleted) == sorted([
        (pool.topic_for_subscription(q1), q1),
        (pool.topic_for_subscription(q2), q2),  # handle's own pool, deduplicated with rev 2's
        (t3, None),
    ])


def test_get_engine_pool_discovery_raises_on_read_errors():
    """PR #39 review: a FAILED env read must raise (retryable), not silently fall back to
    the legacy fixed names — those were never created for a generation-scoped engine, so
    the fallback handle only failed at its first publish, with an untraceable NotFound."""
    from types import SimpleNamespace as NS

    import pytest

    class AgentEngines:
        def get(self, name):
            raise TimeoutError("control plane unreachable")

    client = NS(agent_engines=AgentEngines())
    with pytest.raises(TimeoutError):
        backend._discover_pool(client, "r/reasoningEngines/1", "proj", "w")


# -- per-worker dispatch ---------------------------------------------------------------------


def _per_worker_engine(monkeypatch):
    """A per-worker warm engine over fakes: in-memory roster, recording dispatch, fake jobs."""
    from types import SimpleNamespace as NS

    topic = "projects/p/topics/ratk-w-gen00001-dispatch"
    engine = backend.GeminiEngine(
        resource="r/reasoningEngines/1", spec=AgentSpec(name="w", model="m"),
        project="p", location="l", output_bucket="gs://out", warm=True, topic=topic,
    )
    store = roster_mod.InMemoryRosterStore()
    monkeypatch.setattr(
        engine, "_roster", lambda: roster_mod.PoolRoster("gs://out", topic, store=store)
    )

    class FakeDispatch:
        def __init__(self):
            self.published, self.created, self.deleted = [], [], []

        def publish(self, message, attributes=None):
            self.published.append((message, attributes))

        def create_subscription(self, subscription, **kw):
            self.created.append((subscription, kw))

        def delete_subscription(self, subscription):
            self.deleted.append(subscription)

    dispatch = FakeDispatch()
    monkeypatch.setattr(engine, "_dispatch", lambda: dispatch)

    class FakeAE:
        def __init__(self):
            self.jobs, self.cancelled = [], []

        def run_query_job(self, name, config):
            self.jobs.append(config)
            return NS(job_name=f"op-{len(self.jobs)}")

        def cancel_query_job(self, name, config):
            self.cancelled.append(config["operation_name"])

        def check_query_job(self, name):
            return NS(status="RUNNING")

    ae = FakeAE()
    monkeypatch.setattr(engine, "_agent_engines", lambda: ae)
    return engine, dispatch, ae


def _seeded_tail(monkeypatch, sid="warm-sid"):
    """Patch the backend's stream tail with a sink pre-seeded with a finished turn."""
    seed = InMemorySink(session_id=sid)
    seed.emit(AgentEvent(kind="status", summary="turn started",
                         raw={"event": "turn_started", "worker": "x", "warm": True}))
    seed.emit(AgentEvent(kind="message", summary="working"))
    seed.emit(AgentEvent(kind="result", summary="done", cost_usd=0.1,
                         raw={"subtype": "success", "is_error": False, "num_turns": 3,
                              "session_id": sid}))
    monkeypatch.setattr(backend, "tail_stream", lambda uri, s, **kw: seed.tail(s))
    return seed


def test_fill_pool_gives_every_worker_its_own_channel_and_a_roster_entry(monkeypatch):
    engine, dispatch, ae = _per_worker_engine(monkeypatch)
    engine.fill_pool(2)

    assert len(ae.jobs) == 2 and len(dispatch.created) == 2
    for cfg, (subscription, kw) in zip(ae.jobs, dispatch.created):
        payload = json.loads(cfg["query"])
        assert payload["class_method"] == "async_stream_query"
        # The worker learns its OWN subscription from its job input (never the shared env).
        assert pool.parse_pool_wait(payload["input"]["message"]) == (True, subscription)
        wid = pool.worker_id_from_subscription(subscription)
        assert subscription == pool.worker_subscription(engine._topic, wid)
        # Filtered to this worker's messages; TTL/retention backstops for a dead client.
        assert kw == {"filter": pool.worker_filter(wid),
                      "ttl_s": pool.WORKER_SUBSCRIPTION_TTL_S,
                      "retention_s": pool.WORKER_SUBSCRIPTION_RETENTION_S}
        # The job output lands under jobs/: the one prefix the runtime identity may write.
        assert cfg["output_gcs_uri"] == f"gs://out/jobs/pool-{wid}.jsonl"
    entries = engine._roster().entries()
    assert [e.subscription for e in entries] == [s for s, _ in dispatch.created]
    assert [e.job_name for e in entries] == ["op-1", "op-2"] == engine._pool_jobs
    assert all(e.expires_at - e.submitted_at == pool.DEFAULT_MAX_WAIT_S for e in entries)


def test_fill_pool_drops_the_channel_when_the_job_submit_fails(monkeypatch):
    engine, dispatch, ae = _per_worker_engine(monkeypatch)

    def boom(name, config):
        raise RuntimeError("quota")

    ae.run_query_job = boom
    import pytest

    with pytest.raises(RuntimeError, match="quota"):
        engine.fill_pool(1)
    assert [s for s, _ in dispatch.created] == dispatch.deleted  # no orphan subscription
    assert engine._roster().entries() == []


def test_warm_session_addresses_one_idle_worker_and_drops_its_channel(monkeypatch):
    engine, dispatch, ae = _per_worker_engine(monkeypatch)
    engine.fill_pool(2)
    first, second = engine._roster().entries()
    _seeded_tail(monkeypatch)

    session = backend.GeminiSession(engine, "warm-sid")
    result = asyncio.run(_await(session.run("go")))
    engine._join_background()  # refill + channel cleanup run off the critical path

    # The turn went to the OLDEST idle worker, addressed by attribute (pointers + the run
    # token, never values), and the pool was refilled by one.
    assert dispatch.published == [(
        {"session_id": "warm-sid", "message": "go", "resume": False,
         "gcs_token": "fake-run-token"},
        {"worker": first.worker},
    )]
    remaining = engine._roster().entries()
    assert [e.worker for e in remaining][0] == second.worker and len(remaining) == 2
    assert len(ae.jobs) == 3
    # Its channel was dropped once the worker had the turn (nothing else can ever be
    # published there) — once, not again at completion; nothing was cancelled.
    assert dispatch.deleted == [first.subscription]
    assert ae.cancelled == []
    assert result.text == "done" and result.num_turns == 3
    assert session.status == RunStatus.IDLE and session.stop_reason == StopReason.END_TURN
    assert session._worker is None


def test_warm_session_spawns_a_worker_for_itself_on_an_empty_roster(monkeypatch):
    """A drained pool no longer strands turns: the turn spawns its own (un-rostered)
    worker and is addressed to it, and the usual refill re-warms the pool behind it."""
    engine, dispatch, ae = _per_worker_engine(monkeypatch)
    _seeded_tail(monkeypatch)

    session = backend.GeminiSession(engine, "warm-sid")
    result = asyncio.run(_await(session.run("go")))
    engine._join_background()

    assert len(ae.jobs) == 2  # one for this turn, one refill
    (_, attrs), = dispatch.published
    own, refill = (pool.worker_id_from_subscription(s) for s, _ in dispatch.created)
    assert attrs == {"worker": own}
    # Only the refill is on the roster: the turn's own worker must never be handed out.
    assert [e.worker for e in engine._roster().entries()] == [refill]
    assert result.text == "done"


def test_warm_session_redispatches_when_the_worker_never_starts_the_turn(monkeypatch):
    """Pickup watchdog: the addressed worker stays silent past its deadline → its channel
    (with the undelivered turn) is dropped and its job cancelled, the same payload goes to
    another worker, and the run completes as one Run with one result."""
    engine, dispatch, ae = _per_worker_engine(monkeypatch)
    engine.fill_pool(1)
    (first,) = engine._roster().entries()
    seed = _seeded_tail(monkeypatch)

    async def silent_until_redispatched(uri, sid, **kw):
        while len(dispatch.published) < 2:
            await asyncio.sleep(0.005)
        async for ev in seed.tail(sid):
            yield ev

    monkeypatch.setattr(backend, "tail_stream", silent_until_redispatched)
    monkeypatch.setattr(backend, "pickup_deadline_s", lambda submitted_at, now=None: 0.05)

    session = backend.GeminiSession(engine, "warm-sid")
    result = asyncio.run(_await(session.run("go")))
    engine._join_background()

    workers = [attrs["worker"] for _, attrs in dispatch.published]
    assert workers[0] == first.worker and len(workers) == 2 and workers[1] != first.worker
    assert dispatch.published[0][0] == dispatch.published[1][0]  # same payload, same run
    assert ae.cancelled == [first.job_name]  # the silent worker was stopped
    assert dispatch.deleted[0] == first.subscription  # and its channel dropped first
    assert first.job_name not in engine._pool_jobs
    assert not result.is_error and result.text == "done"
    assert session.status == RunStatus.IDLE


def test_pickup_watched_tail_gives_up_after_max_redispatch():
    async def silent():
        await asyncio.sleep(10)
        yield AgentEvent(kind="result", summary="never")

    calls = []

    def redispatch(attempt):
        calls.append(attempt)
        return 0.02

    async def drive():
        return [ev async for ev in backend._pickup_watched_tail(
            silent, 0.02, redispatch, "sid",
            on_abandon=lambda: calls.append("abandon"), max_redispatch=2,
        )]

    events = asyncio.run(drive())
    assert calls == [1, 2, "abandon"]
    assert [e.kind for e in events] == ["result"]
    raw = events[0].raw
    assert raw["event"] == "pool_pickup_timeout" and raw["is_error"] is True
    assert raw["workers_tried"] == 3 and raw["session_id"] == "sid"


def test_pickup_watched_tail_surfaces_a_failed_redispatch():
    async def silent():
        await asyncio.sleep(10)
        yield AgentEvent(kind="result", summary="never")

    def redispatch(attempt):
        raise RuntimeError("pubsub down")

    async def drive():
        return [ev async for ev in backend._pickup_watched_tail(silent, 0.02, redispatch, "sid")]

    events = asyncio.run(drive())
    assert [e.kind for e in events] == ["result"]
    assert events[0].raw["event"] == "pool_redispatch_failed"
    assert "pubsub down" in events[0].summary


def test_pickup_watched_tail_is_passthrough_once_the_turn_started():
    async def tail():
        yield AgentEvent(kind="status", summary="turn started", raw={"event": "turn_started"})
        await asyncio.sleep(0.05)  # longer than the pickup deadline: must NOT re-dispatch
        yield AgentEvent(kind="message", summary="working")
        yield AgentEvent(kind="result", summary="done", raw={"subtype": "success"})
        yield AgentEvent(kind="message", summary="after the result")  # never delivered

    picked, redispatched = [], []

    async def drive():
        return [ev async for ev in backend._pickup_watched_tail(
            tail, 0.01, lambda attempt: redispatched.append(attempt) or 0.01, "sid",
            on_pickup=lambda: picked.append(1),
        )]

    events = asyncio.run(drive())
    assert [e.kind for e in events] == ["status", "message", "result"]
    assert picked == [1] and redispatched == []


def _get_engine_with_fakes(monkeypatch, env: dict[str, str]):
    """Run the real backend.get_engine() over a fake agentplatform client."""
    import sys
    import types as _types
    from types import SimpleNamespace as NS

    resource = "projects/proj/locations/l/reasoningEngines/1"
    engine_api = _engine_resource(env, name=resource)
    engine_api.display_name = "w"
    client = _FakeClient(engine_api)
    client.agent_engines.list = lambda: [NS(api_resource=engine_api)]
    mod = _types.ModuleType("agentplatform")
    mod.Client = lambda **kw: client
    monkeypatch.setitem(sys.modules, "agentplatform", mod)
    return backend.get_engine("w", project="proj", location="l", warm_pool=True)


def test_get_engine_addresses_a_per_worker_pool_and_warns_on_a_legacy_one(monkeypatch):
    import warnings

    topic = "projects/proj/topics/ratk-w-gen00007-dispatch"
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a per-worker pool is the fixed state: no warning
        engine = _get_engine_with_fakes(
            monkeypatch, {"AGENT_POOL_TOPIC": topic, "AGENT_POOL_MAX_WAIT_S": "3600"}
        )
    assert (engine._topic, engine._subscription, engine._pool_max_wait_s) == (topic, None, 3600.0)

    import pytest

    sub = "projects/proj/subscriptions/ratk-w-gen00006-dispatch-sub"
    with pytest.warns(UserWarning, match="before per-worker dispatch"):
        legacy = _get_engine_with_fakes(monkeypatch, {"AGENT_POOL_SUBSCRIPTION": sub})
    assert (legacy._topic, legacy._subscription) == (pool.topic_for_subscription(sub), sub)
    assert legacy._pool_max_wait_s == pool.DEFAULT_MAX_WAIT_S


def test_retire_pool_sweeps_worker_channels_topic_and_roster(monkeypatch):
    """Cutover/teardown of a per-worker pool: every worker subscription (from the roster
    AND a prefix listing, in case one side is unavailable), then the topic, then the
    roster — so idle workers of the old generation fail their next pull and exit."""
    from remote_agent_toolkit.ports import dispatch as dispatch_mod

    topic = "projects/p/topics/ratk-w-gen00001-dispatch"
    store = roster_mod.InMemoryRosterStore()
    roster = roster_mod.PoolRoster("gs://out", topic, store=store)
    roster.add(roster_mod.WorkerEntry(
        worker="aaa", subscription=pool.worker_subscription(topic, "aaa"), job_name="op1",
        submitted_at=1.0, expires_at=2.0,
    ))
    monkeypatch.setattr(roster_mod, "PoolRoster", lambda *a, **k: roster)

    class FakeDispatch:
        deleted, topics = [], []

        def __init__(self, **kw):
            self.kw = kw

        def list_subscriptions(self, prefix):
            assert prefix == pool.worker_subscription_prefix(topic)
            return [pool.worker_subscription(topic, "bbb")]  # a worker the roster lost track of

        def delete_subscription(self, sub):
            self.deleted.append(sub)

        def delete_topic(self):
            self.topics.append(self.kw["topic"])

    monkeypatch.setattr(dispatch_mod, "PubSubDispatch", FakeDispatch)

    backend._retire_pool(topic, None, "gs://out", None)
    assert sorted(FakeDispatch.deleted) == sorted([
        pool.worker_subscription(topic, "aaa"), pool.worker_subscription(topic, "bbb"),
    ])
    assert FakeDispatch.topics == [topic]
    assert roster.entries() == []


def test_engine_delete_cancels_rostered_workers_of_other_processes(monkeypatch):
    """A reused per-worker engine: idle workers another process submitted are named in the
    roster, so delete() cancels them up front instead of discovering them one blocked
    delete at a time."""
    engine, dispatch, ae = _per_worker_engine(monkeypatch)
    engine._roster().add(roster_mod.WorkerEntry(
        worker="zzz", subscription=pool.worker_subscription(engine._topic, "zzz"),
        job_name="op-other", submitted_at=1.0, expires_at=1e12,
    ))
    ae.delete = lambda name, force: None
    engine.delete()
    assert ae.cancelled == ["op-other"]

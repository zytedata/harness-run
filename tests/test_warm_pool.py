"""The ready pool: fill, claim, refill, expiry, teardown — over the in-memory roster and provider."""

from __future__ import annotations

import asyncio
import time

import pytest
from sandbox_fakes import FakeSandboxProvider, ScriptedWorker, make_engine, result_event

from remote_agent_toolkit import AgentSpec
from remote_agent_toolkit.runtime.gemini import backend
from remote_agent_toolkit.runtime.gemini.provider import SandboxError


async def _await(run):
    return await run


def test_fill_pool_creates_ready_sandboxes_and_records_them():
    provider = FakeSandboxProvider()
    engine = make_engine(provider, warm=True, pool_max_wait_s=1000.0, max_turn_s=50.0)
    assert engine.wait_until_warm(timeout=0.01) is False  # empty roster
    engine.fill_pool(2)
    entries = engine._roster().entries()
    assert len(entries) == 2 and {e.sandbox for e in entries} == set(provider.live())
    for e in entries:
        assert e.template == engine.resource
        assert e.expires_at == pytest.approx(e.created_at + 1000.0)  # the idle life
        assert provider.sandboxes[e.sandbox]["ttl_s"] == 1050.0  # idle life + room for a turn
        assert provider.sandboxes[e.sandbox]["display_name"].startswith("ratk-g-")
    # Every sandbox answered /health before it was recorded.
    assert [c[1] for c in provider.calls] == ["/health", "/health"]
    assert engine.wait_until_warm(timeout=0.01) is True


def test_fill_pool_releases_a_sandbox_that_never_answers(monkeypatch):
    class Unrouted:
        def handle(self, path, body):
            raise RuntimeError("no route")

    provider = FakeSandboxProvider(lambda n: Unrouted())
    monkeypatch.setattr(backend, "READY_WAIT_S", 0.05)
    monkeypatch.setattr(backend.time, "sleep", lambda s: None)
    engine = make_engine(provider, warm=True)
    with pytest.raises(SandboxError, match="never answered"):
        engine.fill_pool(1)
    assert provider.live() == [] and len(provider.deleted) == 1
    assert engine._roster().entries() == []


def test_fill_pool_releases_a_sandbox_the_roster_cannot_record(monkeypatch):
    provider = FakeSandboxProvider()
    engine = make_engine(provider, warm=True)

    def boom(entry):
        raise RuntimeError("bucket denied")

    monkeypatch.setattr(engine._roster(), "add", boom)
    with pytest.raises(RuntimeError, match="bucket denied"):
        engine.fill_pool(1)
    assert provider.live() == []


def test_warm_turn_claims_the_oldest_ready_sandbox_and_refills():
    provider = FakeSandboxProvider()
    engine = make_engine(provider, warm=True)
    engine.fill_pool(2)
    first, second = [e.sandbox for e in engine._roster().entries()]
    result = asyncio.run(_await(engine.start_session().run("go")))
    engine._join_background()
    assert result.text == "done"
    turn = [c for c in provider.calls if c[1] == "/turn"][0]
    assert turn[0] == first and turn[2]["warm"] is True
    assert provider.deleted == [first]  # the used sandbox is gone
    # Refilled: the untouched second sandbox plus one new ready sandbox.
    remaining = {e.sandbox for e in engine._roster().entries()}
    assert second in remaining and len(remaining) == 2
    assert len(provider.live()) == 2


def test_warm_turn_on_an_empty_roster_creates_a_sandbox_for_itself():
    provider = FakeSandboxProvider()
    engine = make_engine(provider, warm=True, max_turn_s=1234.0)
    result = asyncio.run(_await(engine.start_session().run("go")))
    engine._join_background()
    assert result.text == "done"
    turn = [c for c in provider.calls if c[1] == "/turn"][0]
    assert turn[2]["warm"] is False
    created = provider.deleted[0]
    assert created == turn[0]
    # An on-demand sandbox lives max_turn_s; no refill for a turn that took nothing off the pool.
    assert len(provider.live()) == 0 and engine._roster().entries() == []


def test_claim_prunes_expired_entries_and_releases_their_sandboxes():
    provider = FakeSandboxProvider()
    engine = make_engine(provider, warm=True, pool_max_wait_s=0.001)  # entries expire at once
    engine.fill_pool(1)
    stale = engine._roster().entries()[0].sandbox
    time.sleep(0.01)
    sandbox, warm = engine._claim_sandbox()
    engine._join_background()
    assert warm is False and sandbox != stale
    assert stale in provider.deleted and sandbox in provider.live()
    engine._release_sandbox(sandbox)


def test_claim_skips_entries_about_to_expire(monkeypatch):
    provider = FakeSandboxProvider()
    engine = make_engine(provider, warm=True, pool_max_wait_s=backend.CLAIM_MARGIN_S / 2)
    engine.fill_pool(1)
    sandbox, warm = engine._claim_sandbox()
    engine._join_background()
    assert warm is False  # within the margin: treated as expired, a fresh one was created
    engine._release_sandbox(sandbox)


def test_wait_until_warm_is_a_noop_for_non_pool_engines_and_polls_the_roster(monkeypatch):
    assert make_engine(FakeSandboxProvider()).wait_until_warm(timeout=0.0) is True
    provider = FakeSandboxProvider()
    engine = make_engine(provider, warm=True)
    monkeypatch.setattr(backend.time, "sleep", lambda s: engine.fill_pool(1))  # becomes warm mid-wait
    assert engine.wait_until_warm(timeout=5.0) is True


def test_delete_releases_every_sandbox_and_retires_all_versions(monkeypatch):
    provider = FakeSandboxProvider()
    engine = make_engine(provider, warm=True)
    old = provider.add_template("g", create_time=0.5)
    engine.fill_pool(2)
    orphan = provider.create(engine.resource, ttl_s=60, display_name="ratk-g-orphan")  # a crashed client's
    foreign = provider.create(old, ttl_s=60, display_name="ratk-other-x")
    monkeypatch.setattr(backend.time, "sleep", lambda s: None)
    engine.delete(timeout=1.0)
    assert orphan.name in provider.deleted and foreign.name not in provider.deleted
    assert provider.live() == [foreign.name]
    assert engine.resource in provider.deleted_templates
    assert old not in provider.deleted_templates  # still has a sandbox: left with a warning


def test_delete_version_retires_one_template_and_its_rostered_sandboxes(monkeypatch):
    provider = FakeSandboxProvider()
    engine = make_engine(provider, warm=True)
    engine.fill_pool(1)
    monkeypatch.setattr(backend.time, "sleep", lambda s: None)
    with pytest.raises(LookupError):
        engine.delete_version("t9")
    engine.delete_version(engine.version)
    assert provider.live() == [] and provider.deleted_templates == [engine.resource]


def test_retire_templates_gives_up_after_the_timeout(monkeypatch):
    provider = FakeSandboxProvider()
    engine = make_engine(provider)
    other = provider.add_template("g")
    provider.create(other, ttl_s=60, display_name="ratk-g-busy")  # blocks the template delete
    monkeypatch.setattr(backend.time, "sleep", lambda s: None)
    engine._retire_templates([other], timeout=0.0)
    assert other in provider.templates  # not deleted, not raised


def test_turn_that_lands_on_a_dead_pool_sandbox_moves_on():
    provider = FakeSandboxProvider()
    engine = make_engine(provider, warm=True)
    engine.fill_pool(2)
    first = engine._roster().entries()[0].sandbox
    provider.vanish(first)  # OOM / TTL race: gone between the roster and the dispatch
    result = asyncio.run(_await(engine.start_session().run("go")))
    engine._join_background()
    assert result.text == "done"
    turns = [c for c in provider.calls if c[1] == "/turn"]
    assert [t[0] for t in turns][0] == first and len(turns) == 2  # tried, then the next one


def test_get_engine_handle_shares_the_pool_of_the_deploying_process():
    # Two handles over one roster store see the same ready sandboxes (the roster is GCS-shared).
    provider = FakeSandboxProvider()
    deploying = make_engine(provider, warm=True)
    deploying.fill_pool(1)
    other = backend.GeminiEngine(template=deploying.resource, spec=AgentSpec(name="g", model="m"),
                                 project="p", location="l", output_bucket="gs://out", provider=provider,
                                 warm=True, roster_store=deploying._roster_store,
                                 model_service_account="ratk-model@p.iam.gserviceaccount.com")
    assert other.wait_until_warm(timeout=0.01)
    ready = deploying._roster().entries()[0].sandbox
    asyncio.run(_await(other.start_session().run("go")))
    other._join_background()
    assert [c for c in provider.calls if c[1] == "/turn"][0][0] == ready
    assert deploying._roster().entries() == other._roster().entries()  # refilled, visible to both


def test_scripted_worker_pool_end_to_end_with_control_ready():
    provider = FakeSandboxProvider(lambda n: ScriptedWorker(auto=[
        __import__("remote_agent_toolkit").events.AgentEvent(kind="status", summary="ready", raw={"event": "control_ready"}),
        result_event("42"),
    ]))
    engine = make_engine(provider, warm=True)
    engine.fill_pool(1)
    session = engine.start_session()
    result = asyncio.run(_await(session.run("go")))
    assert result.text == "42" and session._control_ready

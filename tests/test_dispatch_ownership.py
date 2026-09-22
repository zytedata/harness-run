"""Who owns a turn's sandbox, and when the sandbox may go (PR #84 review of 2026-09-19, A1/A2/B).

* A1 — a ``/turn`` call whose answer was lost is settled with the SAME sandbox (the worker
  re-acknowledges a turn id it knows); the turn is never replayed elsewhere while the first
  sandbox may be running the agent. When that sandbox cannot be asked any more, the
  dispatch fails with the uncertainty stated and the sandbox is deleted.
* A2 — ``interrupt()`` while the hand-over is in flight cannot stop the dispatch thread, so
  whatever sandbox it lands on (claimed, or running the turn) is deleted when it lands.
* B — the worker mirrors the terminal result before serving it on ``/events``, so the
  client deletes the sandbox only once the durable record holds the result; a failed
  mirror write is stated on the live copy and this client writes the record itself.
"""

from __future__ import annotations

import asyncio
import json
import threading

from sandbox_fakes import FakeSandboxProvider, ScriptedWorker, make_engine, result_event

from harness_run import AgentSpec
from harness_run.events import AgentEvent
from harness_run.ports.blobstore import LocalBlobStore
from harness_run.runtime.sandbox import history
from harness_run.runtime.sandbox.stream import MirrorStream
from harness_run.runtime.sandbox.worker import Worker


def _turn_calls(provider):
    return [sb for sb, path, _b in provider.calls if path == "/turn"]


class _EffectWorker(ScriptedWorker):
    """Records an external side effect the moment a turn is accepted (a push, a PR)."""

    effects: list[str]

    def __init__(self, effects, **kw):
        super().__init__(**kw)
        self.effects = effects

    def handle(self, path, body):
        answer = super().handle(path, body)
        if path == "/turn" and answer["ok"] and not answer.get("already_accepted"):
            self.effects.append(body["turn_id"])
        return answer


class _LostAnswer(FakeSandboxProvider):
    """The first ``/turn`` answer is lost AFTER the worker accepted (a proxy timeout)."""

    def __init__(self, *a, lose: int = 1, **kw):
        super().__init__(*a, **kw)
        self.lose = lose
        self.lost = 0

    def call(self, sandbox, path, body=None, **kw):
        answer = super().call(sandbox, path, body, **kw)
        if path == "/turn" and self.lost < self.lose:
            self.lost += 1
            raise TimeoutError("response lost after turn acceptance")
        return answer


# -- A1: a lost acknowledgement -----------------------------------------------------------


def test_a_lost_turn_answer_is_settled_with_the_same_sandbox_and_never_replayed():
    effects: list[str] = []
    provider = _LostAnswer(lambda n: _EffectWorker(effects, auto=[result_event()]))
    engine = make_engine(provider)
    session = engine.start_session()

    async def go():
        return await session.run("go")

    result = asyncio.run(go())
    engine._join_background()
    assert result.text == "done" and not result.is_error
    assert len(effects) == 1  # the turn ran once
    calls = _turn_calls(provider)
    assert len(calls) == 2 and len(set(calls)) == 1  # asked twice, the same sandbox both times
    assert len(provider.workers) == 1  # no second sandbox was ever created
    assert provider.deleted == calls[:1] and provider.live() == []  # released once, at the result


def test_a_lost_answer_the_sandbox_never_settles_fails_the_dispatch_as_uncertain_and_deletes_it():
    effects: list[str] = []
    provider = _LostAnswer(lambda n: _EffectWorker(effects), lose=10 ** 6)  # every /turn answer is lost
    engine = make_engine(provider)
    session = engine.start_session()

    async def go():
        run = session.run("go")
        events = [ev async for ev in run]
        return run.result, events

    result, events = asyncio.run(go())
    engine._join_background()
    assert result.is_error
    assert events[-1].raw["event"] == "dispatch_uncertain"
    assert "unknown state" in result.text and "not replayed" in result.text
    assert len(effects) == 1 and len(provider.workers) == 1  # accepted once; nothing else tried
    assert len(set(_turn_calls(provider))) == 1 and len(_turn_calls(provider)) >= 2
    (sandbox,) = provider.workers
    assert provider.deleted == [sandbox] and provider.live() == []  # deleted, which stops the run
    assert session._sandbox is None and not session.busy


def test_a_lost_answer_then_a_vanished_sandbox_is_uncertain_not_a_replay():
    effects: list[str] = []
    provider = _LostAnswer(lambda n: _EffectWorker(effects))
    engine = make_engine(provider)
    session = engine.start_session()
    original_call = provider.call

    def vanish_after_loss(sandbox, path, body=None, **kw):
        if path == "/turn" and provider.lost:
            provider.vanish(sandbox)
        return original_call(sandbox, path, body, **kw)

    provider.call = vanish_after_loss

    async def go():
        return await session.run("go")

    result = asyncio.run(go())
    engine._join_background()
    assert result.is_error and "vanished" in result.text and "not replayed" in result.text
    assert len(effects) == 1 and len(provider.workers) == 1


def test_a_worker_from_an_older_image_that_refuses_a_known_turn_id_counts_as_accepted():
    class OldWorker(ScriptedWorker):
        def handle(self, path, body):
            if path == "/turn" and any(t["turn_id"] == body["turn_id"] for t in self.turns):
                return {"ok": False, "error": "turn_id already used on this sandbox"}
            return super().handle(path, body)

    provider = _LostAnswer(lambda n: OldWorker(auto=[result_event()]))
    engine = make_engine(provider)
    result = asyncio.run(_await(engine.start_session().run("go")))
    engine._join_background()
    assert result.text == "done" and len(provider.workers) == 1


def test_the_worker_reacknowledges_a_turn_id_it_knows_and_refuses_a_different_one(tmp_path, monkeypatch):
    held = threading.Event()

    class HeldHarness:
        async def run(self, spec, rc):
            await asyncio.to_thread(held.wait, 5)
            yield result_event("done")

    import harness_run.harness.claude_code as harness_mod

    monkeypatch.setattr(harness_mod, "ClaudeCodeHarness", HeldHarness)
    worker = Worker(workspace_root=str(tmp_path), baked=AgentSpec(harness="claude-code", name="w", model="m"), baked_skills=None,
                    mirror_factory=lambda *a: None)
    first = worker.handle("/turn", {"turn_id": "t1", "session_id": "s"})
    again = worker.handle("/turn", {"turn_id": "t1", "session_id": "s"})
    other = worker.handle("/turn", {"turn_id": "t2", "session_id": "s"})
    assert first["ok"] and "already_accepted" not in first
    assert again == {"ok": True, "turn_id": "t1", "accepted_at": first["accepted_at"],
                     "already_accepted": True, "done": False}
    assert other == {"ok": False, "error": "a turn is running", "running": ["t1"]}
    held.set()
    resp = worker.handle("/events", {"turn_id": "t1", "since": 0, "wait": 5})
    while not resp["done"]:
        resp = worker.handle("/events", {"turn_id": "t1", "since": resp["next"], "wait": 5})
    assert worker.handle("/turn", {"turn_id": "t1", "session_id": "s"})["done"] is True


# -- A2: interrupt() while the hand-over is in flight -------------------------------------


def test_interrupt_during_the_turn_call_deletes_the_sandbox_the_dispatch_lands_on():
    entered, release = threading.Event(), threading.Event()

    class SlowHandOver(FakeSandboxProvider):
        def call(self, sandbox, path, body=None, **kw):
            if path == "/turn":
                entered.set()
                assert release.wait(5)
            return super().call(sandbox, path, body, **kw)

    provider = SlowHandOver(lambda n: ScriptedWorker())
    engine = make_engine(provider)
    session = engine.start_session()

    async def go():
        run = session.run("go")
        assert await asyncio.to_thread(entered.wait, 5)
        try:
            await asyncio.wait_for(session.interrupt(timeout=0.1), 2)
        finally:
            release.set()
        assert run.done and run.result.is_error and "cancelled" in run.result.warning

    asyncio.run(go())
    engine._join_background()
    (sandbox,) = provider.workers
    assert sum(len(w.turns) for w in provider.workers.values()) == 1  # the worker did accept it…
    assert provider.deleted == [sandbox] and provider.live() == []  # …and the sandbox is gone
    assert session._sandbox is None and not session.busy and session._dispatch_slot is None


def test_interrupt_while_a_sandbox_is_still_being_created_deletes_it_and_posts_no_turn():
    entered, release = threading.Event(), threading.Event()

    class SlowCreate(FakeSandboxProvider):
        def create(self, template, **kw):
            entered.set()
            assert release.wait(5)
            return super().create(template, **kw)

    provider = SlowCreate(lambda n: ScriptedWorker())
    engine = make_engine(provider)
    session = engine.start_session()

    async def go():
        run = session.run("go")
        assert await asyncio.to_thread(entered.wait, 5)
        try:
            await asyncio.wait_for(session.interrupt(timeout=0.1), 2)
        finally:
            release.set()
        return run

    asyncio.run(go())
    engine._join_background()
    (sandbox,) = provider.workers
    assert _turn_calls(provider) == []  # the cancelled run's turn was never posted
    assert provider.deleted == [sandbox] and provider.live() == []


def test_a_dispatch_landing_while_the_cancelled_run_completes_still_gets_its_sandbox_deleted():
    """Review follow-up of 2026-09-21: the hand-over lands between ``_on_complete`` starting
    to read state and its check of the dispatch slot. The dispatch thread publishes the
    sandbox and then marks itself finished; if the completion read the sandbox first (None)
    and the finished flag second (set), neither path deleted the sandbox. The ownership
    decision is now one step under the dispatch lock, before the sandbox is read."""
    entered, release = threading.Event(), threading.Event()

    class SlowHandOver(FakeSandboxProvider):
        def call(self, sandbox, path, body=None, **kw):
            if path == "/turn":
                entered.set()
                assert release.wait(5)
            return super().call(sandbox, path, body, **kw)

    provider = SlowHandOver(lambda n: ScriptedWorker())
    engine = make_engine(provider)
    session = engine.start_session()
    armed = threading.Event()  # _on_complete is past its bookkeeping, about to read state
    fired = threading.Event()

    class Interleaved(type(session)):
        # The FIRST read of ``_sandbox`` on the completion path lets the blocked hand-over
        # through and waits until the dispatch thread has published the sandbox and finished
        # — the exact interleaving of the follow-up. The value returned is the one seen
        # before that (None), as the unfixed code would have seen it.
        @property
        def _sandbox(self):
            value = self.__dict__.get("_sandbox_value")
            slot = self.__dict__.get("_dispatch_slot_value")
            if armed.is_set() and slot is not None and not fired.is_set():
                fired.set()
                release.set()
                assert slot.finished.wait(5)
            return value

        @_sandbox.setter
        def _sandbox(self, value):
            self.__dict__["_sandbox_value"] = value

        @property
        def _dispatch_slot(self):
            return self.__dict__.get("_dispatch_slot_value")

        @_dispatch_slot.setter
        def _dispatch_slot(self, value):
            self.__dict__["_dispatch_slot_value"] = value

    session._sandbox, session._dispatch_slot = None, None
    session.__class__ = Interleaved
    real_drop = session._drop_pending_control

    def drop_then_arm():
        n = real_drop()
        armed.set()
        return n

    session._drop_pending_control = drop_then_arm

    async def go():
        run = session.run("go")
        assert await asyncio.to_thread(entered.wait, 5)
        try:
            await asyncio.wait_for(session.interrupt(timeout=0.1), 5)
        finally:
            release.set()
        assert run.done and run.result.is_error and "cancelled" in run.result.warning

    asyncio.run(go())
    engine._join_background()
    (sandbox,) = provider.workers
    assert sum(len(w.turns) for w in provider.workers.values()) == 1  # the worker took the turn…
    assert provider.deleted == [sandbox] and provider.live() == []  # …and exactly one delete followed
    assert session._sandbox is None and not session.busy and session._dispatch_slot is None


# -- B: the result is durable before the sandbox goes -------------------------------------


class _DoneHarness:
    async def run(self, spec, rc):
        yield AgentEvent(kind="message", summary="working")
        yield result_event("done")


def _is_result_batch(data: bytes) -> bool:
    return any(json.loads(line)["kind"] == "result" for line in data.splitlines() if line.strip())


def _real_worker_engine(tmp_path, monkeypatch, store, provider_cls=FakeSandboxProvider):
    """A real Worker per sandbox, mirroring through ``store``; the client reads history from it too."""
    import harness_run.harness.claude_code as harness_mod

    monkeypatch.setattr(harness_mod, "ClaudeCodeHarness", _DoneHarness)
    monkeypatch.setattr(history, "GcsBlobStore", lambda bucket, *a, **kw: store)
    spec = AgentSpec(harness="claude-code", name="g", model="m")

    def factory(name):
        return Worker(
            workspace_root=str(tmp_path / "ws" / name.rsplit("/", 1)[-1]), baked=spec, baked_skills=None,
            mirror_factory=lambda uri, sid, tid: MirrorStream(uri, sid, turn_id=tid, store=store),
            metadata_port=0,
        )

    provider = provider_cls(factory)
    return provider, make_engine(provider, spec=spec)


def test_the_client_sees_and_deletes_at_a_result_that_is_already_in_the_mirror(tmp_path, monkeypatch):
    entered, release, committed = threading.Event(), threading.Event(), threading.Event()

    class SlowStore(LocalBlobStore):
        def put_bytes(self, key, data):
            terminal = _is_result_batch(data)
            if terminal:
                entered.set()
                assert release.wait(5)
            super().put_bytes(key, data)
            if terminal:
                committed.set()

    at_delete = []

    class ObserveDelete(FakeSandboxProvider):
        def delete(self, sandbox):
            at_delete.append({"result_persisted": committed.is_set()})
            super().delete(sandbox)

    store = SlowStore(str(tmp_path / "bucket"))
    provider, engine = _real_worker_engine(tmp_path, monkeypatch, store, ObserveDelete)
    session = engine.start_session()

    async def go():
        run = session.run("go")
        assert await asyncio.to_thread(entered.wait, 5)  # the worker is writing the result…
        await asyncio.sleep(0.2)
        assert not run.done  # …and the client has not been shown it yet
        release.set()
        return await run

    result = asyncio.run(go())
    engine._join_background()
    assert result.text == "done" and not result.is_error
    assert at_delete == [{"result_persisted": True}]
    ended = [e for e in session.history() if e.kind == "result"]
    assert len(ended) == 1 and ended[0].raw["mirror"] == "worker"
    assert session._find_running_turn() is None  # a re-attacher finds the turn ended


def test_a_result_the_worker_could_not_mirror_is_written_by_the_client_before_the_delete(tmp_path, monkeypatch):
    class WorkerWritesFail(LocalBlobStore):
        def put_bytes(self, key, data):
            if _is_result_batch(data) and b'"mirror": "worker"' in data:
                raise RuntimeError("the run-scoped token is dead")
            super().put_bytes(key, data)

    at_delete = []
    store = WorkerWritesFail(str(tmp_path / "bucket"))

    class ObserveDelete(FakeSandboxProvider):
        def delete(self, sandbox):
            lines = [json.loads(line) for key in store.list("events/") for line in store.get_bytes(key).splitlines()]
            at_delete.append([line["raw"]["mirror"] for line in lines if line["kind"] == "result"])
            super().delete(sandbox)

    provider, engine = _real_worker_engine(tmp_path, monkeypatch, store, ObserveDelete)
    session = engine.start_session()

    async def go():
        run = session.run("go")
        events = [ev async for ev in run]
        return run.result, events

    result, events = asyncio.run(go())
    engine._join_background()
    assert result.text == "done" and not result.is_error
    assert events[-1].kind == "result" and events[-1].raw["mirror"] == "failed"  # the live copy says so
    assert at_delete == [["client"]]  # the client's record was there before the sandbox went
    ended = [e for e in session.history() if e.kind == "result"]
    assert len(ended) == 1 and ended[0].raw["mirror"] == "client" and ended[0].summary == "done"
    assert session._find_running_turn() is None


def test_mirror_stream_append_with_wait_reports_the_write_outcome(tmp_path):
    class Flaky(LocalBlobStore):
        fail = False

        def put_bytes(self, key, data):
            if self.fail:
                raise RuntimeError("storage down")
            super().put_bytes(key, data)

    store = Flaky(str(tmp_path / "bucket"))
    stream = MirrorStream("gs://out/events", "sid", turn_id="t1", store=store)
    stream.append(AgentEvent(kind="message", summary="one"))
    assert stream.append(result_event("done"), wait_s=5) is True
    keys = store.list("events/sid/")
    lines = [json.loads(line) for key in keys for line in store.get_bytes(key).splitlines()]
    assert [line["kind"] for line in lines] == ["message", "result"]  # in order, both written
    store.fail = True
    assert stream.append(result_event("again"), wait_s=5) is False
    stream.close()
    assert MirrorStream("not-a-uri", "sid", store=None).append(result_event("x"), wait_s=1) is False


def test_write_turn_mirror_reports_whether_the_store_took_it(tmp_path):
    store = LocalBlobStore(str(tmp_path / "bucket"))
    line = history.mirror_line(result_event("done"))
    assert history.write_turn_mirror("gs://out/events", "sid", [line], now_ms=1, turn_id="t", store=store) is True
    store.put_bytes = lambda key, data: (_ for _ in ()).throw(RuntimeError("storage down"))
    assert history.write_turn_mirror("gs://out/events", "sid", [line], now_ms=2, turn_id="t", store=store) is False


async def _await(run):
    return await run

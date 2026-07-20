"""Local runtime end-to-end with a fake harness (no live model call).

Exercises the full Engine/Session/Run machinery — skills staging, the three consumption
modes (await / async-iter / poll), status & stop_reason mapping, and resume-restore
wiring — by swapping in a harness that yields canned AgentEvents.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from remote_agent_toolkit import AgentSpec, SkillSource, local
from remote_agent_toolkit.events import AgentEvent, RunStatus, StopReason


def _result_ev(text="done", subtype="success", is_error=False, num_turns=3):
    return AgentEvent(
        kind="result", summary=text, cost_usd=0.1, usage={"input_tokens": 5},
        raw={"subtype": subtype, "is_error": is_error, "num_turns": num_turns, "session_id": "claude-sid"},
    )


class FakeHarness:
    """Yields canned events; optional on_run hook to assert the prepared workspace."""

    def __init__(self, events, on_run=None):
        self._events = events
        self._on_run = on_run

    async def run(self, spec, ctx) -> AsyncIterator[AgentEvent]:
        if self._on_run is not None:
            self._on_run(spec, ctx)
        for ev in self._events:
            yield ev


def _skill_src(tmp_path):
    src = tmp_path / "skills_src" / "demo"
    src.mkdir(parents=True)
    (src / "SKILL.md").write_text("---\nname: demo\n---\nA demo skill.")
    return str(tmp_path / "skills_src")


def test_await_returns_result_and_stages_skills(tmp_path):
    seen = {}
    spec = AgentSpec(name="demo", model="m", skills=[SkillSource.local(_skill_src(tmp_path))])
    engine = local.deploy(spec, workdir=str(tmp_path / "wd"))
    engine._harness = FakeHarness(
        [AgentEvent(kind="message", summary="working"), _result_ev(text="final")],
        on_run=lambda s, c: seen.update(
            staged=(c.workspace / ".claude" / "skills" / "demo" / "SKILL.md").is_file()
        ),
    )
    session = engine.start_session()

    async def go():
        return await session.run("build it")

    result = asyncio.run(go())
    assert seen["staged"] is True  # _prepare_workspace staged the local skill
    assert result.text == "final" and result.num_turns == 3 and result.cost_usd == 0.1
    assert result.is_error is False
    assert session.status == RunStatus.IDLE
    assert session.stop_reason == StopReason.END_TURN  # non-checkpoint clean turn


def test_async_iter_streams_events(tmp_path):
    spec = AgentSpec(name="demo", model="m")  # no skills
    engine = local.deploy(spec, workdir=str(tmp_path / "wd"))
    engine._harness = FakeHarness([
        AgentEvent(kind="tool_use", summary="Bash ls"),
        AgentEvent(kind="message", summary="hi"),
        _result_ev(),
    ])
    session = engine.start_session()

    async def collect():
        return [ev async for ev in session.run("go")]

    kinds = [e.kind for e in asyncio.run(collect())]
    # leading workspace_ready status, then the harness events
    assert kinds[0] == "status"
    assert kinds[1:] == ["tool_use", "message", "result"]


def test_poll_until_done(tmp_path):
    spec = AgentSpec(name="demo", model="m")
    engine = local.deploy(spec, workdir=str(tmp_path / "wd"))
    engine._harness = FakeHarness([_result_ev(text="polled")])

    async def go():
        session = engine.start_session()
        run = session.run("go")  # eager-started (loop running) so polling progresses
        for _ in range(500):
            if run.done:
                break
            await asyncio.sleep(0.005)
        return run

    run = asyncio.run(go())
    assert run.done and run.status == RunStatus.IDLE
    assert run.result.text == "polled"


def test_stop_reason_mapping(tmp_path):
    cases = [
        (False, "success", False, StopReason.END_TURN),
        (True, "success", False, StopReason.NEEDS_INPUT),       # checkpoint → awaiting input
        (False, "error_max_turns", False, StopReason.MAX_TURNS),
        (False, "error_max_budget_usd", False, StopReason.BUDGET_EXCEEDED),
        (False, "error_during_execution", True, StopReason.ERROR),
    ]
    for checkpoint, subtype, is_error, expected in cases:
        spec = AgentSpec(name="demo", model="m", checkpoint=checkpoint)
        engine = local.deploy(spec, workdir=str(tmp_path / f"wd-{subtype}-{checkpoint}"))
        engine._harness = FakeHarness([_result_ev(subtype=subtype, is_error=is_error)])
        session = engine.start_session()
        asyncio.run(_await(session.run("go")))
        assert session.stop_reason == expected, (subtype, checkpoint)


async def _await(run):
    return await run


def test_resume_restores_workspace(tmp_path):
    spec = AgentSpec(name="demo", model="m", checkpoint=True)
    engine = local.deploy(spec, workdir=str(tmp_path / "wd"))

    # Pre-seed a checkpoint snapshot for a fixed session id, as a prior turn would have.
    from remote_agent_toolkit.checkpoint.workspace import snapshot
    from remote_agent_toolkit.ports.blobstore import LocalBlobStore

    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "restored.txt").write_text("from a prior turn")
    snapshot(LocalBlobStore(str(engine._blob_root)), "fixedsid", str(seed))

    seen = {}
    engine._harness = FakeHarness(
        [_result_ev()],
        on_run=lambda s, c: seen.update(restored=(c.workspace / "restored.txt").is_file()),
    )
    session = engine.get_session("fixedsid")  # re-attach to the checkpointed session

    asyncio.run(_await(session.send("continue please")))
    assert seen["restored"] is True  # _prepare_workspace restored the snapshot on resume


def test_workspace_accessor_seed_and_collect(tmp_path):
    # The agent cwd is a leaf literally named "workspace" (an anonymous jobs/<uuid> cwd
    # reads as disposable temp and weaker models cd away from it), and Session.workspace
    # is the caller-facing accessor: seed inputs before run(), collect artifacts after —
    # no deriving <workdir>/jobs/<sid> by hand.
    spec = AgentSpec(name="demo", model="m")
    engine = local.deploy(spec, workdir=str(tmp_path / "wd"))
    session = engine.start_session()

    ws = session.workspace
    assert ws.is_dir()  # created on access, ready for seeding
    assert ws == engine._jobs_root / session.session_id / "workspace"
    (ws / "input.txt").write_text("seeded")

    seen = {}

    def on_run(s, c):
        seen.update(
            cwd_name=c.workspace.name,
            same_dir=(c.workspace == ws),
            seeded=(c.workspace / "input.txt").read_text(),
        )
        (c.workspace / "artifact.txt").write_text("produced")

    engine._harness = FakeHarness([_result_ev()], on_run=on_run)
    asyncio.run(_await(session.run("go")))
    assert seen == {"cwd_name": "workspace", "same_dir": True, "seeded": "seeded"}
    assert (session.workspace / "artifact.txt").read_text() == "produced"


def test_start_session_mints_canonical_uuid(tmp_path):
    # Regression: the id LocalEngine mints is passed to the Claude Agent SDK as session_id /
    # resume, which requires a CANONICAL UUID (dashed). uuid4().hex (no dashes) is rejected at
    # runtime with "Invalid session ID. Must be a valid UUID", breaking the live resume path.
    import uuid

    engine = local.deploy(AgentSpec(name="demo", model="m", checkpoint=True),
                          workdir=str(tmp_path / "wd"))
    sid = engine.start_session().session_id
    assert "-" in sid                     # dashed, unlike uuid4().hex
    assert str(uuid.UUID(sid)) == sid     # parses and is already canonical


def test_interactive_decoupled_from_checkpoint(tmp_path):
    # interactive defaults to the checkpoint flag; an explicit spec value overrides it,
    # so an autonomous loop can checkpoint without the "stop and await operator" guidance.
    cases = [
        (dict(checkpoint=True), True),
        (dict(checkpoint=True, interactive=False), False),
        (dict(checkpoint=False, interactive=True), True),
    ]
    for i, (kw, expected) in enumerate(cases):
        seen = {}
        spec = AgentSpec(name="demo", model="m", **kw)
        engine = local.deploy(spec, workdir=str(tmp_path / f"wd-int-{i}"))
        engine._harness = FakeHarness(
            [_result_ev()], on_run=lambda s, c: seen.update(interactive=c.interactive))
        asyncio.run(_await(engine.start_session().run("go")))
        assert seen["interactive"] is expected, kw


def test_checkpoint_blobstore_gcs_env_selection(tmp_path, monkeypatch):
    # AGENT_CHECKPOINT_GCS points the local engine's checkpoint blobs at GCS (same env
    # hook as the gemini runtime); without it the store is the pod-local <workdir>/blobs.
    from remote_agent_toolkit.ports import blobstore as bs

    created = {}

    class StubGcs:
        def __init__(self, bucket, prefix=""):
            created.update(bucket=bucket, prefix=prefix)

    monkeypatch.setattr(bs, "GcsBlobStore", StubGcs)
    monkeypatch.setenv("AGENT_CHECKPOINT_GCS", "gs://ckpt-bkt/some/prefix")
    spec = AgentSpec(name="demo", model="m", checkpoint=True)
    session = local.deploy(spec, workdir=str(tmp_path / "wd-gcs")).start_session()
    assert isinstance(session._blobs, StubGcs)
    assert created == {"bucket": "ckpt-bkt", "prefix": "some/prefix/"}

    monkeypatch.delenv("AGENT_CHECKPOINT_GCS")
    session2 = local.deploy(spec, workdir=str(tmp_path / "wd-local")).start_session()
    assert isinstance(session2._blobs, bs.LocalBlobStore)

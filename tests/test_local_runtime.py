"""Local runtime end-to-end with a fake harness (no live model call).

Exercises the full Engine/Session/Run machinery — skills staging, the three consumption
modes (await / async-iter / poll), status & stop_reason mapping, and resume-restore
wiring — by swapping in a harness that yields canned AgentEvents.
"""

from __future__ import annotations

import asyncio
import os
from typing import AsyncIterator

import pytest

from agent_run import AgentSpec, SkillSource, local
from agent_run.events import AgentEvent, RunStatus, StopReason


def _result_ev(text="done", subtype="success", is_error=False, num_turns=3):
    return AgentEvent(
        kind="result", summary=text, cost_usd=0.1, usage={"input_tokens": 5},
        raw={"subtype": subtype, "is_error": is_error, "num_turns": num_turns, "session_id": "claude-sid"},
    )


class FakeHarness:
    """Yields canned events; optional on_run hook to assert the prepared workspace.

    ``raise_after`` raises once all events are out — models the claude CLI exiting
    non-zero during shutdown, after the terminal result was already streamed.
    """

    def __init__(self, events, on_run=None, raise_after=None):
        self._events = events
        self._on_run = on_run
        self._raise_after = raise_after

    async def run(self, spec, ctx) -> AsyncIterator[AgentEvent]:
        if self._on_run is not None:
            self._on_run(spec, ctx)
        for ev in self._events:
            yield ev
        if self._raise_after is not None:
            raise self._raise_after


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


def test_run_hooks_reach_the_harness(tmp_path):
    seen = {}
    spec = AgentSpec(name="demo", model="m")
    engine = local.deploy(spec, workdir=str(tmp_path / "wd"))
    engine._harness = FakeHarness([_result_ev()], on_run=lambda s, c: seen.update(hooks=c.hooks))
    hooks = {"PreToolUse": []}
    session = engine.start_session()

    async def go():
        await session.run("go", hooks=hooks)

    asyncio.run(go())
    assert seen["hooks"] is hooks


def test_sync_run_forwards_config_and_hooks(tmp_path, monkeypatch):
    from agent_run.config import TurnConfig

    seen = {}
    spec = AgentSpec(name="demo", model="m")
    hooks = {"PreToolUse": []}
    monkeypatch.setattr(
        local.LocalEngine, "_harness_for",
        lambda self, s: FakeHarness([_result_ev()],
                                    on_run=lambda s, c: seen.update(hooks=c.hooks, model=s.model)),
    )
    local.run(spec, "go", workdir=str(tmp_path / "wd"),
              config=TurnConfig(model="m2"), hooks=hooks)
    assert seen == {"hooks": hooks, "model": "m2"}


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
    from agent_run.checkpoint.workspace import snapshot
    from agent_run.ports.blobstore import LocalBlobStore

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


def _seed_snapshot(engine, sid, populate):
    from agent_run.checkpoint.workspace import snapshot
    from agent_run.ports.blobstore import LocalBlobStore

    seed = engine._blob_root.parent / f"seed-{sid}"
    seed.mkdir(parents=True)
    populate(seed)
    snapshot(LocalBlobStore(str(engine._blob_root)), sid, str(seed))


def _resume_events(engine, sid, prompt="continue"):
    session = engine.get_session(sid)

    async def collect():
        return [ev async for ev in session.send(prompt)]

    return asyncio.run(collect())


def test_resume_restores_a_workspace_with_a_venv_symlink(tmp_path):
    # #74: every coding run leaves a ``.venv`` whose ``bin/python`` is an absolute symlink;
    # restoring such a snapshot on another worker aborted, and the fallback then tried to
    # clone the repo into the half-restored directory.
    spec = AgentSpec(name="demo", model="m", checkpoint=True)
    engine = local.deploy(spec, workdir=str(tmp_path / "wd"))

    def populate(seed):
        (seed / "project" / ".venv" / "bin").mkdir(parents=True)
        (seed / "project" / "a.py").write_text("x")
        os.symlink("/usr/local/bin/python3", seed / "project" / ".venv" / "bin" / "python")

    _seed_snapshot(engine, "venvsid", populate)
    seen = {}
    engine._harness = FakeHarness(
        [_result_ev()],
        on_run=lambda s, c: seen.update(
            link=os.readlink(c.workspace / "project" / ".venv" / "bin" / "python"),
            code=(c.workspace / "project" / "a.py").read_text(),
        ),
    )
    events = _resume_events(engine, "venvsid")
    assert seen == {"link": "/usr/local/bin/python3", "code": "x"}
    ready = events[0].raw
    assert ready["event"] == "workspace_ready" and ready["restored"] is True
    assert ready["restore_error"] is None and ready["skipped"] == []


def test_resume_with_a_broken_snapshot_starts_fresh_and_says_so(tmp_path):
    # A snapshot that cannot be extracted is not the caller's clone error and not silence:
    # the turn runs in a clean workspace and ``workspace_ready`` carries the reason.
    spec = AgentSpec(name="demo", model="m", checkpoint=True)
    engine = local.deploy(spec, workdir=str(tmp_path / "wd"))
    _seed_snapshot(engine, "badsid", lambda seed: (seed / "a.txt").write_text("x"))
    (engine._blob_root / "workspace" / "badsid.tar.gz").write_bytes(b"\x1f\x8b not a tarball")

    seen = {}
    engine._harness = FakeHarness(
        [_result_ev()], on_run=lambda s, c: seen.update(files=sorted(os.listdir(c.workspace)))
    )
    events = _resume_events(engine, "badsid")
    assert seen["files"] == []  # a clean directory, nothing half-restored
    ready = events[0].raw
    assert ready["restored"] is False
    assert ready["restore_error"] and "restore failed" in events[0].summary
    assert events[-1].kind == "result"


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


def test_deploy_workspace_is_shared_by_every_session(tmp_path):
    # An engine deployed with workspace= runs its sessions in that directory instead of
    # a per-session jobs/<sid>/workspace: the identical cwd keeps the system-prompt
    # prefix identical across sessions, which is what the prompt cache keys on.
    shared = tmp_path / "shared"
    spec = AgentSpec(name="demo", model="m")
    engine = local.deploy(spec, workdir=str(tmp_path / "wd"), workspace=str(shared))

    cwds = []
    engine._harness = FakeHarness([_result_ev()], on_run=lambda s, c: cwds.append(c.workspace))
    sessions = [engine.start_session() for _ in range(2)]
    for session in sessions:
        assert session.workspace == shared
        asyncio.run(_await(session.run("go")))
    assert cwds == [shared, shared]
    assert not (engine._jobs_root / sessions[0].session_id / "workspace").exists()
    # jobs/<sid>/ is bookkeeping, not the agent cwd: it exists per session even when the
    # agent ran elsewhere, so re-attach discovery still finds these sessions.
    assert [s["session_id"] for s in engine.list_sessions()] == sorted(
        s.session_id for s in sessions
    )


def test_run_forwards_the_chosen_workspace(tmp_path, monkeypatch):
    # local.run() is deploy + start_session + await in one call, so it has to forward the
    # cwd choice; absorbing it would run the agent somewhere the caller never named.
    import agent_run.harness as harness_mod

    shared = tmp_path / "shared"
    seen = {}
    monkeypatch.setattr(
        harness_mod,
        "resolve_harness",
        lambda spec: FakeHarness([_result_ev()], on_run=lambda s, c: seen.update(cwd=c.workspace)),
    )
    local.run(AgentSpec(name="demo", model="m"), "go",
              workdir=str(tmp_path / "wd"), workspace=str(shared))
    assert seen["cwd"] == shared


def test_chosen_workspace_rejects_repos(tmp_path):
    # Repo provisioning clones to a fixed <cwd>/<repo name>, so sharing one cwd across
    # sessions can only work for the first: refuse the combination up front, at deploy and
    # at the session bind that could still introduce repos via its config.
    from agent_run import RepoSource
    from agent_run.config import SessionConfig

    repos = [RepoSource.git("https://github.com/o/r")]
    shared = str(tmp_path / "shared")
    with pytest.raises(ValueError, match="workspace="):
        local.deploy(AgentSpec(name="demo", model="m", repos=repos),
                     workdir=str(tmp_path / "wd"), workspace=shared)

    engine = local.deploy(AgentSpec(name="demo", model="m"),
                          workdir=str(tmp_path / "wd2"), workspace=shared)
    with pytest.raises(ValueError, match="workspace="):
        engine.start_session(config=SessionConfig(repos=repos))


def test_chosen_workspace_is_never_restored_over(tmp_path):
    # A caller-owned cwd outlives the session and may hold files no session wrote, so a
    # resume continues in the directory as it stands: no snapshot is untarred over it,
    # which would roll files back to whatever this session last saw.
    shared = tmp_path / "shared"
    shared.mkdir()
    spec = AgentSpec(name="demo", model="m", checkpoint=True)
    engine = local.deploy(spec, workdir=str(tmp_path / "wd"), workspace=str(shared))

    from agent_run.checkpoint.workspace import snapshot
    from agent_run.ports.blobstore import LocalBlobStore

    stale = tmp_path / "stale"
    stale.mkdir()
    (stale / "notes.txt").write_text("as session 'fixedsid' last saw it")
    snapshot(LocalBlobStore(str(engine._blob_root)), "fixedsid", str(stale))
    (shared / "notes.txt").write_text("newer")

    seen = {}
    engine._harness = FakeHarness(
        [_result_ev()],
        on_run=lambda s, c: seen.update(notes=(c.workspace / "notes.txt").read_text()),
    )
    asyncio.run(_await(engine.get_session("fixedsid").send("continue please")))
    assert seen["notes"] == "newer"


def test_error_result_keeps_accounting(tmp_path):
    # Eval feedback: error results (error_max_turns) reported cost_usd=0.0 / num_turns=0 /
    # usage=None on last_result although the backend result event carried them — exactly the
    # runs that burn the most read as free. An errored run is still a fully accounted run.
    ev = AgentEvent(
        kind="result", summary="(run errored)", cost_usd=6.2966,
        usage={"input_tokens": 240, "cache_creation_input_tokens": 182975},
        raw={"subtype": "error_max_turns", "is_error": True, "num_turns": 121,
             "session_id": "claude-sid"},
    )
    engine = local.deploy(AgentSpec(name="demo", model="m"), workdir=str(tmp_path / "wd"))
    engine._harness = FakeHarness([ev])
    session = engine.start_session()
    asyncio.run(_await(session.run("go")))

    r = session.last_result
    assert r.is_error is True and session.stop_reason == StopReason.MAX_TURNS
    assert r.cost_usd == 6.2966 and r.num_turns == 121
    assert r.usage == {"input_tokens": 240, "cache_creation_input_tokens": 182975}


def test_late_harness_crash_keeps_finished_result(tmp_path):
    # Eval feedback (Fable hims_com run): the run finished — terminal result streamed —
    # then the claude CLI exited 1 during shutdown and the whole run (deliverable text,
    # ~$10 of spend, usage) was reported as an errored, free run. The terminal result is
    # authoritative: keep it, and record the late death as a warning, not an error.
    engine = local.deploy(AgentSpec(name="demo", model="m"), workdir=str(tmp_path / "wd"))
    engine._harness = FakeHarness(
        [_result_ev(text="all done")],
        raise_after=RuntimeError("Command failed with exit code 1"),
    )
    session = engine.start_session()

    async def collect():
        return [ev async for ev in session.run("go")]

    events = asyncio.run(collect())
    r = session.last_result
    assert r.text == "all done" and r.is_error is False
    assert r.cost_usd == 0.1 and r.num_turns == 3 and r.usage == {"input_tokens": 5}
    assert "exit code 1" in r.warning
    assert session.stop_reason == StopReason.END_TURN  # from the result, not the late crash
    # The stream carries the anomaly as a status event, after the terminal result.
    assert [e.kind for e in events][-1] == "status"
    assert "result kept" in events[-1].summary


def test_unknown_cost_reaches_the_caller_as_none(tmp_path):
    """An unpriced run must not read as a free one.

    The harness reports ``cost_usd=None`` when it could not price the model (no price
    data, or OpenRouter reported no charge) and fires a ``cost_unknown`` status event.
    ``RunResult.cost_usd`` carries that ``None`` through, so a caller can tell it apart
    from a turn that really cost nothing.
    """
    ev = _result_ev()
    ev.cost_usd = None
    engine = local.deploy(AgentSpec(name="demo", model="m"), workdir=str(tmp_path / "wd"))
    engine._harness = FakeHarness([ev])
    session = engine.start_session()
    asyncio.run(_await(session.run("go")))

    assert session.last_result.cost_usd is None
    assert session.last_result.num_turns == 3  # the rest of the accounting is unaffected


def test_a_free_run_still_reports_zero(tmp_path):
    """The other side of the same contract: 0.0 stays 0.0 and never becomes None."""
    ev = _result_ev()
    ev.cost_usd = 0.0
    engine = local.deploy(AgentSpec(name="demo", model="m"), workdir=str(tmp_path / "wd"))
    engine._harness = FakeHarness([ev])
    session = engine.start_session()
    asyncio.run(_await(session.run("go")))

    assert session.last_result.cost_usd == 0.0


def test_crash_before_result_is_still_an_error(tmp_path):
    # No terminal result → the exception is the outcome (unchanged semantics).
    engine = local.deploy(AgentSpec(name="demo", model="m"), workdir=str(tmp_path / "wd"))
    engine._harness = FakeHarness([], raise_after=RuntimeError("boom"))
    session = engine.start_session()
    asyncio.run(_await(session.run("go")))
    r = session.last_result
    assert r.is_error is True and "boom" in r.text and r.warning is None
    # The run died before reporting anything, so its spend is unknown, not zero.
    assert r.cost_usd is None
    assert session.stop_reason == StopReason.ERROR


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
    # hook as the sandbox runtime); without it the store is the pod-local <workdir>/blobs.
    from agent_run.ports import blobstore as bs

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


def test_transcript_wires_the_session_store_without_snapshotting(tmp_path):
    # transcript=True alone gives a session store to mirror the transcript to — and the
    # workspace snapshot stays off, so a huge working directory is not archived per turn.
    from agent_run.harness._shared import finalize_checkpoint

    spec = AgentSpec(name="demo", model="m", transcript=True)
    engine = local.deploy(spec, workdir=str(tmp_path / "wd"))
    session = engine.start_session()
    assert session._session_store is not None

    seen = {}
    engine._harness = FakeHarness(
        [_result_ev()], on_run=lambda s, c: seen.update(ctx=c))
    asyncio.run(_await(session.run("go")))
    ctx = seen["ctx"]
    assert finalize_checkpoint(spec, ctx) is None          # nothing snapshotted
    assert not session._blobs.list("workspace/")

    # What the harness mirrors to the store reads back through transcripts().
    entries = [{"uuid": "u1", "type": "user"}, {"uuid": "u2", "type": "assistant"}]
    asyncio.run(ctx.session_store.append({"session_id": session.session_id}, entries))
    asyncio.run(
        ctx.session_store.append(
            {"session_id": session.session_id, "subpath": "subagents/agent-X"},
            [{"uuid": "u3", "type": "user"}],
        )
    )
    assert asyncio.run(session.transcripts()) == {
        "main": entries,
        "subagents/agent-X": [{"uuid": "u3", "type": "user"}],
    }


def test_transcripts_without_persistence_raises(tmp_path):
    session = local.deploy(AgentSpec(name="demo", model="m"),
                           workdir=str(tmp_path / "wd")).start_session()
    with pytest.raises(RuntimeError, match="transcript=True"):
        asyncio.run(session.transcripts())


def test_transcript_only_send_does_not_resume(tmp_path):
    # transcript=True is observational: it wires the store but leaves send() the documented
    # fresh turn. Resume is checkpointing's, and stays in parity with sandbox.
    def resume_sid_of(spec, name):
        seen = {}
        engine = local.deploy(spec, workdir=str(tmp_path / name))
        engine._harness = FakeHarness([_result_ev()], on_run=lambda s, c: seen.update(ctx=c))
        session = engine.start_session()
        asyncio.run(_await(session.send("go")))
        assert seen["ctx"].session_store is not None
        return seen["ctx"].resume_sid

    assert resume_sid_of(AgentSpec(name="d", model="m", transcript=True), "t") is None
    assert resume_sid_of(AgentSpec(name="d", model="m", checkpoint=True), "c") is not None

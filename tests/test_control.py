"""Turn control (issue #44): steer into a running turn, interrupt and continue, interrupt and stop.

Offline throughout. Four layers, each against fakes:

* ``control.py`` — the message type, the in-process channel, the select helper.
* the harnesses — the Claude Agent SDK client and the Codex app-server are faked (see
  ``fakes`` / ``codex_fakes``); the scripts wait for the harness to act (a second
  ``query()``, an ``interrupt()``) the way the live CLIs were observed to.
* the runtimes — ``local`` in-process; the sandbox client + worker over the worker's
  ``/control`` endpoint are covered in ``test_sandbox_runtime``.
* the four session transitions run the same way on both runtimes.
"""

from __future__ import annotations

import asyncio

import pytest
from codex_fakes import (
    agent_message,
    make_async_codex,
    token_usage,
    turn_completed,
    turn_started,
)
from fakes import init_msg, make_sdk_client, result_msg, tool_result_msg

from claude_agent_sdk import AssistantMessage, TextBlock, ToolUseBlock

from agent_run import AgentSpec, local
from agent_run.checkpoint.session_store import BlobSessionStore
from agent_run.conformance import run_harness_conformance
from agent_run.control import (
    ControlledStream,
    ControlMessage,
    LocalControlChannel,
)
from agent_run.events import AgentEvent, RunStatus, StopReason
from agent_run.harness._shared import finalize_checkpoint
from agent_run.harness.claude_code import ClaudeCodeHarness
from agent_run.harness.codex import CodexHarness
from agent_run.harness.context import RunContext
from agent_run.ports.blobstore import LocalBlobStore
from agent_run.runtime._run import build_result


async def _until(pred, what: str, timeout: float = 5.0):
    """Poll ``pred`` until true; a script step that never happens fails loudly."""
    deadline = asyncio.get_running_loop().time() + timeout
    while not pred():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"timed out waiting for: {what}")
        await asyncio.sleep(0.005)


def _assistant(*blocks):
    return AssistantMessage(content=list(blocks), model="m")


def _kinds(events):
    return [(e.kind, (e.raw or {}).get("event") or (e.raw or {}).get("subtype")) for e in events]


# -- control.py -----------------------------------------------------------------------


def test_control_message_round_trip_and_validation():
    steer = ControlMessage(op="steer", message="skip the images", message_id="m-1")
    assert ControlMessage.from_dict(steer.to_dict()) == steer
    stop = ControlMessage.from_dict({"op": "stop", "message": "ignored"})
    assert stop.op == "stop" and stop.message is None and stop.message_id
    with pytest.raises(ValueError, match="unknown control op"):
        ControlMessage.from_dict({"op": "cancel"})
    with pytest.raises(ValueError, match="needs a message"):
        ControlMessage.from_dict({"op": "steer"})
    # The ack event: the text, the caller's id, and whether it came with an interrupt.
    ev = ControlMessage(op="interrupt", message="use the sitemap", message_id="m-2").user_event()
    assert ev.kind == "user" and ev.summary == "use the sitemap"
    assert ev.raw == {"event": "user_message", "message_id": "m-2", "interrupt": True}


def test_controlled_stream_reads_both_sources_and_keeps_items_across_timeouts():
    async def go():
        gate = asyncio.Event()

        async def stream():
            yield "a"
            await gate.wait()
            yield "b"

        channel = LocalControlChannel()
        select = ControlledStream(stream(), channel)
        assert await select.next(1.0) == ("message", "a")
        # Nothing ready: a timeout, and the pending stream read is NOT dropped.
        assert await select.next(0.01) == ("timeout", None)
        channel.send(ControlMessage(op="stop"))
        kind, msg = await select.next(1.0)
        assert kind == "control" and msg.op == "stop"
        gate.set()
        assert await select.next(1.0) == ("message", "b")
        assert await select.next(1.0) == ("end", None)
        await select.close()

    asyncio.run(go())


def test_controlled_stream_hands_control_over_before_a_stream_item():
    # An operator's interrupt must not queue behind a stream item ready in the same tick.
    async def go():
        async def stream():
            yield "a"

        channel = LocalControlChannel()
        channel.send(ControlMessage(op="stop"))
        select = ControlledStream(stream(), channel)
        first, _ = await select.next(1.0)
        assert first == "control"
        assert await select.next(1.0) == ("message", "a")
        await select.close()

    asyncio.run(go())


# -- Claude Code harness ----------------------------------------------------------------


def _claude_ctx(tmp_path, spec, control, **kw):
    return RunContext(
        spec=spec, prompt="build it", job_dir=tmp_path / "job", session_id="sid",
        control=control, **kw,
    )


def _claude_events(script, tmp_path, monkeypatch, control, spec=None, ctx=None):
    import claude_agent_sdk

    client_cls = make_sdk_client(script)
    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", client_cls)
    spec = spec or AgentSpec(name="a", model="m")
    ctx = ctx or _claude_ctx(tmp_path, spec, control)

    async def drive():
        return [ev async for ev in ClaudeCodeHarness().run(spec, ctx)]

    return asyncio.run(drive()), client_cls.instances[-1]


def _second_query(client):
    return _until(lambda: len(client.prompts) >= 2, "the harness to query() the steer")


def _interrupted(client):
    return _until(lambda: client.interrupted, "the harness to interrupt() the client")


def test_claude_steer_is_queried_into_the_turn_and_acked(tmp_path, monkeypatch):
    # Verified live: query() mid-turn reaches the model after the running tool call, the
    # CLI never echoes it, and the turn ends with ONE result. The harness emits the user
    # event itself, and the tool_result → model output sequence proves delivery, so the
    # turn ends at that result with no settle wait.
    control = LocalControlChannel()
    control.send(ControlMessage(op="steer", message="skip the images", message_id="m-1"))
    script = [
        init_msg(),
        _assistant(ToolUseBlock(id="tu1", name="Bash", input={"command": "sleep 8"})),
        _second_query,  # the steer is handed over while the tool runs
        tool_result_msg("tu1", "one"),  # the boundary where the next call is assembled
        _assistant(TextBlock(text="BANANA")),  # that call carried the message
        result_msg(num_turns=2, result="BANANA"),
    ]
    events, client = _claude_events(script, tmp_path, monkeypatch, control)

    assert client.prompts == ["build it", "skip the images"]
    users = [e for e in events if e.kind == "user"]
    assert len(users) == 1 and users[0].summary == "skip the images"
    assert users[0].raw == {"event": "user_message", "message_id": "m-1", "interrupt": False}
    results = [e for e in events if e.kind == "result"]
    assert len(results) == 1 and results[0].summary == "BANANA"
    assert results[0].raw["num_turns"] == 2
    assert not any((e.raw or {}).get("event") == "awaiting_steer" for e in events)
    # The ack precedes the model's reply to it.
    assert events.index(users[0]) < events.index(results[0])


def test_claude_steer_that_missed_the_turn_continues_in_a_new_invocation(tmp_path, monkeypatch):
    # The message arrives as the model ends its turn. Verified live: the CLI then starts
    # a new invocation for it right away (fresh init). The first result is a segment
    # boundary; the run stays one Run with one final result covering both segments.
    control = LocalControlChannel()
    control.send(ControlMessage(op="steer", message="one more thing"))
    script = [
        init_msg(),
        _second_query,
        result_msg(num_turns=2, result="done with the first part"),  # no delivery proof
        init_msg(),  # the CLI took the queued message: a new invocation
        _assistant(TextBlock(text="and the extra thing")),
        result_msg(num_turns=1, result="and the extra thing"),
    ]
    events, client = _claude_events(script, tmp_path, monkeypatch, control)
    results = [e for e in events if e.kind == "result"]
    assert len(results) == 1 and results[0].summary == "and the extra thing"
    assert results[0].raw["num_turns"] == 3  # cumulative across the segments
    assert any((e.raw or {}).get("event") == "awaiting_steer" for e in events)
    assert client.disconnected


def test_claude_steer_without_delivery_proof_settles_after_the_grace(tmp_path, monkeypatch):
    # A result arrives with a steer sent but unproven, and NO new invocation follows: the
    # message was consumed inside the turn. The harness waits the short settle window,
    # then that result is the turn's end (not a hang, not a lost result).
    monkeypatch.setattr(ClaudeCodeHarness, "_STEER_SETTLE_S", 0.05)
    control = LocalControlChannel()
    control.send(ControlMessage(op="steer", message="also do X"))
    script = [init_msg(), _second_query, result_msg(num_turns=2, result="did it all"), "hang"]
    events, _ = _claude_events(script, tmp_path, monkeypatch, control)
    results = [e for e in events if e.kind == "result"]
    assert len(results) == 1 and results[0].summary == "did it all"


def test_claude_interrupt_with_message_continues_in_the_same_client(tmp_path, monkeypatch):
    # Verified live: interrupt() makes the CLI answer with an error_during_execution
    # result; a query() on the same client then runs normally (new init). The harness
    # treats the settle result as a segment boundary and the follow-up as the turn.
    control = LocalControlChannel()
    control.send(ControlMessage(op="interrupt", message="stop, use the sitemap", message_id="m-9"))
    script = [
        init_msg(),
        _assistant(ToolUseBlock(id="tu1", name="Bash", input={"command": "sleep 40"})),
        _interrupted,
        result_msg(subtype="error_during_execution", is_error=True, num_turns=3, result=None),
        init_msg(),
        _assistant(TextBlock(text="OK, sitemap it is")),
        result_msg(num_turns=1, result="OK, sitemap it is"),
    ]
    events, client = _claude_events(script, tmp_path, monkeypatch, control)

    assert client.interrupts == 1
    assert client.prompts == ["build it", "stop, use the sitemap"]
    results = [e for e in events if e.kind == "result"]
    assert len(results) == 1 and results[0].summary == "OK, sitemap it is"
    assert results[0].raw["is_error"] is False and results[0].raw["num_turns"] == 4
    tags = [(e.raw or {}).get("event") for e in events]
    assert "interrupt_requested" in tags and "interrupted_segment" in tags
    user = next(e for e in events if e.kind == "user")
    assert user.raw["interrupt"] is True and user.raw["message_id"] == "m-9"
    # Order: interrupt requested → the settle result demoted → the message delivered.
    assert tags.index("interrupt_requested") < tags.index("interrupted_segment") < events.index(user)


def test_claude_stop_ends_the_turn_as_interrupted_with_a_checkpoint(tmp_path, monkeypatch):
    # Session.interrupt(): the settle result IS the turn's end, re-stamped as a pause
    # (subtype interrupted, not an error), with the model's last text and the turn's
    # accounting; the checkpoint runs inline so the interrupted turn's files survive.
    spec = AgentSpec(name="a", model="m", checkpoint=True)
    blobs = LocalBlobStore(str(tmp_path / "blobs"))
    control = LocalControlChannel()
    control.send(ControlMessage(op="stop"))
    ctx = _claude_ctx(tmp_path, spec, control, blobs=blobs, session_store=BlobSessionStore(blobs))
    ctx.workspace.mkdir(parents=True)

    def write_partial(client):
        (ctx.workspace / "partial.py").write_text("half done")

    script = [
        init_msg(),
        _assistant(TextBlock(text="I am writing the parser")),
        write_partial,
        _interrupted,
        result_msg(subtype="error_during_execution", is_error=True, num_turns=3, result=None,
                   cost=0.42),
    ]
    events, client = _claude_events(script, tmp_path, monkeypatch, control, spec=spec, ctx=ctx)

    result = next(e for e in events if e.kind == "result")
    assert result.raw["subtype"] == "interrupted" and result.raw["is_error"] is False
    assert result.raw["cli_reported_subtype"] == "error_during_execution"
    assert result.summary == "I am writing the parser"
    assert result.raw["num_turns"] == 3 and result.cost_usd == 0.42  # accounting kept
    assert events[-2].raw["event"] == "checkpoint_saved"
    assert events[-1] is result  # consumers stop at the result
    assert blobs.exists("workspace/sid.tar.gz")
    assert client.prompts == ["build it"]  # nothing was sent after the stop
    run_result, stop = build_result(result, "sid", spec)
    assert stop == StopReason.INTERRUPTED and run_result.is_error is False
    # The turn still satisfies the harness contract (init, terminal result, usage shape).
    asyncio.run(run_harness_conformance(lambda: _replay(events)))


async def _replay(events):
    for e in events:
        yield e


def test_claude_stop_with_nothing_running_settles_synthetically(tmp_path, monkeypatch):
    # No settle result comes when the model was idle at the interrupt (e.g. waiting on
    # background tasks). The stop still ends the turn — as interrupted — after the window.
    monkeypatch.setattr(ClaudeCodeHarness, "_INTERRUPT_SETTLE_S", 0.05)
    control = LocalControlChannel()
    control.send(ControlMessage(op="stop"))
    script = [init_msg(), _interrupted, "hang"]
    events, _ = _claude_events(script, tmp_path, monkeypatch, control)
    result = next(e for e in events if e.kind == "result")
    assert result.raw["subtype"] == "interrupted" and result.raw["is_error"] is False
    assert result.raw["interrupt_settled"] is False and result.summary == "(interrupted)"
    assert isinstance(result.usage, dict)  # normalized shape, all None


def test_claude_stop_while_awaiting_background_tasks_ends_at_once(tmp_path, monkeypatch):
    # The model ended its turn with a background task pending; the harness holds the turn
    # open for the CLI's re-invocation. A stop there has nothing to interrupt: the turn
    # ends now, as interrupted, on the result already produced — no settle wait (the
    # window is set long enough to time the test out if it were waited).
    from fakes import task_started_msg

    monkeypatch.setattr(ClaudeCodeHarness, "_INTERRUPT_SETTLE_S", 30.0)
    control = LocalControlChannel()
    script = [
        init_msg(),
        task_started_msg("t1"),
        result_msg(num_turns=2, result="waiting on the crawl"),  # demoted: task pending
        lambda client: control.send(ControlMessage(op="stop")),
        "hang",  # the CLI keeps the stream open for the task
    ]
    events, client = _claude_events(script, tmp_path, monkeypatch, control)
    result = next(e for e in events if e.kind == "result")
    assert result.raw["subtype"] == "interrupted" and result.raw["is_error"] is False
    assert result.summary == "waiting on the crawl" and result.raw["num_turns"] == 2
    assert result.raw["cli_reported_subtype"] == "success"
    assert client.interrupts == 0  # nothing was running


def test_claude_interrupt_with_nothing_running_still_delivers_the_message(tmp_path, monkeypatch):
    monkeypatch.setattr(ClaudeCodeHarness, "_INTERRUPT_SETTLE_S", 0.05)
    control = LocalControlChannel()
    control.send(ControlMessage(op="interrupt", message="carry on with B"))
    script = [
        init_msg(),
        _interrupted,
        _second_query,  # delivered after the settle window with no settle result
        init_msg(),
        result_msg(num_turns=1, result="B done"),
    ]
    events, client = _claude_events(script, tmp_path, monkeypatch, control)
    assert client.prompts == ["build it", "carry on with B"]
    assert [e.kind for e in events if e.kind in ("user", "result")] == ["user", "result"]
    assert events[-1].summary == "B done"


# -- Codex harness ----------------------------------------------------------------------


def _codex_events(script, tmp_path, monkeypatch, control, follow_up=(), spec=None):
    import openai_codex

    client_cls = make_async_codex(script, follow_up_scripts=list(follow_up))
    monkeypatch.setattr(openai_codex, "AsyncCodex", client_cls)
    spec = spec or AgentSpec(name="a", model="gpt-5.6-luna", harness="codex")
    ctx = RunContext(
        spec=spec, prompt="build it", job_dir=tmp_path / "job", session_id="sid",
        secrets={"OPENAI_API_KEY": "sk-test"}, control=control,
    )
    ctx.workspace.mkdir(parents=True, exist_ok=True)

    async def drive():
        return [ev async for ev in CodexHarness().run(spec, ctx)]

    return asyncio.run(drive()), client_cls.instances[-1]


def test_codex_steer_uses_turn_steer(tmp_path, monkeypatch):
    # Verified live: turn/steer lands after the current step and the turn ends with one
    # turn/completed. The user event is emitted when Codex accepts the steer.
    control = LocalControlChannel()
    control.send(ControlMessage(op="steer", message="just say BANANA", message_id="m-1"))
    script = [
        turn_started(),
        lambda client: _until(lambda: client.steers, "the harness to steer()"),
        agent_message("BANANA"),
        token_usage(),
        turn_completed(),
    ]
    events, client = _codex_events(script, tmp_path, monkeypatch, control)
    assert client.steers == ["just say BANANA"]
    user = next(e for e in events if e.kind == "user")
    assert user.raw["message_id"] == "m-1" and user.raw["interrupt"] is False
    result = events[-1]
    assert result.kind == "result" and result.summary == "BANANA"
    assert result.raw["subtype"] == "success" and result.raw["num_turns"] == 1


def test_codex_interrupt_with_message_starts_a_new_turn_on_the_same_thread(tmp_path, monkeypatch):
    # Verified live: interrupt() → turn/completed status=interrupted, and thread.turn()
    # on the same thread continues. The interrupted turn is a segment boundary; the run's
    # accounting spans both turns.
    control = LocalControlChannel()
    control.send(ControlMessage(op="interrupt", message="use the sitemap", message_id="m-2"))
    script = [
        turn_started(),
        token_usage(),
        lambda client: _until(lambda: client.interrupted, "the harness to interrupt()"),
        turn_completed("interrupted"),
    ]
    follow_up = [turn_started(), agent_message("sitemap it is"), token_usage(), turn_completed()]
    events, client = _codex_events(script, tmp_path, monkeypatch, control, follow_up=[follow_up])
    assert [prompt for prompt, _ in client.turns] == ["build it", "use the sitemap"]
    tags = [(e.raw or {}).get("event") for e in events]
    assert "interrupt_requested" in tags and "interrupted_segment" in tags
    user = next(e for e in events if e.kind == "user")
    assert user.raw["interrupt"] is True and user.raw["message_id"] == "m-2"
    results = [e for e in events if e.kind == "result"]
    assert len(results) == 1 and results[0].summary == "sitemap it is"
    assert results[0].raw["subtype"] == "success" and results[0].raw["num_turns"] == 2


def test_codex_stop_ends_the_turn_as_interrupted(tmp_path, monkeypatch):
    control = LocalControlChannel()
    control.send(ControlMessage(op="stop"))
    script = [
        turn_started(),
        agent_message("working on it", phase="commentary"),
        token_usage(),
        lambda client: _until(lambda: client.interrupted, "the harness to interrupt()"),
        turn_completed("interrupted"),
    ]
    events, client = _codex_events(script, tmp_path, monkeypatch, control)
    result = events[-1]
    assert result.kind == "result" and result.raw["subtype"] == "interrupted"
    assert result.raw["is_error"] is False and result.summary == "working on it"
    assert result.raw["num_turns"] == 1 and result.cost_usd is not None
    assert len(client.turns) == 1
    assert build_result(result, "sid", AgentSpec(name="a", model="m"))[1] == StopReason.INTERRUPTED


def test_codex_stop_with_no_settle_settles_synthetically(tmp_path, monkeypatch):
    monkeypatch.setattr(CodexHarness, "_INTERRUPT_SETTLE_S", 0.05)
    control = LocalControlChannel()
    control.send(ControlMessage(op="stop"))

    async def hang(client):
        await asyncio.Event().wait()

    script = [turn_started(), agent_message("partial", phase="commentary"), hang]
    events, _ = _codex_events(script, tmp_path, monkeypatch, control)
    result = events[-1]
    assert result.kind == "result" and result.raw["subtype"] == "interrupted"
    assert result.raw["is_error"] is False and result.summary == "partial"


# -- the four transitions on the local runtime --------------------------------------------


def _result_ev(text="done", subtype="success", is_error=False, num_turns=1):
    return AgentEvent(
        kind="result", summary=text, cost_usd=0.1, usage={"input_tokens": 5},
        raw={"subtype": subtype, "is_error": is_error, "num_turns": num_turns,
             "session_id": "claude-sid"},
    )


class ControlAwareHarness:
    """A harness that behaves like the real ones do with a control message.

    On a fresh turn it starts, then waits for one control message: a steer or an
    interrupt is acknowledged with the user event and answered; a stop ends the turn as
    interrupted after writing a file (so the checkpoint has something to preserve). A
    resumed turn (``ctx.resume_sid``) reports what it found in the restored workspace.
    ``ignore_control`` makes it deaf, standing in for a harness that never settles.
    """

    def __init__(self, ignore_control: bool = False) -> None:
        self.ignore_control = ignore_control
        self.seen: list[ControlMessage] = []

    async def run(self, spec, ctx):
        yield AgentEvent(kind="status", summary="started", raw={"subtype": "init"})
        if ctx.resume_sid:
            found = sorted(p.name for p in ctx.workspace.iterdir())
            yield _result_ev(text=f"resumed; workspace has {found}")
            return
        if self.ignore_control:
            await asyncio.Event().wait()
        msg = await ctx.control.receive()
        self.seen.append(msg)
        if msg.op == "stop":
            (ctx.workspace / "partial.py").write_text("half done")
            fin = finalize_checkpoint(spec, ctx)
            yield AgentEvent(
                kind="result", summary="(interrupted)", cost_usd=0.2, usage={"input_tokens": 9},
                raw={"subtype": "interrupted", "is_error": False, "num_turns": 2,
                     "session_id": "claude-sid"},
            )
            if fin is not None:
                yield fin
            return
        yield msg.user_event()
        yield _result_ev(text=f"saw {msg.op}: {msg.message}", num_turns=2)


def _local_engine(tmp_path, harness, checkpoint=False):
    engine = local.deploy(AgentSpec(name="d", model="m", checkpoint=checkpoint),
                          workdir=str(tmp_path / "wd"))
    engine._harness = harness
    return engine


def test_local_send_into_running_turn_steers_and_returns_the_same_run(tmp_path):
    harness = ControlAwareHarness()
    session = _local_engine(tmp_path, harness).start_session()

    async def go():
        run = session.run("build it")
        await asyncio.sleep(0.02)  # the harness is waiting on control
        assert session.status == RunStatus.RUNNING and session.busy
        same = session.send("skip the images", message_id="m-1")
        assert same is run  # the turn keeps running as the same Run
        events = [e async for e in run]
        return run, events

    run, events = asyncio.run(go())
    assert [m.op for m in harness.seen] == ["steer"]
    user = next(e for e in events if e.kind == "user")
    assert user.summary == "skip the images" and user.raw["message_id"] == "m-1"
    assert run.result.text == "saw steer: skip the images" and run.result.num_turns == 2
    assert session.status == RunStatus.IDLE and session.stop_reason == StopReason.END_TURN


def test_local_send_with_interrupt_reaches_the_harness_as_interrupt(tmp_path):
    harness = ControlAwareHarness()
    session = _local_engine(tmp_path, harness).start_session()

    async def go():
        run = session.run("build it")
        await asyncio.sleep(0.02)
        assert session.send("use the sitemap", interrupt=True) is run
        return await run

    result = asyncio.run(go())
    assert [m.op for m in harness.seen] == ["interrupt"]
    assert result.text == "saw interrupt: use the sitemap"


def test_local_interrupt_checkpoints_and_a_later_send_resumes_from_it(tmp_path):
    harness = ControlAwareHarness()
    engine = _local_engine(tmp_path, harness, checkpoint=True)
    session = engine.start_session()

    async def go():
        run = session.run("build it")
        await asyncio.sleep(0.02)
        await session.interrupt()
        assert run.done
        first = run.result
        assert session.status == RunStatus.IDLE
        assert session.stop_reason == StopReason.INTERRUPTED
        assert first.is_error is False and first.cost_usd == 0.2 and first.num_turns == 2
        # Resumable: the next send() is a normal turn from the interrupted turn's checkpoint,
        # workspace changes included.
        second = await session.send("continue")
        return second

    second = asyncio.run(go())
    assert [m.op for m in harness.seen] == ["stop"]
    assert "partial.py" in second.text
    assert session.stop_reason == StopReason.NEEDS_INPUT


def test_local_run_and_send_guards_while_a_turn_runs(tmp_path):
    from agent_run.config import TurnConfig

    harness = ControlAwareHarness()
    session = _local_engine(tmp_path, harness).start_session()

    async def go():
        run = session.run("build it")
        await asyncio.sleep(0.02)
        with pytest.raises(RuntimeError, match="running a turn"):
            session.run("another one")
        with pytest.raises(ValueError, match="cannot change while it runs"):
            session.send("x", config=TurnConfig(model="m2"))
        with pytest.raises(ValueError, match="cannot change while it runs"):
            session.send("x", secrets={"K": "v"})
        assert not harness.seen  # nothing reached the harness
        await session.interrupt()
        return run

    run = asyncio.run(go())
    assert run.done and session.stop_reason == StopReason.INTERRUPTED


def test_local_interrupt_falls_back_to_cancel_when_the_harness_never_stops(tmp_path):
    session = _local_engine(tmp_path, ControlAwareHarness(ignore_control=True)).start_session()

    async def go():
        run = session.run("build it")
        await asyncio.sleep(0.02)
        await session.interrupt(timeout=0.05)
        return run

    run = asyncio.run(go())
    assert run.done and run.result.is_error is True and run.result.text is None
    assert "without a checkpoint" in run.result.warning
    assert session.stop_reason == StopReason.ERROR


def test_local_interrupt_when_idle_is_a_noop(tmp_path):
    session = _local_engine(tmp_path, ControlAwareHarness()).start_session()
    asyncio.run(session.interrupt())
    assert session.status == RunStatus.PENDING


def test_local_send_when_idle_is_a_new_turn(tmp_path):
    # Unchanged: interrupt= is ignored on an idle session.
    harness = ControlAwareHarness()
    session = _local_engine(tmp_path, harness).start_session()

    async def go():
        run = session.send("first", interrupt=True)
        await asyncio.sleep(0.02)
        session.send("now steer")
        return await run

    result = asyncio.run(go())
    assert result.text == "saw steer: now steer"
    assert harness.seen[0].op == "steer"  # the idle send started the turn; only the steer arrived

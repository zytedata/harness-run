"""Background-task semantics are honored (eval feedback: Monitor fired into the void).

The CLI re-invokes the model when a background task (Bash ``run_in_background``,
Monitor) reaches a terminal state — but only while the message stream stays open. The
one-shot ``query()`` tore the CLI down at the first result; the harness now drives a
``ClaudeSDKClient`` stream and treats a result with pending/undelivered tasks as a
segment boundary, ending the turn only at a result with no pending work (proven live —
see DESIGN.md §7). All offline: the SDK client is faked.
"""

from __future__ import annotations

import asyncio

from claude_agent_sdk import AssistantMessage, TextBlock

from fakes import init_msg, make_sdk_client, result_msg, task_done_msg, task_started_msg

from remote_agent_toolkit import AgentSpec
from remote_agent_toolkit.harness.claude_code import ClaudeCodeHarness, _TaskTracker
from remote_agent_toolkit.harness.context import RunContext
from remote_agent_toolkit.harness.translate import EventTranslator


def _events_of(script, tmp_path, monkeypatch, spec=None):
    import claude_agent_sdk

    client_cls = make_sdk_client(script)
    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", client_cls)
    spec = spec or AgentSpec(name="a", model="m")
    ctx = RunContext(spec=spec, prompt="go", job_dir=tmp_path / "job", session_id="sid")

    async def drive():
        return [ev async for ev in ClaudeCodeHarness().run(spec, ctx)]

    return asyncio.run(drive()), client_cls


# -- the tracker ----------------------------------------------------------------------


def test_tracker_lifecycle():
    tr = _TaskTracker()
    translate = EventTranslator().translate

    def feed(msg):
        for ev in translate(msg):
            tr.observe(ev)

    assert tr.waiting() is None
    feed(task_started_msg("t1"))
    assert tr.waiting() == "pending" and tr.pending == {"t1": "bg work"}
    feed(task_done_msg("t1"))
    # Terminal but the model hasn't been re-invoked yet — the CLI is about to.
    assert tr.waiting() == "undelivered" and tr.pending == {}
    feed(init_msg())  # re-invocation: the notification was delivered
    assert tr.waiting() is None


def test_tracker_clears_terminal_notification_after_later_assistant_output():
    tr = _TaskTracker()
    translate = EventTranslator().translate

    def feed(msg):
        for ev in translate(msg):
            tr.observe(ev)

    feed(task_started_msg("t1"))
    feed(task_done_msg("t1"))
    assert tr.waiting() == "undelivered"

    # This is the common foreground/auto-backgrounded command ordering: the
    # completion arrives mid-invocation, then Claude discusses it and keeps
    # working.  The later assistant output proves the notification was consumed.
    feed(AssistantMessage(content=[TextBlock(text="the task passed")], model="m"))
    assert tr.waiting() is None


def test_translator_task_events():
    translate = EventTranslator().translate
    (started,) = translate(task_started_msg("t9", description="scrapy crawl"))
    assert started.kind == "status" and started.raw["event"] == "task_started"
    assert "scrapy crawl" in started.summary
    (done,) = translate(task_done_msg("t9"))
    assert done.kind == "status" and done.raw["event"] == "task_terminal"
    assert done.raw["status"] == "completed"


# -- the run loop ---------------------------------------------------------------------


def test_turn_stays_open_across_task_reinvocation(tmp_path, monkeypatch):
    # The observed live sequence: model starts a task, ends its turn; the notification
    # arrives; the CLI re-invokes; the SECOND result ends the turn. Exactly one result
    # event must come out (the final one), with the cumulative turn count.
    script = [
        init_msg(),
        task_started_msg("t1"),
        result_msg(num_turns=2, cost=0.02, result="WAITING"),  # segment boundary
        task_done_msg("t1"),
        init_msg(),  # the CLI's re-invocation
        result_msg(num_turns=3, cost=0.05, result="verified, all done"),
    ]
    events, client_cls = _events_of(script, tmp_path, monkeypatch)

    results = [e for e in events if e.kind == "result"]
    assert len(results) == 1
    assert results[0].summary == "verified, all done"
    assert results[0].raw["num_turns"] == 5  # 2 + 3, cumulative across re-invocations
    assert results[0].cost_usd == 0.05  # the CLI's total_cost_usd is already cumulative
    # The demoted first result surfaced as an awaiting_tasks status, before the final one.
    kinds = [(e.kind, (e.raw or {}).get("event")) for e in events]
    assert ("status", "awaiting_tasks") in kinds
    assert kinds.index(("status", "awaiting_tasks")) < kinds.index(("result", None))
    assert client_cls.instances[0].disconnected  # CLI torn down at turn end


def test_active_reinvocation_is_not_subject_to_notification_grace(tmp_path, monkeypatch):
    # Once init proves the CLI re-invoked the model, the notification grace is over. A
    # foreground tool/model step must not inherit that timeout. Record the timeout chosen
    # for each stream read so the state boundary is deterministic without sleeping.
    import remote_agent_toolkit.harness.claude_code as harness_mod

    original_wait_for = harness_mod.asyncio.wait_for
    timeouts = []

    async def recording_wait_for(awaitable, timeout):
        timeouts.append(timeout)
        return await original_wait_for(awaitable, timeout)

    monkeypatch.setattr(harness_mod.asyncio, "wait_for", recording_wait_for)

    script = [
        init_msg(),
        task_started_msg("t1"),
        result_msg(num_turns=2, result="WAITING"),
        task_done_msg("t1"),
        init_msg(),  # notification delivered; the new invocation is now actively working
        result_msg(num_turns=3, result="verified, all done"),
    ]
    events, _ = _events_of(script, tmp_path, monkeypatch)

    results = [e for e in events if e.kind == "result"]
    assert len(results) == 1
    assert results[0].summary == "verified, all done"
    assert results[0].raw["num_turns"] == 5
    assert ClaudeCodeHarness._UNDELIVERED_GRACE_S in timeouts[:-1]
    assert timeouts[-1] is None  # active invocation: foreground work may run indefinitely
    assert not any((e.raw or {}).get("event") == "task_wait_timeout" for e in events)


def test_no_tasks_result_is_final_immediately(tmp_path, monkeypatch):
    events, _ = _events_of([init_msg(), result_msg(result="plain")], tmp_path, monkeypatch)
    assert [e.kind for e in events] == ["status", "result"]
    assert events[-1].summary == "plain"


def test_midturn_task_completion_does_not_demote_later_final_result(tmp_path, monkeypatch):
    script = [
        task_started_msg("t1"),
        task_done_msg("t1"),
        AssistantMessage(content=[TextBlock(text="observed completion")], model="m"),
        result_msg(num_turns=2, result="final answer"),
        "hang",  # must not be read: the result is terminal immediately
    ]
    events, _ = _events_of(script, tmp_path, monkeypatch)

    assert events[-1].kind == "result" and events[-1].summary == "final answer"
    assert not any((e.raw or {}).get("event") == "awaiting_tasks" for e in events)


def test_error_result_ends_turn_even_with_pending_tasks(tmp_path, monkeypatch):
    # An error result (e.g. error_max_turns) is terminal: don't hold the turn open
    # waiting on tasks the conversation can no longer consume.
    script = [
        task_started_msg("t1"),
        result_msg(subtype="error_max_turns", is_error=True, num_turns=9, result=None),
    ]
    events, _ = _events_of(script, tmp_path, monkeypatch)
    assert events[-1].kind == "result" and events[-1].raw["is_error"] is True


def test_pending_wait_times_out_and_last_result_stands(tmp_path, monkeypatch):
    # A task that never completes must not hang the run: background_task_timeout expires,
    # a task_wait_timeout status is emitted, and the stashed result becomes the turn's.
    spec = AgentSpec(name="a", model="m", background_task_timeout=0.01)  # floor is 1s
    script = [
        task_started_msg("t1"),
        result_msg(num_turns=4, result="started the crawl"),
        "hang",
    ]
    events, _ = _events_of(script, tmp_path, monkeypatch, spec=spec)
    assert any((e.raw or {}).get("event") == "task_wait_timeout" for e in events)
    assert events[-1].kind == "result" and events[-1].summary == "started the crawl"
    assert events[-1].raw["num_turns"] == 4


def test_undelivered_grace_does_not_hang_on_midturn_delivery(tmp_path, monkeypatch):
    # A task that went terminal mid-turn is delivered inside the current invocation —
    # no re-invocation follows. The short grace window must expire and finalize the
    # stashed result rather than waiting out the full background_task_timeout.
    monkeypatch.setattr(ClaudeCodeHarness, "_UNDELIVERED_GRACE_S", 0.1)
    script = [
        task_started_msg("t1"),
        task_done_msg("t1"),  # terminal before any result; no init afterwards
        result_msg(num_turns=2, result="handled inline"),
        "hang",
    ]
    events, _ = _events_of(script, tmp_path, monkeypatch)
    assert events[-1].kind == "result" and events[-1].summary == "handled inline"


def test_crash_while_awaiting_tasks_keeps_stashed_result(tmp_path, monkeypatch):
    # If the CLI dies while the turn is held open, the segment result already produced
    # must not be voided (same principle as the runtimes' late-harness-death handling).
    script = [
        task_started_msg("t1"),
        result_msg(num_turns=2, result="work done, awaiting crawl"),
        RuntimeError("Command failed with exit code 1"),
    ]
    events, _ = _events_of(script, tmp_path, monkeypatch)  # must not raise
    assert events[-1].kind == "result" and events[-1].summary == "work done, awaiting crawl"
    assert any((e.raw or {}).get("event") == "late_harness_error" for e in events)

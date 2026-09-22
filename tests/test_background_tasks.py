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

from fakes import (
    init_msg,
    make_sdk_client,
    result_msg,
    task_done_msg,
    task_started_msg,
    tool_result_msg,
)

from harness_run import AgentSpec
from harness_run.harness.claude_code import ClaudeCodeHarness, _TaskTracker
from harness_run.harness.context import RunContext
from harness_run.harness.translate import EventTranslator


def _install_fake_proxy(monkeypatch, *, cost=None, events=(), idle=True, blocked=False):
    """Replace the localhost proxy with a scriptable stand-in; no socket is bound.

    Returns the list of constructor arguments, one entry per proxy created.
    """
    from harness_run.harness import _openrouter_proxy

    created = []

    class FakeProxy:
        def __init__(self, api_key, max_budget_usd=None, expected_model=None, **kwargs):
            created.append(
                {
                    "api_key": api_key,
                    "max_budget_usd": max_budget_usd,
                    "expected_model": expected_model,
                    **kwargs,
                }
            )
            self.base_url = "http://127.0.0.1:1"
            self.client_token = "local-token"
            self.exact_cost_usd = cost
            self.budget_blocked = blocked
            self._pending = list(events)

        def start(self):
            return self

        def close(self):
            return None

        def wait_until_idle(self, _timeout=5.0):
            return idle

        def drain_events(self):
            pending, self._pending = self._pending, []
            return pending

    monkeypatch.setattr(_openrouter_proxy, "OpenRouterProxy", FakeProxy)
    return created


def _events_of(script, tmp_path, monkeypatch, spec=None, secrets=None):
    import claude_agent_sdk

    client_cls = make_sdk_client(script)
    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", client_cls)
    spec = spec or AgentSpec(harness="claude-code", name="a", model="m")
    if (spec.model or "").startswith("openrouter/"):
        from harness_run.harness import _openrouter_proxy

        if _openrouter_proxy.OpenRouterProxy.__module__.startswith("harness_run"):
            # Every OpenRouter turn starts a proxy; keep the offline suite off sockets.
            _install_fake_proxy(monkeypatch)
    ctx = RunContext(
        spec=spec,
        prompt="go",
        job_dir=tmp_path / "job",
        session_id="sid",
        secrets=secrets or {},
    )

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


def test_tracker_clears_terminal_notification_after_boundary_and_assistant_output():
    tr = _TaskTracker()
    translate = EventTranslator().translate

    def feed(msg):
        for ev in translate(msg):
            tr.observe(ev)

    feed(task_started_msg("t1"))
    feed(task_done_msg("t1"))
    assert tr.waiting() == "undelivered"

    # The common mid-invocation ordering (verified live): the completion arrives while
    # a foreground tool runs, its result flows back (the CLI assembles the next model
    # call there, injecting the notification), and the model's output from that call
    # proves delivery.
    feed(tool_result_msg())
    assert tr.waiting() == "undelivered"  # staged: the next call carries it
    feed(AssistantMessage(content=[TextBlock(text="the task passed")], model="m"))
    assert tr.waiting() is None


def test_tracker_keeps_notification_that_landed_during_an_in_flight_call():
    # Assistant output emitted AFTER the notification can come from a model call that
    # was already in flight when it arrived — no tool-result boundary in between. The
    # CLI re-invokes right after the result in that case (verified live, CLI 2.1.223),
    # so the notification must stay undelivered and hold the turn open.
    tr = _TaskTracker()
    translate = EventTranslator().translate

    def feed(msg):
        for ev in translate(msg):
            tr.observe(ev)

    feed(task_started_msg("t1"))
    feed(tool_result_msg())  # the backgrounding Bash call's own result
    feed(task_done_msg("t1"))
    feed(AssistantMessage(content=[TextBlock(text="a long final essay")], model="m"))
    assert tr.waiting() == "undelivered"
    feed(init_msg())  # the CLI's re-invocation delivers it
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
    def seg_model_usage(in_tok, out_tok):
        return {
            "claude-sonnet-5": {
                "inputTokens": in_tok, "outputTokens": out_tok,
                "cacheReadInputTokens": 0, "cacheCreationInputTokens": 0,
                "costUSD": 0.01, "contextWindow": 200_000,
            },
        }

    script = [
        init_msg(),
        task_started_msg("t1"),
        result_msg(  # segment boundary; its model_usage is the spend so far
            num_turns=2, cost=0.02, result="WAITING",
            usage={"input_tokens": 100, "output_tokens": 10},
            model_usage=seg_model_usage(100, 10),
        ),
        task_done_msg("t1"),
        init_msg(),  # the CLI's re-invocation
        result_msg(  # the CLI's model_usage is CUMULATIVE across segments already
            num_turns=3, cost=0.05, result="verified, all done",
            usage={"input_tokens": 40, "output_tokens": 4},  # final segment only
            model_usage=seg_model_usage(140, 14),
        ),
    ]
    events, client_cls = _events_of(script, tmp_path, monkeypatch)

    results = [e for e in events if e.kind == "result"]
    assert len(results) == 1
    assert results[0].summary == "verified, all done"
    assert results[0].raw["num_turns"] == 5  # 2 + 3, cumulative across re-invocations
    assert results[0].cost_usd == 0.05  # the CLI's total_cost_usd is already cumulative
    # usage = the FINAL segment's model_usage normalized (already cumulative — summing
    # segments would double-count); the CLI's flat dict survives as raw["cli_usage"].
    assert results[0].usage == {
        "input_tokens": 140,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "output_tokens": 14,
        "reasoning_output_tokens": None,
    }
    assert results[0].raw["cli_usage"] == {"input_tokens": 40, "output_tokens": 4}
    # The demoted first result surfaced as an awaiting_tasks status, before the final one.
    kinds = [(e.kind, (e.raw or {}).get("event")) for e in events]
    assert ("status", "awaiting_tasks") in kinds
    assert kinds.index(("status", "awaiting_tasks")) < kinds.index(("result", None))
    assert client_cls.instances[0].disconnected  # CLI torn down at turn end


def test_active_reinvocation_is_not_subject_to_notification_grace(tmp_path, monkeypatch):
    # Once init proves the CLI re-invoked the model, the notification grace is over. A
    # foreground tool/model step must not inherit that timeout. Record the timeout chosen
    # for each stream read so the state boundary is deterministic without sleeping.
    import harness_run.harness.claude_code as harness_mod

    # Stream reads go through ControlledStream.next(timeout) (the stream and the operator's
    # control channel are read together); record the timeout each read was given.
    original_next = harness_mod.ControlledStream.next
    timeouts = []

    async def recording_next(self, timeout):
        timeouts.append(timeout)
        return await original_next(self, timeout)

    monkeypatch.setattr(harness_mod.ControlledStream, "next", recording_next)

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
    # The completion lands between model calls (here: during a foreground tool), the
    # tool result boundary hands it to the next call, and the model discusses it. The
    # eventual final result must be terminal immediately — no grace wait in which a
    # stale ScheduleWakeup prompt could spawn a spurious extra segment.
    script = [
        task_started_msg("t1"),
        task_done_msg("t1"),
        tool_result_msg(),
        AssistantMessage(content=[TextBlock(text="observed completion")], model="m"),
        result_msg(num_turns=2, result="final answer"),
        "hang",  # must not be read: the result is terminal immediately
    ]
    events, _ = _events_of(script, tmp_path, monkeypatch)

    assert events[-1].kind == "result" and events[-1].summary == "final answer"
    assert not any((e.raw or {}).get("event") == "awaiting_tasks" for e in events)


def test_completion_during_final_in_flight_call_holds_turn_for_reinvocation(tmp_path, monkeypatch):
    # Verified live (CLI 2.1.223): a task completing while the FINAL model call is in
    # flight is not in that call's context — the CLI re-invokes ~40 ms after the result.
    # The in-flight call's output must not count as delivery proof (no tool-result
    # boundary), so the turn stays open and the re-invoked segment's result wins.
    script = [
        init_msg(),
        task_started_msg("t1"),
        tool_result_msg(),  # the backgrounding Bash call returns
        task_done_msg("t1"),  # completes while the final (essay) call is in flight
        AssistantMessage(content=[TextBlock(text="a long final essay")], model="m"),
        result_msg(num_turns=2, result="a long final essay"),
        init_msg(),  # the CLI's re-invocation, moments after the result
        result_msg(num_turns=1, result="the background task completed"),
    ]
    events, _ = _events_of(script, tmp_path, monkeypatch)

    results = [e for e in events if e.kind == "result"]
    assert len(results) == 1
    assert results[0].summary == "the background task completed"
    assert results[0].raw["num_turns"] == 3
    assert any((e.raw or {}).get("event") == "awaiting_tasks" for e in events)


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
    spec = AgentSpec(harness="claude-code", name="a", model="m", background_task_timeout=0.01)  # floor is 1s
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


# -- structured-output recovery from displaced segment results -------------------------
#
# The reply to a stale background-task notification can become the turn's final message,
# displacing the schema-valid deliverable the agent already emitted (eval feedback:
# agentic-scraping-eval#14 — a killed crawl.log monitor turned a successful run into
# no_deliverable). The final result carries the demoted results' JSON-bearing texts so
# build_result can fall back over them.

_SCHEMA = {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}
_DELIVERABLE = '```json\n{"url": "https://rothys.com"}\n```'


def test_final_result_carries_displaced_deliverable_and_build_result_recovers_it(
    tmp_path, monkeypatch
):
    # The exact Rothy's run-004 shape: deliverable emitted, monitor still pending →
    # demoted; the kill notification re-invokes; the ack becomes the final message.
    spec = AgentSpec(harness="claude-code", name="a", model="m", output_schema=_SCHEMA)
    script = [
        init_msg(),
        task_started_msg("t1", description="tail crawl.log"),
        result_msg(num_turns=2, result=_DELIVERABLE),  # demoted: t1 still pending
        task_done_msg("t1", status="killed"),
        init_msg(),  # the CLI's re-invocation with the stale notification
        result_msg(num_turns=1, result="Noted — the monitor was killed; nothing more to do."),
    ]
    events, _ = _events_of(script, tmp_path, monkeypatch, spec=spec)

    final = events[-1]
    assert final.kind == "result" and "Noted" in final.summary
    assert final.raw["segment_summaries"] == [_DELIVERABLE]

    from harness_run.runtime._run import build_result

    result, _ = build_result(final, "sid", spec)
    assert result.structured_output == {"url": "https://rothys.com"}
    assert result.structured_output_recovered is True
    assert result.text == final.summary  # text stays the real final message


def test_no_segment_summaries_without_output_schema(tmp_path, monkeypatch):
    script = [
        init_msg(),
        task_started_msg("t1"),
        result_msg(num_turns=2, result=_DELIVERABLE),
        task_done_msg("t1"),
        init_msg(),
        result_msg(num_turns=1, result="noted"),
    ]
    events, _ = _events_of(script, tmp_path, monkeypatch)
    assert "segment_summaries" not in events[-1].raw


def test_segment_summaries_newest_first_schema_valid_only_capped(tmp_path, monkeypatch):
    # Four demoted segments with schema-valid JSON, one prose-only, one with junk JSON:
    # only schema-valid texts are kept (junk must not fill the cap and evict a real
    # answer), capped at 3, newest first.
    spec = AgentSpec(harness="claude-code", name="a", model="m", output_schema=_SCHEMA)
    script = [init_msg()]
    texts = ["no json here", '{"quoted_api_response": true}'] + [
        '{"url": "https://example.com/%d"}' % i for i in range(4)
    ]
    for i, text in enumerate(texts):
        script += [
            task_started_msg(f"t{i}"),
            result_msg(num_turns=1, result=text),  # demoted: t<i> still pending
            task_done_msg(f"t{i}"),
            init_msg(),
        ]
    script.append(result_msg(num_turns=1, result="all monitors done"))
    events, _ = _events_of(script, tmp_path, monkeypatch, spec=spec)

    assert events[-1].raw["segment_summaries"] == [
        '{"url": "https://example.com/3"}',
        '{"url": "https://example.com/2"}',
        '{"url": "https://example.com/1"}',
    ]


def test_promoted_stash_excludes_its_own_text_from_summaries(tmp_path, monkeypatch):
    # Timeout path: the newest demoted result IS the final result; only the earlier
    # segment's text rides along as a fallback candidate.
    spec = AgentSpec(harness="claude-code", name="a", model="m", output_schema=_SCHEMA, background_task_timeout=0.01)
    first = '{"url": "https://example.com/first"}'
    second = '{"url": "https://example.com/second"}'
    script = [
        init_msg(),
        task_started_msg("t1"),
        result_msg(num_turns=1, result=first),
        task_done_msg("t1"),
        init_msg(),
        task_started_msg("t2"),
        result_msg(num_turns=1, result=second),  # demoted, then the wait times out
        "hang",
    ]
    events, _ = _events_of(script, tmp_path, monkeypatch, spec=spec)

    final = events[-1]
    assert final.kind == "result" and final.summary == second
    assert final.raw["segment_summaries"] == [first]


def test_openrouter_exact_budget_interrupts_before_another_request(tmp_path, monkeypatch):
    """The proxy's running total is what the cap is measured against."""
    spec = AgentSpec(
        harness="claude-code",
        name="a",
        model="openrouter/deepseek/deepseek-v4-flash",
        max_budget_usd=0.01,
    )
    _install_fake_proxy(monkeypatch, cost=0.0123)
    events, client_cls = _events_of(
        [init_msg(), result_msg(result="done")],
        tmp_path,
        monkeypatch,
        spec=spec,
        secrets={"OPENROUTER_API_KEY": "k"},
    )

    assert client_cls.instances[0].interrupted is True
    assert any((e.raw or {}).get("event") == "limit_interrupt" for e in events)
    assert events[-1].raw["subtype"] == "error_budget_exceeded"
    assert events[-1].cost_usd == 0.0123


def test_openrouter_run_passes_provider_to_proxy(tmp_path, monkeypatch):
    created = _install_fake_proxy(monkeypatch, cost=0.001)
    spec = AgentSpec(
        harness="claude-code",
        name="a",
        model="openrouter/moonshotai/kimi-k3",
        openrouter_provider="moonshotai",
    )

    events, client_cls = _events_of(
        [init_msg(), result_msg(result="done")],
        tmp_path,
        monkeypatch,
        spec=spec,
        secrets={"OPENROUTER_API_KEY": "real-key"},
    )

    # The slug is resolved before the proxy starts, so the proxy sees one routing object.
    assert created and created[0]["routing"] == {
        "only": ["moonshotai"],
        "allow_fallbacks": False,
    }
    # The proxy holds the account key, the cap it enforces, and the only id it will serve.
    assert created[0]["api_key"] == "real-key"
    assert created[0]["max_budget_usd"] == spec.max_budget_usd
    assert created[0]["expected_model"] == "moonshotai/kimi-k3"
    # The CLI is pointed at the proxy and given its token instead of the key.
    env = client_cls.instances[0].options.env
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:1/api"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "local-token"
    assert "real-key" not in env.values()
    assert events[-1].kind == "result"


def test_openrouter_run_passes_routing_to_proxy(tmp_path, monkeypatch):
    created = _install_fake_proxy(monkeypatch, cost=0.001)
    routing = {"only": ["moonshotai", "fireworks"]}
    spec = AgentSpec(
        harness="claude-code",
        name="a",
        model="openrouter/moonshotai/kimi-k3",
        openrouter_routing=routing,
    )

    events, _ = _events_of(
        [init_msg(), result_msg(result="done")],
        tmp_path,
        monkeypatch,
        spec=spec,
        secrets={"OPENROUTER_API_KEY": "real-key"},
    )

    assert created and created[0]["routing"] == routing
    assert events[-1].kind == "result"


def test_openrouter_proxies_without_a_provider(tmp_path, monkeypatch):
    """An unpinned turn still goes through the proxy: that is where cost comes from."""
    from harness_run.events import AgentEvent

    proxy_event = AgentEvent(
        kind="status",
        summary="OpenRouter request: provider=Z.AI model=z-ai/glm-5.3",
        raw={"event": "openrouter_request", "http_status": 200, "cost_usd": 0.004},
    )
    created = _install_fake_proxy(monkeypatch, cost=0.004, events=[proxy_event])
    spec = AgentSpec(harness="claude-code", name="a", model="openrouter/z-ai/glm-5.3")

    events, client_cls = _events_of(
        [init_msg(), result_msg(result="done")],
        tmp_path,
        monkeypatch,
        spec=spec,
        secrets={"OPENROUTER_API_KEY": "real-key"},
    )

    assert created and created[0]["routing"] is None
    env = client_cls.instances[0].options.env
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:1/api"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "local-token"
    assert env["OPENROUTER_API_KEY"] == ""
    assert [e for e in events if (e.raw or {}).get("event") == "openrouter_request"] == [
        proxy_event
    ]
    assert events[-1].cost_usd == 0.004
    assert events[-1].raw["price_source"] == "openrouter"


def test_proxy_402_maps_to_budget_exceeded(tmp_path, monkeypatch):
    """The CLI reports its own failure; the proxy knows the cap was the cause."""
    _install_fake_proxy(monkeypatch, cost=0.02, blocked=True)
    spec = AgentSpec(harness="claude-code", name="a", model="openrouter/z-ai/glm-5.3", max_budget_usd=10.0)

    events, _ = _events_of(
        [init_msg(), result_msg(subtype="error_during_execution", is_error=True)],
        tmp_path,
        monkeypatch,
        spec=spec,
        secrets={"OPENROUTER_API_KEY": "k"},
    )

    assert events[-1].raw["subtype"] == "error_budget_exceeded"
    assert events[-1].raw["budget_enforcement"] == "proxy_402"


def test_metadata_timeout_is_reported_before_the_result(tmp_path, monkeypatch):
    _install_fake_proxy(monkeypatch, cost=0.001, idle=False)
    spec = AgentSpec(harness="claude-code", name="a", model="openrouter/z-ai/glm-5.3")

    events, _ = _events_of(
        [init_msg(), result_msg(result="done")],
        tmp_path,
        monkeypatch,
        spec=spec,
        secrets={"OPENROUTER_API_KEY": "k"},
    )

    kinds = [(e.raw or {}).get("event") for e in events]
    assert "openrouter_metadata_timeout" in kinds
    assert kinds.index("openrouter_metadata_timeout") < len(events) - 1
    assert events[-1].kind == "result"


def _result_event(summary, segment_summaries=None):
    from harness_run.events import AgentEvent

    raw = {"subtype": "success", "is_error": False, "num_turns": 1}
    if segment_summaries is not None:
        raw["segment_summaries"] = segment_summaries
    return AgentEvent(kind="result", summary=summary, raw=raw, cost_usd=0.01)


def test_build_result_terminal_parse_wins_over_segments():
    from harness_run.runtime._run import build_result

    spec = AgentSpec(harness="claude-code", name="a", model="m", output_schema=_SCHEMA)
    ev = _result_event('{"url": "https://final.example"}', ['{"url": "https://old.example"}'])
    result, _ = build_result(ev, "sid", spec)
    assert result.structured_output == {"url": "https://final.example"}
    assert result.structured_output_recovered is False


def test_build_result_skips_schema_invalid_segments():
    from harness_run.runtime._run import build_result

    spec = AgentSpec(harness="claude-code", name="a", model="m", output_schema=_SCHEMA)
    ev = _result_event(
        "wrapping up",
        ['{"url": 42}', '{"url": "https://valid.example"}'],  # newest first; newest invalid
    )
    result, _ = build_result(ev, "sid", spec)
    assert result.structured_output == {"url": "https://valid.example"}
    assert result.structured_output_recovered is True


def test_build_result_ignores_segments_without_schema():
    from harness_run.runtime._run import build_result

    spec = AgentSpec(harness="claude-code", name="a", model="m")
    ev = _result_event("done", ['{"url": "https://x.example"}'])
    result, _ = build_result(ev, "sid", spec)
    assert result.structured_output is None
    assert result.structured_output_recovered is False


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


def test_codex_config_is_ignored_with_a_warning(tmp_path, monkeypatch):
    spec = AgentSpec(name="a", model="m", harness="codex", codex_config={"web_search": "disabled"})
    events, _ = _events_of([init_msg(), result_msg()], tmp_path, monkeypatch, spec=spec)
    warning = next(e for e in events if (e.raw or {}).get("event") == "spec_warning")
    assert "codex_config" in warning.summary
    assert events[-1].kind == "result"

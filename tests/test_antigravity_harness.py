"""Offline tests for the Antigravity binding (``harness_run.harness.antigravity``).

The SDK's ``Agent`` is replaced by the scripted fake in ``antigravity_fakes``; everything
else (config objects, steps, usage records) is the real ``google-antigravity`` package.
"""

from __future__ import annotations

import asyncio
import io
import json
import tarfile

import google.antigravity as sdk
import pytest
from antigravity_fakes import CANCEL_POINT, MAIN, Usage, make_agent, step, tool_call
from google.antigravity import types as sdk_types

from harness_run import AgentSpec, McpServer, SystemPrompt
from harness_run.conformance import run_harness_conformance
from harness_run.control import ControlMessage, LocalControlChannel
from harness_run.harness import pricing, resolve_harness
from harness_run.harness.antigravity import (
    AntigravityHarness,
    AntigravityStepTranslator,
    conversation_id_for,
)
from harness_run.harness.context import RunContext
from harness_run.ports.blobstore import LocalBlobStore
from harness_run.skills import skills_subdir

KEY = {"GEMINI_API_KEY": "test-gemini-key"}


def _spec(**kw):
    kw.setdefault("model", "gemini-4-hydrogen")
    return AgentSpec(name="a", harness="antigravity", **kw)


def _ctx(tmp_path, spec, *, secrets=KEY, **kw):
    ctx = RunContext(spec=spec, prompt="go", job_dir=tmp_path / "job", session_id="sid",
                     secrets=dict(secrets), **kw)
    ctx.workspace.mkdir(parents=True, exist_ok=True)
    return ctx


def _run(tmp_path, monkeypatch, turns, spec=None, *, start=None, **ctx_kw):
    spec = spec or _spec()
    agent_cls = make_agent(turns, start)
    monkeypatch.setattr(sdk, "Agent", agent_cls)
    ctx = _ctx(tmp_path, spec, **ctx_kw)

    async def collect():
        return [e async for e in AntigravityHarness().run(spec, ctx)]

    return asyncio.run(collect()), agent_cls


def _result(events):
    return next(e for e in reversed(events) if e.kind == "result")


SIMPLE_TURN = [
    step("TEXT_RESPONSE", index=0, source="USER", content="go"),
    *tool_call(1, "run_command", {"command_line": "wc -l data.txt"},
               {"exit_code": 0, "combined_output": "3 data.txt\n"}),
    Usage(MAIN, prompt=12_000, cached=8_000, candidates=100, thoughts=50),
    step("TEXT_RESPONSE", index=2, status="ACTIVE", content="Th"),
    step("TEXT_RESPONSE", index=2, content="There are 3 lines.", thinking="count them"),
    Usage(MAIN, prompt=25_000, cached=18_000, candidates=150, thoughts=80),
]


# -- selection & registration ---------------------------------------------------


def test_resolve_harness_and_skills_layout():
    assert type(resolve_harness(_spec())).__name__ == "AntigravityHarness"
    assert skills_subdir("antigravity") == ".agents/skills"


def test_conversation_id_is_stable_hex():
    cid = conversation_id_for("sid")
    assert cid == conversation_id_for("sid") != conversation_id_for("other")
    assert len(cid) == 32 and int(cid, 16) >= 0


def test_sandbox_deploy_rejects_antigravity():
    from harness_run.runtime.sandbox.backend import deploy

    with pytest.raises(ValueError, match="local runtime only"):
        deploy(_spec(), project="p", location="us-central1")


def test_gemini_prices_resolve_from_litellm_gemini_key(monkeypatch):
    monkeypatch.setattr(pricing, "_litellm_prices", lambda: {
        "gemini/gemini-4-neon": {"input_cost_per_token": 3e-6, "output_cost_per_token": 1.5e-5,
                                 "cache_read_input_token_cost": 1.5e-7},
    })
    price = pricing.model_price("gemini-4-neon")
    assert price.source == "litellm" and price.output_per_token == 1.5e-5


def test_gemini_builtin_fallback_prices():
    for model, out in (("gemini-4-neon", 15.0), ("gemini-4-carbon", 10.0),
                       ("gemini-4-argon", 10.0), ("gemini-4-hydrogen", 7.5)):
        price = pricing.model_price(model)
        assert price.source == "builtin" and price.long_context is None
        assert price.output_per_token * 1e6 == pytest.approx(out)


# -- option building ----------------------------------------------------------------


@pytest.mark.parametrize("kw, match", [
    ({"permission_mode": "default"}, "bypassPermissions"),
    ({"allowed_tools": ["view_file"]}, "allowed_tools"),
    ({"disallowed_tools": ["run_command"]}, "allowed_tools"),
    ({"model": "openrouter/google/gemini-4-neon"}, "OpenRouter"),
])
def test_build_options_rejects_unmapped_settings(tmp_path, kw, match):
    spec = _spec(**kw)
    with pytest.raises(ValueError, match=match):
        AntigravityHarness().build_options(spec, _ctx(tmp_path, spec))


def test_build_options_rejects_hooks(tmp_path):
    spec = _spec()
    with pytest.raises(ValueError, match="hooks"):
        AntigravityHarness().build_options(spec, _ctx(tmp_path, spec, hooks={"PreToolUse": []}))


def test_build_options_needs_a_gemini_key(tmp_path, monkeypatch):
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    spec = _spec()
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        AntigravityHarness().build_options(spec, _ctx(tmp_path, spec, secrets={}))


def test_build_options_core_mapping(tmp_path):
    spec = _spec(max_turns=7, reasoning_effort="xhigh", output_schema={"type": "object"})
    ctx = _ctx(tmp_path, spec)
    (ctx.workspace / ".agents" / "skills").mkdir(parents=True)
    opts = AntigravityHarness().build_options(spec, ctx)
    kw = opts.config_kwargs
    target = kw["models"][0]
    assert target.name == "gemini-4-hydrogen"
    assert target.endpoint.api_key == "test-gemini-key"
    assert target.endpoint.options.thinking_level == sdk.ThinkingLevel.EXTRA_HIGH
    assert kw["budget_config"].max_model_calls == 7
    assert kw["budget_config"].scope == sdk_types.BudgetScope.FORWARD_LOOKING
    assert kw["workspaces"] == [str(ctx.workspace)]
    assert kw["skills_paths"] == [str(ctx.workspace / ".agents" / "skills")]
    assert kw["response_schema"] == {"type": "object"}
    assert kw["conversation_id"] == conversation_id_for("sid")
    assert kw["session_continuation_mode"] == sdk_types.SessionContinuationMode.CREATE_ONLY
    assert str(opts.state_dir).startswith(str(ctx.job_dir))
    assert not str(opts.state_dir).startswith(str(ctx.workspace))
    # The config the SDK will validate is constructible as-is.
    sdk.LocalAgentConfig(**kw)


@pytest.mark.parametrize("effort, level, warned", [
    ("low", "low", False), ("medium", "medium", False), ("minimal", "minimal", False),
    ("none", "minimal", True), ("max", "extra_high", True),
])
def test_reasoning_effort_maps_to_thinking_level(tmp_path, effort, level, warned):
    spec = _spec(reasoning_effort=effort)
    opts = AntigravityHarness().build_options(spec, _ctx(tmp_path, spec))
    endpoint = opts.config_kwargs["models"][0].endpoint
    assert endpoint.options.thinking_level.value == level
    assert bool(opts.warnings) is warned


def test_system_prompt_replace_append_and_interactive(tmp_path):
    h = AntigravityHarness()
    replaced = h._system_instructions(_spec(system_prompt="You are X."), interactive=False)
    assert isinstance(replaced, sdk_types.CustomSystemInstructions) and replaced.text == "You are X."
    appended = h._system_instructions(_spec(system_prompt=SystemPrompt(append="Be brief.")), False)
    assert isinstance(appended, sdk_types.TemplatedSystemInstructions)
    assert appended.sections[0].content == "Be brief."
    assert h._system_instructions(_spec(), interactive=False) is None
    inter = h._system_instructions(_spec(), interactive=True)
    assert "INTERACTIVE MODE" in inter.sections[0].content


def test_mcp_servers_map_to_sdk_configs(tmp_path):
    spec = _spec(mcp_servers=[
        McpServer.stdio("tools", "python", ["-m", "srv"]),
        McpServer.remote("docs", "https://example.test/mcp", headers={"X-A": "1"},
                         header_secrets={"Authorization": "DOCS_TOKEN"}),
        McpServer.github(),
    ])
    ctx = _ctx(tmp_path, spec, secrets={**KEY, "DOCS_TOKEN": "Bearer d", "GH_TOKEN": "ghp"})
    servers = AntigravityHarness().build_options(spec, ctx).config_kwargs["mcp_servers"]
    by_name = {s.name: s for s in servers}
    assert by_name["tools"].command == "python" and by_name["tools"].args == ["-m", "srv"]
    assert by_name["docs"].headers == {"X-A": "1", "Authorization": "Bearer d"}
    assert by_name["github"].headers == {"Authorization": "Bearer ghp"}
    # Header secrets and the GitHub token are the harness's, never the agent's env.
    env = AntigravityHarness().build_options(spec, ctx).config_kwargs["env"]
    assert env["DOCS_TOKEN"] == "" and env["GH_TOKEN"] == ""


def test_agent_env_blanks_model_keys_and_keeps_caller_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient")
    spec = _spec(env={"DROP_ME": None, "PLAIN": "1"})
    ctx = _ctx(tmp_path, spec, secrets={**KEY, "GOOGLE_API_KEY": "alias", "MY_TOKEN": "mine"})
    env = AntigravityHarness().build_options(spec, ctx).config_kwargs["env"]
    assert env["GEMINI_API_KEY"] == "" and env["GOOGLE_API_KEY"] == ""
    assert env["ANTHROPIC_API_KEY"] == "" and env["OPENAI_API_KEY"] == ""
    assert env["DROP_ME"] == ""
    assert env["MY_TOKEN"] == "mine" and env["PLAIN"] == "1"


def test_caller_owned_anthropic_key_is_not_blanked(tmp_path):
    spec = _spec()
    ctx = _ctx(tmp_path, spec, secrets={**KEY, "ANTHROPIC_API_KEY": "the-agents-own"})
    env = AntigravityHarness().build_options(spec, ctx).config_kwargs["env"]
    assert env["ANTHROPIC_API_KEY"] == "the-agents-own"


def test_codex_config_is_ignored_with_a_warning(tmp_path):
    spec = _spec(codex_config={"web_search": "disabled"})
    opts = AntigravityHarness().build_options(spec, _ctx(tmp_path, spec))
    assert any("codex_config" in w for w in opts.warnings)


# -- the run loop ---------------------------------------------------------------------


def test_harness_conformance(tmp_path, monkeypatch):
    spec = _spec()
    monkeypatch.setattr(sdk, "Agent", make_agent([SIMPLE_TURN]))
    ctx = _ctx(tmp_path, spec)
    assert asyncio.run(run_harness_conformance(lambda: AntigravityHarness().run(spec, ctx))) is None


def test_simple_turn_events_usage_and_cost(tmp_path, monkeypatch):
    events, agent_cls = _run(tmp_path, monkeypatch, [SIMPLE_TURN])
    kinds = [e.kind for e in events]
    assert kinds == ["status", "tool_use", "tool_result", "thinking", "message", "result"]
    assert events[0].raw["subtype"] == "init"
    assert events[0].raw["data"]["conversation_id"] == MAIN
    use, res = events[1], events[2]
    assert use.raw["tool"] == "run_command" and use.raw["input"] == {"command_line": "wc -l data.txt"}
    assert res.raw["content"] == "3 data.txt\n" and res.raw["is_error"] is False
    result = _result(events)
    assert result.summary == "There are 3 lines."
    assert result.raw["subtype"] == "success" and result.raw["num_turns"] == 2
    assert result.usage == {
        "input_tokens": 7_000,
        "cache_read_input_tokens": 18_000,
        "cache_creation_input_tokens": None,
        "output_tokens": 230,
        "reasoning_output_tokens": 80,
    }
    # gemini-4-hydrogen builtin: 1.50 / 0.075 / 7.50 per 1M.
    expected = (7_000 * 1.5 + 18_000 * 0.075 + 230 * 7.5) / 1e6
    assert result.cost_usd == pytest.approx(expected)
    assert result.raw["price_source"] == "builtin"
    assert agent_cls.instances[0].conversation.sent == ["go"]


def test_failed_command_is_an_error_result(tmp_path, monkeypatch):
    turn = [*tool_call(1, "run_command", {"command_line": "false"},
                       {"exit_code": 1, "combined_output": "boom"}),
            step("TEXT_RESPONSE", index=2, content="It failed.")]
    events, _ = _run(tmp_path, monkeypatch, [turn])
    res = next(e for e in events if e.kind == "tool_result")
    assert res.raw["is_error"] is True and res.summary == "run_command (error)"


def test_structured_output_from_finish_becomes_result_text(tmp_path, monkeypatch):
    turn = [step("TEXT_RESPONSE", index=1, content="Working on it."),
            step("FINISH", index=2, tools=[("finish", {})], structured={"line_count": 3})]
    events, _ = _run(tmp_path, monkeypatch, [turn], _spec(output_schema={"type": "object"}))
    assert json.loads(_result(events).summary) == {"line_count": 3}


def test_subagent_steps_are_tagged_and_never_final(tmp_path, monkeypatch):
    sub = "sub-trajectory"
    turn = [
        *tool_call(1, "invoke_subagent", {"task": "count"}, {"result": "ok"}),
        step("TEXT_RESPONSE", index=0, source="SYSTEM", trajectory=sub, parent=MAIN,
             content="[Message] task from parent"),
        *tool_call(1, "run_command", {"command_line": "wc -l f"}, {"exit_code": 0},
                   trajectory=sub, parent=MAIN),
        Usage(sub, prompt=5_000, candidates=40),
        step("TEXT_RESPONSE", index=2, trajectory=sub, parent=MAIN, content="sub says 3"),
        Usage(MAIN, prompt=10_000, candidates=60),
        step("TEXT_RESPONSE", index=3, content="The answer is 3."),
    ]
    events, _ = _run(tmp_path, monkeypatch, [turn])
    sub_events = [e for e in events if (e.raw or {}).get("subagent")]
    assert {e.kind for e in sub_events} == {"status", "tool_use", "tool_result", "message"}
    assert all(e.raw["trajectory_id"] == sub for e in sub_events)
    result = _result(events)
    assert result.summary == "The answer is 3."
    assert result.raw["subagent_usage"][sub]["input_tokens"] == 5_000
    assert result.usage["input_tokens"] == 15_000  # subagent tokens are in the turn's usage
    assert result.raw["num_turns"] == 1  # main-agent calls only


def test_resumed_session_reports_only_this_turns_usage(tmp_path, monkeypatch):
    start = {MAIN: Usage(MAIN, prompt=50_000, cached=10_000, candidates=500)}
    turn = [Usage(MAIN, prompt=62_000, cached=20_000, candidates=600),
            step("TEXT_RESPONSE", index=9, content="again")]
    events, _ = _run(tmp_path, monkeypatch, [turn], start=start)
    result = _result(events)
    assert result.usage["input_tokens"] == 2_000
    assert result.usage["cache_read_input_tokens"] == 10_000
    assert result.usage["output_tokens"] == 100
    assert result.raw["num_turns"] == 1


def test_max_model_calls_stop_is_error_max_turns(tmp_path, monkeypatch):
    turn = [*tool_call(1, "edit_file", {"file_path": "a"}, {"diff_block": []}),
            Usage(MAIN, prompt=1_000), ("stop", "MAX_MODEL_CALLS_EXCEEDED")]
    events, _ = _run(tmp_path, monkeypatch, [turn], _spec(max_turns=1))
    result = _result(events)
    assert result.raw["subtype"] == "error_max_turns" and result.raw["is_error"] is True


def test_budget_cap_cancels_the_turn(tmp_path, monkeypatch):
    turn = [Usage(MAIN, prompt=1_000_000, candidates=10_000),  # $1.575 on hydrogen
            *tool_call(1, "run_command", {"command_line": "sleep 100"}, {"exit_code": 0})[:1],
            CANCEL_POINT,
            step("TEXT_RESPONSE", index=2, content="never seen")]
    events, agent_cls = _run(tmp_path, monkeypatch, [turn], _spec(max_budget_usd=1.0))
    assert agent_cls.instances[0].conversation.cancels == 1
    assert any((e.raw or {}).get("event") == "limit_interrupt" for e in events)
    # The cancelled call is closed with an error result, so every tool_use is paired.
    res = [e for e in events if e.kind == "tool_result"]
    assert len(res) == 1 and res[0].raw["is_error"] is True
    result = _result(events)
    assert result.raw["subtype"] == "error_budget_exceeded"
    assert result.cost_usd == pytest.approx(1.575)


def test_sdk_error_is_an_error_result(tmp_path, monkeypatch):
    turn = [sdk_types.AntigravityExecutionError("Error 404, models/gemini-4-bogus is not found")]
    events, _ = _run(tmp_path, monkeypatch, [turn], _spec(model="gemini-4-bogus"))
    result = _result(events)
    assert result.raw["subtype"] == "error" and result.raw["is_error"] is True
    assert "404" in result.summary
    assert any((e.raw or {}).get("event") == "cost_unknown" for e in events)


def test_unpriced_model_reports_unknown_cost(tmp_path, monkeypatch):
    turn = [Usage(MAIN, prompt=100), step("TEXT_RESPONSE", index=1, content="hi")]
    events, _ = _run(tmp_path, monkeypatch, [turn], _spec(model="gemini-9-future"))
    assert _result(events).cost_usd is None


# -- operator control -------------------------------------------------------------------


def _with_control(tmp_path, monkeypatch, turns, messages, spec=None):
    """Run with a control channel fed ``messages`` once the first step is out."""
    spec = spec or _spec()
    agent_cls = make_agent(turns)
    monkeypatch.setattr(sdk, "Agent", agent_cls)
    channel = LocalControlChannel()
    ctx = _ctx(tmp_path, spec, control=channel)

    async def collect():
        out = []
        async for e in AntigravityHarness().run(spec, ctx):
            out.append(e)
            if e.kind == "tool_use" and messages:
                for m in messages:
                    channel.send(m)
                messages.clear()
        return out

    return asyncio.run(collect()), agent_cls


def test_interrupt_with_message_continues_the_conversation(tmp_path, monkeypatch):
    first = [*tool_call(1, "run_command", {"command_line": "sleep 30"}, {"exit_code": 0})[:1],
             CANCEL_POINT]
    second = [Usage(MAIN, prompt=500), step("TEXT_RESPONSE", index=3, content="PIVOTED")]
    msg = ControlMessage(op="interrupt", message="pivot", message_id="m1")
    events, agent_cls = _with_control(tmp_path, monkeypatch, [first, second], [msg])
    conv = agent_cls.instances[0].conversation
    assert conv.cancels == 1 and conv.sent == ["go", "pivot"]
    user = [e for e in events if e.kind == "user"]
    assert user and user[0].raw["message_id"] == "m1"
    assert _result(events).summary == "PIVOTED"
    assert _result(events).raw["subtype"] == "success"


def test_stop_ends_the_turn_as_interrupted(tmp_path, monkeypatch):
    first = [*tool_call(1, "run_command", {"command_line": "sleep 30"}, {"exit_code": 0})[:1],
             CANCEL_POINT]
    events, agent_cls = _with_control(tmp_path, monkeypatch, [first],
                                      [ControlMessage(op="stop", message=None, message_id="s")])
    result = _result(events)
    assert result.raw["subtype"] == "interrupted" and result.raw["is_error"] is False
    assert agent_cls.instances[0].conversation.sent == ["go"]


def test_steer_is_delivered_when_the_turn_ends(tmp_path, monkeypatch):
    first = [*tool_call(1, "run_command", {"command_line": "ls"}, {"exit_code": 0}),
             step("TEXT_RESPONSE", index=2, content="listed")]
    second = [step("TEXT_RESPONSE", index=4, content="KESTREL")]
    msg = ControlMessage(op="steer", message="say KESTREL", message_id="st")
    events, agent_cls = _with_control(tmp_path, monkeypatch, [first, second], [msg])
    assert any((e.raw or {}).get("event") == "steer_queued" for e in events)
    assert agent_cls.instances[0].conversation.sent == ["go", "say KESTREL"]
    assert _result(events).summary == "KESTREL"


# -- checkpoint / resume -------------------------------------------------------------------


def _checkpoint_ctx(tmp_path, spec, **kw):
    from harness_run.checkpoint.session_store import BlobSessionStore

    blobs = LocalBlobStore(str(tmp_path / "blobs"))
    return _ctx(tmp_path, spec, blobs=blobs, session_store=BlobSessionStore(blobs), **kw)


def test_checkpoint_archives_and_restores_the_conversation(tmp_path, monkeypatch):
    spec = _spec(checkpoint=True)
    ctx = _checkpoint_ctx(tmp_path, spec)
    h = AntigravityHarness()
    opts = h.build_options(spec, ctx)
    db = opts.state_dir / "save" / f"{opts.conversation_id}.db"
    db.parent.mkdir(parents=True)
    db.write_bytes(b"sqlite")
    (opts.state_dir / "app" / "brain" / opts.conversation_id).mkdir(parents=True)
    h._persist_state(ctx, opts.state_dir)

    # A different worker: same session, fresh job dir.
    ctx2 = _checkpoint_ctx(tmp_path / "w2", spec, resume_sid="sid")
    ctx2.blobs = ctx.blobs
    opts2 = h.build_options(spec, ctx2)
    assert opts2.resume and opts2.conversation_id == opts.conversation_id
    assert opts2.config_kwargs["session_continuation_mode"] == (
        sdk_types.SessionContinuationMode.CREATE_OR_RESUME
    )
    assert h._restore_state(ctx2, opts2) is True
    assert (opts2.state_dir / "save" / f"{opts.conversation_id}.db").read_bytes() == b"sqlite"


def test_restore_without_a_checkpoint_starts_fresh(tmp_path):
    spec = _spec(checkpoint=True)
    ctx = _checkpoint_ctx(tmp_path, spec, resume_sid="sid")
    h = AntigravityHarness()
    assert h._restore_state(ctx, h.build_options(spec, ctx)) is False


def test_restore_rejects_an_archive_escaping_the_state_dir(tmp_path):
    spec = _spec(checkpoint=True)
    ctx = _checkpoint_ctx(tmp_path, spec, resume_sid="sid")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo("../../escaped.txt")
        data = b"x"
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    ctx.blobs.put_bytes("antigravity-conversations/sid/state.tar.gz", buf.getvalue())
    h = AntigravityHarness()
    with pytest.raises(RuntimeError, match="invalid"):
        h._restore_state(ctx, h.build_options(spec, ctx))
    assert not (tmp_path / "escaped.txt").exists()


def test_run_with_checkpoint_saves_and_reports(tmp_path, monkeypatch):
    spec = _spec(checkpoint=True)
    monkeypatch.setattr(sdk, "Agent", make_agent([SIMPLE_TURN]))
    ctx = _checkpoint_ctx(tmp_path, spec)

    async def collect():
        return [e async for e in AntigravityHarness().run(spec, ctx)]

    events = asyncio.run(collect())
    assert any((e.raw or {}).get("event") == "checkpoint_saved" for e in events)
    assert ctx.blobs.exists("antigravity-conversations/sid/state.tar.gz")
    assert events[-1].kind == "result"


def test_fresh_run_clears_a_previous_conversation_in_the_job_dir(tmp_path, monkeypatch):
    spec = _spec()
    ctx = _ctx(tmp_path, spec)
    stale = ctx.job_dir / "antigravity" / "save" / "old.db"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")
    monkeypatch.setattr(sdk, "Agent", make_agent([SIMPLE_TURN]))

    async def collect():
        return [e async for e in AntigravityHarness().run(spec, ctx)]

    asyncio.run(collect())
    assert not stale.exists()


# -- translator corner cases -------------------------------------------------------------


def test_translator_ignores_user_echo_and_repeated_updates():
    t = AntigravityStepTranslator(MAIN)
    out = []
    for s in [step("TEXT_RESPONSE", index=0, source="USER", content="go"),
              step("TEXT_RESPONSE", index=1, status="ACTIVE", content="pa"),
              step("TEXT_RESPONSE", index=1, content="partial"),
              step("TEXT_RESPONSE", index=1, content="partial")]:
        out.extend(t.translate(s))
    assert [e.kind for e in out] == ["message"]
    assert t.final_text == "partial"

"""CodexHarness: option building, event translation, run loop, caps, resume — offline.

The transport (``AsyncCodex``) is faked; notification payloads are the real generated
pydantic models (see ``codex_fakes``), so field names and enum values stay honest.
"""

from __future__ import annotations

import json

import pytest
from codex_fakes import (
    agent_message,
    codex_error,
    command_execution,
    make_async_codex,
    token_usage,
    turn_completed,
    turn_started,
)

from remote_agent_toolkit import AgentSpec, local
from remote_agent_toolkit.harness import pricing
from remote_agent_toolkit.harness.codex import CodexEventTranslator, CodexHarness
from remote_agent_toolkit.harness.context import RunContext
from remote_agent_toolkit.spec import McpServer, SystemPrompt


def _ctx(tmp_path, spec, **kw):
    kw.setdefault("secrets", {"OPENAI_API_KEY": "sk-test"})
    ctx = RunContext(spec=spec, prompt="go", job_dir=tmp_path / "job", session_id="sid", **kw)
    ctx.workspace.mkdir(parents=True, exist_ok=True)
    return ctx


async def _events_of(script, tmp_path, monkeypatch, spec=None, ctx=None):
    import openai_codex

    client_cls = make_async_codex(script)
    monkeypatch.setattr(openai_codex, "AsyncCodex", client_cls)
    spec = spec or AgentSpec(name="a", model="gpt-5.6-luna", harness="codex")
    ctx = ctx or _ctx(tmp_path, spec)
    events = [ev async for ev in CodexHarness().run(spec, ctx)]
    return events, client_cls.instances[-1]


# -- option building ----------------------------------------------------------


def test_build_options_defaults(tmp_path):
    spec = AgentSpec(name="a", model="gpt-5.6-luna", harness="codex")
    ctx = _ctx(tmp_path, spec, secrets={"OPENAI_API_KEY": "sk-1", "MY_TOKEN": "t"})
    opts = CodexHarness().build_options(spec, ctx)

    assert opts.codex_home == ctx.job_dir / "codex_home"
    assert opts.codex_home.is_dir()  # codex refuses a missing CODEX_HOME
    env = opts.codex_config.env
    assert env["CODEX_HOME"] == str(opts.codex_home)
    # OPENAI_API_KEY is harness-consumed (login), not agent env; the caller's own
    # secret is forwarded.
    assert "OPENAI_API_KEY" not in env
    assert env["MY_TOKEN"] == "t"
    assert opts.api_key == "sk-1"
    # The shell sees the caller's secrets (codex's default excludes are lifted), but
    # never the model key.
    ovr = list(opts.codex_config.config_overrides)
    assert "shell_environment_policy.ignore_default_excludes=true" in ovr
    assert any("OPENAI_API_KEY" in o for o in ovr if o.startswith("shell_environment_policy.exclude"))

    from openai_codex import ApprovalMode, Sandbox

    assert opts.thread_args["sandbox"] is Sandbox.full_access  # bypassPermissions default
    assert opts.thread_args["approval_mode"] is ApprovalMode.deny_all
    assert opts.thread_args["model"] == "gpt-5.6-luna"
    assert opts.thread_args["cwd"] == str(ctx.workspace)
    assert not opts.warnings


def test_build_options_permission_and_prompt_mapping(tmp_path):
    from openai_codex import ApprovalMode, Sandbox

    spec = AgentSpec(
        name="a", model="gpt-5.6-luna", harness="codex", permission_mode="default",
        system_prompt=SystemPrompt.inherit("Extra guidance."),
    )
    ctx = _ctx(tmp_path, spec, interactive=True)
    opts = CodexHarness().build_options(spec, ctx)
    assert opts.thread_args["sandbox"] is Sandbox.workspace_write
    assert opts.thread_args["approval_mode"] is ApprovalMode.auto_review
    dev = opts.thread_args["developer_instructions"]
    assert dev.startswith("Extra guidance.")
    assert "INTERACTIVE MODE" in dev  # suffix appended
    assert "base_instructions" not in opts.thread_args

    spec2 = AgentSpec(
        name="a", model="gpt-5.6-luna", harness="codex", system_prompt="Full replace."
    )
    opts2 = CodexHarness().build_options(spec2, _ctx(tmp_path / "b", spec2))
    assert opts2.thread_args["base_instructions"] == "Full replace."


def test_build_options_tool_lists_warn(tmp_path):
    spec = AgentSpec(
        name="a", model="gpt-5.6-luna", harness="codex", allowed_tools=("Bash",)
    )
    opts = CodexHarness().build_options(spec, _ctx(tmp_path, spec))
    assert any("allowed_tools" in w for w in opts.warnings)


def test_build_options_transcript_only_warns(tmp_path):
    # Codex keeps its conversation as a rollout blob written under checkpoint, so a
    # transcript-only spec is a no-op here — say so during the run, not only in the docs.
    spec = AgentSpec(name="a", model="gpt-5.6-luna", harness="codex", transcript=True)
    opts = CodexHarness().build_options(spec, _ctx(tmp_path, spec))
    assert any("transcript=True" in w for w in opts.warnings)

    ckpt = AgentSpec(name="a", model="gpt-5.6-luna", harness="codex", checkpoint=True,
                     transcript=True)
    assert not CodexHarness().build_options(ckpt, _ctx(tmp_path / "b", ckpt)).warnings


def test_build_options_rejects_hooks(tmp_path):
    # Fail closed: a gating hook codex cannot call must not let the turn run.
    spec = AgentSpec(name="a", model="gpt-5.6-luna", harness="codex")
    ctx = _ctx(tmp_path, spec, hooks={"PreToolUse": []})
    with pytest.raises(ValueError, match="no codex equivalent"):
        CodexHarness().build_options(spec, ctx)


def test_build_options_mcp_servers(tmp_path):
    spec = AgentSpec(
        name="a", model="gpt-5.6-luna", harness="codex",
        mcp_servers=[
            McpServer.github(),
            McpServer.remote("zyte", "https://mcp.example.com", headers={"X-K": "v"}),
            McpServer.stdio("loc", "svc", ["--fast"]),
        ],
    )
    ctx = _ctx(tmp_path, spec, secrets={"OPENAI_API_KEY": "sk", "GH_TOKEN": "gh-secret"})
    opts = CodexHarness().build_options(spec, ctx)
    ovr = "\n".join(opts.codex_config.config_overrides)
    assert 'mcp_servers.github.url="https://api.githubcopilot.com/mcp/"' in ovr
    assert "bearer_token_env_var" in ovr
    assert "gh-secret" not in ovr  # token never rides argv
    assert opts.codex_config.env["RATK_GITHUB_MCP_TOKEN"] == "gh-secret"
    assert "GH_TOKEN" not in opts.codex_config.env  # consumed, not agent env
    assert 'mcp_servers.zyte.url="https://mcp.example.com"' in ovr
    assert 'mcp_servers.zyte.http_headers={"X-K" = "v"}' in ovr
    assert 'mcp_servers.loc.command="svc"' in ovr
    assert 'mcp_servers.loc.args=["--fast"]' in ovr


def test_build_options_output_schema(tmp_path):
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    spec = AgentSpec(name="a", model="gpt-5.6-luna", harness="codex", output_schema=schema)
    opts = CodexHarness().build_options(spec, _ctx(tmp_path, spec))
    assert opts.run_args["output_schema"] == schema


def test_build_options_reasoning_effort(tmp_path):
    spec = AgentSpec(name="a", model="gpt-5.6-luna", harness="codex", reasoning_effort="high")
    opts = CodexHarness().build_options(spec, _ctx(tmp_path, spec))
    assert opts.run_args["effort"] == "high"
    assert not opts.warnings
    # The Claude-only "max" maps to codex's ceiling, with a warning.
    spec2 = AgentSpec(name="a", model="gpt-5.6-luna", harness="codex", reasoning_effort="max")
    opts2 = CodexHarness().build_options(spec2, _ctx(tmp_path / "b", spec2))
    assert opts2.run_args["effort"] == "xhigh"
    assert any("reasoning_effort" in w for w in opts2.warnings)
    # Unset sends nothing (SDK/thread default).
    spec3 = AgentSpec(name="a", model="gpt-5.6-luna", harness="codex")
    opts3 = CodexHarness().build_options(spec3, _ctx(tmp_path / "c", spec3))
    assert "effort" not in opts3.run_args


async def test_run_passes_effort_to_turn(tmp_path, monkeypatch):
    script = [turn_started(), agent_message("ok"), turn_completed("completed")]
    spec = AgentSpec(name="a", model="gpt-5.6-luna", harness="codex", reasoning_effort="high")
    _events, client = await _events_of(script, tmp_path, monkeypatch, spec=spec)
    assert client.turns[0][1]["effort"] == "high"


# -- pricing ------------------------------------------------------------------
# The autouse conftest fixture keeps the LiteLLM fetch off the network; the baked
# table is the fallback under test unless a test stubs the dataset itself.


def test_builtin_price_fallback():
    # luna: $1/M in, $0.10/M cached, $6/M out
    price = pricing.model_price("gpt-5.6-luna")
    assert price is not None and price.source == "builtin"
    cost = price.cost_usd(1_000_000, 500_000, 100_000)
    assert cost == pytest.approx(0.5 * 1.0 + 0.5 * 0.10 + 0.1 * 6.0)
    assert pricing.model_price("some-future-model") is None


def test_litellm_price_wins_over_builtin(monkeypatch):
    dataset = {
        # A model the baked table doesn't know — priced anyway.
        "openai/gpt-9-hyperion": {
            "input_cost_per_token": 2e-06,
            "cache_read_input_token_cost": 2e-07,
            "output_cost_per_token": 8e-06,
        },
        # A baked model with a (hypothetically) updated live price — litellm wins.
        "gpt-5.6-luna": {
            "input_cost_per_token": 9e-06,
            "output_cost_per_token": 9e-06,
            # no cache_read price listed → cached tokens bill as fresh input
        },
    }
    monkeypatch.setattr(pricing, "_litellm_prices", lambda: dataset)

    p = pricing.model_price("gpt-9-hyperion")  # found via the openai/ prefix
    assert p is not None and p.source == "litellm"
    assert p.cost_usd(1_000_000, 0, 0) == pytest.approx(2.0)

    p2 = pricing.model_price("gpt-5.6-luna")
    assert p2 is not None and p2.source == "litellm"
    assert p2.input_per_token == pytest.approx(9e-06)
    assert p2.cached_input_per_token == pytest.approx(9e-06)  # falls back to input price


def test_litellm_fetch_failure_is_cached_and_falls_back():
    # conftest stubbed the fetch to fail; resolution still succeeds via the baked table
    # and the failed fetch is attempted only once per process.
    assert pricing._litellm_prices() is None
    assert pricing.model_price("gpt-5.6-terra").source == "builtin"


# -- translator ---------------------------------------------------------------


def test_translator_command_execution():
    tr = CodexEventTranslator()
    (use,) = tr.translate(command_execution(started=True, command="ls -la"))
    assert use.kind == "tool_use" and "ls -la" in use.summary
    assert use.raw["tool"] == "commandExecution"
    (res,) = tr.translate(command_execution(started=False, exit_code=2, output="boom"))
    assert res.kind == "tool_result" and res.raw["is_error"] is True
    assert res.raw["content"] == "boom" and res.raw["exit_code"] == 2


def test_translator_final_message_prefers_final_phase():
    tr = CodexEventTranslator()
    list(tr.translate(agent_message("progress note", phase="commentary")))
    assert tr.final_text == "progress note"
    list(tr.translate(agent_message("THE ANSWER", phase="final_answer")))
    list(tr.translate(agent_message("post-final commentary", phase="commentary")))
    assert tr.final_text == "THE ANSWER"


def test_translator_error_notification():
    tr = CodexEventTranslator()
    (ev,) = tr.translate(codex_error("stream hiccup", will_retry=True))
    assert ev.kind == "status"
    assert ev.raw["event"] == "codex_error" and ev.raw["will_retry"] is True


# -- run loop -----------------------------------------------------------------


async def test_run_happy_path(tmp_path, monkeypatch):
    script = [
        turn_started(),
        command_execution(started=True),
        command_execution(started=False),
        token_usage(in_tok=1000, out_tok=100),
        agent_message("all done"),
        token_usage(in_tok=2000, cached=1000, out_tok=50),
        turn_completed("completed"),
    ]
    events, client = await _events_of(script, tmp_path, monkeypatch)
    assert client.login_keys == ["sk-test"]  # fresh CODEX_HOME → API-key login
    assert client.turns[0][0] == "go"
    assert client.closed

    result = events[-1]
    assert result.kind == "result" and result.raw["is_error"] is False
    assert result.summary == "all done"
    assert result.raw["num_turns"] == 2  # one per token-usage notification
    assert result.raw["thread_id"] == "thr-fake"
    assert result.usage["input_tokens"] == 3000
    assert result.usage["cached_input_tokens"] == 1000
    assert result.usage["output_tokens"] == 150
    # luna: (3000-1000)*1 + 1000*0.1 + 150*6 per MTok
    assert result.cost_usd == pytest.approx((2000 * 1.0 + 1000 * 0.1 + 150 * 6.0) / 1e6)
    kinds = [e.kind for e in events]
    assert kinds[:1] == ["status"] and "tool_use" in kinds and "tool_result" in kinds


async def test_run_skips_login_when_auth_exists(tmp_path, monkeypatch):
    spec = AgentSpec(name="a", model="gpt-5.6-luna", harness="codex")
    ctx = _ctx(tmp_path, spec)
    home = ctx.job_dir / "codex_home"
    home.mkdir(parents=True)
    (home / "auth.json").write_text("{}")
    events, client = await _events_of(
        [turn_started(), agent_message("hi"), token_usage(), turn_completed()],
        tmp_path, monkeypatch, spec=spec, ctx=ctx,
    )
    assert client.login_keys == []


async def test_run_no_credentials_raises(tmp_path, monkeypatch):
    spec = AgentSpec(name="a", model="gpt-5.6-luna", harness="codex")
    ctx = _ctx(tmp_path, spec, secrets={})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="no OpenAI credentials"):
        await _events_of([turn_completed()], tmp_path, monkeypatch, spec=spec, ctx=ctx)


async def test_run_failed_turn(tmp_path, monkeypatch):
    script = [turn_started(), token_usage(), turn_completed("failed", error="model exploded")]
    events, _ = await _events_of(script, tmp_path, monkeypatch)
    result = events[-1]
    assert result.kind == "result" and result.raw["is_error"] is True
    assert result.raw["subtype"] == "error"
    assert result.summary == "model exploded"
    assert result.raw["num_turns"] == 1  # accounting kept on error results


async def test_run_max_turns_interrupts(tmp_path, monkeypatch):
    spec = AgentSpec(name="a", model="gpt-5.6-luna", harness="codex", max_turns=2)
    script = [
        turn_started(),
        agent_message("working", phase="commentary"),
        token_usage(),
        token_usage(),  # hits the cap → interrupt
        turn_completed("interrupted"),
    ]
    events, client = await _events_of(script, tmp_path, monkeypatch, spec=spec)
    assert client.interrupted
    assert any((e.raw or {}).get("event") == "limit_interrupt" for e in events)
    result = events[-1]
    assert result.raw["subtype"] == "error_max_turns" and result.raw["is_error"] is True
    assert result.summary == "working"  # best text so far is kept
    assert result.raw["num_turns"] == 2


async def test_run_budget_interrupts(tmp_path, monkeypatch):
    spec = AgentSpec(name="a", model="gpt-5.6-luna", harness="codex", max_budget_usd=0.005)
    script = [
        turn_started(),
        token_usage(in_tok=1_000_000, out_tok=0),  # $1 on luna → over budget
        turn_completed("interrupted"),
    ]
    events, client = await _events_of(script, tmp_path, monkeypatch, spec=spec)
    assert client.interrupted
    result = events[-1]
    assert result.raw["subtype"] == "error_budget_exceeded"
    assert result.cost_usd == pytest.approx(1.0)


async def test_run_unknown_model_warns_and_skips_budget(tmp_path, monkeypatch):
    spec = AgentSpec(
        name="a", model="gpt-9-hyperion", harness="codex", max_budget_usd=0.0001
    )
    script = [turn_started(), token_usage(in_tok=10_000_000), agent_message("ok"),
              turn_completed()]
    events, client = await _events_of(script, tmp_path, monkeypatch, spec=spec)
    assert any((e.raw or {}).get("event") == "cost_unknown" for e in events)
    assert not client.interrupted  # budget can't be enforced without a price
    result = events[-1]
    assert result.raw["subtype"] == "success" and result.cost_usd is None


async def test_run_stream_without_completion_raises(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="without a turn/completed"):
        await _events_of([turn_started(), agent_message("hi")], tmp_path, monkeypatch)


# -- checkpoint / resume ------------------------------------------------------


def _fake_rollout(codex_home, thread_id="thr-fake"):
    p = codex_home / "sessions" / "2026" / "07" / "21" / f"rollout-x-{thread_id}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('{"line": 1}\n')
    return p


async def test_checkpoint_persists_thread_and_resume_restores(tmp_path, monkeypatch):
    from remote_agent_toolkit.checkpoint.session_store import BlobSessionStore
    from remote_agent_toolkit.ports.blobstore import LocalBlobStore

    blobs = LocalBlobStore(str(tmp_path / "blobs"))
    spec = AgentSpec(name="a", model="gpt-5.6-luna", harness="codex", checkpoint=True)
    ctx = _ctx(
        tmp_path, spec, blobs=blobs, session_store=BlobSessionStore(blobs)
    )

    # The fake writes a rollout as a turn side-effect (as the real CLI does).
    def write_rollout(client):
        _fake_rollout(ctx.job_dir / "codex_home")

    script = [turn_started(), write_rollout, agent_message("done"), token_usage(),
              turn_completed()]
    events, _ = await _events_of(script, tmp_path, monkeypatch, spec=spec, ctx=ctx)
    assert any((e.raw or {}).get("event") == "checkpoint_saved" for e in events)
    meta = json.loads(blobs.get_bytes("codex-threads/sid/meta.json").decode())
    assert meta["thread_id"] == "thr-fake"
    assert blobs.get_bytes("codex-threads/sid/rollout.jsonl") == b'{"line": 1}\n'

    # Resume: a NEW job dir (another worker) restores the rollout and resumes the thread.
    ctx2 = _ctx(
        tmp_path / "w2", spec, blobs=blobs, session_store=BlobSessionStore(blobs),
        resume_sid="sid",
    )
    script2 = [turn_started(), agent_message("resumed reply"), token_usage(),
               turn_completed()]
    events2, client2 = await _events_of(script2, tmp_path, monkeypatch, spec=spec, ctx=ctx2)
    assert client2.thread_resumes and client2.thread_resumes[0][0] == "thr-fake"
    assert not client2.thread_starts
    restored = ctx2.job_dir / "codex_home" / meta["relpath"]
    assert restored.read_bytes() == b'{"line": 1}\n'
    assert any((e.raw or {}).get("event") == "thread_resumed" for e in events2)


async def test_resume_without_persisted_thread_starts_fresh(tmp_path, monkeypatch):
    from remote_agent_toolkit.checkpoint.session_store import BlobSessionStore
    from remote_agent_toolkit.ports.blobstore import LocalBlobStore

    blobs = LocalBlobStore(str(tmp_path / "blobs"))
    spec = AgentSpec(name="a", model="gpt-5.6-luna", harness="codex", checkpoint=True)
    ctx = _ctx(
        tmp_path, spec, blobs=blobs, session_store=BlobSessionStore(blobs),
        resume_sid="never-seen",
    )
    script = [turn_started(), agent_message("hi"), token_usage(), turn_completed()]
    _, client = await _events_of(script, tmp_path, monkeypatch, spec=spec, ctx=ctx)
    assert client.thread_starts and not client.thread_resumes


# -- wiring -------------------------------------------------------------------


def test_spec_round_trips_harness():
    spec = AgentSpec(name="a", model="gpt-5.6-luna", harness="codex")
    assert AgentSpec.from_dict(spec.to_dict()).harness == "codex"
    assert AgentSpec.from_dict({"name": "a", "model": "m"}).harness == "claude-code"


def test_local_engine_resolves_codex_harness(tmp_path):
    spec = AgentSpec(name="a", model="gpt-5.6-luna", harness="codex")
    engine = local.deploy(spec, workdir=str(tmp_path))
    assert type(engine._harness).__name__ == "CodexHarness"


def test_skills_subdir_mapping():
    from remote_agent_toolkit.skills import skills_subdir

    assert skills_subdir("codex") == ".agents/skills"
    assert skills_subdir("claude-code") == ".claude/skills"
    assert skills_subdir("anything-else") == ".claude/skills"

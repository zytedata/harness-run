"""CodexHarness: option building, event translation, run loop, caps, resume — offline.

The transport (``AsyncCodex``) is faked; notification payloads are the real generated
pydantic models (see ``codex_fakes``), so field names and enum values stay honest.
"""

from __future__ import annotations

import json
import pathlib

import pytest
from codex_fakes import (
    agent_message,
    codex_error,
    collab_agent_tool_call,
    command_execution,
    make_async_codex,
    sub_agent_activity,
    token_usage,
    turn_completed,
    turn_started,
)

from harness_run import AgentSpec, local
from harness_run.events import AgentEvent
from harness_run.harness import pricing
from harness_run.harness.codex import CodexEventTranslator, CodexHarness
from harness_run.harness.context import RunContext
from harness_run.spec import McpServer, SystemPrompt

_UNSET = object()


def _ctx(tmp_path, spec, **kw):
    kw.setdefault("secrets", {"OPENAI_API_KEY": "sk-test"})
    ctx = RunContext(spec=spec, prompt="go", job_dir=tmp_path / "job", session_id="sid", **kw)
    ctx.workspace.mkdir(parents=True, exist_ok=True)
    return ctx


async def _events_of(
    script,
    tmp_path,
    monkeypatch,
    spec=None,
    ctx=None,
    proxy_cost=None,
    proxy_provider=None,
    proxy_init=None,
    proxy_blocked=False,
    proxy_idle=True,
    resolved=_UNSET,
):
    import openai_codex
    from harness_run.harness import _openrouter_proxy

    client_cls = (
        make_async_codex(script)
        if resolved is _UNSET
        else make_async_codex(script, resolved=resolved)
    )
    monkeypatch.setattr(openai_codex, "AsyncCodex", client_cls)
    spec = spec or AgentSpec(name="a", model="gpt-5.6-luna", harness="codex")
    ctx = ctx or _ctx(tmp_path, spec)
    if (spec.model or "").startswith("openrouter/"):

        class FakeProxy:
            def __init__(self, api_key, max_budget_usd=None, expected_model=None, **kwargs):
                if proxy_init is not None:
                    proxy_init.append(
                        (
                            (api_key, max_budget_usd, expected_model),
                            kwargs,
                        )
                    )
                self.base_url = "http://127.0.0.1:1"
                self.client_token = "local-test-token"
                self.exact_cost_usd = proxy_cost
                self.budget_blocked = proxy_blocked
                self._drained = False

            def start(self):
                return self

            def close(self):
                return None

            def wait_until_idle(self, _timeout=5.0):
                return proxy_idle

            def drain_events(self):
                if self._drained or proxy_provider is None:
                    return []
                self._drained = True
                return [
                    AgentEvent(
                        kind="status",
                        summary=f"OpenRouter request: provider={proxy_provider}",
                        raw={
                            "event": "openrouter_request",
                            "provider": proxy_provider,
                            "cost_usd": proxy_cost,
                        },
                    )
                ]

        monkeypatch.setattr(_openrouter_proxy, "OpenRouterProxy", FakeProxy)
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
    assert any(
        "OPENAI_API_KEY" in o for o in ovr if o.startswith("shell_environment_policy.exclude")
    )

    from openai_codex import ApprovalMode, Sandbox

    assert opts.thread_args["sandbox"] is Sandbox.full_access  # bypassPermissions default
    assert opts.thread_args["approval_mode"] is ApprovalMode.deny_all
    assert opts.thread_args["model"] == "gpt-5.6-luna"
    assert opts.thread_args["cwd"] == str(ctx.workspace)
    assert not opts.warnings


def test_build_options_permission_and_prompt_mapping(tmp_path):
    from openai_codex import ApprovalMode, Sandbox

    spec = AgentSpec(
        name="a",
        model="gpt-5.6-luna",
        harness="codex",
        permission_mode="default",
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


def test_plan_is_read_only_with_no_approvals(tmp_path):
    from openai_codex import ApprovalMode, Sandbox

    spec = AgentSpec(name="a", model="gpt-5.6-luna", harness="codex", permission_mode="plan")
    opts = CodexHarness().build_options(spec, _ctx(tmp_path, spec))
    assert opts.thread_args["sandbox"] is Sandbox.read_only
    assert opts.thread_args["approval_mode"] is ApprovalMode.deny_all

    spec2 = AgentSpec(
        name="a", model="gpt-5.6-luna", harness="codex", system_prompt="Full replace."
    )
    opts2 = CodexHarness().build_options(spec2, _ctx(tmp_path / "b", spec2))
    assert opts2.thread_args["base_instructions"] == "Full replace."


def test_build_options_tool_lists_are_rejected(tmp_path):
    spec = AgentSpec(name="a", model="gpt-5.6-luna", harness="codex", allowed_tools=("Bash",))
    with pytest.raises(ValueError, match="allowed_tools"):
        CodexHarness().build_options(spec, _ctx(tmp_path, spec))


def test_build_options_transcript_only_warns(tmp_path):
    # Codex keeps its conversation as a rollout blob written under checkpoint, so a
    # transcript-only spec is a no-op here — say so during the run, not only in the docs.
    spec = AgentSpec(name="a", model="gpt-5.6-luna", harness="codex", transcript=True)
    opts = CodexHarness().build_options(spec, _ctx(tmp_path, spec))
    assert any("transcript=True" in w for w in opts.warnings)

    ckpt = AgentSpec(
        name="a", model="gpt-5.6-luna", harness="codex", checkpoint=True, transcript=True
    )
    assert not CodexHarness().build_options(ckpt, _ctx(tmp_path / "b", ckpt)).warnings


def test_build_options_rejects_hooks(tmp_path):
    # Fail closed: a gating hook codex cannot call must not let the turn run.
    spec = AgentSpec(name="a", model="gpt-5.6-luna", harness="codex")
    ctx = _ctx(tmp_path, spec, hooks={"PreToolUse": []})
    with pytest.raises(ValueError, match="no codex equivalent"):
        CodexHarness().build_options(spec, ctx)


def test_build_options_mcp_servers(tmp_path):
    spec = AgentSpec(
        name="a",
        model="gpt-5.6-luna",
        harness="codex",
        mcp_servers=[
            McpServer.github(),
            McpServer.remote("remote", "https://mcp.example.com", headers={"X-K": "v"}),
            McpServer.stdio("loc", "svc", ["--fast"]),
        ],
    )
    ctx = _ctx(tmp_path, spec, secrets={"OPENAI_API_KEY": "sk", "GH_TOKEN": "gh-secret"})
    opts = CodexHarness().build_options(spec, ctx)
    ovr = "\n".join(opts.codex_config.config_overrides)
    assert 'mcp_servers.github.url="https://api.githubcopilot.com/mcp/"' in ovr
    assert "bearer_token_env_var" in ovr
    assert "gh-secret" not in ovr  # token never rides argv
    assert opts.codex_config.env["HARNESS_RUN_GITHUB_MCP_TOKEN"] == "gh-secret"
    assert "GH_TOKEN" not in opts.codex_config.env  # consumed, not agent env
    assert 'mcp_servers.remote.url="https://mcp.example.com"' in ovr
    assert 'mcp_servers.remote.http_headers={"X-K" = "v"}' in ovr
    assert 'mcp_servers.loc.command="svc"' in ovr
    assert 'mcp_servers.loc.args=["--fast"]' in ovr


def test_build_options_codex_config_is_appended_last(tmp_path):
    spec = AgentSpec(
        name="a",
        model="openrouter/moonshotai/kimi-k3",
        harness="codex",
        mcp_servers=[McpServer.stdio("loc", "svc")],
        codex_config={
            "sandbox_workspace_write.network_access": True,
            "sandbox_workspace_write.writable_roots": ["/home/me/.cache/uv"],
            "web_search": "disabled",
            "model_context_window": 4096,
        },
    )
    ctx = _ctx(tmp_path, spec, secrets={"OPENROUTER_API_KEY": "sk-or"})
    ovr = list(CodexHarness().build_options(spec, ctx).codex_config.config_overrides)
    assert ovr[-4:] == [
        "sandbox_workspace_write.network_access=true",
        'sandbox_workspace_write.writable_roots=["/home/me/.cache/uv"]',
        'web_search="disabled"',
        "model_context_window=4096",
    ]
    # The OpenRouter routing sets web_search too; the caller's copy comes last, so it wins.
    assert ovr.count('web_search="disabled"') == 2
    assert 'mcp_servers.loc.command="svc"' in ovr[:-4]


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
    # Normalized (disjoint) buckets: the wire's inclusive input minus its cached part.
    # No call reported cache_write, so cache_creation stays None (not a fake 0).
    assert result.usage == {
        "input_tokens": 2000,
        "cache_read_input_tokens": 1000,
        "cache_creation_input_tokens": None,
        "output_tokens": 150,
        "reasoning_output_tokens": 0,
    }
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
        tmp_path,
        monkeypatch,
        spec=spec,
        ctx=ctx,
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
    spec = AgentSpec(name="a", model="gpt-9-hyperion", harness="codex", max_budget_usd=0.0001)
    script = [turn_started(), token_usage(in_tok=10_000_000), agent_message("ok"), turn_completed()]
    events, client = await _events_of(script, tmp_path, monkeypatch, spec=spec)
    unknown = [e for e in events if (e.raw or {}).get("event") == "cost_unknown"]
    assert len(unknown) == 1  # reported once, before the turn — not again at the end
    assert "no price data" in unknown[0].summary
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
    from harness_run.checkpoint.session_store import BlobSessionStore
    from harness_run.ports.blobstore import LocalBlobStore

    blobs = LocalBlobStore(str(tmp_path / "blobs"))
    spec = AgentSpec(name="a", model="gpt-5.6-luna", harness="codex", checkpoint=True)
    ctx = _ctx(tmp_path, spec, blobs=blobs, session_store=BlobSessionStore(blobs))

    # The fake writes a rollout as a turn side-effect (as the real CLI does).
    def write_rollout(client):
        _fake_rollout(ctx.job_dir / "codex_home")

    script = [turn_started(), write_rollout, agent_message("done"), token_usage(), turn_completed()]
    events, _ = await _events_of(script, tmp_path, monkeypatch, spec=spec, ctx=ctx)
    assert any((e.raw or {}).get("event") == "checkpoint_saved" for e in events)
    meta = json.loads(blobs.get_bytes("codex-threads/sid/meta.json").decode())
    assert meta["thread_id"] == "thr-fake"
    assert blobs.get_bytes("codex-threads/sid/rollout.jsonl") == b'{"line": 1}\n'

    # Resume: a NEW job dir (another worker) restores the rollout and resumes the thread.
    ctx2 = _ctx(
        tmp_path / "w2",
        spec,
        blobs=blobs,
        session_store=BlobSessionStore(blobs),
        resume_sid="sid",
    )
    script2 = [turn_started(), agent_message("resumed reply"), token_usage(), turn_completed()]
    events2, client2 = await _events_of(script2, tmp_path, monkeypatch, spec=spec, ctx=ctx2)
    assert client2.thread_resumes and client2.thread_resumes[0][0] == "thr-fake"
    assert not client2.thread_starts
    restored = ctx2.job_dir / "codex_home" / meta["relpath"]
    assert restored.read_bytes() == b'{"line": 1}\n'
    assert any((e.raw or {}).get("event") == "thread_resumed" for e in events2)


@pytest.mark.parametrize("relpath", [
    "../escape.jsonl",
    "sessions/../../escape.jsonl",
    "/tmp/absolute-escape.jsonl",
    "notsessions/2026/07/21/rollout-x-thr.jsonl",
    "sessions/2026/07/21/rollout-x-thr.txt",
    "sessions\\..\\escape.jsonl",
    123,
])
async def test_resume_ignores_a_rollout_path_outside_codex_home_sessions(
    tmp_path, monkeypatch, relpath
):
    """meta.json sits under the run's own token prefix, so the previous turn's agent could
    have edited it: a relpath that escapes CODEX_HOME/sessions must not be written, and the
    session starts fresh instead."""
    from harness_run.checkpoint.session_store import BlobSessionStore
    from harness_run.ports.blobstore import LocalBlobStore

    blobs = LocalBlobStore(str(tmp_path / "blobs"))
    blobs.put_bytes("codex-threads/sid/meta.json",
                    json.dumps({"thread_id": "thr-evil", "relpath": relpath}).encode())
    blobs.put_bytes("codex-threads/sid/rollout.jsonl", b"OWNED\n")
    spec = AgentSpec(name="a", model="gpt-5.6-luna", harness="codex", checkpoint=True)
    ctx = _ctx(tmp_path / "w2", spec, blobs=blobs, session_store=BlobSessionStore(blobs),
               resume_sid="sid")
    before = {p for p in tmp_path.rglob("*") if p.is_file()}
    script = [turn_started(), agent_message("fresh"), token_usage(), turn_completed()]
    events, client = await _events_of(script, tmp_path, monkeypatch, spec=spec, ctx=ctx)
    assert client.thread_starts and not client.thread_resumes
    assert not any((e.raw or {}).get("event") == "thread_resumed" for e in events)
    written = {p for p in tmp_path.rglob("*") if p.is_file()} - before
    assert not any(p.read_bytes() == b"OWNED\n" for p in written), written
    assert not pathlib.Path("/tmp/absolute-escape.jsonl").exists()


async def test_resume_ignores_a_symlinked_sessions_dir_that_escapes(tmp_path, monkeypatch):
    """A symlink under CODEX_HOME/sessions pointing elsewhere must not be followed either."""
    from harness_run.checkpoint.session_store import BlobSessionStore
    from harness_run.ports.blobstore import LocalBlobStore

    blobs = LocalBlobStore(str(tmp_path / "blobs"))
    blobs.put_bytes("codex-threads/sid/meta.json", json.dumps(
        {"thread_id": "thr-evil", "relpath": "sessions/link/rollout-x-thr-evil.jsonl"}).encode())
    blobs.put_bytes("codex-threads/sid/rollout.jsonl", b"OWNED\n")
    spec = AgentSpec(name="a", model="gpt-5.6-luna", harness="codex", checkpoint=True)
    ctx = _ctx(tmp_path / "w2", spec, blobs=blobs, session_store=BlobSessionStore(blobs),
               resume_sid="sid")
    outside = tmp_path / "outside"
    outside.mkdir()
    sessions = ctx.job_dir / "codex_home" / "sessions"
    sessions.mkdir(parents=True)
    (sessions / "link").symlink_to(outside, target_is_directory=True)
    script = [turn_started(), agent_message("fresh"), token_usage(), turn_completed()]
    _events, client = await _events_of(script, tmp_path, monkeypatch, spec=spec, ctx=ctx)
    assert client.thread_starts and not client.thread_resumes
    assert list(outside.iterdir()) == []


async def test_resume_ignores_a_sessions_dir_that_is_itself_a_symlink(tmp_path, monkeypatch):
    """If CODEX_HOME/sessions itself points elsewhere, a well-formed relpath must still be
    refused: the allowed root is CODEX_HOME (resolved) + literal ``sessions``, so resolving the
    destination through the link lands outside it."""
    from harness_run.checkpoint.session_store import BlobSessionStore
    from harness_run.ports.blobstore import LocalBlobStore

    blobs = LocalBlobStore(str(tmp_path / "blobs"))
    blobs.put_bytes("codex-threads/sid/meta.json", json.dumps(
        {"thread_id": "thr-evil", "relpath": "sessions/2026/07/21/rollout-x-thr-evil.jsonl"}).encode())
    blobs.put_bytes("codex-threads/sid/rollout.jsonl", b"OWNED\n")
    spec = AgentSpec(name="a", model="gpt-5.6-luna", harness="codex", checkpoint=True)
    ctx = _ctx(tmp_path / "w2", spec, blobs=blobs, session_store=BlobSessionStore(blobs),
               resume_sid="sid")
    outside = tmp_path / "outside"
    outside.mkdir()
    codex_home = ctx.job_dir / "codex_home"
    codex_home.mkdir(parents=True)
    (codex_home / "sessions").symlink_to(outside, target_is_directory=True)
    script = [turn_started(), agent_message("fresh"), token_usage(), turn_completed()]
    _events, client = await _events_of(script, tmp_path, monkeypatch, spec=spec, ctx=ctx)
    assert client.thread_starts and not client.thread_resumes
    assert list(outside.rglob("*")) == []


async def test_resume_without_persisted_thread_starts_fresh(tmp_path, monkeypatch):
    from harness_run.checkpoint.session_store import BlobSessionStore
    from harness_run.ports.blobstore import LocalBlobStore

    blobs = LocalBlobStore(str(tmp_path / "blobs"))
    spec = AgentSpec(name="a", model="gpt-5.6-luna", harness="codex", checkpoint=True)
    ctx = _ctx(
        tmp_path,
        spec,
        blobs=blobs,
        session_store=BlobSessionStore(blobs),
        resume_sid="never-seen",
    )
    script = [turn_started(), agent_message("hi"), token_usage(), turn_completed()]
    _, client = await _events_of(script, tmp_path, monkeypatch, spec=spec, ctx=ctx)
    assert client.thread_starts and not client.thread_resumes


# -- subagent usage (recovered from rollout files) -----------------------------

# Codex reports no subagent token usage on the wire (openai/codex#14642); the harness
# reads each subagent thread's rollout file and bills the LAST token_count line's
# cumulative total_token_usage (deltas only, via the state file, on later turns).

_SUB_TID = "01a00000-0000-0000-0000-000000000001"  # rollout ids are 36-char UUIDs


def _token_count_line(total: dict) -> str:
    return json.dumps(
        {
            "timestamp": "t",
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "total_token_usage": total,
                    "last_token_usage": total,
                    "model_context_window": 258_400,
                },
                "rate_limits": None,
            },
        }
    )


def _write_rollout(codex_home, thread_id, totals, parent_thread_id=None, model=None):
    """A rollout whose token_count lines carry cumulative ``totals``, in order.

    ``parent_thread_id`` set makes it a subagent rollout (that session_meta marker is
    how the harness identifies one); ``None`` makes it a parent-thread rollout.
    ``model`` writes the turn_context line each real turn opens with.
    """
    p = codex_home / "sessions" / "2026" / "07" / "21" / f"rollout-x-{thread_id}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    meta = {"id": thread_id, "source": "vscode"}
    if parent_thread_id is not None:
        meta["parent_thread_id"] = parent_thread_id
        meta["source"] = {"subagent": {"thread_spawn": {"parent_thread_id": parent_thread_id}}}
    lines = [json.dumps({"timestamp": "t", "type": "session_meta", "payload": meta})]
    if model is not None:
        lines.append(json.dumps(
            {"timestamp": "t", "type": "turn_context", "payload": {"model": model, "effort": "high"}}
        ))
    lines += [_token_count_line(t) for t in totals]
    p.write_text("\n".join(lines) + "\n")
    return p


def _write_subagent_rollout(codex_home, thread_id, totals, model=None):
    return _write_rollout(codex_home, thread_id, totals, parent_thread_id="thr-fake", model=model)


async def test_subagent_usage_merged_into_result(tmp_path, monkeypatch):
    spec = AgentSpec(name="a", model="gpt-5.6-luna", harness="codex")
    ctx = _ctx(tmp_path, spec)

    def write_rollout(client):
        _write_subagent_rollout(
            ctx.job_dir / "codex_home",
            _SUB_TID,
            [
                {"input_tokens": 100, "cached_input_tokens": 0, "output_tokens": 10,
                 "reasoning_output_tokens": 0, "total_tokens": 110},
                # cumulative; only this LAST line may be billed
                {"input_tokens": 3000, "cached_input_tokens": 1000,
                 "cache_write_input_tokens": 500, "output_tokens": 200,
                 "reasoning_output_tokens": 20, "total_tokens": 3200},
            ],
        )

    script = [
        turn_started(),
        collab_agent_tool_call(started=True, agents={_SUB_TID: "running"}),
        sub_agent_activity(agent_thread_id=_SUB_TID),
        write_rollout,
        collab_agent_tool_call(agents={_SUB_TID: "completed"}),
        agent_message("done"),
        token_usage(in_tok=1000, out_tok=100),
        turn_completed(),
    ]
    events, _ = await _events_of(script, tmp_path, monkeypatch, spec=spec, ctx=ctx)
    result = events[-1]
    # Parent (1000 in / 100 out) + subagent rollout total, both normalized to
    # disjoint buckets (subagent fresh input: 3000 - 1000 cached - 500 written).
    assert result.usage == {
        "input_tokens": 1000 + 1500,
        "cache_read_input_tokens": 0 + 1000,
        "cache_creation_input_tokens": 500,  # parent reported none (None) — not summed as 0
        "output_tokens": 100 + 200,
        "reasoning_output_tokens": 0 + 20,
    }
    assert result.raw["subagent_usage"] == {
        _SUB_TID: {
            "input_tokens": 1500,
            "cache_read_input_tokens": 1000,
            "cache_creation_input_tokens": 500,
            "output_tokens": 200,
            "reasoning_output_tokens": 20,
            "model": None,  # no turn_context in this rollout → parent's model assumed
        }
    }
    # Subagent spend priced with the parent model's price, on the inclusive wire numbers.
    # luna per MTok: fresh 1.0 / cached 0.1 / output 6.0.
    parent = (1000 * 1.0 + 100 * 6.0) / 1e6
    sub = ((3000 - 1000) * 1.0 + 1000 * 0.1 + 200 * 6.0) / 1e6
    assert result.cost_usd == pytest.approx(parent + sub)
    # The completed collab item surfaced as a status event with the per-agent statuses.
    sub_events = [e for e in events if (e.raw or {}).get("event") == "codex_subagents"]
    assert sub_events and sub_events[-1].raw["agent_statuses"] == {_SUB_TID: "completed"}
    assert sub_events[-1].raw["tool"] == "spawnAgent"  # plain str, not the SDK enum
    # All tracked threads ended terminal: no partial-usage warning.
    for kind in ("subagent_usage_partial", "subagent_usage_missing"):
        assert not [e for e in events if (e.raw or {}).get("event") == kind]


async def test_subagent_nonterminal_status_warns_partial_usage(tmp_path, monkeypatch):
    script = [
        turn_started(),
        collab_agent_tool_call(agents={_SUB_TID: "running"}),
        agent_message("done"),
        token_usage(),
        turn_completed(),
    ]
    events, _ = await _events_of(script, tmp_path, monkeypatch)
    partial = [e for e in events if (e.raw or {}).get("event") == "subagent_usage_partial"]
    assert partial and partial[0].raw["threads"] == [_SUB_TID]


def test_collect_subagent_usage_bills_deltas_across_turns(tmp_path):
    # Rollout totals are thread-cumulative and the job dir persists across a session's
    # turns, so a second collection must bill only what the first did not.
    harness = CodexHarness()
    home = tmp_path / "codex_home"
    parent = "01a00000-0000-0000-0000-00000000feed"
    _write_rollout(  # the current parent's own rollout: excluded by filename AND meta
        home, parent,
        [{"input_tokens": 9, "cached_input_tokens": 0, "output_tokens": 9,
          "reasoning_output_tokens": 0, "total_tokens": 18}],
    )
    _write_rollout(  # an EARLIER turn's parent thread (live-caught bug: a session's
        # next turn starts a new thread id, so "everything but the current parent"
        # would bill this as a subagent) — no parent_thread_id marker, never billed
        home, "01a00000-0000-0000-0000-0000000feed0",
        [{"input_tokens": 7, "cached_input_tokens": 0, "output_tokens": 7,
          "reasoning_output_tokens": 0, "total_tokens": 14}],
    )
    _write_rollout(
        home, _SUB_TID,
        [{"input_tokens": 100, "cached_input_tokens": 40, "output_tokens": 10,
          "reasoning_output_tokens": 1, "total_tokens": 110}],
        parent_thread_id=parent,
    )
    first, found = harness._collect_subagent_usage(home, parent)
    assert set(first) == {_SUB_TID}  # only the session_meta-marked subagent bills
    assert found == {_SUB_TID}  # parent rollouts are not "found" subagents either
    assert first[_SUB_TID]["input_tokens"] == 100
    assert first[_SUB_TID]["cache_write_input_tokens"] is None

    # Same turn state, nothing new: nothing more to bill — but the rollout is still
    # FOUND (a zero-delta thread must not read as a missing rollout).
    assert harness._collect_subagent_usage(home, parent) == ({}, {_SUB_TID})

    # The subagent thread runs again (cumulative totals grow): only the delta bills.
    _write_subagent_rollout(
        home, _SUB_TID,
        [{"input_tokens": 100, "cached_input_tokens": 40, "output_tokens": 10,
          "reasoning_output_tokens": 1, "total_tokens": 110},
         {"input_tokens": 350, "cached_input_tokens": 140, "output_tokens": 35,
          "reasoning_output_tokens": 3, "total_tokens": 385}],
    )
    second, _ = harness._collect_subagent_usage(home, parent)
    assert second == {
        _SUB_TID: {
            "input_tokens": 250,
            "cached_input_tokens": 100,
            "cache_write_input_tokens": None,
            "output_tokens": 25,
            "reasoning_output_tokens": 2,
            "model": None,
        }
    }


def test_translator_tracks_subagent_threads():
    tr = CodexEventTranslator()
    list(tr.translate(collab_agent_tool_call(started=True, agents={_SUB_TID: "pendingInit"})))
    other = "01a00000-0000-0000-0000-000000000002"
    list(tr.translate(sub_agent_activity(agent_thread_id=other)))
    list(tr.translate(collab_agent_tool_call(agents={_SUB_TID: "completed"})))
    # collab statuses override; activity-only threads are tracked with status unknown.
    assert tr.subagent_threads == {_SUB_TID: "completed", other: None}


async def test_cache_write_tokens_normalize_to_cache_creation(tmp_path, monkeypatch):
    script = [
        turn_started(),
        agent_message("hi"),
        token_usage(in_tok=1000, cached=300, cache_write=200, out_tok=50),
        turn_completed(),
    ]
    events, _ = await _events_of(script, tmp_path, monkeypatch)
    assert events[-1].usage == {
        "input_tokens": 500,  # 1000 wire-inclusive - 300 cached - 200 written
        "cache_read_input_tokens": 300,
        "cache_creation_input_tokens": 200,
        "output_tokens": 50,
        "reasoning_output_tokens": 0,
    }


# -- openrouter provider ------------------------------------------------------

# Every override asserted here was validated live against OpenRouter on 2026-08-20 (all
# four blessed models completed a tool-using turn); these tests pin the wiring so a
# refactor can't silently drop a piece the provider requires.

_OR_MODEL = "openrouter/moonshotai/kimi-k3"


def _or_spec(**kw):
    return AgentSpec(name="a", model=_OR_MODEL, harness="codex", **kw)


def test_openrouter_build_options_emits_provider_config(tmp_path):
    spec = _or_spec()
    ctx = _ctx(tmp_path, spec, secrets={"OPENROUTER_API_KEY": "sk-or-1"})
    opts = CodexHarness().build_options(spec, ctx)
    ovr = "\n".join(opts.codex_config.config_overrides)

    assert opts.openrouter is True
    assert 'model_provider="openrouter"' in ovr
    assert 'model_providers.openrouter.base_url="https://openrouter.ai/api/v1"' in ovr
    assert 'model_providers.openrouter.env_key="OPENROUTER_API_KEY"' in ovr
    # Codex 0.147 dropped wire_api="chat"; responses is the only wire left.
    assert 'model_providers.openrouter.wire_api="responses"' in ovr
    # Metadata is read by the proxy, which sets the header on its own request.
    assert "X-OpenRouter-Metadata" not in ovr
    # Codex's web-search tool carries a field OpenRouter 400s on.
    assert 'web_search="disabled"' in ovr
    # The prefix is ours, not OpenRouter's: codex gets the provider-relative id.
    assert opts.thread_args["model"] == "moonshotai/kimi-k3"
    # Known model → real context window instead of codex's fallback metadata.
    assert "model_context_window=1048576" in ovr


def test_openrouter_run_can_use_local_metadata_proxy(tmp_path):
    spec = _or_spec()
    ctx = _ctx(tmp_path, spec, secrets={"OPENROUTER_API_KEY": "real-key"})
    opts = CodexHarness().build_options(
        spec,
        ctx,
        openrouter_base_url="http://127.0.0.1:4321/api/v1",
        openrouter_client_token="local-token",
    )
    overrides = "\n".join(opts.codex_config.config_overrides)

    assert 'base_url="http://127.0.0.1:4321/api/v1"' in overrides
    assert 'env_key="HARNESS_RUN_OPENROUTER_PROXY_TOKEN"' in overrides
    assert opts.codex_config.env["OPENROUTER_API_KEY"] == ""
    assert opts.codex_config.env["HARNESS_RUN_OPENROUTER_PROXY_TOKEN"] == "local-token"
    assert opts.api_key == "real-key"


def test_openrouter_key_rides_env_never_argv(tmp_path):
    spec = _or_spec()
    ctx = _ctx(tmp_path, spec, secrets={"OPENROUTER_API_KEY": "sk-or-secret"})
    opts = CodexHarness().build_options(spec, ctx)

    assert opts.api_key == "sk-or-secret"
    # env_key indirection: the provider reads the value from the app-server's env.
    assert opts.codex_config.env["OPENROUTER_API_KEY"] == "sk-or-secret"
    assert "sk-or-secret" not in "\n".join(opts.codex_config.config_overrides)
    # ...and it stays out of the agent's own shell.
    assert "allow_login_shell=false" in opts.codex_config.config_overrides
    assert (
        "OPENROUTER_API_KEY"
        in [
            o
            for o in opts.codex_config.config_overrides
            if o.startswith("shell_environment_policy.exclude")
        ][0]
    )


def _shell_excludes(opts):
    (override,) = [
        o for o in opts.codex_config.config_overrides
        if o.startswith("shell_environment_policy.exclude=")
    ]
    return json.loads(override.split("=", 1)[1])


def test_codex_excludes_unrelated_ambient_model_credentials(tmp_path):
    spec = _or_spec()
    ctx = _ctx(tmp_path, spec, secrets={"OPENROUTER_API_KEY": "k"})
    excludes = _shell_excludes(CodexHarness().build_options(spec, ctx))
    assert {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"} <= set(excludes)

    ctx = _ctx(tmp_path, spec, secrets={"OPENROUTER_API_KEY": "k",
                                      "ANTHROPIC_API_KEY": "agent-owned"})
    opts = CodexHarness().build_options(spec, ctx)
    assert opts.codex_config.env["ANTHROPIC_API_KEY"] == "agent-owned"
    assert "ANTHROPIC_API_KEY" not in _shell_excludes(opts)


def test_spec_env_none_excludes_the_variable_from_the_shell(tmp_path):
    spec = _or_spec(env={"ZYTE_API_KEY": None})
    ctx = _ctx(tmp_path, spec, secrets={"OPENROUTER_API_KEY": "k", "ZYTE_API_KEY": "s"})
    opts = CodexHarness().build_options(spec, ctx)
    assert "ZYTE_API_KEY" not in opts.codex_config.env
    assert "ZYTE_API_KEY" in _shell_excludes(opts)


def test_openrouter_defaults_reasoning_effort(tmp_path):
    """OpenRouter's Responses endpoint refuses a turn with reasoning disabled."""
    spec = _or_spec()
    opts = CodexHarness().build_options(
        spec, _ctx(tmp_path, spec, secrets={"OPENROUTER_API_KEY": "k"})
    )
    assert opts.run_args["effort"] == "low"  # spec left it unset

    spec = _or_spec(reasoning_effort="none")
    opts = CodexHarness().build_options(
        spec, _ctx(tmp_path, spec, secrets={"OPENROUTER_API_KEY": "k"})
    )
    assert opts.run_args["effort"] == "low"
    assert any("mandatory" in w for w in opts.warnings)

    spec = _or_spec(reasoning_effort="high")
    opts = CodexHarness().build_options(
        spec, _ctx(tmp_path, spec, secrets={"OPENROUTER_API_KEY": "k"})
    )
    assert opts.run_args["effort"] == "high"  # a real level passes through


def test_openai_model_gets_no_openrouter_config(tmp_path):
    spec = AgentSpec(name="a", model="gpt-5.6-luna", harness="codex")
    opts = CodexHarness().build_options(spec, _ctx(tmp_path, spec))
    ovr = "\n".join(opts.codex_config.config_overrides)
    assert opts.openrouter is False
    assert "model_provider" not in ovr and "web_search" not in ovr
    assert opts.thread_args["model"] == "gpt-5.6-luna"
    assert "effort" not in opts.run_args  # unchanged for the OpenAI path


def test_unknown_openrouter_model_skips_context_window(tmp_path):
    spec = AgentSpec(name="a", model="openrouter/some/other-model", harness="codex")
    opts = CodexHarness().build_options(
        spec, _ctx(tmp_path, spec, secrets={"OPENROUTER_API_KEY": "k"})
    )
    ovr = "\n".join(opts.codex_config.config_overrides)
    assert 'model_provider="openrouter"' in ovr  # still routed
    assert "model_context_window" not in ovr  # codex's fallback metadata applies


def test_openrouter_output_schema_is_also_asked_for_in_words(tmp_path):
    """OpenRouter accepts the json_schema format but doesn't enforce it (GLM-5.3 ignored it)."""
    schema = {"type": "object", "properties": {"answer": {"type": "integer"}}}
    spec = _or_spec(output_schema=schema)
    opts = CodexHarness().build_options(
        spec, _ctx(tmp_path, spec, secrets={"OPENROUTER_API_KEY": "k"})
    )

    assert opts.run_args["output_schema"] == schema  # still sent, in case it starts enforcing
    dev = opts.thread_args["developer_instructions"]
    assert "FINAL MESSAGE FORMAT" in dev
    assert '"answer"' in dev  # the schema itself is in the instruction


def test_openrouter_schema_instruction_appends_to_existing_prompt(tmp_path):
    """It must not clobber SystemPrompt.append or the interactive suffix."""
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    spec = _or_spec(system_prompt=SystemPrompt.inherit(append="HOUSE RULE."), output_schema=schema)
    ctx = _ctx(tmp_path, spec, secrets={"OPENROUTER_API_KEY": "k"}, interactive=True)
    dev = CodexHarness().build_options(spec, ctx).thread_args["developer_instructions"]

    assert dev.startswith("HOUSE RULE.")
    assert "INTERACTIVE MODE" in dev
    assert dev.index("HOUSE RULE.") < dev.index("FINAL MESSAGE FORMAT")


def test_openai_output_schema_is_not_steered(tmp_path):
    """The OpenAI path enforces the format server-side; no prompt text is added."""
    spec = AgentSpec(
        name="a",
        model="gpt-5.6-luna",
        harness="codex",
        output_schema={"type": "object", "properties": {"a": {"type": "string"}}},
    )
    opts = CodexHarness().build_options(spec, _ctx(tmp_path, spec))
    assert "developer_instructions" not in opts.thread_args


def test_openrouter_and_mcp_overrides_coexist(tmp_path):
    """Provider config and MCP config share one override list — neither may drop the other."""
    spec = _or_spec(mcp_servers=(McpServer.remote("docs", "https://example.test/mcp"),))
    ctx = _ctx(tmp_path, spec, secrets={"OPENROUTER_API_KEY": "k"})
    ovr = "\n".join(CodexHarness().build_options(spec, ctx).codex_config.config_overrides)

    assert 'model_provider="openrouter"' in ovr
    assert "mcp_servers.docs.url" in ovr


async def test_openrouter_run_skips_login(tmp_path, monkeypatch):
    """Auth is the provider's env_key; `codex login` would store OpenAI credentials."""
    spec = _or_spec()
    ctx = _ctx(tmp_path, spec, secrets={"OPENROUTER_API_KEY": "sk-or-1"})
    script = [turn_started(), agent_message("done"), token_usage(), turn_completed()]
    events, client = await _events_of(script, tmp_path, monkeypatch, spec=spec, ctx=ctx)

    assert client.login_keys == []
    assert events[-1].kind == "result" and events[-1].raw["is_error"] is False
    assert events[-1].raw["model"] == _OR_MODEL  # the result reports the caller's id


async def test_openrouter_run_passes_provider_to_proxy(tmp_path, monkeypatch):
    seen = []
    spec = _or_spec(openrouter_provider="moonshotai")
    ctx = _ctx(tmp_path, spec, secrets={"OPENROUTER_API_KEY": "sk-or-1"})
    script = [turn_started(), agent_message("done"), token_usage(), turn_completed()]

    await _events_of(
        script,
        tmp_path,
        monkeypatch,
        spec=spec,
        ctx=ctx,
        proxy_init=seen,
    )

    # The slug is resolved before the proxy starts, so the proxy sees one routing object.
    assert seen and seen[0][1]["routing"] == {
        "only": ["moonshotai"],
        "allow_fallbacks": False,
    }
    # The proxy holds the account key, the spec's cap, and the id it will serve — the
    # prefix already stripped, because that is what the CLI puts in the request body.
    assert seen[0][0] == ("sk-or-1", spec.max_budget_usd, "moonshotai/kimi-k3")


async def test_openrouter_run_passes_routing_to_proxy(tmp_path, monkeypatch):
    seen = []
    routing = {"order": ["moonshotai", "fireworks"], "allow_fallbacks": True}
    spec = _or_spec(openrouter_routing=routing)
    ctx = _ctx(tmp_path, spec, secrets={"OPENROUTER_API_KEY": "sk-or-1"})
    script = [turn_started(), agent_message("done"), token_usage(), turn_completed()]

    await _events_of(
        script,
        tmp_path,
        monkeypatch,
        spec=spec,
        ctx=ctx,
        proxy_init=seen,
    )

    assert seen and seen[0][1]["routing"] == routing


async def test_openrouter_run_without_key_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    spec = _or_spec()
    ctx = _ctx(tmp_path, spec, secrets={})
    with pytest.raises(RuntimeError, match="no OpenRouter credentials"):
        await _events_of([turn_completed()], tmp_path, monkeypatch, spec=spec, ctx=ctx)


async def test_openrouter_budget_is_enforceable(tmp_path, monkeypatch):
    """OpenRouter's billed cost drives the budget cap."""
    spec = _or_spec(max_budget_usd=0.01)
    ctx = _ctx(tmp_path, spec, secrets={"OPENROUTER_API_KEY": "k"})
    script = [
        turn_started(),
        agent_message("partial"),
        token_usage(in_tok=10_000, out_tok=10_000),
        turn_completed("completed"),
    ]
    events, client = await _events_of(
        script,
        tmp_path,
        monkeypatch,
        spec=spec,
        ctx=ctx,
        proxy_cost=0.02,
    )
    assert client.interrupted
    assert any(e.raw and e.raw.get("event") == "limit_interrupt" for e in events)
    assert events[-1].raw["subtype"] == "error_budget_exceeded"
    assert not any(e.raw and e.raw.get("event") == "cost_unknown" for e in events)


async def test_openrouter_exact_cost_and_provider_come_from_proxy(tmp_path, monkeypatch):
    spec = _or_spec(max_budget_usd=0.01)
    ctx = _ctx(tmp_path, spec, secrets={"OPENROUTER_API_KEY": "k"})
    script = [
        turn_started(),
        agent_message("done"),
        token_usage(in_tok=10, out_tok=1),
        turn_completed(),
    ]
    events, _ = await _events_of(
        script,
        tmp_path,
        monkeypatch,
        spec=spec,
        ctx=ctx,
        proxy_cost=0.0123,
        proxy_provider="Moonshot AI",
    )

    request = next(e for e in events if (e.raw or {}).get("event") == "openrouter_request")
    result = next(e for e in events if e.kind == "result")
    assert request.raw["provider"] == "Moonshot AI"
    assert result.cost_usd == 0.0123
    assert result.raw["price_source"] == "openrouter"
    assert result.raw["subtype"] == "error_budget_exceeded"


async def test_proxy_402_is_reported_as_a_budget_failure(tmp_path, monkeypatch):
    """The proxy refused a request after the cap was spent; the run stopped for budget.

    Whatever the CLI made of that 402, the reason is the budget, so the result says so
    even though the harness's own mid-turn check never tripped.
    """
    spec = _or_spec(max_budget_usd=0.01)
    ctx = _ctx(tmp_path, spec, secrets={"OPENROUTER_API_KEY": "k"})
    script = [turn_started(), agent_message("done"), token_usage(), turn_completed()]

    events, _ = await _events_of(
        script,
        tmp_path,
        monkeypatch,
        spec=spec,
        ctx=ctx,
        proxy_cost=0.004,  # under the cap: only the 402 says the budget stopped this
        proxy_blocked=True,
    )

    result = next(e for e in events if e.kind == "result")
    assert result.raw["subtype"] == "error_budget_exceeded"
    assert result.raw["is_error"] is True
    assert result.cost_usd == 0.004  # the charges already recorded still count


async def test_metadata_timeout_is_reported(tmp_path, monkeypatch):
    """The charge may still be in flight when the turn ends; say so rather than guess."""
    spec = _or_spec()
    ctx = _ctx(tmp_path, spec, secrets={"OPENROUTER_API_KEY": "k"})
    script = [turn_started(), agent_message("done"), token_usage(), turn_completed()]

    events, _ = await _events_of(
        script,
        tmp_path,
        monkeypatch,
        spec=spec,
        ctx=ctx,
        proxy_cost=0.001,
        proxy_idle=False,
    )

    timeout = next(
        e for e in events if (e.raw or {}).get("event") == "openrouter_metadata_timeout"
    )
    assert "timed out" in timeout.summary
    assert events[-1].kind == "result"  # the turn still finishes


def test_openrouter_models_are_not_priced_by_the_table():
    """The price table has no OpenRouter entries: those turns report OpenRouter's charge.

    An estimate cannot match it — OpenRouter routes one id to providers whose prices
    differ by up to 2.5x, so a per-model number only fits whoever served the call.
    """
    assert not [key for key in pricing._BUILTIN_PER_MTOK if key.startswith("openrouter/")]


async def test_openrouter_turn_reports_no_cost_when_openrouter_reports_none(
    tmp_path, monkeypatch
):
    """Even with a LiteLLM entry for the id, an unreported charge means no cost.

    LiteLLM does carry some ``openrouter/`` keys, so this asserts the binding never
    reaches for one, and says so in an event instead of inventing a number.
    """
    monkeypatch.setattr(
        pricing,
        "_litellm_prices",
        lambda: {
            "openrouter/moonshotai/kimi-k3": {
                "input_cost_per_token": 9e-6,
                "output_cost_per_token": 9e-5,
                "cache_read_input_token_cost": 9e-7,
            }
        },
    )
    spec = _or_spec()
    ctx = _ctx(tmp_path, spec, secrets={"OPENROUTER_API_KEY": "k"})
    script = [
        turn_started(),
        agent_message("done"),
        token_usage(in_tok=10_000, out_tok=10_000),
        turn_completed(),
    ]
    events, _ = await _events_of(
        script, tmp_path, monkeypatch, spec=spec, ctx=ctx, proxy_cost=None
    )
    result = next(e for e in events if e.kind == "result")

    assert result.cost_usd is None
    assert result.raw["price_source"] is None
    unknown = [e for e in events if (e.raw or {}).get("event") == "cost_unknown"]
    assert len(unknown) == 1
    assert "did not report a charge" in unknown[0].summary


def test_openrouter_key_is_harness_consumed(tmp_path):
    """The key reaches the provider, not the agent's shell.

    Two mechanisms, both needed: ``runtime_env`` never forwards a harness-consumed
    secret (so an unused one can't leak either), while the key this turn DOES use is put
    back deliberately for ``env_key`` — and kept out of the agent's shell by
    ``shell_environment_policy.exclude``.
    """
    from harness_run.harness._shared import harness_consumed_secret_names, runtime_env

    spec = _or_spec()
    assert harness_consumed_secret_names(spec) == {"OPENAI_API_KEY", "OPENROUTER_API_KEY"}
    ctx = _ctx(tmp_path, spec, secrets={"OPENROUTER_API_KEY": "k", "MY_TOKEN": "visible"})
    assert "OPENROUTER_API_KEY" not in runtime_env(spec, ctx)

    opts = CodexHarness().build_options(spec, ctx)
    assert opts.codex_config.env["OPENROUTER_API_KEY"] == "k"  # for env_key
    assert opts.codex_config.env["MY_TOKEN"] == "visible"  # caller's own secret: agent's
    excludes = [
        o
        for o in opts.codex_config.config_overrides
        if o.startswith("shell_environment_policy.exclude")
    ][0]
    assert "OPENROUTER_API_KEY" in excludes


def test_unused_model_auth_key_never_reaches_the_agent(tmp_path):
    """An OpenAI key passed on an OpenRouter turn (or vice versa) is still not the agent's."""
    spec = _or_spec()
    ctx = _ctx(tmp_path, spec, secrets={"OPENROUTER_API_KEY": "k", "OPENAI_API_KEY": "sk-unused"})
    env = CodexHarness().build_options(spec, ctx).codex_config.env
    assert "OPENAI_API_KEY" not in env


def test_openrouter_provider_keeps_the_known_context_window(tmp_path):
    spec = AgentSpec(
        name="a",
        model=_OR_MODEL,
        harness="codex",
        openrouter_provider="moonshotai",
    )
    opts = CodexHarness().build_options(
        spec, _ctx(tmp_path, spec, secrets={"OPENROUTER_API_KEY": "k"})
    )
    overrides = "\n".join(opts.codex_config.config_overrides)

    assert opts.thread_args["model"] == "moonshotai/kimi-k3"
    assert "model_context_window=1048576" in overrides
    assert not opts.warnings


# -- routing attribution ------------------------------------------------------

# The result event otherwise reports what we ASKED for. `thread.read()` is the app-server's
# own record, so it is the only real answer to "did the override take effect" — which
# matters because on the Responses wire OpenRouter never names the upstream provider.


def _routing_of(events):
    return next((e.raw for e in events if (e.raw or {}).get("event") == "model_routing"), None)


async def test_routing_event_reports_the_resolved_provider(tmp_path, monkeypatch):
    spec = _or_spec()
    ctx = _ctx(tmp_path, spec, secrets={"OPENROUTER_API_KEY": "k"})
    events, _ = await _events_of(
        [turn_started(), agent_message("hi"), token_usage(), turn_completed()],
        tmp_path,
        monkeypatch,
        spec=spec,
        ctx=ctx,
        resolved={"model_provider": "openrouter", "model": "moonshotai/kimi-k3"},
    )

    routing = _routing_of(events)
    assert routing is not None
    assert routing["resolved_model_provider"] == "openrouter"
    assert routing["resolved_model"] == "moonshotai/kimi-k3"
    assert routing["asked_provider"] == "openrouter"
    assert routing["matches_request"] is True


async def test_routing_event_flags_a_mismatch(tmp_path, monkeypatch):
    """The failure this exists to catch: config ignored, turn silently served elsewhere."""
    spec = _or_spec()
    ctx = _ctx(tmp_path, spec, secrets={"OPENROUTER_API_KEY": "k"})
    events, _ = await _events_of(
        [turn_started(), agent_message("hi"), token_usage(), turn_completed()],
        tmp_path,
        monkeypatch,
        spec=spec,
        ctx=ctx,
        resolved={"model_provider": "openai", "model": "gpt-5.6-luna"},
    )

    routing = _routing_of(events)
    assert routing["resolved_model_provider"] == "openai"
    assert routing["asked_provider"] == "openrouter"
    assert routing["matches_request"] is False


async def test_openai_path_reports_its_own_routing(tmp_path, monkeypatch):
    events, _ = await _events_of(
        [turn_started(), agent_message("hi"), token_usage(), turn_completed()],
        tmp_path,
        monkeypatch,
    )
    routing = _routing_of(events)
    assert routing["asked_provider"] == "openai"
    assert routing["matches_request"] is True


async def test_attribution_failure_never_fails_the_turn(tmp_path, monkeypatch):
    """A server that won't answer `read()` costs us the event, not the run."""
    spec = _or_spec()
    ctx = _ctx(tmp_path, spec, secrets={"OPENROUTER_API_KEY": "k"})
    events, _ = await _events_of(
        [turn_started(), agent_message("done"), token_usage(), turn_completed()],
        tmp_path,
        monkeypatch,
        spec=spec,
        ctx=ctx,
        resolved=None,  # read() raises
    )

    assert _routing_of(events) is None
    assert events[-1].kind == "result" and events[-1].raw["is_error"] is False


# -- wiring -------------------------------------------------------------------


def test_spec_round_trips_harness():
    spec = AgentSpec(name="a", model="gpt-5.6-luna", harness="codex")
    assert AgentSpec.from_dict(spec.to_dict()).harness == "codex"


def test_local_engine_resolves_codex_harness(tmp_path):
    spec = AgentSpec(name="a", model="gpt-5.6-luna", harness="codex")
    engine = local.deploy(spec, workdir=str(tmp_path))
    assert type(engine._harness).__name__ == "CodexHarness"


def test_skills_subdir_mapping():
    from harness_run.skills import skills_subdir

    assert skills_subdir("codex") == ".agents/skills"
    assert skills_subdir("claude-code") == ".claude/skills"
    with pytest.raises(ValueError, match="no skills layout known"):
        skills_subdir("anything-else")


async def test_subagent_spend_crossing_the_budget_is_not_a_success(tmp_path, monkeypatch):
    # Subagent usage is only known at turn end, after the mid-turn budget checks; a turn
    # it pushes over max_budget_usd must end error_budget_exceeded, not success.
    spec = AgentSpec(name="a", model="gpt-5.6-luna", harness="codex", max_budget_usd=0.005)
    ctx = _ctx(tmp_path, spec)

    def write_rollout(client):
        _write_subagent_rollout(
            ctx.job_dir / "codex_home",
            _SUB_TID,
            [{"input_tokens": 1_000_000, "cached_input_tokens": 0, "output_tokens": 0,
              "reasoning_output_tokens": 0, "total_tokens": 1_000_000}],  # $1 on luna
        )

    script = [
        turn_started(),
        collab_agent_tool_call(started=True, agents={_SUB_TID: "running"}),
        write_rollout,
        collab_agent_tool_call(agents={_SUB_TID: "completed"}),
        agent_message("done"),
        token_usage(in_tok=1000, out_tok=100),  # parent alone is well under the cap
        turn_completed(),
    ]
    events, client = await _events_of(script, tmp_path, monkeypatch, spec=spec, ctx=ctx)
    result = events[-1]
    assert not client.interrupted  # nothing to interrupt: the turn had already ended
    assert result.raw["subtype"] == "error_budget_exceeded"
    assert result.cost_usd > 1.0
    assert ("status", "limit_exceeded") in [
        (e.kind, (e.raw or {}).get("event")) for e in events
    ]


async def test_subagent_on_another_model_is_priced_at_its_own_rate(tmp_path, monkeypatch):
    # A subagent's role config can put it on a different model; its rollout's
    # turn_context names it, and that model's price — not the parent's — bills its spend.
    spec = AgentSpec(name="a", model="gpt-5.6-luna", harness="codex")
    ctx = _ctx(tmp_path, spec)
    sub_total = {"input_tokens": 2000, "cached_input_tokens": 0, "output_tokens": 100,
                 "reasoning_output_tokens": 0, "total_tokens": 2100}

    def write_rollout(client):
        _write_subagent_rollout(
            ctx.job_dir / "codex_home", _SUB_TID, [sub_total], model="gpt-5.6-sol"
        )

    script = [
        turn_started(),
        collab_agent_tool_call(started=True, agents={_SUB_TID: "running"}),
        write_rollout,
        collab_agent_tool_call(agents={_SUB_TID: "completed"}),
        agent_message("done"),
        token_usage(in_tok=1000, out_tok=100),
        turn_completed(),
    ]
    events, _ = await _events_of(script, tmp_path, monkeypatch, spec=spec, ctx=ctx)
    result = events[-1]
    assert result.raw["subagent_usage"][_SUB_TID]["model"] == "gpt-5.6-sol"
    # luna: 1.0 / 0.1 / 6.0 per MTok; sol: 5.0 / 0.5 / 30.0.
    parent = (1000 * 1.0 + 100 * 6.0) / 1e6
    on_sol = (2000 * 5.0 + 100 * 30.0) / 1e6
    assert result.cost_usd == pytest.approx(parent + on_sol)
    assert not [e for e in events if (e.raw or {}).get("event") == "subagent_price_unknown"]


async def test_subagent_on_an_unpriced_model_makes_cost_unknown(tmp_path, monkeypatch):
    # A cost is a real number or absent, never a guess (README contract): a subagent on
    # a model pricing doesn't know nulls cost_usd — with a warning — instead of being
    # approximated at the parent's rate (which could even trip the budget cap falsely).
    # Its tokens still count: usage and raw["subagent_usage"] stay complete.
    spec = AgentSpec(name="a", model="gpt-5.6-luna", harness="codex", max_budget_usd=0.001)
    ctx = _ctx(tmp_path, spec)
    sub_total = {"input_tokens": 2000, "cached_input_tokens": 0, "output_tokens": 100,
                 "reasoning_output_tokens": 0, "total_tokens": 2100}

    def write_rollout(client):
        _write_subagent_rollout(
            ctx.job_dir / "codex_home", _SUB_TID, [sub_total], model="gpt-99-unpriced"
        )

    script = [
        turn_started(),
        collab_agent_tool_call(started=True, agents={_SUB_TID: "running"}),
        write_rollout,
        collab_agent_tool_call(agents={_SUB_TID: "completed"}),
        agent_message("done"),
        token_usage(in_tok=100, out_tok=10),  # parent alone is under the tiny cap
        turn_completed(),
    ]
    events, _ = await _events_of(script, tmp_path, monkeypatch, spec=spec, ctx=ctx)
    result = events[-1]
    assert result.cost_usd is None
    assert result.raw["subtype"] == "success"  # unknown cost must not trip the budget
    assert result.raw["subagent_usage"][_SUB_TID]["model"] == "gpt-99-unpriced"
    assert result.usage["input_tokens"] == 100 + 2000  # tokens still fully counted
    warnings = [e for e in events if (e.raw or {}).get("event") == "subagent_price_unknown"]
    assert len(warnings) == 1 and warnings[0].raw["thread_id"] == _SUB_TID
    assert "unknown" in warnings[0].summary


async def test_wire_seen_subagent_without_a_rollout_warns_usage_missing(tmp_path, monkeypatch):
    # The tripwire for the rollout recovery's reliance on non-public details: the wire
    # announced a subagent thread, but no rollout under CODEX_HOME/sessions accounts for
    # it — usage/cost are undercounting and the run must say so (without failing).
    spec = AgentSpec(name="a", model="gpt-5.6-luna", harness="codex")
    script = [
        turn_started(),
        collab_agent_tool_call(started=True, agents={_SUB_TID: "running"}),
        collab_agent_tool_call(agents={_SUB_TID: "completed"}),
        agent_message("done"),
        token_usage(in_tok=1000, out_tok=100),
        turn_completed(),
    ]
    events, _ = await _events_of(script, tmp_path, monkeypatch, spec=spec)
    missing = [e for e in events if (e.raw or {}).get("event") == "subagent_usage_missing"]
    assert len(missing) == 1 and missing[0].raw["threads"] == [_SUB_TID]
    result = events[-1]
    assert result.kind == "result" and not result.raw["is_error"]
    assert "subagent_usage" not in result.raw  # nothing recovered — and no fake zeros

"""ClaudeCodeHarness.build_options: spec → ClaudeAgentOptions wiring (no model call)."""

from __future__ import annotations

from pathlib import Path

from remote_agent_toolkit import (
    DEFAULT_MAX_BUFFER_SIZE,
    AgentSpec,
    McpServer,
    RepoSource,
    SystemPrompt,
)
import pytest

from remote_agent_toolkit.events import AgentEvent
from remote_agent_toolkit.harness._shared import harness_consumed_secret_names
from remote_agent_toolkit.harness.claude_code import ClaudeCodeHarness
from remote_agent_toolkit.harness.context import RunContext


def _ctx(spec, **kw):
    return RunContext(spec=spec, prompt="hi", job_dir=Path("/tmp/x"), session_id="sid", **kw)


def test_system_prompt_inherit_appends_to_preset():
    spec = AgentSpec(
        name="a", model="claude-sonnet-4-6", system_prompt=SystemPrompt.inherit(append="Be terse.")
    )
    opts = ClaudeCodeHarness().build_options(spec, _ctx(spec))
    assert opts.system_prompt == {"type": "preset", "preset": "claude_code", "append": "Be terse."}
    # cwd is the "workspace" leaf under the job dir, not the anonymous job dir itself.
    assert opts.cwd == "/tmp/x/workspace" and opts.model == "claude-sonnet-4-6"


def test_system_prompt_plain_string_replaces():
    spec = AgentSpec(name="a", model="m", system_prompt="You are X.")
    opts = ClaudeCodeHarness().build_options(spec, _ctx(spec))
    assert opts.system_prompt == "You are X."


def test_interactive_appends_suffix():
    spec = AgentSpec(name="a", model="m")  # system_prompt None
    opts = ClaudeCodeHarness().build_options(spec, _ctx(spec, interactive=True))
    assert opts.system_prompt["preset"] == "claude_code"
    assert "INTERACTIVE MODE" in opts.system_prompt["append"]


def test_default_allowed_tools_when_unset_and_disallowed_passthrough():
    spec = AgentSpec(name="a", model="m", disallowed_tools=["WebSearch"])
    opts = ClaudeCodeHarness().build_options(spec, _ctx(spec))
    assert "Bash" in opts.allowed_tools and "Read" in opts.allowed_tools
    assert opts.disallowed_tools == ["WebSearch"]


def test_reasoning_effort_passthrough_and_floor():
    spec = AgentSpec(name="a", model="m", reasoning_effort="high")
    assert ClaudeCodeHarness().build_options(spec, _ctx(spec)).effort == "high"
    # Codex-only levels floor to Claude's lowest; unset leaves the SDK default.
    for codex_only in ("minimal", "none"):
        low = AgentSpec(name="a", model="m", reasoning_effort=codex_only)
        assert ClaudeCodeHarness().build_options(low, _ctx(low)).effort == "low"
    unset = AgentSpec(name="a", model="m")
    assert ClaudeCodeHarness().build_options(unset, _ctx(unset)).effort is None


def test_max_buffer_size_reaches_the_sdk_and_beats_its_1mib_default():
    # The SDK caps ONE stdout message at 1 MiB and raises inside its read loop past that,
    # killing the turn with no result; the spec's cap must actually reach the transport.
    spec = AgentSpec(name="a", model="m")
    assert ClaudeCodeHarness().build_options(spec, _ctx(spec)).max_buffer_size == (
        DEFAULT_MAX_BUFFER_SIZE
    )
    raised = AgentSpec(name="a", model="m", max_buffer_size=64 * 1024 * 1024)
    opts = ClaudeCodeHarness().build_options(raised, _ctx(raised))
    assert opts.max_buffer_size == 64 * 1024 * 1024


def test_github_mcp_built_from_resolved_secret_only():
    spec = AgentSpec(name="a", model="m", mcp_servers=[McpServer.github()])
    # No token resolved → server skipped (never errors, never logs a token).
    assert ClaudeCodeHarness().build_options(spec, _ctx(spec)).mcp_servers == {}
    # Token present in resolved secrets → http MCP block with Bearer auth.
    opts = ClaudeCodeHarness().build_options(spec, _ctx(spec, secrets={"GH_TOKEN": "ghp_x"}))
    gh = opts.mcp_servers["github"]
    assert gh["type"] == "http" and gh["headers"]["Authorization"] == "Bearer ghp_x"


def test_mcp_config_is_strict():
    # Only spec.mcp_servers reaches the agent; a `.mcp.json` in the workspace does not.
    spec = AgentSpec(name="a", model="m")
    assert ClaudeCodeHarness().build_options(spec, _ctx(spec)).strict_mcp_config is True


def test_hooks_passed_through():
    # PreToolUse fires under every permission mode — including bypassPermissions, which
    # shadows can_use_tool — so hooks are the seam for observing every tool call.
    spec = AgentSpec(name="a", model="m")
    hooks = {"PreToolUse": []}
    assert ClaudeCodeHarness().build_options(spec, _ctx(spec, hooks=hooks)).hooks is hooks
    assert ClaudeCodeHarness().build_options(spec, _ctx(spec)).hooks is None


def test_remote_mcp_passthrough():
    spec = AgentSpec(
        name="a",
        model="m",
        mcp_servers=[McpServer.remote("zyte", "https://mcp.zyte.com", {"X-Key": "v"})],
    )
    opts = ClaudeCodeHarness().build_options(spec, _ctx(spec))
    assert opts.mcp_servers["zyte"] == {
        "type": "http",
        "url": "https://mcp.zyte.com",
        "headers": {"X-Key": "v"},
    }


def test_agent_visible_secrets_forwarded_to_env():
    # A caller's own API key (per-invocation) plus non-secret spec.env both reach the agent env.
    spec = AgentSpec(name="a", model="m", env={"FOO": "bar"})
    opts = ClaudeCodeHarness().build_options(spec, _ctx(spec, secrets={"SH_APIKEY": "sekret"}))
    assert opts.env["SH_APIKEY"] == "sekret" and opts.env["FOO"] == "bar"


def test_caller_path_is_the_base_not_discarded():
    # Regression (eval-harness feedback issue 2): the uv-dir prepend rebuilt PATH from
    # os.environ, silently discarding a caller-supplied spec.env["PATH"].
    spec = AgentSpec(name="a", model="m", env={"PATH": "/my/venv/bin:/usr/bin"})
    opts = ClaudeCodeHarness().build_options(spec, _ctx(spec))
    path = opts.env["PATH"]
    assert path.endswith(":/my/venv/bin:/usr/bin")  # caller PATH survives as the base
    assert path != "/my/venv/bin:/usr/bin"  # ...with the uv dir prepended on top


def test_ctx_env_layered_after_spec_env():
    # Runtime-resolved env (e.g. the local packages venv) is applied after spec.env.
    spec = AgentSpec(name="a", model="m", env={"FOO": "spec", "BAR": "spec"})
    ctx = _ctx(spec, env={"FOO": "runtime", "VIRTUAL_ENV": "/w/venv"})
    opts = ClaudeCodeHarness().build_options(spec, ctx)
    assert opts.env["FOO"] == "runtime" and opts.env["BAR"] == "spec"
    assert opts.env["VIRTUAL_ENV"] == "/w/venv"


def test_harness_consumed_secrets_excluded_from_agent_env():
    # Repo push token + GitHub MCP token are consumed by git/MCP, so they must NOT appear as
    # environment variables the agent can read; the caller's own key still does.
    spec = AgentSpec(
        name="a",
        model="m",
        repos=[RepoSource.git("https://github.com/acme/x", auth="BB_TOKEN")],
        mcp_servers=[McpServer.github()],
    )
    ctx = _ctx(spec, secrets={"BB_TOKEN": "t", "GH_TOKEN": "ghp_x", "SH_APIKEY": "k"})
    opts = ClaudeCodeHarness().build_options(spec, ctx)
    assert "BB_TOKEN" not in opts.env  # repo auth -> embedded in origin, not the env
    assert "GH_TOKEN" not in opts.env  # github MCP -> header, not the env
    assert opts.env["SH_APIKEY"] == "k"  # the agent's own key is forwarded
    # ...but the consumers still receive their tokens (routed, not dropped).
    assert opts.mcp_servers["github"]["headers"]["Authorization"] == "Bearer ghp_x"


def test_checkpoint_session_wiring():
    spec = AgentSpec(name="a", model="m", checkpoint=True)
    store = object()
    opts = ClaudeCodeHarness().build_options(spec, _ctx(spec, session_store=store))
    assert opts.session_store is store and opts.session_store_flush is True
    assert opts.session_id == "sid"  # pinned for a new conversation
    # On resume, `resume` is set instead of a fresh session_id.
    opts2 = ClaudeCodeHarness().build_options(
        spec, _ctx(spec, session_store=store, resume_sid="old")
    )
    assert opts2.resume == "old"


def test_interactive_false_omits_suffix():
    spec = AgentSpec(name="a", model="m")  # system_prompt None
    opts = ClaudeCodeHarness().build_options(spec, _ctx(spec, interactive=False))
    # Claude Code's own prompt, with no INTERACTIVE MODE injection.
    assert opts.system_prompt == {"type": "preset", "preset": "claude_code"}


# -- openrouter models --------------------------------------------------------

# OpenRouter serves an Anthropic-compatible endpoint, so the Claude CLI can reach it. Two
# things make it work: the base URL plus a bearer token, and a custom catalogue entry (the
# CLI otherwise refuses the id with `unrecognized_model`). Verified live on all four models.

_OR = "openrouter/moonshotai/kimi-k3"


def test_openrouter_points_the_cli_at_openrouter():
    spec = AgentSpec(name="a", model=_OR)
    ctx = _ctx(spec, secrets={"OPENROUTER_API_KEY": "sk-or-1"})
    opts = ClaudeCodeHarness().build_options(spec, ctx)

    assert opts.env["ANTHROPIC_BASE_URL"] == "https://openrouter.ai/api"
    assert opts.env["ANTHROPIC_AUTH_TOKEN"] == "sk-or-1"
    # Metadata is read by the relay, which sets the header on its own request.
    assert "ANTHROPIC_CUSTOM_HEADERS" not in opts.env
    # Without this the CLI rejects the id outright.
    assert opts.env["ANTHROPIC_CUSTOM_MODEL_OPTION"] == "moonshotai/kimi-k3"
    assert opts.env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "1048576"
    # The prefix is the toolkit's; the CLI gets the provider-relative id.
    assert opts.model == "moonshotai/kimi-k3"
    # The CLI's own fallback prices are wrong for these ids. The harness applies the
    # exact OpenRouter charge after each response.
    assert opts.max_budget_usd is None


def test_output_schema_uses_claude_sdk_format_and_steers_openrouter():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
    }
    claude = AgentSpec(name="a", model="claude-haiku-4-5", output_schema=schema)
    claude_opts = ClaudeCodeHarness().build_options(claude, _ctx(claude))
    assert claude_opts.output_format == {"type": "json_schema", "schema": schema}
    assert "FINAL MESSAGE FORMAT" not in str(claude_opts.system_prompt)

    openrouter = AgentSpec(
        name="a",
        model=_OR,
        output_schema=schema,
        system_prompt=SystemPrompt.inherit(append="HOUSE RULE."),
    )
    or_opts = ClaudeCodeHarness().build_options(
        openrouter,
        _ctx(openrouter, secrets={"OPENROUTER_API_KEY": "k"}, interactive=True),
    )
    assert or_opts.output_format == {"type": "json_schema", "schema": schema}
    append = or_opts.system_prompt["append"]
    assert append.startswith("HOUSE RULE.")
    assert "INTERACTIVE MODE" in append
    assert "FINAL MESSAGE FORMAT" in append
    assert '"answer"' in append


def test_openrouter_blanks_the_auth_sources_that_outrank_the_token():
    """A deployed engine bakes CLAUDE_CODE_USE_VERTEX, which would win over the token."""
    spec = AgentSpec(name="a", model=_OR, env={"CLAUDE_CODE_USE_VERTEX": "1"})
    ctx = _ctx(spec, secrets={"OPENROUTER_API_KEY": "k"})
    env = ClaudeCodeHarness().build_options(spec, ctx).env

    assert env["CLAUDE_CODE_USE_VERTEX"] == ""
    assert env["CLAUDE_CODE_USE_BEDROCK"] == ""
    assert env["CLAUDE_CODE_USE_FOUNDRY"] == ""
    assert env["ANTHROPIC_API_KEY"] == ""
    # The SDK layers this env over the worker's own, so an ambient key would otherwise
    # reach the CLI process even when the relay holds the real one.
    assert env["OPENROUTER_API_KEY"] == ""


def test_openrouter_without_a_key_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    spec = AgentSpec(name="a", model=_OR)
    with pytest.raises(RuntimeError, match="no OpenRouter credentials"):
        ClaudeCodeHarness().build_options(spec, _ctx(spec, secrets={}))


def test_openrouter_key_is_removed_before_bash_tools():
    """The CLI receives auth, while its Bash wrapper removes it from tool processes."""
    spec = AgentSpec(name="a", model=_OR)
    assert harness_consumed_secret_names(spec) == {"OPENROUTER_API_KEY"}
    ctx = _ctx(spec, secrets={"OPENROUTER_API_KEY": "k", "MY_TOKEN": "visible"})
    env = ClaudeCodeHarness().build_options(spec, ctx).env
    assert env["MY_TOKEN"] == "visible"  # the caller's own secret still reaches the agent
    assert env["ANTHROPIC_AUTH_TOKEN"] == "k"
    wrapper = Path(env["CLAUDE_CODE_SHELL_PREFIX"])
    assert wrapper.stat().st_mode & 0o777 == 0o700
    body = wrapper.read_text()
    assert "unset ANTHROPIC_AUTH_TOKEN ANTHROPIC_CUSTOM_HEADERS OPENROUTER_API_KEY" in body


def test_claude_model_is_untouched_by_any_of_this():
    spec = AgentSpec(name="a", model="claude-haiku-4-5")
    env = ClaudeCodeHarness().build_options(spec, _ctx(spec)).env
    assert "ANTHROPIC_BASE_URL" not in env
    assert "ANTHROPIC_CUSTOM_MODEL_OPTION" not in env


def test_openrouter_cost_is_recomputed_from_usage():
    """The CLI prices these models from its own catalogue and gets it badly wrong.

    Measured live: for deepseek-v4-flash the CLI reported $0.271 on a turn that really
    cost $0.0045 — 60x over. Left alone, `max_budget_usd` would fire almost immediately.
    """
    from remote_agent_toolkit.harness import pricing

    price = pricing.model_price("openrouter/deepseek/deepseek-v4-flash")
    assert price is not None
    event = AgentEvent(
        kind="result",
        summary="done",
        cost_usd=0.271,  # what the CLI claimed
        usage={
            "input_tokens": 20_000,
            "output_tokens": 100,
            "cache_read_input_tokens": 500,
            "cache_creation_input_tokens": 0,
        },
        raw={"subtype": "success"},
    )
    out = ClaudeCodeHarness()._final_result(event, turns_total=2, price=price)

    # 20000 fresh + 500 cached input, 100 output, at flash rates.
    expected = (20_000 * 0.084 + 500 * 0.0168 + 100 * 0.168) / 1e6
    assert out.cost_usd == pytest.approx(expected)
    assert out.raw["cli_reported_cost_usd"] == 0.271  # kept for comparison
    assert out.raw["price_source"] == "builtin"


def test_openrouter_exact_cost_wins_and_enforces_budget():
    event = AgentEvent(
        kind="result",
        summary="done",
        cost_usd=0.271,
        usage={"input_tokens": 20_000, "output_tokens": 100},
        raw={"subtype": "success", "is_error": False},
    )
    out = ClaudeCodeHarness()._final_result(
        event,
        turns_total=2,
        exact_openrouter_cost=0.0123,
        max_budget_usd=0.01,
    )

    assert out.cost_usd == 0.0123
    assert out.raw["price_source"] == "openrouter"
    assert out.raw["cli_reported_cost_usd"] == 0.271
    assert out.raw["subtype"] == "error_budget_exceeded"
    assert out.raw["is_error"] is True
    assert out.raw["budget_enforcement"] == "after_model_response"


def test_openrouter_no_final_text_is_an_error():
    event = AgentEvent(
        kind="result",
        summary="(no final text)",
        cost_usd=0.01,
        raw={"subtype": "success", "is_error": False},
    )
    out = ClaudeCodeHarness()._final_result(
        event,
        turns_total=1,
        exact_openrouter_cost=0.001,
        max_budget_usd=1.0,
    )
    assert out.raw["subtype"] == "error_no_final_text"
    assert out.raw["is_error"] is True


def test_claude_models_keep_the_cli_cost():
    """Only the OpenRouter path is repriced; Claude's own number is authoritative."""
    event = AgentEvent(kind="result", summary="done", cost_usd=0.42, usage={}, raw={})
    out = ClaudeCodeHarness()._final_result(event, turns_total=1, price=None)
    assert out.cost_usd == 0.42
    assert "cli_reported_cost_usd" not in out.raw


@pytest.mark.parametrize("provider", ["moonshotai", None])
def test_openrouter_turns_use_the_local_relay_on_claude(provider):
    """Pinned or not, the CLI talks to the relay and never holds the account key."""
    spec = AgentSpec(name="a", model=_OR, openrouter_provider=provider)
    opts = ClaudeCodeHarness().build_options(
        spec,
        _ctx(spec, secrets={"OPENROUTER_API_KEY": "real-key"}),
        openrouter_base_url="http://127.0.0.1:1234/api",
        openrouter_client_token="local-token",
    )

    assert opts.env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:1234/api"
    assert opts.env["ANTHROPIC_AUTH_TOKEN"] == "local-token"
    assert "real-key" not in opts.env.values()


def test_relay_402_is_reported_as_a_budget_failure():
    """Whatever the CLI made of the 402, the run stopped because of the cap."""
    event = AgentEvent(
        kind="result",
        summary="failed",
        cost_usd=0.02,
        usage={},
        raw={"subtype": "error_during_execution", "is_error": True},
    )
    out = ClaudeCodeHarness()._final_result(
        event, turns_total=1, exact_openrouter_cost=0.02, budget_blocked=True
    )
    assert out.raw["subtype"] == "error_budget_exceeded"
    assert out.raw["budget_enforcement"] == "relay_402"
    assert out.raw["cli_reported_subtype"] == "error_during_execution"


def test_unpriced_openrouter_model_reports_no_cost():
    """An unknown id has no fallback price, so the CLI's invented figure is dropped."""
    event = AgentEvent(kind="result", summary="done", cost_usd=0.271, usage={}, raw={})
    out = ClaudeCodeHarness()._final_result(
        event, turns_total=1, price=None, unpriced_openrouter=True
    )
    assert out.cost_usd is None
    assert out.raw["cli_reported_cost_usd"] == 0.271

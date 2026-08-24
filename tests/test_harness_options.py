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
from remote_agent_toolkit.harness._shared import (
    harness_consumed_secret_names,
    openrouter_cost_unknown,
    openrouter_provider_routing,
    openrouter_schema_steer,
)
from remote_agent_toolkit.harness.claude_code import ClaudeCodeHarness
from remote_agent_toolkit.harness.context import RunContext


_JOB_DIR: Path


@pytest.fixture(autouse=True)
def _job_dir(tmp_path):
    """Give every ctx a writable job dir under tmp_path.

    ``build_options`` writes the OpenRouter shell wrapper under the job dir, so a fixed
    path would create files on the host (and fails outright where /tmp is read-only).
    """
    global _JOB_DIR
    _JOB_DIR = tmp_path / "job"


def _ctx(spec, **kw):
    return RunContext(spec=spec, prompt="hi", job_dir=_JOB_DIR, session_id="sid", **kw)


def test_system_prompt_inherit_appends_to_preset():
    spec = AgentSpec(
        name="a", model="claude-sonnet-4-6", system_prompt=SystemPrompt.inherit(append="Be terse.")
    )
    opts = ClaudeCodeHarness().build_options(spec, _ctx(spec))
    assert opts.system_prompt == {"type": "preset", "preset": "claude_code", "append": "Be terse."}
    # cwd is the "workspace" leaf under the job dir, not the anonymous job dir itself.
    assert opts.cwd == str(_JOB_DIR / "workspace") and opts.model == "claude-sonnet-4-6"


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
    # Metadata is read by the proxy, which sets the header on its own request.
    assert "ANTHROPIC_CUSTOM_HEADERS" not in opts.env
    # Without this the CLI rejects the id outright.
    assert opts.env["ANTHROPIC_CUSTOM_MODEL_OPTION"] == "moonshotai/kimi-k3"
    assert opts.env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "1048576"
    # The prefix is the toolkit's; the CLI gets the provider-relative id.
    assert opts.model == "moonshotai/kimi-k3"
    # The CLI's own fallback prices are wrong for these ids. The harness applies the
    # exact OpenRouter charge after each response.
    assert opts.max_budget_usd is None


def test_unlisted_openrouter_model_gets_no_context_window():
    """An id the table does not carry still runs; the CLI then assumes its own 200k."""
    spec = AgentSpec(name="a", model="openrouter/some-vendor/brand-new-model")
    opts = ClaudeCodeHarness().build_options(
        spec, _ctx(spec, secrets={"OPENROUTER_API_KEY": "sk-or-1"})
    )

    assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS" not in opts.env
    assert opts.model == "some-vendor/brand-new-model"


def test_openrouter_cost_unknown_event_names_the_model():
    event = openrouter_cost_unknown(_OR)

    assert event.raw == {"event": "cost_unknown", "model": _OR}
    assert "did not report a charge" in event.summary
    assert "max_budget_usd could not be enforced" in event.summary


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
    # reach the CLI process even when the proxy holds the real one.
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
    assert "unset ANTHROPIC_AUTH_TOKEN OPENROUTER_API_KEY" in body


def test_claude_model_is_untouched_by_any_of_this():
    spec = AgentSpec(name="a", model="claude-haiku-4-5")
    env = ClaudeCodeHarness().build_options(spec, _ctx(spec)).env
    assert "ANTHROPIC_BASE_URL" not in env
    assert "ANTHROPIC_CUSTOM_MODEL_OPTION" not in env


def test_openrouter_cost_is_never_estimated():
    """No charge from OpenRouter means no cost, never a price-table estimate.

    The CLI's own figure is wrong here — measured live, it reported $0.271 for a
    deepseek-v4-flash turn that really cost $0.0045 — and an estimate would be wrong in a
    way nobody could see, because the charge depends on which provider served the call.
    """
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
    out = ClaudeCodeHarness()._final_result(event, turns_total=2, openrouter=True)

    assert out.cost_usd is None
    assert out.raw["cli_reported_cost_usd"] == 0.271  # kept for comparison
    assert "price_source" not in out.raw


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


def _no_final_text_event():
    return AgentEvent(
        kind="result",
        summary="(no final text)",
        cost_usd=0.01,
        raw={"subtype": "success", "is_error": False},
    )


@pytest.mark.parametrize("max_budget_usd", [1.0, None])
def test_openrouter_no_final_text_is_an_error(max_budget_usd):
    """The reclassification follows the OpenRouter turn, not the budget cap."""
    out = ClaudeCodeHarness()._final_result(
        _no_final_text_event(),
        turns_total=1,
        openrouter=True,
        exact_openrouter_cost=0.001,
        max_budget_usd=max_budget_usd,
    )
    assert out.raw["subtype"] == "error_no_final_text"
    assert out.raw["cli_reported_subtype"] == "success"
    assert out.raw["is_error"] is True


def test_claude_no_final_text_is_left_alone():
    """A native Claude turn keeps whatever the CLI reported."""
    out = ClaudeCodeHarness()._final_result(
        _no_final_text_event(),
        turns_total=1,
        openrouter=False,
        max_budget_usd=1.0,
    )
    assert out.raw["subtype"] == "success"
    assert out.raw["is_error"] is False


def test_claude_models_keep_the_cli_cost():
    """Only the OpenRouter path is touched; Claude's own number is authoritative."""
    event = AgentEvent(kind="result", summary="done", cost_usd=0.42, usage={}, raw={})
    out = ClaudeCodeHarness()._final_result(event, turns_total=1, openrouter=False)
    assert out.cost_usd == 0.42
    assert "cli_reported_cost_usd" not in out.raw


@pytest.mark.parametrize("provider", ["moonshotai", None])
def test_openrouter_turns_use_the_local_proxy_on_claude(provider):
    """Pinned or not, the CLI talks to the proxy and never holds the account key."""
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


def test_proxy_402_is_reported_as_a_budget_failure():
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
    assert out.raw["budget_enforcement"] == "proxy_402"
    assert out.raw["cli_reported_subtype"] == "error_during_execution"


def test_unpriced_openrouter_model_reports_no_cost():
    """Any id the proxy recorded no charge for drops the CLI's invented figure."""
    event = AgentEvent(kind="result", summary="done", cost_usd=0.271, usage={}, raw={})
    out = ClaudeCodeHarness()._final_result(event, turns_total=1, openrouter=True)
    assert out.cost_usd is None
    assert out.raw["cli_reported_cost_usd"] == 0.271


# -- one routing value --------------------------------------------------------
#
# ``openrouter_provider`` is the shorthand for pinning one provider. It is resolved to a
# routing object once, above the proxy, so nothing below has to ask which of the two
# fields the caller used.


def test_a_pinned_provider_becomes_a_closed_routing_object():
    spec = AgentSpec(name="a", model=_OR, openrouter_provider="moonshotai")
    assert openrouter_provider_routing(spec) == {
        "only": ["moonshotai"],
        "allow_fallbacks": False,
    }


def test_a_routing_object_is_passed_through_unchanged():
    routing = {"order": ["moonshotai", "fireworks"], "allow_fallbacks": True}
    spec = AgentSpec(name="a", model=_OR, openrouter_routing=routing)
    resolved = openrouter_provider_routing(spec)

    assert resolved == routing
    assert resolved is not spec.openrouter_routing


def test_no_provider_preference_resolves_to_nothing():
    assert openrouter_provider_routing(AgentSpec(name="a", model=_OR)) is None


# -- one schema steer ---------------------------------------------------------
#
# Both bindings ask an OpenRouter model for bare JSON in words, because OpenRouter passes
# the native structured-output format on without enforcing it. Same text, same conditions.


def test_the_schema_steer_states_the_schema_for_an_openrouter_turn():
    spec = AgentSpec(name="a", model=_OR, output_schema={"type": "object"})
    steer = openrouter_schema_steer(spec)

    assert "FINAL MESSAGE FORMAT" in steer
    assert '{"type":"object"}' in steer
    # It is appended to a prompt, so it brings its own separation.
    assert steer.startswith("\n\n")


@pytest.mark.parametrize(
    "spec",
    [
        AgentSpec(name="a", model="claude-sonnet-4-5", output_schema={"type": "object"}),
        AgentSpec(name="a", model=_OR),
    ],
    ids=["native-model", "no-schema"],
)
def test_the_schema_steer_is_empty_when_it_does_not_apply(spec):
    assert openrouter_schema_steer(spec) == ""

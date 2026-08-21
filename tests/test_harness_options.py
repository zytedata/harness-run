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
from remote_agent_toolkit.harness.claude_code import ClaudeCodeHarness
from remote_agent_toolkit.harness.context import RunContext


def _ctx(spec, **kw):
    return RunContext(spec=spec, prompt="hi", job_dir=Path("/tmp/x"), session_id="sid", **kw)


def test_system_prompt_inherit_appends_to_preset():
    spec = AgentSpec(name="a", model="claude-sonnet-4-6",
                     system_prompt=SystemPrompt.inherit(append="Be terse."))
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
    spec = AgentSpec(name="a", model="m",
                     mcp_servers=[McpServer.remote("zyte", "https://mcp.zyte.com", {"X-Key": "v"})])
    opts = ClaudeCodeHarness().build_options(spec, _ctx(spec))
    assert opts.mcp_servers["zyte"] == {"type": "http", "url": "https://mcp.zyte.com", "headers": {"X-Key": "v"}}


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
    assert path != "/my/venv/bin:/usr/bin"          # ...with the uv dir prepended on top


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
        name="a", model="m",
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
    opts2 = ClaudeCodeHarness().build_options(spec, _ctx(spec, session_store=store, resume_sid="old"))
    assert opts2.resume == "old"


def test_interactive_false_omits_suffix():
    spec = AgentSpec(name="a", model="m")  # system_prompt None
    opts = ClaudeCodeHarness().build_options(spec, _ctx(spec, interactive=False))
    # Claude Code's own prompt, with no INTERACTIVE MODE injection.
    assert opts.system_prompt == {"type": "preset", "preset": "claude_code"}

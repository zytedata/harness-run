"""ClaudeCodeHarness.build_options: spec → ClaudeAgentOptions wiring (no model call)."""

from __future__ import annotations

from pathlib import Path

from remote_agent_toolkit import AgentSpec, McpServer, SystemPrompt
from remote_agent_toolkit.harness.claude_code import ClaudeCodeHarness
from remote_agent_toolkit.harness.context import RunContext


def _ctx(spec, **kw):
    return RunContext(spec=spec, prompt="hi", job_dir=Path("/tmp/x"), session_id="sid", **kw)


def test_system_prompt_inherit_appends_to_preset():
    spec = AgentSpec(name="a", model="claude-sonnet-4-6",
                     system_prompt=SystemPrompt.inherit(append="Be terse."))
    opts = ClaudeCodeHarness().build_options(spec, _ctx(spec))
    assert opts.system_prompt == {"type": "preset", "preset": "claude_code", "append": "Be terse."}
    assert opts.cwd == "/tmp/x" and opts.model == "claude-sonnet-4-6"


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


def test_github_mcp_built_from_resolved_secret_only():
    spec = AgentSpec(name="a", model="m", mcp_servers=[McpServer.github()])
    # No token resolved → server skipped (never errors, never logs a token).
    assert ClaudeCodeHarness().build_options(spec, _ctx(spec)).mcp_servers == {}
    # Token present in resolved secrets → http MCP block with Bearer auth.
    opts = ClaudeCodeHarness().build_options(spec, _ctx(spec, secrets={"GH_TOKEN": "ghp_x"}))
    gh = opts.mcp_servers["github"]
    assert gh["type"] == "http" and gh["headers"]["Authorization"] == "Bearer ghp_x"


def test_remote_mcp_passthrough():
    spec = AgentSpec(name="a", model="m",
                     mcp_servers=[McpServer.remote("zyte", "https://mcp.zyte.com", {"X-Key": "v"})])
    opts = ClaudeCodeHarness().build_options(spec, _ctx(spec))
    assert opts.mcp_servers["zyte"] == {"type": "http", "url": "https://mcp.zyte.com", "headers": {"X-Key": "v"}}


def test_secrets_forwarded_to_env_but_never_in_options_elsewhere():
    spec = AgentSpec(name="a", model="m", secrets=["ZYTE_API_KEY"], env={"FOO": "bar"})
    opts = ClaudeCodeHarness().build_options(spec, _ctx(spec, secrets={"ZYTE_API_KEY": "sekret"}))
    assert opts.env["ZYTE_API_KEY"] == "sekret" and opts.env["FOO"] == "bar"


def test_checkpoint_session_wiring():
    spec = AgentSpec(name="a", model="m", checkpoint=True)
    store = object()
    opts = ClaudeCodeHarness().build_options(spec, _ctx(spec, session_store=store))
    assert opts.session_store is store and opts.session_store_flush is True
    assert opts.session_id == "sid"  # pinned for a new conversation
    # On resume, `resume` is set instead of a fresh session_id.
    opts2 = ClaudeCodeHarness().build_options(spec, _ctx(spec, session_store=store, resume_sid="old"))
    assert opts2.resume == "old"

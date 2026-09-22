"""Secret references, not values, belong in durable configuration (S11-B)."""
import json

import pytest

from agent_run import AgentSpec, McpServer, SessionConfig
from agent_run.config import apply_session_config
from agent_run.harness.claude_code import ClaudeCodeHarness
from agent_run.harness.codex import CodexHarness
from agent_run.harness.context import RunContext


@pytest.mark.parametrize("header", ["Authorization", "authorization", "Cookie", "X-API-Key"])
def test_authentication_headers_cannot_be_static_config(header):
    with pytest.raises(ValueError, match="header_secrets") as error:
        McpServer.remote("audit", "https://audit.invalid", headers={header: "AUDIT_FAKE_SECRET"})
    assert "AUDIT_FAKE_SECRET" not in str(error.value)


@pytest.mark.parametrize("field", ["env", "extra_env"])
def test_recognized_credential_env_names_require_runtime_secrets(field):
    with pytest.raises(ValueError, match="secrets"):
        if field == "env":
            AgentSpec(name="audit", model="dummy", env={"AUDIT_API_KEY": "AUDIT_FAKE_SECRET"})
        else:
            SessionConfig(extra_env={"AUDIT_API_KEY": "AUDIT_FAKE_SECRET"})


def test_header_reference_roundtrip_and_runtime_resolution(tmp_path):
    server = McpServer.remote("audit", "https://audit.invalid", headers={"Accept": "application/json"},
                             header_secrets={"Authorization": "MCP_AUTH"})
    config = SessionConfig(mcp_servers=[server], extra_env={"MODE": "audit"})
    config = SessionConfig.from_dict(config.to_dict())
    spec = apply_session_config(AgentSpec(name="audit", model="dummy"), config)
    token = "Bearer AUDIT_FAKE_SECRET"
    ctx = RunContext(spec=spec, prompt="dummy", job_dir=tmp_path, session_id="audit",
                     secrets={"MCP_AUTH": token})
    claude = ClaudeCodeHarness().build_options(spec, ctx)
    assert claude.mcp_servers["audit"]["headers"]["Authorization"] == token
    assert "MCP_AUTH" not in claude.env
    codex = CodexHarness().build_options(spec, ctx)
    overrides = list(codex.codex_config.config_overrides)
    assert any("env_http_headers" in item for item in overrides)
    assert not any("AUDIT_FAKE_SECRET" in item for item in overrides)
    injected = [name for name, value in codex.codex_config.env.items() if value == token]
    assert len(injected) == 1
    exclusions = next(item for item in overrides if item.startswith("shell_environment_policy.exclude="))
    assert injected[0] in exclusions and "MCP_AUTH" in exclusions
    assert "AUDIT_FAKE_SECRET" not in json.dumps(config.to_dict())
    assert "AUDIT_FAKE_SECRET" not in json.dumps(spec.to_dict())


@pytest.mark.parametrize("harness", [ClaudeCodeHarness, CodexHarness])
def test_missing_header_secret_fails_without_ambient_fallback(tmp_path, monkeypatch, harness):
    monkeypatch.setenv("MCP_AUTH", "AUDIT_FAKE_AMBIENT_SECRET")
    server = McpServer.remote("audit", "https://audit.invalid",
                             header_secrets={"Authorization": "MCP_AUTH"})
    spec = AgentSpec(name="audit", model="dummy", mcp_servers=[server])
    ctx = RunContext(spec=spec, prompt="dummy", job_dir=tmp_path, session_id="audit")
    with pytest.raises(ValueError, match="MCP header secret"):
        harness().build_options(spec, ctx)

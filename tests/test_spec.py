"""P0 round-trip tests for the serializable AgentSpec and its value types."""

from __future__ import annotations

import pytest

from remote_agent_toolkit import AgentSpec, McpServer, SkillSource, SystemPrompt


def _example_spec() -> AgentSpec:
    return AgentSpec(
        name="spider-builder",
        model="claude-sonnet-4-6",
        system_prompt=SystemPrompt.inherit(append="Prefer the Zyte web-scraping skills."),
        skills=[SkillSource.git("https://github.com/zytedata/claude-skills", ref="0.2.0")],
        mcp_servers=[McpServer.github(), McpServer.stdio("local", "echo", ["hi"])],
        allowed_tools=["Bash", "Read"],
        secrets=["ZYTE_API_KEY", "GH_PAT"],
        permission_mode="bypassPermissions",
        max_turns=42,
        max_budget_usd=5.0,
        checkpoint=True,
        env={"FOO": "bar"},
    )


def test_agentspec_dict_round_trip() -> None:
    spec = _example_spec()
    restored = AgentSpec.from_dict(spec.to_dict())
    assert restored == spec


def test_agentspec_coerces_lists_to_tuples() -> None:
    spec = _example_spec()
    assert isinstance(spec.skills, tuple)
    assert isinstance(spec.mcp_servers, tuple)
    assert isinstance(spec.secrets, tuple)
    assert isinstance(spec.allowed_tools, tuple)


def test_agentspec_yaml_round_trip() -> None:
    pytest.importorskip("yaml")
    spec = _example_spec()
    restored = AgentSpec.from_yaml(spec.to_yaml())
    assert restored == spec


def test_skillsource_git_smoke() -> None:
    s = SkillSource.git("https://github.com/zytedata/claude-skills", ref="0.2.0")
    assert s.kind == "git"
    assert s.url == "https://github.com/zytedata/claude-skills"
    assert s.ref == "0.2.0"
    assert s.subdir == "skills"
    assert SkillSource.from_dict(s.to_dict()) == s

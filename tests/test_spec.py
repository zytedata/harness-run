"""P0 round-trip tests for the serializable AgentSpec and its value types."""

from __future__ import annotations

import pytest

from remote_agent_toolkit import (
    DEFAULT_MAX_BUFFER_SIZE,
    AgentSpec,
    McpServer,
    RepoSource,
    SkillSource,
    SystemPrompt,
)


def _example_spec() -> AgentSpec:
    return AgentSpec(
        name="spider-builder",
        model="claude-sonnet-4-6",
        system_prompt=SystemPrompt.inherit(append="Prefer the Zyte web-scraping skills."),
        skills=[SkillSource.git("https://github.com/zytedata/claude-skills", ref="0.2.0")],
        repos=[RepoSource.git("https://github.com/acme/spiders", ref="main", auth="GH_TOKEN")],
        mcp_servers=[McpServer.github(), McpServer.stdio("local", "echo", ["hi"])],
        allowed_tools=["Bash", "Read"],
        permission_mode="bypassPermissions",
        max_turns=42,
        max_budget_usd=5.0,
        reasoning_effort="high",
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
    assert isinstance(spec.repos, tuple)
    assert isinstance(spec.mcp_servers, tuple)
    assert isinstance(spec.allowed_tools, tuple)


def test_reposource_auth_round_trip() -> None:
    r = RepoSource.git("https://bitbucket.org/acme/spiders", ref="dev", auth="BITBUCKET_API_TOKEN")
    assert r.url == "https://bitbucket.org/acme/spiders"
    assert r.ref == "dev"
    assert r.auth == "BITBUCKET_API_TOKEN"
    assert RepoSource.from_dict(r.to_dict()) == r
    # Public repo: no auth carried.
    pub = RepoSource.git("https://github.com/acme/pub")
    assert pub.auth is None and "auth" not in pub.to_dict()


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


def test_agentspec_reasoning_effort_round_trip() -> None:
    # None (default) is omitted from the dict and survives the trip.
    spec = AgentSpec(name="a", model="m")
    assert spec.reasoning_effort is None
    assert "reasoning_effort" not in spec.to_dict()
    assert AgentSpec.from_dict(spec.to_dict()).reasoning_effort is None
    explicit = AgentSpec(name="a", model="m", reasoning_effort="xhigh")
    assert AgentSpec.from_dict(explicit.to_dict()).reasoning_effort == "xhigh"


def test_agentspec_max_buffer_size_round_trip_and_default() -> None:
    # The default is the toolkit's own generous cap, NOT the SDK's 1 MiB (which a real
    # message — a base64 image, a large tool result — exceeds and dies mid-turn on).
    assert AgentSpec(name="a", model="m").max_buffer_size == DEFAULT_MAX_BUFFER_SIZE
    assert DEFAULT_MAX_BUFFER_SIZE > 1024 * 1024
    explicit = AgentSpec(name="a", model="m", max_buffer_size=4 * 1024 * 1024)
    assert AgentSpec.from_dict(explicit.to_dict()).max_buffer_size == 4 * 1024 * 1024
    # A spec serialized before the field existed takes the default, not the old 1 MiB.
    older = {k: v for k, v in explicit.to_dict().items() if k != "max_buffer_size"}
    assert AgentSpec.from_dict(older).max_buffer_size == DEFAULT_MAX_BUFFER_SIZE


@pytest.mark.parametrize("bad", [0, -1])
def test_agentspec_rejects_non_positive_max_buffer_size(bad: int) -> None:
    # Would reject every message; caught at construction rather than as a turn failure
    # that reads like the payload's fault.
    with pytest.raises(ValueError, match="max_buffer_size must be > 0"):
        AgentSpec(name="a", model="m", max_buffer_size=bad)


def test_agentspec_interactive_round_trip() -> None:
    # None (default) follows checkpoint, is omitted from the dict, and survives the trip.
    spec = _example_spec()
    assert spec.interactive is None
    assert "interactive" not in spec.to_dict()
    assert AgentSpec.from_dict(spec.to_dict()).interactive is None
    # Explicit False (checkpoint WITHOUT the operator-pause guidance) round-trips.
    explicit = AgentSpec(name="a", model="m", checkpoint=True, interactive=False)
    restored = AgentSpec.from_dict(explicit.to_dict())
    assert restored.interactive is False and restored.checkpoint is True

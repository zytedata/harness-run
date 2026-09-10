"""SessionConfig / TurnConfig semantics: sparse overlays over the deployed AgentSpec.

The deploy/session/turn scope model (README): a config field left at INHERIT keeps the
value from the layer below; a set field replaces it wholesale (extra_env is the one
additive exception). Deploy-only facts (name, packages, engine env, harness availability)
have no config field, so changing them per run is unrepresentable.
"""

from __future__ import annotations

import pytest

from remote_agent_toolkit import (
    INHERIT,
    AgentSpec,
    RepoSource,
    SessionConfig,
    SkillSource,
    SystemPrompt,
    TurnConfig,
)
from remote_agent_toolkit.config import (
    apply_session_config,
    apply_turn_config,
    validate_harness_choice,
)


def _spec(**kw) -> AgentSpec:
    base = dict(
        name="a", model="m-deploy", system_prompt="DEPLOY-PROMPT",
        env={"KEEP": "1"}, packages=("lxml",), max_budget_usd=5.0,
    )
    base.update(kw)
    return AgentSpec(**base)


# ---------------------------------------------------------------- overlay semantics


def test_inherit_keeps_deployed_values_and_set_fields_replace():
    spec = _spec()
    cfg = SessionConfig(model="m-session", max_budget_usd=2.0)
    eff = apply_session_config(spec, cfg)
    assert eff.model == "m-session" and eff.max_budget_usd == 2.0
    # Everything left at INHERIT comes from the deployed spec.
    assert eff.system_prompt == "DEPLOY-PROMPT" and eff.name == "a" and eff.packages == ("lxml",)
    # No config at all is the identity.
    assert apply_session_config(spec, None) is spec
    assert apply_session_config(spec, SessionConfig()) is spec


def test_explicit_none_is_a_value_not_inherit():
    # None means "harness default", NOT "inherit the deployed prompt" — the reason the
    # sentinel exists at all.
    eff = apply_session_config(_spec(), SessionConfig(system_prompt=None))
    assert eff.system_prompt is None
    assert SessionConfig().system_prompt is INHERIT


def test_extra_env_adds_on_top_of_deployed_env():
    eff = apply_session_config(_spec(), SessionConfig(extra_env={"NEW": "2", "KEEP": "override"}))
    assert eff.env == {"KEEP": "override", "NEW": "2"}
    # The deployed env object is not mutated.
    assert _spec().env == {"KEEP": "1"}


def test_deploy_only_fields_are_unrepresentable():
    with pytest.raises(TypeError):
        SessionConfig(packages=("x",))
    with pytest.raises(TypeError):
        SessionConfig(name="other")
    with pytest.raises(TypeError):
        TurnConfig(repos=(RepoSource.git("https://h/r.git"),))  # world field: session-only


def test_turn_config_layers_over_the_session():
    spec = _spec()
    session_eff = apply_session_config(spec, SessionConfig(model="m-session", reasoning_effort="low"))
    turn_eff = apply_turn_config(session_eff, TurnConfig(model="m-turn"))
    assert turn_eff.model == "m-turn"              # turn wins where set
    assert turn_eff.reasoning_effort == "low"      # session shows through where not
    assert turn_eff.max_budget_usd == 5.0          # deploy shows through below both
    assert apply_turn_config(session_eff, None) is session_eff


def test_max_buffer_size_is_tunable_per_session_and_per_turn():
    # An agent that reads big things (an image, a huge tool result) can be given room
    # without a redeploy — and one turn can ask for more than its session's default.
    spec = _spec()
    session_eff = apply_session_config(spec, SessionConfig(max_buffer_size=8 * 1024 * 1024))
    assert session_eff.max_buffer_size == 8 * 1024 * 1024
    turn_eff = apply_turn_config(session_eff, TurnConfig(max_buffer_size=64 * 1024 * 1024))
    assert turn_eff.max_buffer_size == 64 * 1024 * 1024
    # Validation applies to an overlaid value too (the overlay rebuilds the spec).
    with pytest.raises(ValueError, match="max_buffer_size must be > 0"):
        apply_turn_config(spec, TurnConfig(max_buffer_size=0))


def test_openrouter_provider_can_be_set_and_cleared_per_turn():
    spec = _spec(model="openrouter/moonshotai/kimi-k3")
    session_eff = apply_session_config(
        spec, SessionConfig(openrouter_provider="moonshotai")
    )
    assert session_eff.openrouter_provider == "moonshotai"

    turn_eff = apply_turn_config(session_eff, TurnConfig(openrouter_provider="fireworks"))
    assert turn_eff.openrouter_provider == "fireworks"
    assert TurnConfig.from_dict(TurnConfig(openrouter_provider="fireworks").to_dict()) == (
        TurnConfig(openrouter_provider="fireworks")
    )

    unpinned = apply_turn_config(session_eff, TurnConfig(openrouter_provider=None))
    assert unpinned.openrouter_provider is None
    openai = apply_turn_config(
        session_eff,
        TurnConfig(model="gpt-5.6-luna", openrouter_provider=None),
    )
    assert openai.model == "gpt-5.6-luna"

    with pytest.raises(ValueError, match="requires an openrouter/ model"):
        apply_turn_config(session_eff, TurnConfig(model="gpt-5.6-luna"))


def test_openrouter_routing_can_be_set_and_cleared_per_turn():
    spec = _spec(model="openrouter/moonshotai/kimi-k3")
    routing = {"order": ["moonshotai", "fireworks"], "allow_fallbacks": True}
    session_eff = apply_session_config(spec, SessionConfig(openrouter_routing=routing))
    assert session_eff.openrouter_routing == routing

    turn = TurnConfig(openrouter_routing={"only": ["moonshotai"]})
    turn_eff = apply_turn_config(session_eff, turn)
    assert turn_eff.openrouter_routing == {"only": ["moonshotai"]}
    assert TurnConfig.from_dict(turn.to_dict()) == turn

    unpinned = apply_turn_config(session_eff, TurnConfig(openrouter_routing=None))
    assert unpinned.openrouter_routing is None

    # One turn can swap the whole routing object for the single-slug pin.
    pinned = apply_turn_config(
        session_eff, TurnConfig(openrouter_routing=None, openrouter_provider="moonshotai")
    )
    assert pinned.openrouter_provider == "moonshotai" and pinned.openrouter_routing is None

    with pytest.raises(ValueError, match="cannot be combined"):
        apply_turn_config(session_eff, TurnConfig(openrouter_provider="moonshotai"))
    with pytest.raises(ValueError, match="requires an openrouter/ model"):
        apply_turn_config(session_eff, TurnConfig(model="gpt-5.6-luna"))


# ---------------------------------------------------------------- serialization


def test_openrouter_model_switches_per_turn_on_a_codex_engine():
    """One deployed engine, several providers: the model override is all it takes.

    This is the mechanism the remote probe drives live — a codex engine baked with one
    OpenRouter model serving turns on the others, with no redeploy.
    """
    baked = AgentSpec(
        name="a", model="openrouter/deepseek/deepseek-v4-flash", harness="codex",
        harnesses=("codex",),
    )
    session_eff = apply_session_config(baked, SessionConfig(harness="codex"))
    turn_eff = apply_turn_config(session_eff, TurnConfig(model="openrouter/moonshotai/kimi-k3"))

    assert turn_eff.model == "openrouter/moonshotai/kimi-k3"
    assert turn_eff.harness == "codex"
    validate_harness_choice(baked, turn_eff)  # the harness is baked; the model is free
    # An OpenAI model on the same engine is just another override.
    assert apply_turn_config(session_eff, TurnConfig(model="gpt-5.6-luna")).model == "gpt-5.6-luna"


def test_codex_config_can_be_set_and_cleared_per_turn():
    spec = AgentSpec(name="a", model="gpt-5.6-luna", harness="codex")
    session_eff = apply_session_config(
        spec, SessionConfig(codex_config={"sandbox_workspace_write.network_access": True})
    )
    turn_eff = apply_turn_config(session_eff, TurnConfig(codex_config={"web_search": "disabled"}))
    assert turn_eff.codex_config == {"web_search": "disabled"}  # replaced, not merged
    assert apply_turn_config(session_eff, TurnConfig(codex_config=None)).codex_config is None


def test_sparse_round_trip_preserves_set_and_only_set_fields():
    cfg = SessionConfig(
        repos=[RepoSource.git("https://h/r.git", ref="heal-1", auth="tok")],
        skills=[SkillSource.git("https://h/s.git")],
        system_prompt=SystemPrompt.inherit(append="EXTRA"),
        model="m2",
        allowed_tools=None,  # explicitly None — must survive as null
    )
    d = cfg.to_dict()
    assert set(d) == {"repos", "skills", "system_prompt", "model", "allowed_tools"}
    assert d["allowed_tools"] is None
    assert SessionConfig.from_dict(d) == cfg
    assert SessionConfig().to_dict() == {}
    t = TurnConfig(max_turns=3)
    assert TurnConfig.from_dict(t.to_dict()) == t


def test_unknown_field_fails_closed():
    # A newer client staging a field this revision doesn't know must FAIL, not silently
    # drop it (the client/engine same-revision rule).
    with pytest.raises(ValueError, match="same toolkit revision"):
        SessionConfig.from_dict({"model": "m", "warp_drive": True})
    with pytest.raises(ValueError, match="same toolkit revision"):
        TurnConfig.from_dict({"skills": []})  # a session-only field is unknown to TurnConfig


def test_credentialed_repo_url_is_refused_at_construction():
    # Configs are staged as GCS objects referenced from persisted payloads AND kept for
    # post-mortem — they must never smuggle a pre-authenticated clone URL.
    with pytest.raises(ValueError, match="auth_user"):
        SessionConfig(repos=[RepoSource.git("https://x-token-auth:SECRET@bitbucket.org/o/r.git")])


# ---------------------------------------------------------------- harness availability


def test_baked_harnesses_defaults_to_the_single_harness():
    assert _spec().baked_harnesses == ("claude-code",)
    both = _spec(harnesses=("claude-code", "codex"))
    assert both.baked_harnesses == ("claude-code", "codex")
    # Round-trips through the serialized form (what deploy pickles into the engine).
    assert AgentSpec.from_dict(both.to_dict()).baked_harnesses == ("claude-code", "codex")


def test_default_harness_must_be_among_the_baked_ones():
    with pytest.raises(ValueError, match="must be in harnesses"):
        AgentSpec(name="a", model="m", harness="codex", harnesses=("claude-code",))


def test_harness_selection_is_validated_against_the_image():
    baked = _spec(harnesses=("claude-code", "codex"))
    ok = apply_session_config(baked, SessionConfig(harness="codex"))
    validate_harness_choice(baked, ok)  # baked → fine

    single = _spec()  # claude-code only
    bad = apply_session_config(single, SessionConfig(harness="codex"))
    with pytest.raises(ValueError, match="not baked into this engine"):
        validate_harness_choice(single, bad)

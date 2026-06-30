"""Offline tests for the Gemini deploy/packaging builder (no GCP, no network).

Covers the pure builders (build_requirements / build_env / build_engine_config) and the
local-only skill staging path. stage_agent is intentionally NOT exercised here — it
copytrees the whole installed package, which is slow and irrelevant to these contracts.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from remote_agent_toolkit.spec import AgentSpec, SkillSource

# ``gemini.__init__`` binds the name ``deploy`` to ``backend.deploy`` (a function), which
# shadows the submodule on the package object — import the module by its full path instead.
deploy = importlib.import_module("remote_agent_toolkit.runtime.gemini.deploy")


def _spec(**overrides) -> AgentSpec:
    base = {"name": "test-agent", "model": "claude-sonnet-4-6"}
    base.update(overrides)
    return AgentSpec(**base)


def test_build_requirements_includes_base_and_spec_packages() -> None:
    reqs = deploy.build_requirements(_spec(packages=["pandas==2.2.*"]))

    # Base deps are present (a representative sampling).
    assert any(r.startswith("claude-agent-sdk") for r in reqs)
    assert any(r.startswith("google-adk") for r in reqs)
    assert any(r.startswith("a2a-sdk") for r in reqs)
    assert "uv>=0.5" in reqs
    # Spec-declared baked dep is appended.
    assert "pandas==2.2.*" in reqs
    # The toolkit ships via extra_packages, never as a requirement.
    assert "remote-agent-toolkit" not in reqs
    assert not any("remote-agent-toolkit" in r for r in reqs)


def test_build_env_secrets_and_buckets() -> None:
    spec = _spec(secrets=["ZYTE_API_KEY", "GH_TOKEN"], checkpoint=True)

    # No output_bucket, model override None -> uses spec.model, no GCS env.
    env = deploy.build_env(spec, model=None)
    assert env["IS_SANDBOX"] == "1"
    assert env["AGENT_JOBS_ROOT"]  # set (non-empty)
    assert env["CLAUDE_AGENT_MODEL"] == spec.model

    # Secrets become secret_ref dicts; env-var name == secret name (1:1).
    assert env["ZYTE_API_KEY"] == {"secret": "ZYTE_API_KEY", "version": "latest"}
    assert env["GH_TOKEN"] == {"secret": "GH_TOKEN", "version": "latest"}

    # Plaintext-vs-ref distinction: secrets are dicts, the model env is a plain str.
    assert isinstance(env["ZYTE_API_KEY"], dict)
    assert isinstance(env["CLAUDE_AGENT_MODEL"], str)

    # Without a bucket, checkpoint has nowhere to write -> no checkpoint/artifact env.
    assert "AGENT_CHECKPOINT_GCS" not in env
    assert "AGENT_ARTIFACTS_GCS" not in env

    # With a bucket, both land under it.
    env2 = deploy.build_env(spec, output_bucket="gs://bkt")
    assert env2["AGENT_CHECKPOINT_GCS"] == "gs://bkt/checkpoints"
    assert env2["AGENT_ARTIFACTS_GCS"] == "gs://bkt/artifacts"


def test_build_env_model_override() -> None:
    spec = _spec(model="claude-sonnet-4-6")
    env = deploy.build_env(spec, model="claude-haiku-4-5")
    assert env["CLAUDE_AGENT_MODEL"] == "claude-haiku-4-5"


def test_build_env_vertex_routing_default_and_api_key_opt_out() -> None:
    # Default (no ANTHROPIC_API_KEY secret) + a project => route Claude through Vertex.
    env = deploy.build_env(_spec(), project="proj", vertex_region="global")
    assert env["CLAUDE_CODE_USE_VERTEX"] == "1"
    assert env["ANTHROPIC_VERTEX_PROJECT_ID"] == "proj"
    assert env["CLOUD_ML_REGION"] == "global"

    # Listing ANTHROPIC_API_KEY as a secret opts out of Vertex routing (uses the key).
    env2 = deploy.build_env(_spec(secrets=["ANTHROPIC_API_KEY"]), project="proj")
    assert "CLAUDE_CODE_USE_VERTEX" not in env2
    assert env2["ANTHROPIC_API_KEY"] == {"secret": "ANTHROPIC_API_KEY", "version": "latest"}

    # No project => no Vertex routing env (e.g. a unit context).
    assert "CLAUDE_CODE_USE_VERTEX" not in deploy.build_env(_spec())


def test_stage_skills_local(tmp_path: Path) -> None:
    # Fake local skill dir: one folder with a SKILL.md.
    src = tmp_path / "src"
    skill = src / "my-skill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("hello skill")

    spec = _spec(skills=(SkillSource.local(str(src)),))
    dest = tmp_path / "dest"

    names = deploy.stage_skills(spec, str(dest))

    assert names == ["my-skill"]
    staged = dest / "my-skill" / "SKILL.md"
    assert staged.is_file()
    assert staged.read_text() == "hello skill"


def test_build_engine_config_shape() -> None:
    spec = _spec()
    cfg = deploy.build_engine_config(
        spec,
        project="proj",
        location="us-central1",
        staging_bucket="gs://staging",
        extra_packages=["remote_agent_toolkit"],
    )

    assert cfg["agent_framework"] == "google-adk"
    assert cfg["python_version"] == "3.12"
    assert cfg["display_name"] == spec.name
    assert isinstance(cfg["requirements"], list) and cfg["requirements"]
    assert isinstance(cfg["env_vars"], dict)
    assert cfg["extra_packages"] == ["remote_agent_toolkit"]

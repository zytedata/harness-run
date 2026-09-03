"""Offline tests for the Gemini deploy/packaging builder (no GCP, no network).

Covers the pure builders (build_requirements / build_env / build_engine_config) and the
local-only skill staging path. stage_agent is intentionally NOT exercised here — it
copytrees the whole installed package, which is slow and irrelevant to these contracts.
"""

from __future__ import annotations

from pathlib import Path

from remote_agent_toolkit.runtime.gemini import _deploy as deploy
from remote_agent_toolkit.spec import AgentSpec, SkillSource


def _spec(**overrides) -> AgentSpec:
    base = {"name": "test-agent", "model": "claude-sonnet-4-6"}
    base.update(overrides)
    return AgentSpec(**base)


def test_build_requirements_includes_base_and_spec_packages() -> None:
    reqs = deploy.build_requirements(_spec(packages=["pandas==2.2.*"]))

    # Base deps are present (a representative sampling).
    assert "claude-agent-sdk==0.2.130" in reqs
    assert any(r.startswith("google-adk") for r in reqs)
    assert any(r.startswith("a2a-sdk") for r in reqs)
    assert "uv>=0.5" in reqs
    # Spec-declared baked dep is appended.
    assert "pandas==2.2.*" in reqs
    # The toolkit ships via extra_packages, never as a requirement.
    assert "remote-agent-toolkit" not in reqs
    assert not any("remote-agent-toolkit" in r for r in reqs)


def test_constraints_file_parses_and_pins_platform_layer() -> None:
    cons = deploy.load_constraints()
    # Every pickle-coupled package must carry a pin (verify_deploy_env relies on it).
    for name in deploy._PICKLE_COUPLED:
        assert name in cons
    # Representative platform pins.
    assert "google-adk" in cons
    assert "opentelemetry-sdk" in cons


def test_build_requirements_merges_constraints() -> None:
    reqs = deploy.build_requirements(_spec())

    aip = next(r for r in reqs if r.startswith("google-cloud-aiplatform"))
    assert "[adk,agent_engines]" in aip  # extras survive the merge
    assert "==" in aip and ">=1.154" in aip  # base floor and constraint pin intersect
    assert "<2" in aip  # aiplatform 2.0 (2026-08) is excluded until validated
    # Constrained packages nothing requires directly are appended as pinned requirements.
    assert any(r.startswith("google-auth==") for r in reqs)
    # Unconstrained base deps stay untouched.
    assert "uv>=0.5" in reqs


def test_build_requirements_spec_repin_merges_into_one_line() -> None:
    # pip rejects duplicate requirement names — a spec re-pin must merge, not duplicate.
    reqs = deploy.build_requirements(_spec(packages=["jsonschema>=4"]))
    js = [r for r in reqs if r.startswith("jsonschema")]
    assert js == ["jsonschema>=4"]


def test_build_requirements_conflicting_spec_pin_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="unsatisfiable"):
        deploy.build_requirements(_spec(packages=["pydantic==2.0.0"]))


def test_build_requirements_url_requirement_passes_through() -> None:
    url = "mypkg @ https://example.com/mypkg-1.0-py3-none-any.whl"
    reqs = deploy.build_requirements(_spec(packages=[url]))
    assert url in reqs


def test_verify_deploy_env_accepts_this_venv() -> None:
    # The dev venv is held to the same contract as an operator's deploy venv.
    deploy.verify_deploy_env()


def test_verify_deploy_env_raises_on_pickle_coupled_skew(monkeypatch) -> None:
    import importlib.metadata

    import pytest

    real = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "0.0.1" if name == "cloudpickle" else real(name),
    )
    with pytest.raises(RuntimeError, match="cloudpickle"):
        deploy.verify_deploy_env()


def test_build_env_no_baked_secrets_and_buckets() -> None:
    spec = _spec(checkpoint=True, env={"FEATURE_FLAG": "on"})

    # No output_bucket, model override None -> uses spec.model, no GCS env.
    env = deploy.build_env(spec, model=None)
    assert env["IS_SANDBOX"] == "1"
    assert env["AGENT_JOBS_ROOT"]  # set (non-empty)
    assert env["CLAUDE_AGENT_MODEL"] == spec.model
    # One uvicorn worker process in the platform harness: the default (cpu_count + 1) costs
    # ~3 GiB of private memory per job container before the agent runs.
    assert env["NUM_WORKERS"] == "1"

    # No secrets are baked into the engine: every value is a plain str (non-secret machinery),
    # never a Secret Manager secret_ref dict. Secrets travel per-invocation instead.
    assert all(isinstance(v, str) for v in env.values())

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
    # Default (use_vertex=True) + a project => route Claude through Vertex (no key in env).
    env = deploy.build_env(_spec(), project="proj", vertex_region="global")
    assert env["CLAUDE_CODE_USE_VERTEX"] == "1"
    assert env["ANTHROPIC_VERTEX_PROJECT_ID"] == "proj"
    assert env["CLOUD_ML_REGION"] == "global"

    # use_vertex=False opts out (API-key mode): the key is supplied per-invocation, never baked.
    env2 = deploy.build_env(_spec(), project="proj", use_vertex=False)
    assert "CLAUDE_CODE_USE_VERTEX" not in env2
    assert "ANTHROPIC_API_KEY" not in env2  # never baked into the engine

    # No project => no Vertex routing env (e.g. a unit context).
    assert "CLAUDE_CODE_USE_VERTEX" not in deploy.build_env(_spec())


def test_build_env_pool_max_wait() -> None:
    """pool_max_wait_s must reach the worker env: the wait loop reads AGENT_POOL_MAX_WAIT_S
    from the WORKER process, and before this knob nothing plumbed it there (the var was
    unreachable — callers resorted to monkeypatching build_env)."""
    import pytest

    pool_kw = dict(warm_pool=True, pool_subscription="projects/p/subscriptions/s")

    env = deploy.build_env(_spec(), **pool_kw, pool_max_wait_s=7200)
    assert env["AGENT_POOL_SUBSCRIPTION"] == "projects/p/subscriptions/s"
    assert env["AGENT_POOL_MAX_WAIT_S"] == "7200"

    # Unset -> absent from the env, so the worker-side default (pool.DEFAULT_MAX_WAIT_S)
    # stays authoritative.
    assert "AGENT_POOL_MAX_WAIT_S" not in deploy.build_env(_spec(), **pool_kw)
    # Not a pool worker -> the knob is meaningless; never baked.
    assert "AGENT_POOL_MAX_WAIT_S" not in deploy.build_env(_spec(), pool_max_wait_s=7200)

    for bad in (0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="pool_max_wait_s"):
            deploy.build_env(_spec(), **pool_kw, pool_max_wait_s=bad)


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
    # Async-only toolkit: no standing container (an idle engine must not bill for compute).
    assert cfg["min_instances"] == 0


def test_build_engine_config_resource_limits() -> None:
    """resource_limits passthrough (2026-07-29 OOM mitigation): set → forwarded verbatim;
    omitted → absent from the config so the platform default (4 cpu / 4Gi) stays
    authoritative."""
    kw = dict(project="proj", location="us-central1", staging_bucket="gs://staging",
              extra_packages=[])
    cfg = deploy.build_engine_config(_spec(), **kw)
    assert "resource_limits" not in cfg

    limits = {"cpu": "8", "memory": "16Gi"}
    cfg = deploy.build_engine_config(_spec(), resource_limits=limits, **kw)
    assert cfg["resource_limits"] == {"cpu": "8", "memory": "16Gi"}
    assert cfg["resource_limits"] is not limits  # defensive copy


def test_build_engine_config_pool_max_wait_passthrough() -> None:
    cfg = deploy.build_engine_config(
        _spec(),
        project="proj",
        location="us-central1",
        staging_bucket="gs://staging",
        extra_packages=[],
        warm_pool=True,
        pool_subscription="projects/p/subscriptions/s",
        pool_max_wait_s=3600,
    )
    assert cfg["env_vars"]["AGENT_POOL_MAX_WAIT_S"] == "3600"


def test_validate_resource_limits_rejects_malformed() -> None:
    import pytest

    deploy.validate_resource_limits({"cpu": "4", "memory": "32Gi"})  # max memory OK
    for bad in (
        {"cpu": "4"},  # missing memory
        {"cpu": "4", "memory": "16Gi", "gpu": "1"},  # unknown key
        {"cpu": "3", "memory": "16Gi"},  # unsupported cpu value
        {"cpu": "4", "memory": "16G"},  # not Gi syntax
        {"cpu": "4", "memory": "64Gi"},  # above the 32Gi cap
        {"cpu": "4", "memory": "4096Mi"},  # Mi not supported by the platform contract
    ):
        with pytest.raises(ValueError):
            deploy.validate_resource_limits(bad)


def test_build_requirements_codex_bakes_sdk() -> None:
    reqs = deploy.build_requirements(_spec(model="gpt-5.6-luna", harness="codex"))
    assert "openai-codex==0.147.0" in reqs
    # A claude-harness engine doesn't carry the codex CLI binary.
    assert not any(r.startswith("openai-codex") for r in deploy.build_requirements(_spec()))


def test_build_env_codex_skips_vertex_routing() -> None:
    env = deploy.build_env(
        _spec(model="gpt-5.6-luna", harness="codex"), project="p", use_vertex=True
    )
    # OpenAI models have no Vertex path; the key travels per-invocation instead.
    assert "CLAUDE_CODE_USE_VERTEX" not in env
    assert "ANTHROPIC_VERTEX_PROJECT_ID" not in env
    assert "OPENAI_API_KEY" not in env  # never baked


def test_build_env_openrouter_bakes_no_credentials() -> None:
    """An OpenRouter model changes nothing at deploy time: the key travels per-invocation."""
    spec = _spec(model="openrouter/moonshotai/kimi-k3", harness="codex")
    env = deploy.build_env(spec, project="p", use_vertex=True)

    assert "OPENROUTER_API_KEY" not in env  # never baked into the engine
    assert "CLAUDE_CODE_USE_VERTEX" not in env  # no Vertex path for a codex engine
    # The model id reaches the worker through the pickled spec; this env var is the
    # informational copy, and it must carry the caller's full id (prefix included).
    assert env["CLAUDE_AGENT_MODEL"] == "openrouter/moonshotai/kimi-k3"


def test_build_requirements_openrouter_needs_only_the_codex_sdk() -> None:
    """OpenRouter is reached through codex's config, so no extra wheel is baked."""
    reqs = deploy.build_requirements(
        _spec(model="openrouter/z-ai/glm-5.3", harness="codex")
    )
    assert any(r.startswith("openai-codex") for r in reqs)
    assert not any("openrouter" in r.lower() for r in reqs)


def test_claude_openrouter_engine_bakes_vertex_but_the_harness_blanks_it(tmp_path) -> None:
    """The risky interaction on the remote runtime, pinned.

    A claude-code engine bakes `CLAUDE_CODE_USE_VERTEX=1`, and that switch outranks
    `ANTHROPIC_AUTH_TOKEN` in the CLI's auth order. So an OpenRouter turn on a deployed
    engine only works because the harness blanks it in the subprocess env.
    """
    from remote_agent_toolkit.harness.claude_code import ClaudeCodeHarness
    from remote_agent_toolkit.harness.context import RunContext

    spec = _spec(model="openrouter/moonshotai/kimi-k3")  # claude-code, the default
    baked = deploy.build_env(spec, project="p", use_vertex=True)
    assert baked["CLAUDE_CODE_USE_VERTEX"] == "1"  # deploy still bakes it
    assert "OPENROUTER_API_KEY" not in baked  # and never the key

    ctx = RunContext(
        spec=spec, prompt="hi", job_dir=tmp_path / "job", session_id="sid",
        secrets={"OPENROUTER_API_KEY": "k"}, env=baked,
    )
    env = ClaudeCodeHarness().build_options(spec, ctx).env
    assert env["CLAUDE_CODE_USE_VERTEX"] == ""  # blanked for the turn
    assert env["ANTHROPIC_AUTH_TOKEN"] == "k"


def test_openrouter_model_round_trips_through_the_baked_spec() -> None:
    """The worker rebuilds the spec from a dict; the prefix must survive that trip."""
    from remote_agent_toolkit import AgentSpec

    spec = _spec(model="openrouter/deepseek/deepseek-v4-pro", harness="codex")
    assert AgentSpec.from_dict(spec.to_dict()).model == "openrouter/deepseek/deepseek-v4-pro"


def test_deploy_submodule_import_does_not_shadow_the_deploy_function() -> None:
    """``gemini.deploy`` must stay callable after any submodule import.

    Importing a submodule binds it as an attribute of its parent package, so a module
    named ``gemini/deploy.py`` would overwrite the ``deploy`` *function* that
    ``gemini/__init__.py`` re-exports — making the second ``gemini.deploy(...)`` call
    (the first triggers the packaging module's lazy import) fail with
    ``TypeError: 'module' object is not callable``. Hence ``_deploy``.
    """
    import pkgutil

    from remote_agent_toolkit import gemini

    # Name check, not an import check: the submodules pull in google-adk/agentplatform, which
    # these offline tests deliberately don't have. A collision is decidable from names alone.
    submodules = {m.name for m in pkgutil.iter_modules(gemini.__path__)}
    assert not submodules & set(gemini.__all__), (
        f"submodule(s) {sorted(submodules & set(gemini.__all__))} collide with gemini.__all__"
    )
    assert callable(gemini.deploy)


def test_build_adk_app_pins_deploy_project() -> None:
    """The pickled AdkApp must carry the DEPLOY target, not the deployer's local default.

    AdkApp snapshots the aiplatform global config at construction and the engine worker
    exports all telemetry to the snapshotted project — a stray local default (gcloud's
    ``other-project``) baked into the pickle gave every span/metric export a persistent
    403 (2026-08-03→06 incident). build_adk_app must override whatever the environment
    left in the global config, and refuse to ship a mismatch.
    """
    import pytest

    pytest.importorskip("agentplatform")
    import google.cloud.aiplatform as aiplatform

    from remote_agent_toolkit.runtime.gemini.backend import build_adk_app

    # Simulate the incident: some earlier code path left an unrelated default behind.
    aiplatform.init(project="stray-local-default", location="europe-west1")

    app = build_adk_app(_spec(), project="deploy-target", location="us-central1")

    assert app._tmpl_attrs["project"] == "deploy-target"
    assert app._tmpl_attrs["location"] == "us-central1"

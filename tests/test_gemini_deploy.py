"""Offline tests for the sandbox deploy path: the image builder and ``deploy`` / ``get_engine``.

No Docker, no GCP: the registry check is stubbed, the platform is ``FakeSandboxProvider``,
the deploy record goes to a ``LocalBlobStore``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sandbox_fakes import FakeSandboxProvider

from remote_agent_toolkit.ports.blobstore import LocalBlobStore
from remote_agent_toolkit.runtime.gemini import _image, backend, handoff
from remote_agent_toolkit.spec import AgentSpec, SkillSource


def _spec(**overrides) -> AgentSpec:
    base = {"name": "test-agent", "model": "claude-sonnet-4-6"}
    base.update(overrides)
    return AgentSpec(**base)


# -- the image ---------------------------------------------------------------------------


def test_build_requirements_includes_base_and_spec_packages() -> None:
    reqs = _image.build_requirements(_spec(packages=["pandas==2.2.*"]))
    assert "claude-agent-sdk==0.2.130" in reqs
    assert "uv>=0.5" in reqs and "google-cloud-storage" in reqs
    assert "pandas==2.2.*" in reqs
    assert not any("remote-agent-toolkit" in r for r in reqs)  # the toolkit is COPYed, not installed
    assert not any(r.startswith(("google-adk", "google-cloud-aiplatform", "a2a-sdk", "cloudpickle")) for r in reqs)


def test_build_requirements_spec_repin_merges_into_one_line() -> None:
    reqs = _image.build_requirements(_spec(packages=["jsonschema>=4"]))
    assert [r for r in reqs if r.startswith("jsonschema")] == ["jsonschema>=4"]


def test_build_requirements_conflicting_spec_pin_raises() -> None:
    with pytest.raises(ValueError, match="unsatisfiable"):
        _image.build_requirements(_spec(packages=["claude-agent-sdk==0.1.0"]))


def test_build_requirements_url_requirement_passes_through() -> None:
    url = "mypkg @ https://example.com/mypkg-1.0-py3-none-any.whl"
    assert url in _image.build_requirements(_spec(packages=[url]))


def test_build_requirements_codex_bakes_the_sdk_only_when_baked() -> None:
    assert _image.CODEX_REQUIREMENT not in _image.build_requirements(_spec())
    assert _image.CODEX_REQUIREMENT in _image.build_requirements(_spec(harness="codex"))
    assert _image.CODEX_REQUIREMENT in _image.build_requirements(_spec(harnesses=("claude-code", "codex")))


def test_render_dockerfile_contract() -> None:
    text = _image.render_dockerfile(_spec(), has_skills=True)
    assert text.startswith("# Generated")
    assert "FROM python:3.12-slim-bookworm" in text
    assert "COPY skills /opt/toolkit/skills" in text
    assert "USER agent" in text and "EXPOSE 8080" in text
    assert 'CMD ["python", "-m", "remote_agent_toolkit.runtime.gemini.worker"]' in text
    assert "RATK_BAKED_SPEC=/opt/toolkit/spec.json" in text
    assert "COPY skills" not in _image.render_dockerfile(_spec(), has_skills=False)


def test_stage_skills_local(tmp_path: Path) -> None:
    src = tmp_path / "src"
    (src / "my-skill").mkdir(parents=True)
    (src / "my-skill" / "SKILL.md").write_text("hello skill")
    names = _image.stage_skills(_spec(skills=(SkillSource.local(str(src)),)), str(tmp_path / "dest"))
    assert names == ["my-skill"]
    assert (tmp_path / "dest" / "my-skill" / "SKILL.md").read_text() == "hello skill"


def test_stage_build_context_is_complete_and_content_addressed(tmp_path: Path) -> None:
    src = tmp_path / "src"
    (src / "sk").mkdir(parents=True)
    (src / "sk" / "SKILL.md").write_text("s")
    spec = _spec(skills=(SkillSource.local(str(src)),), packages=("httpx",))
    ctx, digest = _image.stage_build_context(spec)
    assert (ctx / "Dockerfile").is_file() and (ctx / "requirements.txt").is_file()
    assert (ctx / "remote_agent_toolkit" / "runtime" / "gemini" / "worker.py").is_file()
    assert (ctx / "skills" / "sk" / "SKILL.md").is_file()
    import json
    assert json.loads((ctx / "spec.json").read_text())["name"] == "test-agent"
    assert "httpx" in (ctx / "requirements.txt").read_text()
    assert len(digest) == 16
    # Deterministic: the same spec from the same toolkit hashes the same; a change does not.
    _, again = _image.stage_build_context(spec)
    assert again == digest
    _, other = _image.stage_build_context(_spec(packages=("httpx", "rich")))
    assert other != digest
    # No skills: no skills/ dir in the context and no COPY line.
    ctx2, _ = _image.stage_build_context(_spec())
    assert not (ctx2 / "skills").exists() and "COPY skills" not in (ctx2 / "Dockerfile").read_text()


def test_image_uri_and_default_repo() -> None:
    assert _image.default_image_repo("proj", "us-central1") == "us-central1-docker.pkg.dev/proj/ratk"
    assert _image.image_uri("r/repo/", "My Agent_1", "abc") == "r/repo/ratk-my-agent_1:abc"
    assert _image.image_uri("r/repo", "ratk-smoke", "abc") == "r/repo/ratk-smoke:abc"  # no double prefix


def test_validate_resource_limits_rejects_malformed() -> None:
    _image.validate_resource_limits({"cpu": "8", "memory": "16Gi"})
    for bad in ({"cpu": "4"}, {"cpu": "16", "memory": "8Gi"}, {"cpu": "4", "memory": "8G"},
                {"cpu": "4", "memory": "8Gi", "gpu": "1"}, {"cpu": "0", "memory": "8Gi"}):
        with pytest.raises(ValueError):
            _image.validate_resource_limits(bad)


def test_build_and_push_needs_docker(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(_image.shutil, "which", lambda name: None)
    assert _image.image_exists("r/x:1") is False
    with pytest.raises(RuntimeError, match="Docker CLI"):
        _image.build_and_push(tmp_path, "r/x:1")


# -- deploy / get_engine / list_engines -------------------------------------------------------


@pytest.fixture
def records(tmp_path, monkeypatch):
    store = LocalBlobStore(str(tmp_path / "blobs"))
    monkeypatch.setattr(handoff, "GcsBlobStore", lambda bucket, *a, **kw: store)
    monkeypatch.setattr(handoff, "ensure_handoff_lifecycle", lambda *a, **kw: True)
    return store


@pytest.fixture
def no_docker(monkeypatch):
    """The image is 'already in the registry', so deploy never shells out."""
    built = []
    monkeypatch.setattr(_image, "image_exists", lambda uri: True)
    monkeypatch.setattr(_image, "build_and_push", lambda ctx, uri, log=None: built.append(uri) or uri)
    return built


def test_deploy_creates_a_template_writes_the_record_and_returns_a_handle(records, no_docker):
    provider = FakeSandboxProvider()
    spec = _spec(packages=("httpx",))
    log = []
    engine = backend.deploy(spec, "proj", "us-central1", output_bucket="gs://out", provider=provider,
                            resource_limits={"cpu": "8", "memory": "16Gi"}, log=log.append)
    assert no_docker == []  # image existed: no build
    (tpl,) = provider.list_templates()
    assert tpl["display_name"] == "test-agent" and tpl["cpu"] == "8" and tpl["memory"] == "16Gi"
    assert tpl["image_uri"].startswith("us-central1-docker.pkg.dev/proj/ratk/ratk-test-agent:")
    assert engine.resource == tpl["name"] and engine.version == "t1" and engine.name == "test-agent"
    assert engine.spec == spec and engine._spec_known and engine._use_vertex
    assert engine._model_service_account == "ratk-model@proj.iam.gserviceaccount.com"
    record = handoff.load_deploy_record("gs://out", "test-agent", "t1", store=records)
    assert record["spec"] == spec.to_dict() and record["image"] == tpl["image_uri"]
    assert record["resource_limits"] == {"cpu": "8", "memory": "16Gi"} and record["warm_pool"] is False
    assert any("creating template" in line for line in log)


def test_deploy_builds_when_the_image_is_missing_and_reuses_a_matching_template(records, no_docker, monkeypatch):
    provider = FakeSandboxProvider()
    monkeypatch.setattr(_image, "image_exists", lambda uri: False)
    first = backend.deploy(_spec(), "proj", "us-central1", output_bucket="gs://out", provider=provider)
    assert len(no_docker) == 1 and no_docker[0].startswith("us-central1-docker.pkg.dev/proj/ratk/")
    # Same spec, same toolkit: same image, and the newest template already serves it.
    second = backend.deploy(_spec(), "proj", "us-central1", output_bucket="gs://out", provider=provider)
    assert second.resource == first.resource and len(provider.list_templates()) == 1
    # A different image (a new spec) mints a new version; the previous one is retired.
    third = backend.deploy(_spec(packages=("rich",)), "proj", "us-central1", output_bucket="gs://out",
                           provider=provider)
    third._join_background()
    assert third.resource != first.resource
    assert provider.deleted_templates == [first.resource]
    assert [t["name"] for t in provider.list_templates()] == [third.resource]


def test_deploy_with_image_skips_the_build_and_fills_the_pool(records, no_docker, monkeypatch):
    from remote_agent_toolkit.runtime.gemini import roster as roster_mod

    monkeypatch.setattr(_image, "image_exists", lambda uri: pytest.fail("no registry check with image="))
    monkeypatch.setattr(roster_mod, "GcsRosterStore",
                        lambda bucket, credentials=None: roster_mod.InMemoryRosterStore())
    provider = FakeSandboxProvider()
    engine = backend.deploy(_spec(), "proj", "l", warm_pool=True, pool_size=2, pool_max_wait_s=600,
                            output_bucket="gs://out", provider=provider, image="r/ratk-x:custom")
    assert provider.list_templates()[0]["image_uri"] == "r/ratk-x:custom"
    assert len(provider.live()) == 2 and engine.wait_until_warm(timeout=1)
    for sb in provider.list():
        assert sb["ttl_s"] == 600 + backend.DEFAULT_MAX_TURN_S  # idle life + room for a turn
    assert handoff.load_deploy_record("gs://out", "test-agent", "t1", store=records)["pool_max_wait_s"] == 600


def test_deploy_validation_happens_before_any_side_effect(records, no_docker):
    provider = FakeSandboxProvider()
    with pytest.raises(ValueError, match="warm_pool=True"):
        backend.deploy(_spec(), "p", "l", pool_max_wait_s=10, provider=provider)
    with pytest.raises(ValueError, match="positive, finite"):
        backend.deploy(_spec(), "p", "l", warm_pool=True, pool_max_wait_s=float("nan"), provider=provider)
    with pytest.raises(ValueError, match="resource_limits"):
        backend.deploy(_spec(), "p", "l", resource_limits={"cpu": "32", "memory": "8Gi"}, provider=provider)
    assert provider.list_templates() == [] and no_docker == []


def test_deploy_api_key_mode_records_no_model_account(records, no_docker):
    provider = FakeSandboxProvider()
    engine = backend.deploy(_spec(), "proj", "l", output_bucket="gs://out", provider=provider, use_vertex=False)
    assert engine._use_vertex is False and engine._model_service_account is None
    assert handoff.load_deploy_record("gs://out", "test-agent", "t1", store=records)["model_service_account"] is None


def test_get_engine_resolves_the_newest_template_and_reads_the_record(records, no_docker):
    provider = FakeSandboxProvider()
    backend.deploy(_spec(max_turns=7), "proj", "l", output_bucket="gs://out", provider=provider,
                   warm_pool=True, pool_size=0, model_service_account="custom@proj.iam.gserviceaccount.com")
    engine = backend.get_engine("test-agent", "proj", "l", output_bucket="gs://out", provider=provider)
    assert engine._spec_known and engine.spec.max_turns == 7 and engine.version == "t1"
    assert engine._warm is True and engine._model_service_account == "custom@proj.iam.gserviceaccount.com"
    assert backend.get_engine("test-agent", "proj", "l", output_bucket="gs://out", provider=provider,
                              warm_pool=False)._warm is False


def test_get_engine_without_a_record_is_addressing_only(no_docker, monkeypatch):
    monkeypatch.setattr(handoff, "load_deploy_record", lambda *a, **kw: None)
    provider = FakeSandboxProvider()
    old = provider.add_template("test-agent", create_time=1.0)
    new = provider.add_template("test-agent", create_time=2.0)
    engine = backend.get_engine("test-agent", "proj", "l", provider=provider)
    assert engine.resource == new and not engine._spec_known and engine.spec.model == ""
    assert engine.versions() == ["t2", "t1"]
    assert [r["current"] for r in engine.revisions()] == [True, False]
    pinned = backend.get_engine("test-agent", "proj", "l", version="t1", provider=provider)
    assert pinned.resource == old and pinned.version == "t1"
    with pytest.raises(LookupError, match="no version"):
        backend.get_engine("test-agent", "proj", "l", version="t9", provider=provider)
    with pytest.raises(LookupError, match="no deployed engine"):
        backend.get_engine("other", "proj", "l", provider=provider)


def test_get_engine_rejects_removed_kwargs():
    with pytest.raises(TypeError, match="SessionConfig"):
        backend.get_engine("n", "p", "l", spec=_spec())
    with pytest.raises(TypeError, match="deploy-time"):
        backend.get_engine("n", "p", "l", scoped_gcs=False)
    with pytest.raises(TypeError, match="unexpected"):
        backend.get_engine("n", "p", "l", bogus=1)


def test_list_engines_groups_templates_by_name():
    provider = FakeSandboxProvider()
    provider.add_template("a", create_time=1.0)
    newest = provider.add_template("a", create_time=2.0)
    provider.add_template("b")
    rows = {r["name"]: r for r in backend.list_engines("p", "l", provider=provider)}
    assert rows["a"]["versions"] == 2 and rows["a"]["resource"] == newest and rows["b"]["versions"] == 1

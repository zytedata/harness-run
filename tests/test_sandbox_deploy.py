"""Offline tests for the sandbox deploy path: the image builder and ``deploy`` / ``get_engine``.

No Docker, no GCP: the registry check is stubbed, the platform is ``FakeSandboxProvider``,
the deploy record goes to a ``LocalBlobStore``.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest
from sandbox_fakes import FakeSandboxProvider

from harness_run.ports.blobstore import LocalBlobStore
from harness_run.runtime.sandbox import _image, backend, handoff
from harness_run.spec import AgentSpec, SkillSource


def _spec(**overrides) -> AgentSpec:
    base = {"name": "test-agent", "model": "claude-sonnet-4-6", "harness": "claude-code"}
    base.update(overrides)
    return AgentSpec(**base)


# -- the image ---------------------------------------------------------------------------


def test_build_requirements_includes_base_and_spec_packages() -> None:
    reqs = _image.build_requirements(_spec(packages=["pandas==2.2.*"]))
    assert "claude-agent-sdk==0.2.162" in reqs
    assert "uv>=0.5" in reqs and "google-cloud-storage" in reqs
    assert "pandas==2.2.*" in reqs
    assert not any("harness-run" in r for r in reqs)  # the toolkit is COPYed, not installed
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
    assert 'CMD ["python", "-m", "harness_run.runtime.sandbox.worker"]' in text
    assert "HARNESS_RUN_BAKED_SPEC=/opt/toolkit/spec.json" in text
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
    assert (ctx / "harness_run" / "runtime" / "sandbox" / "worker.py").is_file()
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


def _modes(root: Path) -> dict[str, int]:
    return {str(p.relative_to(root)): p.stat().st_mode & 0o777 for p in root.rglob("*") if not p.is_symlink()}


def test_stage_build_context_modes_do_not_depend_on_the_umask(tmp_path: Path) -> None:
    """Field report: deploying under ``umask 077`` baked a 0600 spec.json and 0700 skill dirs
    into the image; the worker (the non-root ``agent`` user) could not read its own spec."""
    src = tmp_path / "src"
    (src / "sk").mkdir(parents=True)
    (src / "sk" / "SKILL.md").write_text("s")
    (src / "sk" / "run.sh").write_text("#!/bin/sh\n")
    (src / "sk" / "run.sh").chmod(0o700)
    spec = _spec(skills=(SkillSource.local(str(src)),))
    old = os.umask(0o077)
    try:
        ctx, digest = _image.stage_build_context(spec)
    finally:
        os.umask(old)
    modes = _modes(ctx)
    assert modes["spec.json"] == 0o644 and modes["requirements.txt"] == 0o644 and modes["Dockerfile"] == 0o644
    assert modes["skills"] == 0o755 and modes["skills/sk"] == 0o755 and modes["skills/sk/SKILL.md"] == 0o644
    assert modes["skills/sk/run.sh"] == 0o755  # executables stay executable, but become readable
    assert all(m in (0o644, 0o755) for m in modes.values()), {k: oct(m) for k, m in modes.items() if m not in (0o644, 0o755)}
    # Same content under a permissive umask: same context, same digest.
    old = os.umask(0o022)
    try:
        ctx2, digest2 = _image.stage_build_context(spec)
    finally:
        os.umask(old)
    assert digest2 == digest and _modes(ctx2) == modes
    # The exec bit is content: a script that stops being executable is a different image.
    (src / "sk" / "run.sh").chmod(0o600)
    _, digest3 = _image.stage_build_context(spec)
    assert digest3 != digest


def test_repo_of_strips_only_the_tag() -> None:
    assert _image.repo_of("us-central1-docker.pkg.dev/p/r/harness-run-x:abc") == "us-central1-docker.pkg.dev/p/r/harness-run-x"
    assert _image.repo_of("localhost:5000/harness-run-x:abc") == "localhost:5000/harness-run-x"
    assert _image.repo_of("localhost:5000/harness-run-x") == "localhost:5000/harness-run-x"


def test_resolve_digest_asks_the_registry_and_falls_back(monkeypatch) -> None:
    uri = "r.example/p/repo/harness-run-x:abc"
    sha = "sha256:" + "a" * 64
    answers: dict[str, tuple[int, str]] = {}

    def fake_run(cmd, *, capture=False):
        key = cmd[1] if cmd[1] != "buildx" else "buildx"
        code, out = answers.get(key, (1, ""))
        return subprocess.CompletedProcess(cmd, code, stdout=out, stderr="")

    monkeypatch.setattr(_image, "_run", fake_run)
    monkeypatch.setattr(_image.shutil, "which", lambda name: "/usr/bin/docker")
    # buildx imagetools answers: that is the digest.
    answers = {"buildx": (0, sha + "\n")}
    assert _image.resolve_digest(uri) == f"r.example/p/repo/harness-run-x@{sha}"
    # No buildx: a single-manifest `docker manifest inspect -v` object.
    answers = {"manifest": (0, json.dumps({"Ref": uri, "Descriptor": {"digest": sha}}))}
    assert _image.resolve_digest(uri) == f"r.example/p/repo/harness-run-x@{sha}"
    # ... or a one-entry list, as newer CLIs print it.
    answers = {"manifest": (0, json.dumps([{"Ref": uri, "Descriptor": {"digest": sha}}]))}
    assert _image.resolve_digest(uri) == f"r.example/p/repo/harness-run-x@{sha}"
    # A multi-platform index has no digest here; unparsable or failing output resolves to None.
    answers = {"manifest": (0, json.dumps([{"Descriptor": {"digest": sha}}, {"Descriptor": {"digest": sha}}]))}
    assert _image.resolve_digest(uri) is None
    answers = {"manifest": (0, "not json")}
    assert _image.resolve_digest(uri) is None
    answers = {}
    assert _image.resolve_digest(uri) is None
    monkeypatch.setattr(_image.shutil, "which", lambda name: None)
    assert _image.resolve_digest(uri) is None


def test_image_uri_and_default_repo() -> None:
    assert _image.default_image_repo("proj", "us-central1") == "us-central1-docker.pkg.dev/proj/harness-run"
    assert _image.image_uri("r/repo/", "My Agent_1", "abc") == "r/repo/harness-run-my-agent_1:abc"
    assert _image.image_uri("r/repo", "harness-run-smoke", "abc") == "r/repo/harness-run-smoke:abc"  # no double prefix


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
    monkeypatch.setattr(_image, "resolve_digest", lambda uri: None)  # the tag fallback; see the digest test
    return built


def test_deploy_pins_the_template_to_the_registry_digest(records, no_docker, monkeypatch):
    """Field report: a rebuild pushed a different image under the same content tag (the modes
    fix above; the digest could not tell), the new template named the tag, and its sandboxes
    kept running the image nodes had cached under it. The template names the digest now."""
    sha = "sha256:" + "b" * 64
    resolved = []
    monkeypatch.setattr(_image, "resolve_digest", lambda uri: resolved.append(uri) or f"{_image.repo_of(uri)}@{sha}")
    provider = FakeSandboxProvider()
    spec = _spec()
    engine = backend.deploy(spec, "proj", "us-central1", provider=provider)
    assert resolved and resolved[0].startswith("us-central1-docker.pkg.dev/proj/harness-run/harness-run-test-agent:")
    tpl = provider.templates[engine.resource]
    assert tpl["image_uri"] == f"us-central1-docker.pkg.dev/proj/harness-run/harness-run-test-agent@{sha}"
    record = handoff.load_deploy_record("gs://proj-agent-output", "test-agent", engine.version, store=records)
    assert record["image"] == tpl["image_uri"]  # the record names the same reference
    # An unchanged spec resolves to the same digest: the newest template is reused, no new version.
    again = backend.deploy(spec, "proj", "us-central1", provider=provider)
    assert again.resource == engine.resource and len(provider.templates) == 1
    # A tag the registry re-pointed at another image is a different digest: a new version.
    other = "sha256:" + "c" * 64
    monkeypatch.setattr(_image, "resolve_digest", lambda uri: f"{_image.repo_of(uri)}@{other}")
    third = backend.deploy(spec, "proj", "us-central1", provider=provider)
    assert third.resource != engine.resource and provider.templates[third.resource]["image_uri"].endswith(other)


def test_deploy_creates_a_template_writes_the_record_and_returns_a_handle(records, no_docker):
    provider = FakeSandboxProvider()
    spec = _spec(packages=("httpx",))
    log = []
    engine = backend.deploy(spec, "proj", "us-central1", output_bucket="gs://out", provider=provider,
                            resource_limits={"cpu": "8", "memory": "16Gi"}, log=log.append)
    assert no_docker == []  # image existed: no build
    (tpl,) = provider.list_templates()
    assert tpl["display_name"] == "test-agent" and tpl["cpu"] == "8" and tpl["memory"] == "16Gi"
    assert tpl["image_uri"].startswith("us-central1-docker.pkg.dev/proj/harness-run/harness-run-test-agent:")
    assert engine.resource == tpl["name"] and engine.version == "t1" and engine.name == "test-agent"
    assert engine.spec == spec and engine._spec_known and engine._use_vertex
    assert engine._model_service_account == "harness-run-model@proj.iam.gserviceaccount.com"
    record = handoff.load_deploy_record("gs://out", "test-agent", "t1", store=records)
    assert record["spec"] == spec.to_dict() and record["image"] == tpl["image_uri"]
    assert record["resource_limits"] == {"cpu": "8", "memory": "16Gi"} and record["warm_pool"] is False
    assert any("creating template" in line for line in log)


def test_deploy_builds_when_the_image_is_missing_and_reuses_a_matching_template(records, no_docker, monkeypatch):
    provider = FakeSandboxProvider()
    monkeypatch.setattr(_image, "image_exists", lambda uri: False)
    first = backend.deploy(_spec(), "proj", "us-central1", output_bucket="gs://out", provider=provider)
    assert len(no_docker) == 1 and no_docker[0].startswith("us-central1-docker.pkg.dev/proj/harness-run/")
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


def test_deploy_rejects_removed_and_unknown_kwargs_before_any_side_effect(records, no_docker):
    # PR #84 review, E2: deploy() swallowed **kwargs while get_engine() rejected them, so a
    # migrated deploy script passing service_account= or new_engine= believed it set something.
    provider = FakeSandboxProvider()
    with pytest.raises(TypeError, match=r"service_account=.*removed with the sandbox runtime"):
        backend.deploy(_spec(), "proj", "l", image="img:1", provider=provider, service_account="x@p.iam")
    with pytest.raises(TypeError, match=r"new_engine=.*removed"):
        backend.deploy(_spec(), "proj", "l", image="img:1", provider=provider, new_engine=True)
    with pytest.raises(TypeError, match=r"unexpected keyword argument\(s\) \['pool_sizee'\]"):
        backend.deploy(_spec(), "proj", "l", image="img:1", provider=provider, pool_sizee=3)
    assert provider.list_templates() == [] and no_docker == []


def test_deploy_reuses_a_template_only_when_its_egress_flag_matches(records, no_docker):
    # PR #84 review, C: redeploying the same image with internet_access=False used to return
    # the existing template with egress on and report a no-op.
    provider = FakeSandboxProvider()
    on = backend.deploy(_spec(), "proj", "l", image="img:1", output_bucket="gs://out", provider=provider,
                        internet_access=True)
    off = backend.deploy(_spec(), "proj", "l", image="img:1", output_bucket="gs://out", provider=provider,
                         internet_access=False)
    off._join_background()
    assert off.resource != on.resource
    assert provider.templates[off.resource]["internet_access"] is False
    record = handoff.load_deploy_record("gs://out", "test-agent", off.version, store=records)
    assert record["internet_access"] is False  # the deploy record says what was asked
    again = backend.deploy(_spec(), "proj", "l", image="img:1", output_bucket="gs://out", provider=provider,
                           internet_access=False)
    assert again.resource == off.resource  # a matching flag is a no-op again
    # A listing that does not report the flag is trusted for the default (on) only.
    provider.templates[again.resource]["internet_access"] = None
    assert backend.deploy(_spec(), "proj", "l", image="img:1", output_bucket="gs://out", provider=provider,
                          internet_access=False).resource != again.resource


def test_deploy_with_image_skips_the_build_and_fills_the_pool(records, no_docker, monkeypatch):
    from harness_run.runtime.sandbox import roster as roster_mod

    monkeypatch.setattr(_image, "image_exists", lambda uri: pytest.fail("no registry check with image="))
    monkeypatch.setattr(roster_mod, "GcsRosterStore",
                        lambda bucket, credentials=None: roster_mod.InMemoryRosterStore())
    provider = FakeSandboxProvider()
    engine = backend.deploy(_spec(), "proj", "l", warm_pool=True, pool_size=2, pool_max_wait_s=600,
                            output_bucket="gs://out", provider=provider, image="r/harness-run-x:custom")
    assert provider.list_templates()[0]["image_uri"] == "r/harness-run-x:custom"
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
    # The gap is announced at lookup, where the operator looks (field report on #84: a deploy
    # interrupted mid-poll left an ACTIVE template with no record; turns failed days later).
    with pytest.warns(UserWarning, match="version t2 has no deploy record under gs://proj-agent-output.*re-run sandbox.deploy"):
        engine = backend.get_engine("test-agent", "proj", "l", provider=provider)
    assert engine.resource == new and not engine._spec_known and engine.spec.model == ""
    assert engine.versions() == ["t2", "t1"]
    assert [r["current"] for r in engine.revisions()] == [True, False]
    with pytest.warns(UserWarning, match="version t1 has no deploy record"):
        pinned = backend.get_engine("test-agent", "proj", "l", version="t1", provider=provider)
    assert pinned.resource == old and pinned.version == "t1"
    with pytest.raises(LookupError, match="no version"):
        backend.get_engine("test-agent", "proj", "l", version="t9", provider=provider)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no bucket to look in: nothing to warn about
        backend.get_engine("test-agent", provider=provider)
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


def test_get_engine_skips_failed_and_provisioning_templates(records, no_docker):
    # A deploy whose template ended FAILED (field note on #84) must not become "the newest version".
    provider = FakeSandboxProvider()
    backend.deploy(_spec(), "proj", "l", output_bucket="gs://out", provider=provider)
    failed = provider.add_template("test-agent", image="img:2", create_time=time.time() + 100)
    provider.templates[failed]["state"] = "FAILED"
    with pytest.warns(UserWarning, match="not ACTIVE; resolving to t1"):
        engine = backend.get_engine("test-agent", "proj", "l", output_bucket="gs://out", provider=provider)
    assert engine.version == "t1" and engine._spec_known
    assert [r["state"] for r in engine.revisions()] == ["FAILED", "ACTIVE"]
    provider.templates[engine.resource]["state"] = "FAILED"
    with pytest.raises(LookupError, match="no ACTIVE version: t2 is FAILED, t1 is FAILED"):
        backend.get_engine("test-agent", "proj", "l", output_bucket="gs://out", provider=provider)


def test_deploy_reports_template_progress_and_retires_failed_versions(records, no_docker, monkeypatch):
    provider = FakeSandboxProvider()
    failed = provider.add_template("test-agent", image="img:0", create_time=1.0)
    provider.templates[failed]["state"] = "FAILED"
    lines = []
    monkeypatch.setattr(backend.time, "sleep", lambda s: None)
    engine = backend.deploy(_spec(), "proj", "l", output_bucket="gs://out", provider=provider, log=lines.append)
    engine._join_background()
    assert any("PROVISIONING" in line for line in lines) and any("is ACTIVE" in line for line in lines)
    assert failed in provider.deleted_templates  # the failed attempt is retired with the previous versions

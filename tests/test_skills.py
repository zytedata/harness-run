"""Tests for skills.provision (local sources + last-wins merge)."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness_run.skills import provision
from harness_run.spec import SkillSource


def _make_skill(root: Path, name: str, body: str) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(body)
    return root


def test_provision_local_single_source(tmp_path) -> None:
    src = _make_skill(tmp_path / "src", "my-skill", "hello skill")
    dest = tmp_path / "job"

    staged = provision((SkillSource.local(str(src)),), str(dest))

    assert staged.names == ["my-skill"]
    assert staged.sources == [{"kind": "local", "path": str(src), "skills": ["my-skill"]}]
    staged = dest / ".claude" / "skills" / "my-skill" / "SKILL.md"
    assert staged.read_text() == "hello skill"


def test_provision_last_wins_merge(tmp_path) -> None:
    base = _make_skill(tmp_path / "base", "shared", "base version")
    override = _make_skill(tmp_path / "override", "shared", "override version")

    dest = tmp_path / "job"
    staged = provision(
        (SkillSource.local(str(base)), SkillSource.local(str(override))),
        str(dest),
    )

    assert staged.names == ["shared"]
    staged = dest / ".claude" / "skills" / "shared" / "SKILL.md"
    assert staged.read_text() == "override version"  # later source wins


def test_provision_empty_source_raises(tmp_path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        provision((SkillSource.local(str(empty)),), str(tmp_path / "job"))


def test_provision_builtin_not_implemented(tmp_path) -> None:
    with pytest.raises(NotImplementedError):
        provision((SkillSource.builtin("bundle"),), str(tmp_path / "job"))


# -- git sources: provenance, pins, auth, token hygiene ---------------------------------


def _git(repo: Path, *args: str) -> str:
    import subprocess

    return subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=str(repo), check=True, capture_output=True, text=True,
    ).stdout.strip()


def _skill_repo(tmp_path: Path) -> tuple[Path, str, str]:
    """A local skill repository with two commits; returns (repo, first commit, HEAD)."""
    repo = tmp_path / "skills-repo"
    (repo / "skills" / "wkdb").mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    (repo / "skills" / "wkdb" / "SKILL.md").write_text("v1")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "v1")
    first = _git(repo, "rev-parse", "HEAD")
    _git(repo, "tag", "v1")
    (repo / "skills" / "wkdb" / "SKILL.md").write_text("v2")
    _git(repo, "commit", "-qam", "v2")
    return repo, first, _git(repo, "rev-parse", "HEAD")


def test_provision_git_source_records_the_resolved_commit_and_drops_the_clone(tmp_path, monkeypatch) -> None:
    import tempfile

    repo, _first, head = _skill_repo(tmp_path)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()

    staged = provision((SkillSource.git(str(repo)),), str(tmp_path / "job"))

    assert staged.names == ["wkdb"]
    assert staged.sources == [{
        "kind": "git", "url": str(repo), "ref": None, "commit": head, "subdir": "skills", "skills": ["wkdb"],
    }]
    assert (tmp_path / "job" / ".claude" / "skills" / "wkdb" / "SKILL.md").read_text() == "v2"
    # The provenance is written next to the staged folders (a restored workspace reports it) ...
    from harness_run.skills import SOURCES_FILE, read_sources

    assert read_sources(tmp_path / "job" / ".claude" / "skills") == staged.sources
    assert (tmp_path / "job" / ".claude" / "skills" / SOURCES_FILE).is_file()
    # ... and the temporary clone (whose config would hold a token) is gone.
    assert list((tmp_path / "tmp").iterdir()) == []


def test_provision_git_source_pins_a_tag_and_a_commit(tmp_path) -> None:
    repo, first, head = _skill_repo(tmp_path)

    by_tag = provision((SkillSource.git(str(repo), ref="v1"),), str(tmp_path / "a"))
    assert by_tag.sources[0]["commit"] == first and by_tag.sources[0]["ref"] == "v1"
    assert (tmp_path / "a" / ".claude" / "skills" / "wkdb" / "SKILL.md").read_text() == "v1"

    by_sha = provision((SkillSource.git(str(repo), ref=first),), str(tmp_path / "b"))  # not a branch
    assert by_sha.sources[0]["commit"] == first
    assert (tmp_path / "b" / ".claude" / "skills" / "wkdb" / "SKILL.md").read_text() == "v1"

    with pytest.raises(RuntimeError, match="git clone failed"):
        provision((SkillSource.git(str(repo), ref="no-such-ref"),), str(tmp_path / "c"))


def test_provision_git_source_with_a_missing_subdir_is_a_config_error(tmp_path) -> None:
    repo, _first, _head = _skill_repo(tmp_path)
    with pytest.raises(FileNotFoundError, match="under other/"):
        provision((SkillSource.git(str(repo), subdir="other"),), str(tmp_path / "job"))


def test_provision_git_auth_names_a_secret_the_turn_must_supply(tmp_path, monkeypatch) -> None:
    from harness_run.integrations import git as git_mod

    seen = {}

    def fake_clone(url, ref=None, dest=None, token=None, auth_user=None):
        seen.update(url=url, ref=ref, token=token, auth_user=auth_user)
        (Path(dest) / "skills" / "s").mkdir(parents=True)
        (Path(dest) / "skills" / "s" / "SKILL.md").write_text("x")
        return Path(dest)

    monkeypatch.setattr(git_mod, "clone", fake_clone)
    monkeypatch.setattr(git_mod, "head_commit", lambda d: "abc123")
    source = SkillSource.git("https://git.example/org/skills", ref="1.0", auth="SKILLS_TOKEN", auth_user="bot")

    # Named but not supplied: loud failure, never an anonymous attempt.
    with pytest.raises(ValueError, match="SKILLS_TOKEN"):
        provision((source,), str(tmp_path / "a"), secrets={})
    assert not seen

    staged = provision((source,), str(tmp_path / "b"), secrets={"SKILLS_TOKEN": "s3cret"})
    assert seen == {"url": source.url, "ref": "1.0", "token": "s3cret", "auth_user": "bot"}
    assert staged.sources[0]["commit"] == "abc123"
    # The record carries the source and its resolution, never the token.
    assert "s3cret" not in (tmp_path / "b" / ".claude" / "skills" / ".harness-run-sources.json").read_text()


def test_provision_sources_record_overrides_the_computed_provenance(tmp_path) -> None:
    """The worker's baked fast path copies from the image's flat dir but reports the deploy's record."""
    from harness_run.skills import read_sources

    src = _make_skill(tmp_path / "baked", "my-skill", "hello skill")
    deploy_record = [{"kind": "git", "url": "https://h/s", "ref": "0.2", "commit": "f" * 40,
                      "subdir": "skills", "skills": ["my-skill"]}]
    staged = provision((SkillSource.local(str(src)),), str(tmp_path / "job"), sources_record=deploy_record)
    assert staged.sources == deploy_record
    assert read_sources(tmp_path / "job" / ".claude" / "skills") == deploy_record
    assert read_sources(tmp_path / "nowhere") is None and read_sources(None) is None


def test_build_result_exposes_the_stamped_skill_sources() -> None:
    from harness_run.events import AgentEvent
    from harness_run.runtime._run import build_result
    from harness_run.spec import AgentSpec

    spec = AgentSpec(harness="claude-code", name="a", model="m")
    sources = [{"kind": "git", "url": "https://h/s", "ref": "main", "commit": "a" * 40, "subdir": "skills",
                "skills": ["wkdb"]}]
    ev = AgentEvent(kind="result", summary="done", raw={"subtype": "success", "skill_sources": sources})
    result, _ = build_result(ev, "sid", spec)
    assert result.skill_sources == sources
    bare, _ = build_result(AgentEvent(kind="result", summary="done", raw={"subtype": "success"}), "sid", spec)
    assert bare.skill_sources is None

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

    names = provision((SkillSource.local(str(src)),), str(dest))

    assert names == ["my-skill"]
    staged = dest / ".claude" / "skills" / "my-skill" / "SKILL.md"
    assert staged.read_text() == "hello skill"


def test_provision_last_wins_merge(tmp_path) -> None:
    base = _make_skill(tmp_path / "base", "shared", "base version")
    override = _make_skill(tmp_path / "override", "shared", "override version")

    dest = tmp_path / "job"
    names = provision(
        (SkillSource.local(str(base)), SkillSource.local(str(override))),
        str(dest),
    )

    assert names == ["shared"]
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

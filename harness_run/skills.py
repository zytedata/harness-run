"""Discover + provision skills from ``SkillSource[]`` (DESIGN.md §8, §12).

Both harnesses discover Agent Skills from ``<name>/SKILL.md`` folders under a per-harness
project path in the cwd — ``.claude/skills`` for Claude Code (with ``setting_sources``
including ``"project"``), ``.agents/skills`` for Codex (its repo-level discovery path;
the same SKILL.md format). We therefore resolve each ``SkillSource`` (git clone / local
path / builtin) to a directory of skill folders and copy each folder into the per-job cwd
before starting the engine. Lifted from the PoC ``skills.py``; git is used via
``integrations.git`` lazily.

Merge policy for name collisions across sources (the open DESIGN.md §12 question): **last
wins**. Sources are processed in order, and a later source's skill of the same name
overwrites an earlier one (``shutil.copytree`` after ``rmtree``). This makes a base + extra
layering predictable: list the base first, overrides last.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .spec import SkillSource

# Where each harness discovers project-level skills, relative to the agent cwd.
_SKILLS_SUBDIR = {"claude-code": ".claude/skills", "codex": ".agents/skills"}


def skills_subdir(harness: str) -> str:
    """The cwd-relative skills directory for ``harness``.

    Raises for an unknown harness rather than assuming one: silently using Claude's
    layout would stage skills where the running CLI never looks for them, and the agent
    would start with none of the skills the spec declared.
    """
    try:
        return _SKILLS_SUBDIR[harness]
    except KeyError:
        raise ValueError(
            f"no skills layout known for harness {harness!r} "
            f"(expected one of {sorted(_SKILLS_SUBDIR)})"
        ) from None


def discover_skill_names(src: Path) -> list[str]:
    """Return the names of skills available in ``src`` (dirs containing a SKILL.md)."""
    if not src.is_dir():
        return []
    return sorted(
        p.name
        for p in src.iterdir()
        if p.is_dir() and (p / "SKILL.md").is_file()
    )


def _resolve_source_dir(source: SkillSource) -> Path:
    """Resolve a ``SkillSource`` to a local directory of skill folders."""
    if source.kind == "local":
        if not source.path:
            raise ValueError("local SkillSource requires a path")
        return Path(source.path)
    if source.kind == "git":
        if not source.url:
            raise ValueError("git SkillSource requires a url")
        from .integrations import git  # lazy: subprocess-backed, no third-party dep

        clone_dir = git.clone(source.url, ref=source.ref)
        return clone_dir / source.subdir
    if source.kind == "builtin":
        raise NotImplementedError("builtin skill bundles are not shipped yet")
    raise ValueError(f"unknown SkillSource kind: {source.kind!r}")


def provision(
    sources: tuple[SkillSource, ...], dest: str, subdir: str = ".claude/skills"
) -> list[str]:
    """Resolve & stage ``sources`` into ``dest/<subdir>/``; return the staged names.

    ``subdir`` is the harness's discovery path (see :func:`skills_subdir`). For each
    source, every skill folder (a dir containing ``SKILL.md``) is copied to
    ``dest/<subdir>/<name>/`` (overwriting any existing). On name collision across
    sources, later sources win (see module docstring). Raises ``FileNotFoundError`` if a
    source stages zero skills — a source that yields nothing is a config error.

    Returns the merged, sorted list of provisioned skill names.
    """
    dest_root = Path(dest) / subdir
    dest_root.mkdir(parents=True, exist_ok=True)

    provisioned: set[str] = set()
    for source in sources:
        src_dir = _resolve_source_dir(source)
        names = discover_skill_names(src_dir)
        if not names:
            raise FileNotFoundError(
                f"No skills found under {src_dir} (expected <name>/SKILL.md folders)"
            )
        for name in names:
            dst = dest_root / name
            if dst.exists():
                shutil.rmtree(dst)  # last source wins on collision
            shutil.copytree(src_dir / name, dst)
            provisioned.add(name)

    return sorted(provisioned)

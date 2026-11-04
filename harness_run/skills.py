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

**Provenance.** Staging records what each source resolved to — for a git source the
commit its ``ref`` pointed at — as :attr:`StagedSkills.sources`, and writes the same list
to :data:`SOURCES_FILE` next to the staged folders, so a workspace restored from a
checkpoint still tells which skill versions it carries. With floating refs this is the
only way to say afterwards which skill version a run used.

**Private sources.** A git source's ``auth`` names a per-invocation secret (``secrets``);
the token authenticates the clone only. The clone itself is a temporary directory that
is removed once the skill folders are copied out, so the token (which git keeps in the
clone's ``.git/config``) never reaches the agent's workspace.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

if TYPE_CHECKING:
    from .spec import SkillSource

# Where each harness discovers project-level skills, relative to the agent cwd.
_SKILLS_SUBDIR = {"claude-code": ".claude/skills", "codex": ".agents/skills"}

# The provenance record written next to the staged skill folders (a file, so the harnesses'
# folder-based discovery never sees it).
SOURCES_FILE = ".harness-run-sources.json"


@dataclass(frozen=True)
class StagedSkills:
    """What :func:`provision` staged: the merged skill names and each source's provenance.

    ``sources`` has one entry per declared source, in order: for a git source ``{"kind":
    "git", "url", "ref", "commit", "subdir", "skills": [...]}`` (``commit`` is the full id
    the ref resolved to), for a local one ``{"kind": "local", "path", "skills": [...]}``.
    """

    names: list[str]
    sources: list[dict]


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


def _git_token(source: SkillSource, secrets: Mapping[str, str] | None) -> str | None:
    """The token a git source's ``auth`` names; a named-but-missing secret is a config error."""
    if not source.auth:
        return None
    token = (secrets or {}).get(source.auth)
    if not token:
        raise ValueError(
            f"SkillSource {source.url} names the secret {source.auth!r} for its clone, "
            "but run(secrets=...) did not supply it"
        )
    return token


def _copy_skills(src_dir: Path, dest_root: Path, provisioned: set[str], what: str) -> list[str]:
    names = discover_skill_names(src_dir)
    if not names:
        raise FileNotFoundError(f"No skills found in {what} (expected <name>/SKILL.md folders)")
    for name in names:
        dst = dest_root / name
        if dst.exists():
            shutil.rmtree(dst)  # last source wins on collision
        shutil.copytree(src_dir / name, dst)
        provisioned.add(name)
    return names


def stage_sources(
    sources: Sequence[SkillSource],
    dest_root: Path,
    secrets: Mapping[str, str] | None = None,
    *,
    sources_record: list[dict] | None = None,
) -> StagedSkills:
    """Resolve ``sources`` and copy every skill folder FLAT into ``dest_root``.

    The deploy-time image bake and the run-time workspace staging share this: the layout
    is ``dest_root/<name>/SKILL.md`` plus :data:`SOURCES_FILE`. ``sources_record``
    replaces the computed provenance in that file (the worker's baked fast path copies
    from the image's flat dir but records what the deploy resolved). Raises
    ``FileNotFoundError`` if a source stages zero skills, ``ValueError`` for a missing
    ``auth`` secret and ``RuntimeError`` for a failed clone — a source that yields nothing
    is a config error, never silently skipped.
    """
    dest_root.mkdir(parents=True, exist_ok=True)
    provisioned: set[str] = set()
    record: list[dict] = []
    for source in sources:
        if source.kind == "local":
            if not source.path:
                raise ValueError("local SkillSource requires a path")
            names = _copy_skills(Path(source.path), dest_root, provisioned, source.path)
            record.append({"kind": "local", "path": source.path, "skills": names})
        elif source.kind == "git":
            if not source.url:
                raise ValueError("git SkillSource requires a url")
            from .integrations import git  # lazy: subprocess-backed, no third-party dep

            token = _git_token(source, secrets)
            clone_dir = Path(tempfile.mkdtemp(prefix="harness-run-skills-"))
            try:
                git.clone(source.url, ref=source.ref, dest=clone_dir, token=token,
                          auth_user=source.auth_user)
                commit = git.head_commit(clone_dir)
                what = f"{source.url}" + (f"@{source.ref}" if source.ref else "") + f" under {source.subdir}/"
                names = _copy_skills(clone_dir / source.subdir, dest_root, provisioned, what)
            finally:
                shutil.rmtree(clone_dir, ignore_errors=True)  # the clone's config holds the token
            record.append({"kind": "git", "url": source.url, "ref": source.ref, "commit": commit,
                           "subdir": source.subdir, "skills": names})
        elif source.kind == "builtin":
            raise NotImplementedError("builtin skill bundles are not shipped yet")
        else:
            raise ValueError(f"unknown SkillSource kind: {source.kind!r}")
    if sources_record is not None:
        record = list(sources_record)
    write_sources(dest_root, record)
    return StagedSkills(names=sorted(provisioned), sources=record)


def provision(
    sources: Sequence[SkillSource],
    dest: str,
    subdir: str = ".claude/skills",
    secrets: Mapping[str, str] | None = None,
    *,
    sources_record: list[dict] | None = None,
) -> StagedSkills:
    """Resolve & stage ``sources`` into ``dest/<subdir>/``; return what was staged.

    ``subdir`` is the harness's discovery path (see :func:`skills_subdir`). For each
    source, every skill folder (a dir containing ``SKILL.md``) is copied to
    ``dest/<subdir>/<name>/`` (overwriting any existing). On name collision across
    sources, later sources win (see module docstring). ``secrets`` resolves the git
    sources' ``auth``. See :func:`stage_sources` for the failure modes.
    """
    return stage_sources(sources, Path(dest) / subdir, secrets, sources_record=sources_record)


def write_sources(skills_dir: Path, record: list[dict]) -> None:
    """Write the provenance ``record`` as :data:`SOURCES_FILE` under ``skills_dir``."""
    Path(skills_dir).mkdir(parents=True, exist_ok=True)
    (Path(skills_dir) / SOURCES_FILE).write_text(json.dumps(record, indent=1) + "\n")


def read_sources(skills_dir: Path | None) -> list[dict] | None:
    """The provenance record staged under ``skills_dir``, or ``None`` when there is none.

    The file lives in the workspace, which the agent can edit: what a restored workspace
    reports is as trustworthy as the agent. The first turn's ``workspace_ready`` event
    carries the resolution the worker made itself.
    """
    if skills_dir is None:
        return None
    path = Path(skills_dir) / SOURCES_FILE
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return record if isinstance(record, list) else None

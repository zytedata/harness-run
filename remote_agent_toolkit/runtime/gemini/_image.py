"""The sandbox image: Dockerfile generation, build and push (DESIGN.md §13.1).

``gemini.deploy`` turns a spec into a container image the platform runs as a custom
container: Debian/glibc, Python 3.12, git, the toolkit with its harness CLIs (the
``claude-agent-sdk`` wheel ships the glibc ``claude`` binary; ``openai-codex`` bundles the
Codex CLI when that harness is baked), the resolved skills, ``spec.packages``, and the
baked spec itself. The image runs the worker (``worker.py``) as a non-root user on port
8080. ``dev/Dockerfile`` and the production image are the same contract.

The **image tag is a content digest** of everything that goes into the image (the
toolkit's own source, the spec, the requirements, the Dockerfile) — a deploy of an
unchanged spec from an unchanged toolkit finds its image already in the registry and
skips the build. Docker is driven through the CLI (``docker build`` / ``docker push``);
the registry login is the operator's (``gcloud auth configure-docker``).

Stdlib-only at import; ``packaging`` (a toolkit dependency) is imported lazily.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...spec import AgentSpec

# What every sandbox image installs besides the toolkit itself.
BASE_REQUIREMENTS: tuple[str, ...] = (
    "claude-agent-sdk==0.2.130",  # keep in lockstep with pyproject (the [local] extra)
    "google-cloud-storage",  # the event mirror / checkpoints on the run-scoped token
    "google-auth",
    "uv>=0.5",  # the agent installs more at run time with uv
    "pyyaml",
    "jsonschema",
)
CODEX_REQUIREMENT = "openai-codex==0.147.0"  # keep in lockstep with pyproject

PYTHON_VERSION = "3.12"
IMAGE_USER = "agent"
TOOLKIT_DIR = "/opt/toolkit"
WORKSPACE_ROOT = "/workspace"
DEFAULT_REPO_ID = "ratk"

# Where a build can be parametrized for tests: the CLI the module shells out to.
DOCKER = "docker"


def default_image_repo(project: str, location: str) -> str:
    """The Artifact Registry Docker repo ``deploy`` pushes to when ``image_repo=`` is omitted."""
    return f"{location}-docker.pkg.dev/{project}/{DEFAULT_REPO_ID}"


def validate_resource_limits(resource_limits: dict[str, str]) -> None:
    """Fail fast on a malformed ``resource_limits`` (before the build).

    Exactly the keys ``cpu`` and ``memory``; cpu a whole number of vCPUs from 1 to 8 (the
    platform's ceiling, measured 2026-09-11: "Request CPU exceeds maximum allowed: 8.0
    vCPU"), memory ``<n>Gi``. Values are strings (Kubernetes quantity syntax).
    """
    if set(resource_limits) != {"cpu", "memory"}:
        raise ValueError(
            f"resource_limits must have exactly the keys 'cpu' and 'memory'; "
            f"got {sorted(resource_limits)}"
        )
    cpu, memory = str(resource_limits["cpu"]), str(resource_limits["memory"])
    if not (cpu.isdigit() and 1 <= int(cpu) <= 8):
        raise ValueError(f"resource_limits cpu must be a whole number of vCPUs, 1..8; got {cpu!r}")
    if not (memory.endswith("Gi") and memory[:-2].isdigit() and 1 <= int(memory[:-2]) <= 64):
        raise ValueError(f"resource_limits memory must be '1Gi'..'64Gi'; got {memory!r}")


def _unsatisfiable(req: Any) -> bool:
    """True when a merged specifier set excludes one of its own ``==``/``===`` pins."""
    from packaging.version import InvalidVersion, Version

    for s in req.specifier:
        if s.operator in ("==", "===") and not s.version.endswith(".*"):
            try:
                version = Version(s.version)
            except InvalidVersion:
                continue
            if not req.specifier.contains(version, prereleases=True):
                return True
    return False


def build_requirements(spec: AgentSpec) -> list[str]:
    """The image's ``requirements.txt`` lines: the base layer, then ``spec.packages``.

    A spec re-pin of a base dependency merges into one line (pip rejects duplicate names);
    an unsatisfiable merge (a spec pin the base excludes) raises here, not minutes into
    the build. Non-PEP 508 lines and direct URLs pass through verbatim.
    """
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.utils import canonicalize_name

    extra = (CODEX_REQUIREMENT,) if "codex" in spec.baked_harnesses else ()
    entries: list = []
    by_name: dict = {}
    for line in (*BASE_REQUIREMENTS, *extra, *spec.packages):
        try:
            req = Requirement(line)
        except InvalidRequirement:
            entries.append(line)
            continue
        if req.url:
            entries.append(line)
            continue
        key = canonicalize_name(req.name)
        if key in by_name:
            by_name[key].specifier &= req.specifier
            by_name[key].extras |= req.extras
        else:
            by_name[key] = req
            entries.append(req)
    for req in by_name.values():
        if _unsatisfiable(req):
            raise ValueError(
                f"unsatisfiable requirement {str(req)!r} after merging spec.packages with "
                "the image's base requirements; align the spec's pin with the base"
            )
    return [str(e) for e in entries]


def render_dockerfile(spec: AgentSpec, *, has_skills: bool) -> str:
    """The Dockerfile of ``spec``'s image (pure; the build context layout is :func:`stage_build_context`)."""
    lines = [
        "# Generated by remote-agent-toolkit (runtime/gemini/_image.py) — do not edit.",
        f"FROM python:{PYTHON_VERSION}-slim-bookworm",
        "RUN apt-get update \\",
        "    && apt-get install -y --no-install-recommends git ca-certificates curl procps build-essential \\",
        "    && rm -rf /var/lib/apt/lists/*",
        f"COPY requirements.txt {TOOLKIT_DIR}/requirements.txt",
        f"RUN pip install --no-cache-dir -r {TOOLKIT_DIR}/requirements.txt",
        f"COPY remote_agent_toolkit {TOOLKIT_DIR}/remote_agent_toolkit",
    ]
    if has_skills:
        lines.append(f"COPY skills {TOOLKIT_DIR}/skills")
    lines += [
        f"COPY spec.json {TOOLKIT_DIR}/spec.json",
        f"RUN useradd --uid 1000 --create-home --shell /bin/bash {IMAGE_USER} \\",
        f"    && mkdir -p {WORKSPACE_ROOT} && chown -R {IMAGE_USER}:{IMAGE_USER} {WORKSPACE_ROOT}",
        f"USER {IMAGE_USER}",
        f"ENV HOME=/home/{IMAGE_USER} \\",
        "    PORT=8080 \\",
        f"    PYTHONPATH={TOOLKIT_DIR} \\",
        f"    RATK_WORKSPACE_ROOT={WORKSPACE_ROOT} \\",
        f"    RATK_BAKED_SPEC={TOOLKIT_DIR}/spec.json \\",
        "    PYTHONUNBUFFERED=1 \\",
        "    PYTHONDONTWRITEBYTECODE=1",
        f"WORKDIR {WORKSPACE_ROOT}",
        "EXPOSE 8080",
        'CMD ["python", "-m", "remote_agent_toolkit.runtime.gemini.worker"]',
        "",
    ]
    return "\n".join(lines)


def stage_skills(spec: AgentSpec, dest_dir: str) -> list[str]:
    """Resolve ``spec.skills`` into a FLAT ``dest_dir`` of skill folders; return the names.

    Each staged entry is a directory containing ``SKILL.md``. The flat layout is what the
    worker's baked-skills fast path provisions from (``worker._find_baked_skills``). Per
    ``SkillSource.kind``: ``local`` copies from ``path``; ``git`` clones ``url`` at ``ref``
    and reads ``<clone>/<subdir>``; ``builtin`` is not shipped yet. Last source wins on a
    name collision (matches ``skills.provision``).
    """
    from ...skills import discover_skill_names

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    staged: set[str] = set()
    for source in spec.skills:
        if source.kind == "local":
            if not source.path:
                raise ValueError("local SkillSource requires a path")
            src_dir = Path(source.path)
        elif source.kind == "git":
            if not source.url:
                raise ValueError("git SkillSource requires a url")
            from ...integrations import git

            clone_dir = git.clone(source.url, ref=source.ref)
            src_dir = clone_dir / source.subdir
        elif source.kind == "builtin":
            raise NotImplementedError("builtin skills not shipped yet")
        else:
            raise ValueError(f"unknown SkillSource kind: {source.kind!r}")
        for name in discover_skill_names(src_dir):
            dst = dest / name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src_dir / name, dst)
            staged.add(name)
    return sorted(staged)


def _toolkit_package_dir() -> Path:
    import remote_agent_toolkit

    return Path(remote_agent_toolkit.__file__).resolve().parent


def _tree_digest(root: Path, hasher: Any) -> None:
    """Feed every file under ``root`` (sorted, relative names + bytes) into ``hasher``."""
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        hasher.update(str(path.relative_to(root)).encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")


def stage_build_context(spec: AgentSpec) -> tuple[Path, str]:
    """Build the Docker context for ``spec`` in a fresh temp dir; return ``(dir, digest)``.

    The context holds ``Dockerfile``, ``requirements.txt``, ``spec.json``, the toolkit
    package (copied from this venv — the image runs THIS toolkit revision) and, when the
    spec declares skills, the flat ``skills/`` dir. ``digest`` is a 16-hex content hash of
    the whole context: the image tag.
    """
    stage = Path(tempfile.mkdtemp(prefix="ratk-image-"))
    shutil.copytree(
        _toolkit_package_dir(), stage / "remote_agent_toolkit",
        ignore=shutil.ignore_patterns("__pycache__", ".venv", "*.pyc"),
    )
    has_skills = bool(stage_skills(spec, str(stage / "skills")))
    if not has_skills:
        shutil.rmtree(stage / "skills", ignore_errors=True)
    (stage / "requirements.txt").write_text("\n".join(build_requirements(spec)) + "\n")
    (stage / "spec.json").write_text(json.dumps(spec.to_dict(), sort_keys=True, indent=1))
    (stage / "Dockerfile").write_text(render_dockerfile(spec, has_skills=has_skills))
    hasher = hashlib.sha256()
    _tree_digest(stage, hasher)
    return stage, hasher.hexdigest()[:16]


def image_uri(image_repo: str, spec_name: str, digest: str) -> str:
    """``<repo>/ratk-<name>:<digest>`` — the tag is the content digest of the build context."""
    import re

    slug = re.sub(r"[^a-z0-9._-]", "-", spec_name.lower()).strip("-.") or "agent"
    return f"{image_repo.rstrip('/')}/ratk-{slug}:{digest}"


def _run(cmd: list[str], *, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=False, text=True, capture_output=capture)


def image_exists(uri: str) -> bool:
    """Whether ``uri`` is already in the registry (``docker manifest inspect``)."""
    if shutil.which(DOCKER) is None:
        return False
    return _run([DOCKER, "manifest", "inspect", uri], capture=True).returncode == 0


def build_and_push(context: Path, uri: str, *, log: Any = None) -> str:
    """``docker build`` the context as ``uri`` and push it; return the pushed reference.

    Requires the Docker CLI and a registry login for the repo (``gcloud auth
    configure-docker <region>-docker.pkg.dev``). Output streams to the caller's terminal.
    """
    if shutil.which(DOCKER) is None:
        raise RuntimeError(
            "gemini.deploy builds the sandbox image with the Docker CLI, which is not on "
            "PATH. Install Docker (or build and push the image yourself and pass image=)."
        )
    if log:
        log(f"building {uri} from {context}")
    build = _run([DOCKER, "build", "--platform", "linux/amd64", "-f", str(context / "Dockerfile"),
                  "-t", uri, str(context)])
    if build.returncode != 0:
        raise RuntimeError(f"docker build failed (exit {build.returncode}) for {uri}")
    if log:
        log(f"pushing {uri}")
    push = _run([DOCKER, "push", uri])
    if push.returncode != 0:
        raise RuntimeError(
            f"docker push failed (exit {push.returncode}) for {uri}; is the registry "
            "login configured (gcloud auth configure-docker <region>-docker.pkg.dev) and "
            "does the identity hold artifactregistry.writer on the repo?"
        )
    return uri

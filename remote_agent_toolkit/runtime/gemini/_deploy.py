"""Packaging + deploy contracts for Gemini Agent Runtime.

Encodes the §6 deploy contracts so consumers inherit them for free:

* uv shipped as a Python requirement + its bin dir prepended to PATH (build-script
  filesystem changes do NOT persist into the runtime container; only ``git`` survives).
* glibc base, Python 3.12 (the ``claude-agent-sdk`` wheel ships a self-contained glibc
  ELF ``claude`` binary; no Alpine/musl).
* ``a2a-sdk>=0.3.4,<0.4`` (1.x is incompatible with ADK 2.3.0).
* Only ``/tmp`` is writable → agent cwd under ``/tmp/agent-jobs/<session-id>/workspace``
  (the ``workspace`` leaf is the cross-backend cwd contract; see ``RunContext.workspace``).
* ``IS_SANDBOX=1`` to allow ``bypassPermissions`` under root.
* ``min_instances=0`` default: the toolkit only uses the async ``run_query_job`` path, where
  every job provisions its own worker — a min-instances container would serve only the (unused)
  sync query path while billing continuously for idle compute.
* ``resource_limits`` optional passthrough (platform default ``{"cpu": "4", "memory": "4Gi"}``):
  raise the memory for agents whose turns build dependencies or import big projects — under the
  4Gi default the platform's job runner can OOM-kill the worker mid-turn (observed live
  2026-07-28/29), and its automatic retry replays the turn from scratch.

Lifted & generalized from the PoC ``deploy/deploy_agent_engine.py`` (DESIGN.md §8):
everything is now driven off the declarative :class:`AgentSpec` (Zyte specifics dropped).

The Google SDK is never imported here — :func:`build_engine_config` returns a plain
``dict`` of kwargs for ``agentplatform.types.AgentEngineConfig(**kwargs)``, which the
backend constructs lazily. This module stays importable with ZERO third-party deps at
import time (stdlib only at module scope; any third-party import is lazy inside a body).
"""

from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...spec import AgentSpec

# Runtime third-party requirements baked into every engine. The toolkit package itself is
# staged separately as an ``extra_packages`` entry, so it is intentionally NOT listed here.
# Generalized from the PoC: scrapy/zyte-api (Zyte-specific) are dropped; agent-declared
# baked deps (``spec.packages``) are appended by ``build_requirements``.
_BASE_REQUIREMENTS: tuple[str, ...] = (
    # 1.154+ so the runtime can unpickle the ``agentplatform.agent_engines`` AdkApp the
    # client stages (the pre-rename ``vertexai`` template module is a different class);
    # <2 because aiplatform 2.0 (2026-08) is not validated here yet — migrate deliberately
    # (the constraints.txt pin keeps engine builds on 1.165.x either way).
    "google-cloud-aiplatform[adk,agent_engines]>=1.154,<2",
    "cloudpickle",
    "pydantic",
    "claude-agent-sdk==0.2.130",  # keep in lockstep with pyproject and constraints
    "google-adk>=1.5",  # floor of the agentplatform AdkApp template
    # uv as a PYTHON dependency, not via an install script: build-script filesystem changes
    # don't persist into the runtime container, but requirements always do. The uv console
    # script lands next to the runtime python -> already on PATH (skills shell out to it).
    "uv>=0.5",
    # 1.x is incompatible with ADK; pin to the verified-compatible range.
    "a2a-sdk>=0.3.4,<0.4",
    "google-cloud-storage",
    # Warm-pool dispatch: pool workers pull turn assignments from a Pub/Sub subscription.
    "google-cloud-pubsub",
    # Native per-step observability: structured Cloud Logging entries.
    "google-cloud-logging",
    # Cloud Trace export for enable_tracing=True (ADK no-ops tracing without these).
    "google-cloud-trace",
    "opentelemetry-sdk",
    "opentelemetry-exporter-otlp-proto-http",
    "pyyaml",
    "jsonschema",
)


# Constraints for the platform-critical layer (pip ``-c`` semantics, merged into the
# requirements by ``build_requirements`` — the engine build offers no hook for a real
# constraints file: extra_packages are extracted only at container runtime, after pip).
_CONSTRAINTS_PATH = Path(__file__).parent / "constraints.txt"

# The engine build unpickles the AdkApp object the deploying client pickled, so these
# packages must match between the deploy venv and the engine — enforced by
# :func:`verify_deploy_env` against their ``constraints.txt`` pins.
_PICKLE_COUPLED: tuple[str, ...] = ("google-cloud-aiplatform", "cloudpickle", "pydantic")


def load_constraints() -> dict:
    """Parse ``constraints.txt`` into ``{canonical package name: SpecifierSet}``."""
    from packaging.requirements import Requirement  # lazy: keep import-time stdlib-only
    from packaging.utils import canonicalize_name

    out: dict = {}
    for raw in _CONSTRAINTS_PATH.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            req = Requirement(line)
            out[canonicalize_name(req.name)] = req.specifier
    return out


def _unsatisfiable(req) -> bool:
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
    """Compute the runtime ``requirements`` list, applying the §6 build contracts.

    Returns the base third-party deps every engine needs (uv, the a2a-sdk pin,
    claude-agent-sdk, google-adk, the GCP/otel stack) followed by the agent's own declared
    baked deps (``spec.packages``), with the ``constraints.txt`` pins merged in (pip ``-c``
    semantics, applied client-side). The toolkit package is shipped via ``extra_packages``,
    so ``remote-agent-toolkit`` is deliberately absent here. Order is first-seen; a spec
    re-pin of a base dep merges into one line (pip rejects duplicate names outright).

    Raises ``ValueError`` when a merge is unsatisfiable (e.g. ``spec.packages`` pins a
    platform-critical package to a version ``constraints.txt`` excludes) — fail fast
    here, not 4 billable minutes into the engine build.
    """
    from packaging.requirements import InvalidRequirement, Requirement  # lazy
    from packaging.utils import canonicalize_name

    # The Codex SDK bundles a pinned codex CLI binary (tens of MB): baked only when the
    # deployment offers the codex harness (spec.harness or spec.harnesses) — harness
    # AVAILABILITY is a deploy-time fact; sessions select among what is baked. Keep in
    # lockstep with pyproject.
    extra = ("openai-codex==0.147.0",) if "codex" in spec.baked_harnesses else ()

    entries: list = []  # str (opaque pass-through) or Requirement, in first-seen order
    by_name: dict = {}
    for line in (*_BASE_REQUIREMENTS, *extra, *spec.packages):
        try:
            req = Requirement(line)
        except InvalidRequirement:
            entries.append(line)  # not PEP 508 (e.g. a pip option) — pass through as-is
            continue
        if req.url:
            entries.append(line)  # direct URL: no specifier to merge, keep verbatim
            continue
        key = canonicalize_name(req.name)
        if key in by_name:
            by_name[key].specifier &= req.specifier
            by_name[key].extras |= req.extras
        else:
            by_name[key] = req
            entries.append(req)

    constraints = load_constraints()
    for key, req in by_name.items():
        if key in constraints:
            req.specifier &= constraints[key]
    # Constrained packages nothing requires directly (e.g. google-auth) are installed as
    # transitive deps either way — appending them as pinned requirements is equivalent.
    for key, specifier in constraints.items():
        if key not in by_name:
            entries.append(f"{key}{specifier}")

    for req in by_name.values():
        if _unsatisfiable(req):
            raise ValueError(
                f"unsatisfiable requirement {str(req)!r} after merging spec.packages with "
                f"the platform constraints ({_CONSTRAINTS_PATH}); align the spec's pin "
                "with the constraint or refresh the constraint deliberately"
            )
    return [str(e) for e in entries]


def verify_deploy_env() -> None:
    """Fail fast when the deploy venv diverges from the pickle-coupled pins.

    The engine build unpickles the ``agentplatform`` AdkApp that THIS venv pickles, so
    for :data:`_PICKLE_COUPLED` the venv version and the engine's ``constraints.txt`` pin
    must agree — a skew can produce an engine that fails to unpickle (or misbehaves)
    only at build/runtime, after the ~4 min billable build. Called by ``deploy`` before
    any side effect.
    """
    import importlib.metadata

    from packaging.version import Version  # lazy: keep import-time stdlib-only

    constraints = load_constraints()
    problems: list[str] = []
    for name in _PICKLE_COUPLED:
        specifier = constraints.get(name)
        if specifier is None:
            continue
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        if not specifier.contains(Version(installed), prereleases=True):
            problems.append(f"{name}: venv has {installed}, constraints.txt wants {specifier}")
    if problems:
        raise RuntimeError(
            "deploy venv out of sync with the pickle-coupled engine pins — the engine "
            "build unpickles what this venv pickles, so they must match. Either sync the "
            f"venv or refresh runtime/gemini/constraints.txt: {'; '.join(problems)}"
        )


def build_env(
    spec: AgentSpec,
    *,
    project: str | None = None,
    model: str | None = None,
    output_bucket: str | None = None,
    use_vertex: bool = True,
    vertex_region: str = "global",
    warm_pool: bool = False,
    pool_subscription: str | None = None,
    pool_max_wait_s: float | None = None,
) -> dict:
    """Build the engine ``env_vars`` dict from ``spec`` (generalizes the PoC ``_env_vars``).

    Pure: performs no GCP calls. Returns only **non-secret machinery** — model routing, the
    sandbox/cwd flags, artifact/checkpoint buckets, the pool subscription. **No secrets are
    baked into the engine**: credentials are passed per-invocation to ``run``/``send`` and
    travel with the turn, so one deployed engine safely serves many tenants (DESIGN.md §3.5).

    Args:
        model: Overrides ``spec.model`` for the embedded harness when given.
        output_bucket: A ``gs://`` base prefix. Required for artifact upload and (when
            ``spec.checkpoint``) for checkpointing; see the body for why checkpointing is
            skipped without it.
        use_vertex: Route the model through Vertex (the RE service agent's own identity, no API
            key in the agent env). Set ``False`` for API-key mode (key supplied per-invocation).
        warm_pool / pool_subscription: when both set, the engine acts as a pool worker that
            pulls turn assignments from ``pool_subscription``.
        pool_max_wait_s: how long an idle pool worker waits for an assignment before it
            exits (default: ``pool.DEFAULT_MAX_WAIT_S``, a day). Baked into the engine env
            as ``AGENT_POOL_MAX_WAIT_S`` for the worker's wait loop to read; only
            meaningful with ``warm_pool``.
    """
    # NaN would pass a bare `<= 0` check and make the worker's deadline arithmetic always
    # false — every worker idle-expires instantly, silently recreating the permanently-cold
    # pool this knob exists to prevent; inf would defeat the documented billing bound.
    if pool_max_wait_s is not None and not (
        math.isfinite(pool_max_wait_s) and pool_max_wait_s > 0
    ):
        raise ValueError(
            f"pool_max_wait_s must be a positive, finite number of seconds; "
            f"got {pool_max_wait_s!r}"
        )
    env: dict = {
        # The engine refuses bypassPermissions under root; Agent Runtime may run as root.
        "IS_SANDBOX": "1",
        # Only /tmp is reliably writable in the managed runtime → per-job cwd lives here.
        "AGENT_JOBS_ROOT": "/tmp/agent-jobs",
        # Model the embedded Claude Code harness uses (resolved at runtime by the agent).
        "CLAUDE_AGENT_MODEL": model or spec.model,
        # The platform's serving harness (uvicorn, in the container base image) reads this
        # and defaults to ``os.cpu_count() + 1`` worker PROCESSES — 10-11 on the nodes seen
        # live — each importing the whole stack privately (spawn start method, no
        # copy-on-write): ~300 MiB apiece, ~3 GiB of a 4Gi worker before the agent ran
        # (measured 2026-09-03). A query job serves exactly one request per container, so
        # the extra workers only cost memory; one worker brings the baseline to ~450 MiB
        # and startup CPU from minutes to seconds. The harness handles ``"1"`` explicitly.
        "NUM_WORKERS": "1",
    }

    # Produced files are uploaded under the output bucket after each run.
    if output_bucket:
        env["AGENT_ARTIFACTS_GCS"] = f"{output_bucket}/artifacts"
        # Per-session event mirror at events/<sid>/*.jsonl — BOTH the durable history
        # (Session.history(); the platform's own job output isn't session-keyed for warm
        # turns and keeps only the last cold job) AND the live channel: the worker streams
        # batches as events happen and the client tails the listing (stream.py). Cloud
        # Logging is emit-only (ops/debug), never tailed.
        env["AGENT_EVENTS_GCS"] = f"{output_bucket}/events"

    # Checkpoint/resume mirrors each turn's conversation + workspace to GCS, and a
    # transcript-only spec mirrors the conversation alone. Either needs a bucket to write
    # to, so we only enable it when an output_bucket is supplied; otherwise the flag is
    # silently a no-op (no place to write to).
    if (spec.checkpoint or spec.transcript) and output_bucket:
        env["AGENT_CHECKPOINT_GCS"] = f"{output_bucket}/checkpoints"

    # Claude model auth. Default: route Claude through Vertex, so the engine authenticates as
    # its OWN GCP identity (the RE service agent) — no API key in the agent's environment. For
    # API-key mode (use_vertex=False) the caller passes ANTHROPIC_API_KEY per-invocation.
    # The Codex harness calls the OpenAI API directly (no Vertex path for OpenAI models);
    # its OPENAI_API_KEY travels per-invocation, so the vertex routing vars are set only
    # when a non-codex harness is baked (they are inert for codex turns either way).
    if use_vertex and project and any(h != "codex" for h in spec.baked_harnesses):
        env["CLAUDE_CODE_USE_VERTEX"] = "1"
        env["ANTHROPIC_VERTEX_PROJECT_ID"] = project
        env["CLOUD_ML_REGION"] = vertex_region

    # Warm-pool worker config: a pooled job pulls turn assignments from this subscription.
    if warm_pool and pool_subscription:
        env["AGENT_POOL_SUBSCRIPTION"] = pool_subscription
        if pool_max_wait_s is not None:
            env["AGENT_POOL_MAX_WAIT_S"] = str(pool_max_wait_s)

    return env


def stage_skills(spec: AgentSpec, dest_dir: str) -> list[str]:
    """Resolve ``spec.skills`` into a FLAT ``dest_dir`` of skill folders; return the names.

    Each staged entry is a directory containing ``SKILL.md``. The flat layout lets the skills
    be baked into the engine image (as an ``extra_packages`` dir) and copied into the per-job
    cwd at runtime. Resolution per ``SkillSource.kind``:

    * ``"local"``   — read skill folders from ``source.path``.
    * ``"git"``     — clone ``source.url`` at ``source.ref``, read from ``<clone>/<subdir>``.
    * ``"builtin"`` — not shipped yet (raises ``NotImplementedError``).

    On name collision across sources, the last source wins (matches ``skills.provision``).
    Returns the sorted list of staged skill names.
    """
    from ...skills import discover_skill_names  # local: keep import-time stdlib-only

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
            from ...integrations import git  # lazy: subprocess-backed, no third-party dep

            clone_dir = git.clone(source.url, ref=source.ref)
            src_dir = clone_dir / source.subdir
        elif source.kind == "builtin":
            raise NotImplementedError("builtin skills not shipped yet")
        else:
            raise ValueError(f"unknown SkillSource kind: {source.kind!r}")

        for name in discover_skill_names(src_dir):
            dst = dest / name
            if dst.exists():
                shutil.rmtree(dst)  # last source wins on collision
            shutil.copytree(src_dir / name, dst)
            staged.add(name)

    return sorted(staged)


def stage_agent(spec: AgentSpec) -> tuple[str, list[str]]:
    """Build the deploy bundle in a fresh temp dir (generalizes the PoC ``_stage_packages``).

    Stages the toolkit package + the resolved skills FLAT so they bundle at the tar top
    level — ``extra_packages`` paths are resolved relative to cwd, and a nested ``src/...``
    arcname would not be importable as ``remote_agent_toolkit``. Steps:

    * Locate the installed ``remote_agent_toolkit`` package dir (via its ``__file__``) and
      ``copytree`` it into ``<stage>/remote_agent_toolkit`` (skipping ``__pycache__`` /
      ``.venv`` / ``*.pyc``).
    * Stage ``spec.skills`` into ``<stage>/skills`` via :func:`stage_skills`.

    Returns ``(stage_dir, extra_packages)`` where ``extra_packages`` is a list of RELATIVE
    paths (``["remote_agent_toolkit"]``, plus ``"skills"`` only if any were staged). The
    caller is expected to ``chdir`` to ``stage_dir`` before deploying.
    """
    import remote_agent_toolkit  # lazy: locate the installed package dir at deploy time

    pkg_dir = Path(remote_agent_toolkit.__file__).resolve().parent
    stage = Path(tempfile.mkdtemp(prefix="rat-deploy-"))

    shutil.copytree(
        pkg_dir,
        stage / "remote_agent_toolkit",
        ignore=shutil.ignore_patterns("__pycache__", ".venv", "*.pyc"),
    )

    extra_packages = ["remote_agent_toolkit"]
    if stage_skills(spec, str(stage / "skills")):
        extra_packages.append("skills")

    return str(stage), extra_packages


def validate_resource_limits(resource_limits: dict[str, str]) -> None:
    """Fail fast on a malformed ``resource_limits`` (BEFORE the ~4 min billable build).

    Mirrors the platform contract (``ReasoningEngineSpec.deploymentSpec.resourceLimits``):
    exactly the keys ``cpu`` and ``memory``, cpu in 1/2/4/6/8, memory ``<n>Gi`` up to 32.
    Values are strings (Cloud Run quantity syntax), e.g. ``{"cpu": "4", "memory": "16Gi"}``.
    """
    if set(resource_limits) != {"cpu", "memory"}:
        raise ValueError(
            f"resource_limits must have exactly the keys 'cpu' and 'memory'; "
            f"got {sorted(resource_limits)}"
        )
    cpu, memory = str(resource_limits["cpu"]), str(resource_limits["memory"])
    if cpu not in ("1", "2", "4", "6", "8"):
        raise ValueError(f"resource_limits cpu must be one of '1','2','4','6','8'; got {cpu!r}")
    if not (memory.endswith("Gi") and memory[:-2].isdigit() and 1 <= int(memory[:-2]) <= 32):
        raise ValueError(f"resource_limits memory must be '1Gi'..'32Gi'; got {memory!r}")


def build_engine_config(
    spec: AgentSpec,
    *,
    project: str,
    location: str,
    staging_bucket: str,
    extra_packages: list[str],
    model: str | None = None,
    output_bucket: str | None = None,
    use_vertex: bool = True,
    vertex_region: str = "global",
    warm_pool: bool = False,
    pool_subscription: str | None = None,
    pool_max_wait_s: float | None = None,
    min_instances: int = 0,
    max_instances: int = 1,
    resource_limits: dict[str, str] | None = None,
) -> dict:
    """Build the kwargs dict for ``agentplatform.types.AgentEngineConfig(**kwargs)``.

    Does NOT import the SDK — returns a plain ``dict`` so the backend constructs the config
    lazily. ``project`` / ``location`` are accepted for caller symmetry (the genai client
    carries them); the staged ``extra_packages`` (from :func:`stage_agent`) are passed in.
    No ``build_options`` / install scripts — the toolkit needs no node (uv is a requirement).

    ``resource_limits`` sets the engine container's CPU/memory (query-job workers run with
    it too). Omitted → the platform default, ``{"cpu": "4", "memory": "4Gi"}`` — 4Gi is
    shared by the harness CLI, the ADK app, and everything the agent's tools spawn, and
    memory-heavy agent work (dependency builds, big imports) can OOM-kill the worker
    mid-turn; raise it (up to ``"32Gi"``) for such agents. The kwarg is omitted from the
    config when None so the platform default stays authoritative.
    """
    if resource_limits is not None:
        validate_resource_limits(resource_limits)
    extra: dict = {"resource_limits": dict(resource_limits)} if resource_limits else {}
    return {
        **extra,
        "display_name": spec.name,
        "description": f"remote-agent-toolkit agent: {spec.name}",
        "staging_bucket": staging_bucket,
        # google-adk framework enables the long-running query-job path.
        "agent_framework": "google-adk",
        "python_version": "3.12",
        "requirements": build_requirements(spec),
        "extra_packages": extra_packages,
        "env_vars": build_env(
            spec,
            project=project,
            model=model,
            output_bucket=output_bucket,
            use_vertex=use_vertex,
            vertex_region=vertex_region,
            warm_pool=warm_pool,
            pool_subscription=pool_subscription,
            pool_max_wait_s=pool_max_wait_s,
        ),
        "min_instances": min_instances,
        "max_instances": max_instances,
    }

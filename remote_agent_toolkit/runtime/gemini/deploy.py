"""Packaging + deploy contracts for Gemini Agent Runtime.

Encodes the §6 deploy contracts so consumers inherit them for free:

* uv shipped as a Python requirement + its bin dir prepended to PATH (build-script
  filesystem changes do NOT persist into the runtime container; only ``git`` survives).
* glibc base, Python 3.12 (the ``claude-agent-sdk`` wheel ships a self-contained glibc
  ELF ``claude`` binary; no Alpine/musl).
* ``a2a-sdk>=0.3.4,<0.4`` (1.x is incompatible with ADK 2.3.0).
* Only ``/tmp`` is writable → per-job cwd under ``/tmp/agent-jobs/<session-id>``.
* ``IS_SANDBOX=1`` to allow ``bypassPermissions`` under root.
* ``min_instances=0`` default: the toolkit only uses the async ``run_query_job`` path, where
  every job provisions its own worker — a min-instances container would serve only the (unused)
  sync query path while billing continuously for idle compute.

Lifted & generalized from the PoC ``deploy/deploy_agent_engine.py`` (DESIGN.md §8):
everything is now driven off the declarative :class:`AgentSpec` (Zyte specifics dropped).

The Google SDK is never imported here — :func:`build_engine_config` returns a plain
``dict`` of kwargs for ``vertexai._genai.types.AgentEngineConfig(**kwargs)``, which the
backend constructs lazily. This module stays importable with ZERO third-party deps at
import time (stdlib only at module scope; any third-party import is lazy inside a body).
"""

from __future__ import annotations

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
    "google-cloud-aiplatform[adk,agent_engines]>=1.110",
    "cloudpickle",
    "pydantic",
    "claude-agent-sdk>=0.2.0",
    "google-adk>=1.0",
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


def build_requirements(spec: AgentSpec) -> list[str]:
    """Compute the runtime ``requirements`` list, applying the §6 build contracts.

    Returns the base third-party deps every engine needs (uv, the a2a-sdk pin,
    claude-agent-sdk, google-adk, the GCP/otel stack) followed by the agent's own declared
    baked deps (``spec.packages``). The toolkit package is shipped via ``extra_packages``,
    so ``remote-agent-toolkit`` is deliberately absent here. Order is preserved and dupes
    are dropped (a spec may re-pin a base dep).
    """
    seen: set[str] = set()
    out: list[str] = []
    for req in (*_BASE_REQUIREMENTS, *spec.packages):
        if req not in seen:
            seen.add(req)
            out.append(req)
    return out


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
    """
    env: dict = {
        # The engine refuses bypassPermissions under root; Agent Runtime may run as root.
        "IS_SANDBOX": "1",
        # Only /tmp is reliably writable in the managed runtime → per-job cwd lives here.
        "AGENT_JOBS_ROOT": "/tmp/agent-jobs",
        # Model the embedded Claude Code harness uses (resolved at runtime by the agent).
        "CLAUDE_AGENT_MODEL": model or spec.model,
    }

    # Produced files are uploaded under the output bucket after each run.
    if output_bucket:
        env["AGENT_ARTIFACTS_GCS"] = f"{output_bucket}/artifacts"
        # Durable per-session event history: each turn mirrors its events to
        # events/<sid>/<ts>.jsonl (the platform's own job output isn't session-keyed for
        # warm turns and keeps only the last cold job) — read via Session.history().
        env["AGENT_EVENTS_GCS"] = f"{output_bucket}/events"

    # Checkpoint/resume mirrors each turn's conversation + workspace to GCS. It needs a
    # bucket to write to, so we only enable it when an output_bucket is supplied; otherwise
    # the flag is silently a no-op (no place to checkpoint to).
    if spec.checkpoint and output_bucket:
        env["AGENT_CHECKPOINT_GCS"] = f"{output_bucket}/checkpoints"

    # Claude model auth. Default: route Claude through Vertex, so the engine authenticates as
    # its OWN GCP identity (the RE service agent) — no API key in the agent's environment. For
    # API-key mode (use_vertex=False) the caller passes ANTHROPIC_API_KEY per-invocation.
    if use_vertex and project:
        env["CLAUDE_CODE_USE_VERTEX"] = "1"
        env["ANTHROPIC_VERTEX_PROJECT_ID"] = project
        env["CLOUD_ML_REGION"] = vertex_region

    # Warm-pool worker config: a pooled job pulls turn assignments from this subscription.
    if warm_pool and pool_subscription:
        env["AGENT_POOL_SUBSCRIPTION"] = pool_subscription

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
    min_instances: int = 0,
    max_instances: int = 1,
) -> dict:
    """Build the kwargs dict for ``vertexai._genai.types.AgentEngineConfig(**kwargs)``.

    Does NOT import vertexai — returns a plain ``dict`` so the backend constructs the config
    lazily. ``project`` / ``location`` are accepted for caller symmetry (the genai client
    carries them); the staged ``extra_packages`` (from :func:`stage_agent`) are passed in.
    No ``build_options`` / install scripts — the toolkit needs no node (uv is a requirement).
    """
    return {
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
        ),
        "min_instances": min_instances,
        "max_instances": max_instances,
    }

"""Gemini Agent Runtime control plane: deploy / get_engine / list_engines — P2.

These encode the §6 platform contracts (uv/glibc/IS_SANDBOX/min_instances/warm pool,
runtime secret resolution, the two-identity IAM model). ``deploy`` mints a new version
of the engine identified by ``spec.name``; app code addresses engines by name via
``get_engine`` and never deploys (DESIGN.md §3.3).

P0: signatures only. The Google SDK (``vertexai`` / ``google-cloud-aiplatform``) is
imported lazily inside the bodies; bodies raise ``NotImplementedError`` until P2.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ...spec import AgentSpec
    from ..base import Engine


def deploy(
    spec: AgentSpec,
    project: str,
    location: str,
    warm_pool: bool = False,
    **kw: Any,
) -> Engine:
    """Deploy ``spec`` to Gemini Agent Runtime, minting a new engine version (P2).

    Ops/CI action. Packages the agent (see ``deploy.py``), applies the §6 deploy
    contracts automatically, optionally provisions a warm pool (``pool.py`` +
    ``DispatchTransport``), and returns a ``GeminiEngine`` handle.
    """
    raise NotImplementedError(
        "gemini.deploy is a P2 capability: package the spec via runtime/gemini/deploy.py, "
        "apply the §6 platform contracts, optionally start the warm pool, register the "
        "engine under spec.name and return a GeminiEngine."
    )


def get_engine(
    name: str,
    project: str | None = None,
    location: str | None = None,
    version: str | None = None,
) -> Engine:
    """Look up a deployed engine by ``name`` (the app-code hot path; P2).

    Returns the latest deployed version, or a pinned ``version``. Never deploys.
    """
    raise NotImplementedError(
        "gemini.get_engine is a P2 capability: resolve spec.name -> Agent Engine "
        "resource (latest or pinned version) and return a GeminiEngine. Never deploys."
    )


def list_engines(project: str, location: str) -> list:
    """Discover deployed engines in ``project``/``location`` (control plane; P2)."""
    raise NotImplementedError(
        "gemini.list_engines is a P2 capability: enumerate deployed Agent Engines "
        "in the given project/location."
    )

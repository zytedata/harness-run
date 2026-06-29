"""Packaging + deploy contracts for Gemini Agent Runtime — P2.

Encodes the §6 deploy contracts so consumers inherit them for free:

* uv shipped as a Python requirement + its bin dir prepended to PATH (build-script
  filesystem changes do NOT persist into the runtime container; only ``git`` survives).
* glibc base, Python 3.12 (the ``claude-agent-sdk`` wheel ships a self-contained glibc
  ELF ``claude`` binary; no Alpine/musl).
* ``a2a-sdk>=0.3.4,<0.4`` (1.x is incompatible with ADK 2.3.0).
* Only ``/tmp`` is writable → per-job cwd under ``/tmp/agent-jobs/<session-id>``.
* ``IS_SANDBOX=1`` to allow ``bypassPermissions`` under root.
* ``min_instances>=1`` for real long sync jobs; min=0 is fine for the async path.

Lifted & generalized from the PoC ``deploy/deploy_agent_engine.py`` (DESIGN.md §8).
The Google SDK is imported lazily inside the bodies. P0: signatures/stubs only.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ...spec import AgentSpec


def build_requirements(spec: AgentSpec) -> list[str]:
    """Compute the runtime ``requirements`` list, applying the §6 build contracts (P2)."""
    raise NotImplementedError(
        "P2: derive the runtime requirements (uv, a2a-sdk pin, claude-agent-sdk, "
        "google-adk, plus spec-driven extras) per the §6 deploy contracts."
    )


def package_agent(spec: AgentSpec, **kw: Any) -> Any:
    """Build the deployable ADK agent object + deploy kwargs from ``spec`` (P2).

    Pins the Claude session id up front, sets ``IS_SANDBOX=1`` and the per-job cwd, and
    wires runtime secret resolution (never pickles secrets/paths — DESIGN.md §3.5).
    """
    raise NotImplementedError(
        "P2: package the ClaudeCodeHarness as a deployable ADK agent with the §6 "
        "contracts applied (uv on PATH, IS_SANDBOX, /tmp cwd, runtime resolution)."
    )

"""``SecretResolver`` port + adapters (DESIGN.md §3.5, §6, §7).

Secret *name* → value, resolved at runtime (never pickled into a deployed engine). On
Gemini Agent Runtime, Secret Manager access authorizes against the RE service agent,
not the operator SA (DESIGN.md §6). Protocol is stdlib-only; ``GcpSecretResolver``
imports the Secret Manager client lazily. ``EnvSecretResolver`` is fully implemented
(stdlib ``os.environ``).
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable


@runtime_checkable
class SecretResolver(Protocol):
    """Resolve a secret *name* to its value at runtime."""

    def resolve(self, name: str) -> str:
        """Return the value of secret ``name``."""
        ...


class GcpSecretResolver:
    """Secret Manager-backed :class:`SecretResolver` (not implemented yet).

    Authorizes against the RE service agent at runtime (DESIGN.md §6). The Secret
    Manager client is imported lazily inside ``resolve``. Note: on the deployed engine the
    platform already injects ``spec.secrets`` as env vars, so the runtime path uses
    ``EnvSecretResolver``; this is only for reading secrets at the control plane.
    """

    def __init__(self, project: str) -> None:
        self.project = project

    def resolve(self, name: str) -> str:
        raise NotImplementedError(
            "GcpSecretResolver is not implemented yet: read the latest version of secret "
            "`name` from Secret Manager in self.project (authorized against the RE service agent)."
        )


class EnvSecretResolver:
    """Environment-variable :class:`SecretResolver` for local dev (stdlib-only).

    Resolves a secret ``name`` from ``os.environ``. This is a real implementation,
    not a stub.
    """

    def resolve(self, name: str) -> str:
        """Return ``os.environ[name]``; raise ``KeyError`` if unset."""
        try:
            return os.environ[name]
        except KeyError as exc:
            raise KeyError(f"secret {name!r} not found in the environment") from exc

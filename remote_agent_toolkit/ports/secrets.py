"""``SecretResolver`` port + adapters (DESIGN.md §3.5, §6, §7).

Secret *name* → value, resolved at the *control plane* (never pickled into a deployed
engine). These adapters help a caller turn secret names into the per-invocation ``secrets``
dict passed to ``run``/``send``; the runtime itself no longer resolves secrets (they arrive
already-resolved with the invocation). ``EnvSecretResolver`` reads ``os.environ`` (handy for
local dev); ``GcpSecretResolver`` (not implemented yet) would read Secret Manager, authorizing
against the RE service agent (DESIGN.md §6). Protocol is stdlib-only; the Secret Manager
client is imported lazily.
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

    For a caller who keeps secrets in Secret Manager: resolve names to values at the control
    plane, then hand them to ``run``/``send`` as the per-invocation ``secrets`` dict. The
    Secret Manager client is imported lazily inside ``resolve``.
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

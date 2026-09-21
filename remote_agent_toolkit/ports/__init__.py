"""Ports & adapters (DESIGN.md §7).

Every platform dependency is a ``typing.Protocol`` with concrete adapters shipped for
prod (GCP) and dev (local/in-memory). The protocols are stdlib-only; adapter bodies
import their third-party SDK lazily so importing a port module needs no GCP deps.
"""

from __future__ import annotations

from .blobstore import BlobStore, GcsBlobStore, LocalBlobStore
from .secrets import EnvSecretResolver, GcpSecretResolver, SecretResolver

__all__ = [
    "BlobStore",
    "GcsBlobStore",
    "LocalBlobStore",
    "SecretResolver",
    "GcpSecretResolver",
    "EnvSecretResolver",
]

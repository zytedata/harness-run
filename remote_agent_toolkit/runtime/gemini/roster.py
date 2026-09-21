"""The ready pool's roster: which idle sandboxes are waiting for a turn (DESIGN.md §13.1).

A turn wants a sandbox that already answers — creating one takes ~2 s plus 12–31 s until
the proxy routes to it (measured). So the control plane keeps a **ready pool**: sandboxes
created ahead of time, health-checked, and recorded here, one small GCS object per idle
sandbox under ``pool/<template-id>/idle/``, written by the client that created it and
**removed atomically by the client that dispatches to it** (a delete with a generation
precondition: of two clients racing for the same sandbox exactly one succeeds, the other
moves on to the next entry). Any client process driving the engine shares the roster.

Entries are ordered by creation time and picked oldest first. ``expires_at`` is the
sandbox's TTL (set at creation; the platform enforces it and an exec does not extend it —
measured), so an expired entry is pruned without asking the platform. The roster is a
coordination device, not a security boundary: with no per-sandbox IAM, whoever can
execute on the host instance can drive any sandbox, so the client identity is the trust
boundary either way.

``google.cloud.storage`` is imported lazily by :class:`GcsRosterStore`; an injected store
keeps the logic testable offline (:class:`InMemoryRosterStore`).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from ...ports.blobstore import parse_gcs_uri


def roster_prefix(template: str) -> str:
    """Bucket-relative key prefix of a template's idle-sandbox roster."""
    return f"pool/{template.rsplit('/', 1)[-1]}/idle/"


@dataclass(frozen=True)
class SandboxEntry:
    """One idle, ready sandbox, as the client that created it recorded it."""

    sandbox: str  # full resource name
    template: str  # the template it was created from
    created_at: float  # epoch seconds
    expires_at: float  # epoch seconds: the TTL the platform enforces

    @property
    def id(self) -> str:
        return self.sandbox.rsplit("/", 1)[-1]

    def to_json(self) -> bytes:
        return json.dumps(asdict(self)).encode("utf-8")

    @classmethod
    def from_json(cls, data: bytes) -> SandboxEntry:
        d = json.loads(data.decode("utf-8"))
        return cls(
            sandbox=str(d["sandbox"]),
            template=str(d["template"]),
            created_at=float(d["created_at"]),
            expires_at=float(d["expires_at"]),
        )


class RosterStore(Protocol):
    """The generation-aware object operations the roster needs (a thin GCS subset)."""

    def list(self, prefix: str) -> list[tuple[str, int, str | None]]:
        """``(object_name, generation, inline_entry)`` for every object under ``prefix``."""
        ...

    def get(self, name: str) -> bytes | None:
        ...

    def put(self, name: str, data: bytes) -> None:
        """Write ``data`` as the body AND as the ``entry`` metadata value."""
        ...

    def delete_if(self, name: str, generation: int) -> bool:
        """Delete ``name`` only if it still has ``generation``; ``False`` if it did not."""
        ...


class GcsRosterStore:
    """:class:`RosterStore` over one GCS bucket, authorized as the client identity."""

    def __init__(self, bucket: str, credentials: Any | None = None) -> None:
        self._bucket_name = bucket
        self._credentials = credentials
        self._client: Any = None

    def _bucket(self) -> Any:
        if self._client is None:
            from google.cloud import storage  # lazy

            kwargs = {"credentials": self._credentials} if self._credentials else {}
            self._client = storage.Client(**kwargs)
        return self._client.bucket(self._bucket_name)

    def list(self, prefix: str) -> list[tuple[str, int, str | None]]:
        bucket = self._bucket()
        return [
            (b.name, int(b.generation), (b.metadata or {}).get("entry"))
            for b in self._client.list_blobs(bucket, prefix=prefix)
        ]

    def get(self, name: str) -> bytes | None:
        from google.api_core.exceptions import NotFound

        try:
            return self._bucket().blob(name).download_as_bytes()
        except NotFound:
            return None

    def put(self, name: str, data: bytes) -> None:
        blob = self._bucket().blob(name)
        blob.metadata = {"entry": data.decode("utf-8")}
        blob.upload_from_string(data, content_type="application/json")

    def delete_if(self, name: str, generation: int) -> bool:
        from google.api_core.exceptions import NotFound, PreconditionFailed

        try:
            self._bucket().blob(name).delete(if_generation_match=generation)
        except (NotFound, PreconditionFailed):
            return False
        return True


class InMemoryRosterStore:
    """In-process :class:`RosterStore` for tests; generations count per-object writes."""

    def __init__(self) -> None:
        self._objects: dict[str, tuple[bytes, int]] = {}
        self._next_generation = 1

    def list(self, prefix: str) -> list[tuple[str, int, str | None]]:
        return sorted(
            (name, gen, data.decode("utf-8"))
            for name, (data, gen) in self._objects.items() if name.startswith(prefix)
        )

    def get(self, name: str) -> bytes | None:
        found = self._objects.get(name)
        return found[0] if found else None

    def put(self, name: str, data: bytes) -> None:
        self._objects[name] = (data, self._next_generation)
        self._next_generation += 1

    def delete_if(self, name: str, generation: int) -> bool:
        found = self._objects.get(name)
        if found is None or found[1] != generation:
            return False
        del self._objects[name]
        return True


class PoolRoster:
    """The idle-sandbox roster of one template (see the module docstring)."""

    def __init__(
        self,
        output_bucket: str,
        template: str,
        credentials: Any | None = None,
        store: RosterStore | None = None,
    ) -> None:
        bucket, base = parse_gcs_uri(output_bucket)
        self._prefix = f"{base + '/' if base else ''}{roster_prefix(template)}"
        self._store: RosterStore = store if store is not None else GcsRosterStore(bucket, credentials)

    def _key(self, sandbox_id: str) -> str:
        return f"{self._prefix}{sandbox_id}.json"

    def add(self, entry: SandboxEntry) -> None:
        """Record ``entry`` as idle and ready (called by the client that created it)."""
        self._store.put(self._key(entry.id), entry.to_json())

    def remove(self, sandbox_id: str) -> None:
        """Drop ``sandbox_id`` whatever its state (teardown; no precondition)."""
        for name, generation, _ in self._store.list(self._key(sandbox_id)):
            self._store.delete_if(name, generation)

    def _read(self) -> list[tuple[str, int, SandboxEntry]]:
        rows = []
        for name, generation, inline in self._store.list(self._prefix):
            data = inline.encode("utf-8") if inline else self._store.get(name)
            if data is None:
                continue  # claimed by someone else between list and read
            try:
                entry = SandboxEntry.from_json(data)
            except (ValueError, KeyError, TypeError):
                continue  # not ours / corrupt: never dispatch on it
            rows.append((name, generation, entry))
        rows.sort(key=lambda row: row[2].created_at)
        return rows

    def entries(self) -> list[SandboxEntry]:
        """Every recorded idle sandbox, oldest first (expired ones included)."""
        return [entry for _, _, entry in self._read()]

    def prune_expired(self, now: float | None = None) -> list[SandboxEntry]:
        """Remove entries whose sandbox has surely expired; return the ones removed."""
        now = time.time() if now is None else now
        pruned = []
        for name, generation, entry in self._read():
            if entry.expires_at <= now and self._store.delete_if(name, generation):
                pruned.append(entry)
        return pruned

    def claim(self, now: float | None = None, margin_s: float = 0.0) -> tuple[SandboxEntry | None, list[SandboxEntry]]:
        """Atomically take the oldest idle sandbox: ``(entry_or_None, expired_entries_pruned)``.

        One listing: expired entries met on the way are deleted and returned so the caller
        can drop the sandboxes; entries another client won (the precondition delete fails)
        are skipped. ``margin_s`` treats an entry expiring within that many seconds as
        expired — a turn must not start on a sandbox about to be deleted under it.
        """
        now = time.time() if now is None else now
        pruned: list[SandboxEntry] = []
        claimed: SandboxEntry | None = None
        for name, generation, entry in self._read():
            if entry.expires_at <= now + margin_s:
                if self._store.delete_if(name, generation):
                    pruned.append(entry)
                continue
            if claimed is None and self._store.delete_if(name, generation):
                claimed = entry
        return claimed, pruned

    def clear(self) -> list[SandboxEntry]:
        """Remove every entry (the pool is being retired); return what was recorded."""
        cleared = []
        for name, generation, entry in self._read():
            if self._store.delete_if(name, generation):
                cleared.append(entry)
        return cleared

"""The warm pool's idle-worker roster (per-worker dispatch; ``pool.py``).

With one subscription per worker, the control plane has to know *which* idle worker to
address a turn to — across every client process that drives the engine. The roster is
that shared state: one small GCS object per idle worker under the pool's
``pool/<generation>/idle/`` prefix (``pool.roster_prefix``), written by the client that
submitted the worker in ``fill_pool`` and **removed atomically by the client that
dispatches to it** (a delete with a generation precondition: of two clients racing for the
same worker exactly one succeeds, the other moves on to the next entry).

Entries are ordered by submission time and picked oldest first — the worker most likely
to have finished its boot. Readiness is not tracked here on purpose: a worker that has
not booted yet still receives its turn (the message waits on its subscription), and a
worker that never boots is caught by the client's pickup watchdog, which re-dispatches.
Entries past their idle expiry (the worker exited without a turn) are pruned on the way.

SECURITY: the roster decides where a turn — its pointers and run-scoped token — is sent,
so it must be writable by the client identity only. The runtime identity's bucket
bindings (README IAM table) cover ``jobs/`` and ``events/ratk-``, never ``pool/``; a
worker that could re-add itself here could receive another run's turn. The roster is the
one piece of pool state the worker never reads or writes.

``google.cloud.storage`` is imported lazily by :class:`GcsRosterStore`; an injected store
keeps the logic testable offline (:class:`InMemoryRosterStore`).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from ...ports.blobstore import parse_gcs_uri
from .pool import roster_prefix


@dataclass(frozen=True)
class WorkerEntry:
    """One idle pool worker, as the client that submitted it recorded it."""

    worker: str  # the random worker id (pool.new_worker_id)
    subscription: str  # its own dispatch subscription path
    job_name: str | None  # the platform operation of its query job (cancel on teardown)
    submitted_at: float  # epoch seconds the job was submitted
    expires_at: float  # epoch seconds after which the worker has surely idle-expired

    def to_json(self) -> bytes:
        return json.dumps(asdict(self)).encode("utf-8")

    @classmethod
    def from_json(cls, data: bytes) -> WorkerEntry:
        d = json.loads(data.decode("utf-8"))
        return cls(
            worker=str(d["worker"]),
            subscription=str(d["subscription"]),
            job_name=d.get("job_name"),
            submitted_at=float(d["submitted_at"]),
            expires_at=float(d["expires_at"]),
        )


class RosterStore(Protocol):
    """The generation-aware object operations the roster needs (a thin GCS subset)."""

    def list(self, prefix: str) -> list[tuple[str, int]]:
        """``(object_name, generation)`` for every object under ``prefix``."""
        ...

    def get(self, name: str) -> bytes | None:
        """The object's bytes, or ``None`` when it is gone."""
        ...

    def put(self, name: str, data: bytes) -> None:
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

    def list(self, prefix: str) -> list[tuple[str, int]]:
        bucket = self._bucket()
        return [(b.name, int(b.generation)) for b in self._client.list_blobs(bucket, prefix=prefix)]

    def get(self, name: str) -> bytes | None:
        from google.api_core.exceptions import NotFound

        try:
            return self._bucket().blob(name).download_as_bytes()
        except NotFound:
            return None

    def put(self, name: str, data: bytes) -> None:
        self._bucket().blob(name).upload_from_string(data, content_type="application/json")

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

    def list(self, prefix: str) -> list[tuple[str, int]]:
        return sorted(
            (name, gen) for name, (_, gen) in self._objects.items() if name.startswith(prefix)
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
    """The idle-worker roster of one pool generation (see the module docstring)."""

    def __init__(
        self,
        output_bucket: str,
        topic: str,
        credentials: Any | None = None,
        store: RosterStore | None = None,
    ) -> None:
        bucket, base = parse_gcs_uri(output_bucket)
        self._prefix = f"{base + '/' if base else ''}{roster_prefix(topic)}"
        self._store: RosterStore = store if store is not None else GcsRosterStore(bucket, credentials)

    def _key(self, worker_id: str) -> str:
        return f"{self._prefix}{worker_id}.json"

    def add(self, entry: WorkerEntry) -> None:
        """Record ``entry`` as idle (called by the client that just submitted the worker)."""
        self._store.put(self._key(entry.worker), entry.to_json())

    def remove(self, worker_id: str) -> None:
        """Drop ``worker_id`` whatever its state (teardown; no precondition)."""
        for name, generation in self._store.list(self._key(worker_id)):
            self._store.delete_if(name, generation)

    def _read(self) -> list[tuple[str, int, WorkerEntry]]:
        rows = []
        for name, generation in self._store.list(self._prefix):
            data = self._store.get(name)
            if data is None:
                continue  # claimed by someone else between list and read
            try:
                entry = WorkerEntry.from_json(data)
            except (ValueError, KeyError, TypeError):
                continue  # not ours / corrupt: never dispatch on it
            rows.append((name, generation, entry))
        rows.sort(key=lambda row: row[2].submitted_at)
        return rows

    def entries(self) -> list[WorkerEntry]:
        """Every recorded idle worker, oldest first (expired ones included)."""
        return [entry for _, _, entry in self._read()]

    def prune_expired(self, now: float | None = None) -> list[WorkerEntry]:
        """Remove entries whose worker has surely idle-expired; return the ones removed."""
        now = time.time() if now is None else now
        pruned = []
        for name, generation, entry in self._read():
            if entry.expires_at <= now and self._store.delete_if(name, generation):
                pruned.append(entry)
        return pruned

    def claim(self, now: float | None = None) -> WorkerEntry | None:
        """Atomically take the oldest idle worker; ``None`` when the roster is empty.

        Skips expired entries (see :meth:`prune_expired`) and entries another client won
        the race for (the precondition delete fails, we move on to the next).
        """
        now = time.time() if now is None else now
        for name, generation, entry in self._read():
            if entry.expires_at <= now:
                continue
            if self._store.delete_if(name, generation):
                return entry
        return None

    def clear(self) -> list[WorkerEntry]:
        """Remove every entry (the pool is being retired); return what was recorded."""
        cleared = []
        for name, generation, entry in self._read():
            if self._store.delete_if(name, generation):
                cleared.append(entry)
        return cleared

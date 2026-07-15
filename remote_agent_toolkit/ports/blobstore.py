"""``BlobStore`` port + adapters (DESIGN.md §7).

Backs checkpoint, workspace tar, and artifacts. Protocol is stdlib-only;
``GcsBlobStore`` imports ``google.cloud.storage`` lazily.

``list(prefix)`` was added to the protocol (beyond the original
put/get/exists surface) because the Claude SDK ``SessionStore`` adapter
(:class:`~remote_agent_toolkit.checkpoint.session_store.BlobSessionStore`)
writes append-only batch objects under a per-session prefix and must enumerate
them in lexical order to reconstruct a transcript. Prefix listing is the one
extra capability that need demands, so it lives on the port.
"""

from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class BlobStore(Protocol):
    """Blob/tree storage over an opaque key space."""

    def put_bytes(self, key: str, data: bytes) -> None:
        """Store ``data`` at ``key``."""
        ...

    def get_bytes(self, key: str) -> bytes:
        """Load the bytes stored at ``key``."""
        ...

    def put_tree(self, key: str, local_dir: str) -> None:
        """Archive directory ``local_dir`` and store it under ``key`` (e.g. workspace tar)."""
        ...

    def get_tree(self, key: str, local_dir: str) -> None:
        """Restore the tree stored at ``key`` into ``local_dir``."""
        ...

    def exists(self, key: str) -> bool:
        """Whether a blob/tree exists at ``key``."""
        ...

    def list(self, prefix: str) -> list[str]:
        """List keys that start with ``prefix``, sorted lexically.

        Required by the ``SessionStore`` adapter, which stores append-only batch
        objects under a per-session prefix and replays them in lexical (==
        chronological) order.
        """
        ...

    def delete(self, key: str) -> None:
        """Delete the blob at ``key`` (no error if absent).

        Required by the per-invocation secrets handoff: the worker deletes the
        staged secrets object the moment it has read it.
        """
        ...


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    """Split ``gs://bucket/some/prefix`` (scheme optional) into ``(bucket, prefix)``."""
    rest = uri[len("gs://"):] if uri.startswith("gs://") else uri
    bucket, _, prefix = rest.partition("/")
    return bucket, prefix.strip("/")


def _normalize_key(root: Path, key: str) -> Path:
    """Resolve ``key`` under ``root``, rejecting traversal that escapes ``root``."""
    # Keys may contain "/" and are treated as relative paths under root.
    candidate = (root / key).resolve()
    root_resolved = root.resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ValueError(f"key escapes store root: {key!r}")
    return candidate


class GcsBlobStore:
    """GCS-backed :class:`BlobStore` (runtime identity = the RE service agent).

    ``google.cloud.storage`` is imported lazily inside the methods; the client and
    bucket handle are cached on first use.
    """

    def __init__(self, bucket: str, prefix: str = "") -> None:
        self.bucket = bucket
        self.prefix = prefix
        self._client: Any = None
        self._bucket_obj: Any = None

    def _get_bucket(self) -> Any:
        if self._bucket_obj is None:
            from google.cloud import storage  # lazy: keep import-time dep-free

            self._client = storage.Client()
            self._bucket_obj = self._client.bucket(self.bucket)
        return self._bucket_obj

    def _object_name(self, key: str) -> str:
        # ``prefix`` is concatenated verbatim (it already carries any trailing "/");
        # strip a leading "/" so we never produce an object name with a leading slash.
        return f"{self.prefix}{key}".lstrip("/")

    def put_bytes(self, key: str, data: bytes) -> None:
        self._get_bucket().blob(self._object_name(key)).upload_from_string(data)

    def get_bytes(self, key: str) -> bytes:
        from google.api_core.exceptions import NotFound  # lazy

        blob = self._get_bucket().blob(self._object_name(key))
        try:
            return blob.download_as_bytes()
        except NotFound as exc:
            raise KeyError(key) from exc

    def put_tree(self, key: str, local_dir: str) -> None:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            # arcname="." stores the dir *contents* so restore needs no extra nesting.
            tar.add(local_dir, arcname=".")
        buf.seek(0)
        self._get_bucket().blob(self._object_name(key)).upload_from_file(
            buf, content_type="application/gzip"
        )

    def get_tree(self, key: str, local_dir: str) -> None:
        data = self.get_bytes(key)
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            tar.extractall(local_dir, filter="data")

    def exists(self, key: str) -> bool:
        return self._get_bucket().blob(self._object_name(key)).exists()

    def list(self, prefix: str) -> list[str]:
        bucket = self._get_bucket()
        obj_prefix = self._object_name(prefix)
        # Strip the store prefix back off so callers see keys, not object names.
        strip = len(self._object_name(""))
        names = [b.name[strip:] for b in self._client.list_blobs(bucket, prefix=obj_prefix)]
        return sorted(names)

    def delete(self, key: str) -> None:
        from google.api_core.exceptions import NotFound  # lazy

        try:
            self._get_bucket().blob(self._object_name(key)).delete()
        except NotFound:
            pass  # already gone — deletion is idempotent


class LocalBlobStore:
    """Filesystem-backed :class:`BlobStore` for local dev.

    Keys are relative paths under ``root``; ``..`` segments that would escape the
    root are rejected.
    """

    def __init__(self, root: str) -> None:
        self.root = Path(root)

    def put_bytes(self, key: str, data: bytes) -> None:
        path = _normalize_key(self.root, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def get_bytes(self, key: str) -> bytes:
        path = _normalize_key(self.root, key)
        try:
            return path.read_bytes()
        except FileNotFoundError as exc:
            raise KeyError(key) from exc

    def put_tree(self, key: str, local_dir: str) -> None:
        path = _normalize_key(self.root, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(str(path), mode="w:gz") as tar:
            # arcname="." stores the dir *contents* so restore needs no extra nesting.
            tar.add(local_dir, arcname=".")

    def get_tree(self, key: str, local_dir: str) -> None:
        path = _normalize_key(self.root, key)
        if not path.exists():
            raise KeyError(key)
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        with tarfile.open(str(path), mode="r:gz") as tar:
            tar.extractall(local_dir, filter="data")

    def exists(self, key: str) -> bool:
        return _normalize_key(self.root, key).exists()

    def list(self, prefix: str) -> list[str]:
        root = self.root.resolve()
        if not root.is_dir():
            return []
        keys: list[str] = []
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                rel = os.path.relpath(os.path.join(dirpath, fn), root)
                # Normalize to "/"-separated keys regardless of host separator.
                rel = rel.replace(os.sep, "/")
                if rel.startswith(prefix):
                    keys.append(rel)
        return sorted(keys)

    def delete(self, key: str) -> None:
        path = _normalize_key(self.root, key)
        try:
            path.unlink()
        except FileNotFoundError:
            pass  # already gone — deletion is idempotent

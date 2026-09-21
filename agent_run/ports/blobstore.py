"""``BlobStore`` port + adapters (DESIGN.md §7).

Backs checkpoint, workspace tar, and artifacts. Protocol is stdlib-only;
``GcsBlobStore`` imports ``google.cloud.storage`` lazily.

``list(prefix)`` was added to the protocol (beyond the original
put/get/exists surface) because the Claude SDK ``SessionStore`` adapter
(:class:`~agent_run.checkpoint.session_store.BlobSessionStore`)
writes append-only batch objects under a per-session prefix and must enumerate
them in lexical order to reconstruct a transcript. Prefix listing is the one
extra capability that need demands, so it lives on the port.
"""

from __future__ import annotations

import io
import logging
import os
import tarfile
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


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

    def get_tree(self, key: str, local_dir: str) -> list[str]:
        """Restore the tree stored at ``key`` into ``local_dir``.

        Returns the archive member names that were NOT restored because extracting them
        would be unsafe (see :func:`extract_tree`); ``[]`` for a complete restore.
        """
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


# Process-wide credentials for every GcsBlobStore built without its own. The gemini
# WORKER sets the run's scoped token here at turn start (runtime/gemini/scoped_gcs.py) so
# every GCS access of the turn authorizes with that token instead of the runtime identity.
# ``None`` (the default, and what the client process always has) means ADC.
_default_credentials: Any = None


def set_default_gcs_credentials(credentials: Any | None) -> None:
    """Set (or with ``None`` clear) the credentials new ``GcsBlobStore`` instances use."""
    global _default_credentials
    _default_credentials = credentials


def default_gcs_credentials() -> Any | None:
    return _default_credentials


def _restore_member(member: tarfile.TarInfo, dest_path: str) -> tarfile.TarInfo:
    """The ``data`` filter, except that a symlink may point anywhere.

    ``tarfile``'s ``data`` filter refuses a symlink whose target is absolute or outside the
    destination. A workspace snapshot is full of those by construction — every virtualenv
    has ``.venv/bin/python -> /usr/local/bin/python3`` — and a symlink's target is only
    followed at use, never at extraction, so recreating it writes nothing outside the
    destination. Its *name* is still checked (``tar_filter`` keeps the path containment
    and mode stripping), ownership is dropped as ``data`` does, and a later member whose
    path runs *through* the symlink is still refused: the filters resolve member paths
    with ``realpath`` against what is already on disk. Hard links and everything else keep
    the full ``data`` treatment.
    """
    try:
        return tarfile.data_filter(member, dest_path)
    except (tarfile.AbsoluteLinkError, tarfile.LinkOutsideDestinationError):
        if not member.issym():
            raise
    return tarfile.tar_filter(member, dest_path).replace(
        uid=None, gid=None, uname=None, gname=None, deep=False
    )


def extract_tree(tar: tarfile.TarFile, local_dir: str) -> list[str]:
    """Extract a tree archive into ``local_dir``; return the member names skipped as unsafe.

    Members are filtered with :func:`_restore_member`. One the filter refuses (a path that
    escapes the destination, a device node or FIFO, a hard link pointing outside) is
    skipped and logged instead of aborting the whole extraction: a snapshot that restores
    all but a stray special file is worth far more than the fresh workspace the caller
    would otherwise fall back to, and the caller sees the names.
    """
    skipped: list[str] = []

    def restore_filter(member: tarfile.TarInfo, dest_path: str) -> tarfile.TarInfo | None:
        try:
            return _restore_member(member, dest_path)
        except tarfile.FilterError as exc:
            logger.warning("restore: skipping unsafe archive member %r: %s", member.name, exc)
            skipped.append(member.name)
            return None

    tar.extractall(local_dir, filter=restore_filter)
    return skipped


class GcsBlobStore:
    """GCS-backed :class:`BlobStore`.

    Authorizes with ``credentials`` when given, else with the process default set by
    :func:`set_default_gcs_credentials` (the worker's run-scoped token), else with ADC
    (the runtime identity in a worker, the operator identity in a client).
    ``google.cloud.storage`` is imported lazily inside the methods; the client and
    bucket handle are cached on first use.
    """

    def __init__(self, bucket: str, prefix: str = "", credentials: Any | None = None) -> None:
        self.bucket = bucket
        self.prefix = prefix
        self._credentials = credentials
        self._client: Any = None
        self._bucket_obj: Any = None

    def _get_bucket(self) -> Any:
        if self._bucket_obj is None:
            from google.cloud import storage  # lazy: keep import-time dep-free

            creds = self._credentials if self._credentials is not None else _default_credentials
            if creds is not None:
                # A bare access token carries no project; the storage client only needs one
                # for bucket-level calls, which the toolkit never makes with these credentials.
                project = os.environ.get("GOOGLE_CLOUD_PROJECT") or "_"
                self._client = storage.Client(project=project, credentials=creds)
            else:
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

    def get_tree(self, key: str, local_dir: str) -> list[str]:
        data = self.get_bytes(key)
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            return extract_tree(tar, local_dir)

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

    def get_tree(self, key: str, local_dir: str) -> list[str]:
        path = _normalize_key(self.root, key)
        if not path.exists():
            raise KeyError(key)
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        with tarfile.open(str(path), mode="r:gz") as tar:
            return extract_tree(tar, local_dir)

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

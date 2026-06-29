"""``BlobStore`` port + adapters (DESIGN.md §7).

Backs checkpoint, workspace tar, and artifacts. Protocol is stdlib-only;
``GcsBlobStore`` imports ``google.cloud.storage`` lazily. P0: protocol + stubs.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


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


class GcsBlobStore:
    """GCS-backed :class:`BlobStore` (runtime identity = the RE service agent; P1).

    ``google.cloud.storage`` is imported lazily inside the methods.
    """

    def __init__(self, bucket: str, prefix: str = "") -> None:
        self.bucket = bucket
        self.prefix = prefix

    def put_bytes(self, key: str, data: bytes) -> None:
        raise NotImplementedError("P1: write bytes to gs://{bucket}/{prefix}{key}.")

    def get_bytes(self, key: str) -> bytes:
        raise NotImplementedError("P1: read bytes from gs://{bucket}/{prefix}{key}.")

    def put_tree(self, key: str, local_dir: str) -> None:
        raise NotImplementedError("P1: tar local_dir and upload to GCS under key.")

    def get_tree(self, key: str, local_dir: str) -> None:
        raise NotImplementedError("P1: download the tar at key from GCS and extract into local_dir.")

    def exists(self, key: str) -> bool:
        raise NotImplementedError("P1: check object existence in GCS.")


class LocalBlobStore:
    """Filesystem-backed :class:`BlobStore` for local dev (P1)."""

    def __init__(self, root: str) -> None:
        self.root = root

    def put_bytes(self, key: str, data: bytes) -> None:
        raise NotImplementedError("P1: write bytes to {root}/{key}.")

    def get_bytes(self, key: str) -> bytes:
        raise NotImplementedError("P1: read bytes from {root}/{key}.")

    def put_tree(self, key: str, local_dir: str) -> None:
        raise NotImplementedError("P1: tar local_dir into {root}/{key}.")

    def get_tree(self, key: str, local_dir: str) -> None:
        raise NotImplementedError("P1: extract {root}/{key} into local_dir.")

    def exists(self, key: str) -> bool:
        raise NotImplementedError("P1: check path existence under root.")

"""Per-invocation secrets handoff over GCS (control plane stages, worker fetches+deletes).

Secret VALUES must never ride the invocation payload itself: the platform persists a
``run_query_job``'s input verbatim to ``jobs/<sid>_input.jsonl`` in the output bucket, and a
Pub/Sub dispatch message is retained until acked (up to the subscription's retention if no
worker claims it). So the payload carries only a *pointer*: the control plane stages the
secrets JSON at a single-use GCS key, and the worker deletes the object the moment it has
read it (the client also best-effort deletes on run completion, covering a worker that never
ran). Access is limited to the bucket's principals either way — the operator identity and the
RE service agent (which already holds ``objectAdmin`` on the output bucket).

SECURITY: functions here handle secret *values* — never log payloads, keys are fine.
``google.cloud.storage`` is imported lazily via :class:`GcsBlobStore`; passing an explicit
``store`` keeps the logic testable offline.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from ...ports.blobstore import GcsBlobStore, parse_gcs_uri

# Staged secrets live under this prefix inside the output bucket.
_SECRETS_PREFIX = "invocation-secrets"


def _store_for(bucket: str, store: Any | None) -> Any:
    return store if store is not None else GcsBlobStore(bucket)


def stage_secrets(
    output_bucket: str, session_id: str, secrets: dict, store: Any | None = None
) -> str:
    """Write ``secrets`` to a single-use object under ``output_bucket``; return its gs:// URI.

    The key embeds a fresh nonce so a resend (e.g. a retried turn) never reuses or
    overwrites a prior staging.
    """
    bucket, prefix = parse_gcs_uri(output_bucket)
    key = f"{prefix + '/' if prefix else ''}{_SECRETS_PREFIX}/{session_id}-{uuid.uuid4().hex}.json"
    _store_for(bucket, store).put_bytes(key, json.dumps(secrets).encode("utf-8"))
    return f"gs://{bucket}/{key}"


def fetch_and_delete_secrets(uri: str, store: Any | None = None) -> dict:
    """Read the staged secrets at ``uri`` and delete the object immediately.

    Returns ``{}`` when the object is missing (already consumed / cleaned up) or unreadable —
    the run proceeds without secrets rather than crashing; the caller may surface a
    (value-free) warning event.
    """
    bucket, key = parse_gcs_uri(uri)
    blobs = _store_for(bucket, store)
    try:
        data = blobs.get_bytes(key)
    except Exception:  # noqa: BLE001 — missing/unreadable → no secrets, never a crash
        return {}
    try:
        return json.loads(data.decode("utf-8"))
    finally:
        try:
            blobs.delete(key)  # single-use: gone the moment it is read
        except Exception:  # noqa: BLE001 — best-effort; client cleanup is the backstop
            pass


def delete_staged_secrets(uri: str, store: Any | None = None) -> None:
    """Best-effort client-side cleanup of a staged secrets object (worker is the main line)."""
    try:
        bucket, key = parse_gcs_uri(uri)
        _store_for(bucket, store).delete(key)
    except Exception:  # noqa: BLE001 — cleanup must never fail a run
        pass

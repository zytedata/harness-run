"""Per-invocation secrets handoff over GCS (control plane stages, worker fetches).

Secret VALUES must never ride the invocation payload itself: the platform persists a
``run_query_job``'s input verbatim to ``jobs/<sid>_input.jsonl`` in the output bucket, and a
Pub/Sub dispatch message is retained until acked (up to the subscription's retention if no
worker claims it). So the payload carries only a *pointer*: the control plane stages the
secrets JSON at a per-invocation GCS key, and the worker fetches it by that pointer.

Deletion is deferred to the END of the run — never read time. The platform's job runner
(Cloud Run jobs) automatically re-runs a crashed attempt with the SAME payload, and a warm
worker's dispatch message can be redelivered after a worker death — either way the retry
re-fetches the same pointer, so consuming the object on first read strands every retry
without credentials (observed live 2026-07-28/29: mid-turn platform restarts replayed the
turn, found the staging object gone, and failed the repo clone). Cleanup is layered:

1. the worker deletes the object after the turn's terminal result event went out — a
   completed turn (success or an application-level failure result) exits cleanly and is
   never retried, while a killed attempt dies before that point and leaves the object for
   its retry;
2. the client best-effort deletes on run completion (covers a worker that never ran);
3. a bucket lifecycle rule (:func:`ensure_secrets_lifecycle`, applied at deploy) reaps
   whatever both miss — e.g. every attempt crashed AND the client died.

Access is limited to the bucket's principals throughout — the operator identity and the
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

# Staged objects are reaped by the bucket lifecycle rule after this many days (backstop 3;
# GCS lifecycle granularity is whole days, so 1 is the tightest possible).
LIFECYCLE_DAYS = 1


def _store_for(bucket: str, store: Any | None) -> Any:
    return store if store is not None else GcsBlobStore(bucket)


def stage_secrets(
    output_bucket: str, session_id: str, secrets: dict, store: Any | None = None
) -> str:
    """Write ``secrets`` to a per-invocation object under ``output_bucket``; return its gs:// URI.

    The key embeds a fresh nonce so a resend (e.g. a retried turn) never reuses or
    overwrites a prior staging.
    """
    bucket, prefix = parse_gcs_uri(output_bucket)
    key = f"{prefix + '/' if prefix else ''}{_SECRETS_PREFIX}/{session_id}-{uuid.uuid4().hex}.json"
    _store_for(bucket, store).put_bytes(key, json.dumps(secrets).encode("utf-8"))
    return f"gs://{bucket}/{key}"


def fetch_secrets(uri: str, store: Any | None = None) -> dict:
    """Read the staged secrets at ``uri``; the object is left in place.

    NOT deleted on read: a platform retry of a killed attempt re-delivers the same pointer
    and must still find the object (see module docstring). The worker deletes it at the end
    of a completed turn via :func:`delete_staged_secrets`.

    Returns ``{}`` when the object is missing (already cleaned up) or unreadable — the run
    proceeds without secrets rather than crashing; the caller may surface a (value-free)
    warning event.
    """
    bucket, key = parse_gcs_uri(uri)
    try:
        data = _store_for(bucket, store).get_bytes(key)
        return json.loads(data.decode("utf-8"))
    except Exception:  # noqa: BLE001 — missing/unreadable → no secrets, never a crash
        return {}


def delete_staged_secrets(uri: str, store: Any | None = None) -> None:
    """Best-effort cleanup of a staged secrets object.

    Called by the worker after the turn's terminal result went out (the main line) and by
    the client on run completion (backstop); the lifecycle rule reaps anything both miss.
    """
    try:
        bucket, key = parse_gcs_uri(uri)
        _store_for(bucket, store).delete(key)
    except Exception:  # noqa: BLE001 — cleanup must never fail a run
        pass


def secrets_lifecycle_rule(output_bucket: str, days: int = LIFECYCLE_DAYS) -> tuple[str, dict]:
    """The (bucket name, GCS lifecycle rule) that reaps staged secrets older than ``days``.

    The rule matches only the staged-secrets prefix under ``output_bucket``'s own prefix, so
    it can never touch job outputs, checkpoints, or artifacts in the same bucket.
    """
    bucket, prefix = parse_gcs_uri(output_bucket)
    match = f"{prefix + '/' if prefix else ''}{_SECRETS_PREFIX}/"
    return bucket, {
        "action": {"type": "Delete"},
        "condition": {"age": days, "matchesPrefix": [match]},
    }


def ensure_secrets_lifecycle(
    output_bucket: str, days: int = LIFECYCLE_DAYS, bucket_obj: Any | None = None
) -> bool:
    """Idempotently ensure the delete-after-``days`` lifecycle rule on the output bucket.

    The last-resort reaper for staged secrets orphaned by a crash on BOTH sides (every job
    attempt killed and the client gone before its completion cleanup). Returns ``True`` when
    the rule is present (already or newly added), ``False`` on any failure — callers treat
    this as a warning, not an error (the rule is the third cleanup layer, and the operator
    may lack ``storage.buckets.update``). ``bucket_obj`` injects a fake for offline tests.
    """
    name, rule = secrets_lifecycle_rule(output_bucket, days)
    match = rule["condition"]["matchesPrefix"][0]
    try:
        if bucket_obj is None:
            from google.cloud import storage  # lazy: only deploy needs it

            bucket_obj = storage.Client().bucket(name)
            bucket_obj.reload()
        rules = list(bucket_obj.lifecycle_rules or [])
        for existing in rules:
            if (
                existing.get("action", {}).get("type") == "Delete"
                and match in (existing.get("condition", {}).get("matchesPrefix") or [])
            ):
                return True  # a delete rule already covers the prefix (any age)
        bucket_obj.lifecycle_rules = [*rules, rule]
        bucket_obj.patch()
        return True
    except Exception:  # noqa: BLE001 — best-effort backstop; never fail a deploy over it
        return False

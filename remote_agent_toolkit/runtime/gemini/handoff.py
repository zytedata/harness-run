"""GCS handoff between the control plane and the worker (secrets + configs).

The invocation channel is narrow and PERSISTED: the platform keeps a ``run_query_job``'s
input verbatim at ``jobs/<sid>_input.jsonl``, and a Pub/Sub dispatch message is retained
until acked. So neither secret values nor bulky config JSON ride the payload — the control
plane writes a GCS object and the payload carries only its *pointer*. Three object kinds:

* **Per-invocation secrets** (``invocation-secrets/<sid>-<nonce>.json``): secret VALUES.
  Deleted at the END of the turn, never at read time — the platform re-runs a crashed
  attempt with the SAME payload, and a redelivered dispatch must still find the object
  (observed live 2026-07-28/29). Cleanup is layered: worker delete after the terminal
  result → client delete on run completion → a 1-day bucket lifecycle rule.
* **The session config** (``session-config/<sid>.json``): the ``SessionConfig`` overlay
  bound at session start, written ONCE and pointed to by every turn of the session. NOT
  secret (configs are credential-free by construction) and deliberately NOT deleted:
  it is the ground-truth record of what the session was asked to run (post-mortem /
  replay), and it is what ``get_session`` re-attach reads back — a re-attaching process
  cannot substitute a different config. Reaped by a 30-day lifecycle rule.
* **Per-turn configs** (``turn-config/<sid>-<nonce>.json``): the ``TurnConfig`` overlay
  of one turn. Same non-secret, keep-for-debugging treatment as the session config.
* **The control inbox** (``control/<sid>/...`` and ``control-delivered/<sid>/...``, see
  ``control.py``): operator messages into a running turn, deleted by the worker on
  delivery, and the per-message delivered markers. Aged out with the configs.

A worker that is pointed at a config object and cannot read it FAILS the turn (terminal
error result): silently substituting the baked spec would be exactly the
wrong-configuration run the transport exists to prevent. Missing *secrets* degrade
gracefully instead (the run proceeds without them, with a warning event).

Access is limited to the bucket's principals throughout — the operator identity and the
RE service agent (which already holds ``objectAdmin`` on the output bucket).

SECURITY: only the secrets functions handle secret *values* — never log payloads; keys
are fine. ``google.cloud.storage`` is imported lazily via :class:`GcsBlobStore`; passing
an explicit ``store`` keeps the logic testable offline.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from ...ports.blobstore import GcsBlobStore, parse_gcs_uri

# Staged per-invocation secrets live under this prefix inside the output bucket.
_SECRETS_PREFIX = "invocation-secrets"

# Persisted session configs (one per session, written at session start, kept).
_SESSION_CONFIG_PREFIX = "session-config"

# Staged per-turn configs (one per turn, kept for post-mortem debugging).
_TURN_CONFIG_PREFIX = "turn-config"

# The control inbox + delivered markers (control.py); listed here for the lifecycle rule.
_CONTROL_PREFIXES = ("control", "control-delivered")

# Secrets are reaped by the bucket lifecycle rule after this many days (backstop 3;
# GCS lifecycle granularity is whole days, so 1 is the tightest possible).
LIFECYCLE_DAYS = 1

# Configs are kept much longer: they are the debugging/replay record of what a session
# was asked to run. Tiny JSON objects, so retention is cheap.
CONFIG_LIFECYCLE_DAYS = 30


def _store_for(bucket: str, store: Any | None) -> Any:
    return store if store is not None else GcsBlobStore(bucket)


def _put_json(output_bucket: str, key_suffix: str, payload: dict, store: Any | None) -> str:
    bucket, base = parse_gcs_uri(output_bucket)
    key = f"{base + '/' if base else ''}{key_suffix}"
    _store_for(bucket, store).put_bytes(key, json.dumps(payload).encode("utf-8"))
    return f"gs://{bucket}/{key}"


def stage_secrets(
    output_bucket: str, session_id: str, secrets: dict, store: Any | None = None
) -> str:
    """Write ``secrets`` to a per-invocation object under ``output_bucket``; return its gs:// URI.

    The key embeds a fresh nonce so a resend (e.g. a retried turn) never reuses or
    overwrites a prior staging.
    """
    suffix = f"{_SECRETS_PREFIX}/{session_id}-{uuid.uuid4().hex}.json"
    return _put_json(output_bucket, suffix, secrets, store)


def session_config_uri(output_bucket: str, session_id: str) -> str:
    """The stable gs:// URI a session's persisted config lives at (one object per session)."""
    bucket, base = parse_gcs_uri(output_bucket)
    return f"gs://{bucket}/{base + '/' if base else ''}{_SESSION_CONFIG_PREFIX}/{session_id}.json"


def persist_session_config(
    output_bucket: str, session_id: str, config_dict: dict, store: Any | None = None
) -> str:
    """Persist the session's ``SessionConfig.to_dict()`` at its stable key; return the URI.

    Written ONCE at session start (the key has no nonce: a session has exactly one
    config, bound at open). Every turn's payload points here; ``get_session`` re-attach
    reads it back. Never deleted by the toolkit — see the module docstring.
    """
    suffix = f"{_SESSION_CONFIG_PREFIX}/{session_id}.json"
    return _put_json(output_bucket, suffix, config_dict, store)


def load_session_config(
    output_bucket: str, session_id: str, store: Any | None = None
) -> tuple[str, dict] | None:
    """Read back a session's persisted config: ``(uri, config_dict)``, or ``None`` if absent.

    The re-attach path: a fresh process resuming a session by id was not there when the
    config was bound, so it recovers BOTH the pointer its turns must carry and the dict
    for client-side result parsing from the persisted object.
    """
    uri = session_config_uri(output_bucket, session_id)
    bucket, key = parse_gcs_uri(uri)
    try:
        data = _store_for(bucket, store).get_bytes(key)
        return uri, json.loads(data.decode("utf-8"))
    except Exception:  # noqa: BLE001 — absent/unreadable → the session has no config
        return None


def stage_turn_config(
    output_bucket: str, session_id: str, config_dict: dict, store: Any | None = None
) -> str:
    """Stage one turn's ``TurnConfig.to_dict()``; return its gs:// URI (nonce-keyed)."""
    suffix = f"{_TURN_CONFIG_PREFIX}/{session_id}-{uuid.uuid4().hex}.json"
    return _put_json(output_bucket, suffix, config_dict, store)


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


def fetch_config(uri: str, store: Any | None = None) -> dict | None:
    """Read the config object (session or turn) at ``uri``; the object is left in place.

    Returns ``None`` when missing or unreadable. The caller must FAIL the turn on
    ``None``: silently running the baked spec instead of the requested configuration is
    exactly the class of bug this transport exists to prevent.
    """
    bucket, key = parse_gcs_uri(uri)
    try:
        data = _store_for(bucket, store).get_bytes(key)
        return json.loads(data.decode("utf-8"))
    except Exception:  # noqa: BLE001 — missing/unreadable → None; the caller fails the turn
        return None


def delete_staged_secrets(uri: str, store: Any | None = None) -> None:
    """Best-effort cleanup of a staged secrets object.

    Called by the worker after the turn's terminal result went out (the main line) and by
    the client on run completion (backstop); the lifecycle rule reaps anything both miss.
    Config objects are deliberately NOT deleted (see the module docstring).
    """
    try:
        bucket, key = parse_gcs_uri(uri)
        _store_for(bucket, store).delete(key)
    except Exception:  # noqa: BLE001 — cleanup must never fail a run
        pass


def handoff_lifecycle_rules(output_bucket: str) -> tuple[str, list[dict]]:
    """The (bucket name, GCS lifecycle rules) that reap handoff objects.

    Three rules: staged secrets after :data:`LIFECYCLE_DAYS` (they are values; the rule
    is the last-resort reaper behind the worker/client deletes), configs after
    :data:`CONFIG_LIFECYCLE_DAYS` (kept for post-mortem debugging), and the control inbox
    objects + delivered markers after :data:`CONFIG_LIFECYCLE_DAYS` too (the worker
    deletes inbox objects on delivery; the markers only need to outlive a session's
    retries). A separate rule for the control prefixes, so a bucket configured by an
    older revision gets exactly the missing coverage appended. Each rule matches only its
    prefixes under ``output_bucket``'s own prefix, so it can never touch job outputs,
    checkpoints, or artifacts in the same bucket.
    """
    bucket, prefix = parse_gcs_uri(output_bucket)
    base = f"{prefix + '/' if prefix else ''}"
    return bucket, [
        {
            "action": {"type": "Delete"},
            "condition": {
                "age": LIFECYCLE_DAYS,
                "matchesPrefix": [f"{base}{_SECRETS_PREFIX}/"],
            },
        },
        {
            "action": {"type": "Delete"},
            "condition": {
                "age": CONFIG_LIFECYCLE_DAYS,
                "matchesPrefix": [
                    f"{base}{_SESSION_CONFIG_PREFIX}/",
                    f"{base}{_TURN_CONFIG_PREFIX}/",
                ],
            },
        },
        {
            "action": {"type": "Delete"},
            "condition": {
                "age": CONFIG_LIFECYCLE_DAYS,
                "matchesPrefix": [f"{base}{p}/" for p in _CONTROL_PREFIXES],
            },
        },
    ]


def ensure_handoff_lifecycle(output_bucket: str, bucket_obj: Any | None = None) -> bool:
    """Idempotently ensure the handoff lifecycle rules on the output bucket.

    Returns ``True`` when every needed rule is present (already or newly added),
    ``False`` on any failure — callers treat this as a warning, not an error (the rules
    are backstops, and the operator may lack ``storage.buckets.update``). Buckets
    configured by older revisions keep their old rules (harmless: broader deletes of the
    same prefixes); only missing coverage is appended. ``bucket_obj`` injects a fake for
    offline tests.
    """
    name, wanted = handoff_lifecycle_rules(output_bucket)
    try:
        if bucket_obj is None:
            from google.cloud import storage  # lazy: only deploy needs it

            bucket_obj = storage.Client().bucket(name)
            bucket_obj.reload()
        rules = list(bucket_obj.lifecycle_rules or [])
        covered: set[str] = set()
        for existing in rules:
            if existing.get("action", {}).get("type") == "Delete":
                covered.update(existing.get("condition", {}).get("matchesPrefix") or [])
        missing = [
            rule
            for rule in wanted
            if not all(m in covered for m in rule["condition"]["matchesPrefix"])
        ]
        if not missing:
            return True
        bucket_obj.lifecycle_rules = [*rules, *missing]
        bucket_obj.patch()
        return True
    except Exception:  # noqa: BLE001 — best-effort backstop; never fail a deploy over it
        return False

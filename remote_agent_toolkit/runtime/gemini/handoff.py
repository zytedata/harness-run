"""GCS records the control plane keeps next to a session (configs) and their lifecycle.

With the sandbox runtime nothing rides a persisted platform channel any more: the turn
(prompt, configs, secrets, tokens) goes to the worker in one HTTP body. What the client
still writes to GCS is the **record** — small, non-secret JSON, kept for post-mortem and
re-attach:

* **The session config** (``session-config/<sid>.json``): the ``SessionConfig`` overlay
  bound at session start, written ONCE. It is what ``get_session`` re-attach reads back —
  a re-attaching process cannot substitute a different config. Reaped after 30 days.
* **Per-turn configs** (``turn-config/<sid>-<nonce>.json``): the ``TurnConfig`` overlay of
  one turn. Same treatment.
* **The run's token objects** (``invocation-secrets/<sid>-…``, ``scoped_gcs.py``): the
  refresh objects for the run-scoped GCS token and the model token. Values, so deleted by
  the client at turn end and reaped after a day as the backstop.
* **The deploy record** (``deploys/<name>/<template-id>.json``): what ``deploy`` baked
  (spec, model account, resources, pool settings) so ``get_engine`` knows the engine it
  addresses. Kept.

``google.cloud.storage`` is imported lazily via :class:`GcsBlobStore`; passing an explicit
``store`` keeps the logic testable offline.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from ...ports.blobstore import GcsBlobStore, parse_gcs_uri

_SECRETS_PREFIX = "invocation-secrets"  # the run's token refresh objects (scoped_gcs.py)
_SESSION_CONFIG_PREFIX = "session-config"
_TURN_CONFIG_PREFIX = "turn-config"
_DEPLOYS_PREFIX = "deploys"

# Token objects are reaped by the bucket lifecycle rule after this many days (the client
# deletes them at turn end; GCS lifecycle granularity is whole days, so 1 is the tightest).
LIFECYCLE_DAYS = 1
# Configs are the debugging/replay record of what a session was asked to run.
CONFIG_LIFECYCLE_DAYS = 30


def _store_for(bucket: str, store: Any | None) -> Any:
    return store if store is not None else GcsBlobStore(bucket)


def _put_json(output_bucket: str, key_suffix: str, payload: dict, store: Any | None) -> str:
    bucket, base = parse_gcs_uri(output_bucket)
    key = f"{base + '/' if base else ''}{key_suffix}"
    _store_for(bucket, store).put_bytes(key, json.dumps(payload).encode("utf-8"))
    return f"gs://{bucket}/{key}"


def _get_json(uri: str, store: Any | None) -> dict | None:
    """The JSON object at ``uri``; ``None`` for a known missing object; raises otherwise."""
    bucket, key = parse_gcs_uri(uri)
    try:
        data = _store_for(bucket, store).get_bytes(key)
    except KeyError:
        return None
    except Exception:  # noqa: BLE001 — do not expose service errors/credential material
        raise RuntimeError(f"could not read {uri.rsplit('/', 1)[-1]}; refusing fallback") from None
    try:
        parsed = json.loads(data.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("not an object")
    except (ValueError, UnicodeError):
        raise RuntimeError(f"invalid JSON record at {uri.rsplit('/', 1)[-1]}; refusing fallback") from None
    return parsed


def session_config_uri(output_bucket: str, session_id: str) -> str:
    """The stable gs:// URI a session's persisted config lives at (one object per session)."""
    bucket, base = parse_gcs_uri(output_bucket)
    return f"gs://{bucket}/{base + '/' if base else ''}{_SESSION_CONFIG_PREFIX}/{session_id}.json"


def persist_session_config(
    output_bucket: str, session_id: str, config_dict: dict, store: Any | None = None
) -> str:
    """Persist the session's ``SessionConfig.to_dict()`` at its stable key; return the URI."""
    return _put_json(output_bucket, f"{_SESSION_CONFIG_PREFIX}/{session_id}.json", config_dict, store)


def load_session_config(
    output_bucket: str, session_id: str, store: Any | None = None
) -> tuple[str, dict] | None:
    """Read back a session's config as ``(uri, dict)``, or ``None`` only for a known missing object.

    A transient/denied/invalid read raises: it must stay retryable, never be cached as
    "unconfigured".
    """
    uri = session_config_uri(output_bucket, session_id)
    found = _get_json(uri, store)
    return None if found is None else (uri, found)


def stage_turn_config(
    output_bucket: str, session_id: str, config_dict: dict, store: Any | None = None
) -> str:
    """Record one turn's ``TurnConfig.to_dict()``; return its gs:// URI (nonce-keyed)."""
    suffix = f"{_TURN_CONFIG_PREFIX}/{session_id}-{uuid.uuid4().hex}.json"
    return _put_json(output_bucket, suffix, config_dict, store)


def deploy_record_uri(output_bucket: str, name: str, template_id: str) -> str:
    bucket, base = parse_gcs_uri(output_bucket)
    return f"gs://{bucket}/{base + '/' if base else ''}{_DEPLOYS_PREFIX}/{name}/{template_id}.json"


def write_deploy_record(
    output_bucket: str, name: str, template_id: str, record: dict, store: Any | None = None
) -> str:
    """Persist what ``deploy`` baked into template ``template_id`` of engine ``name``."""
    return _put_json(output_bucket, f"{_DEPLOYS_PREFIX}/{name}/{template_id}.json", record, store)


def load_deploy_record(
    output_bucket: str, name: str, template_id: str, store: Any | None = None
) -> dict | None:
    """The deploy record of a template, or ``None`` when there is none (or it is unreadable).

    Best-effort on purpose: a ``get_engine`` handle works without it (addressing only, the
    deploy-baked spec unknown client-side), so a missing or unreadable record degrades to
    that instead of failing the lookup.
    """
    try:
        return _get_json(deploy_record_uri(output_bucket, name, template_id), store)
    except Exception:  # noqa: BLE001
        return None


def handoff_lifecycle_rules(output_bucket: str) -> tuple[str, list[dict]]:
    """The (bucket name, GCS lifecycle rules) that reap the run records.

    Two rules: token objects after :data:`LIFECYCLE_DAYS` (values; the client's delete at
    turn end is the main line), configs after :data:`CONFIG_LIFECYCLE_DAYS`. Each rule
    matches only its prefixes under ``output_bucket``'s own prefix, so it can never touch
    events, checkpoints, artifacts or deploy records in the same bucket.
    """
    bucket, prefix = parse_gcs_uri(output_bucket)
    base = f"{prefix + '/' if prefix else ''}"
    return bucket, [
        {
            "action": {"type": "Delete"},
            "condition": {"age": LIFECYCLE_DAYS, "matchesPrefix": [f"{base}{_SECRETS_PREFIX}/"]},
        },
        {
            "action": {"type": "Delete"},
            "condition": {
                "age": CONFIG_LIFECYCLE_DAYS,
                "matchesPrefix": [f"{base}{_SESSION_CONFIG_PREFIX}/", f"{base}{_TURN_CONFIG_PREFIX}/"],
            },
        },
    ]


def ensure_handoff_lifecycle(
    output_bucket: str, bucket_obj: Any | None = None, *, credentials: Any | None = None
) -> bool:
    """Idempotently ensure the lifecycle rules on the output bucket.

    Returns ``True`` when every needed rule is present (already or newly added), ``False``
    on any failure — callers treat this as a warning (the rules are backstops, and the
    operator may lack ``storage.buckets.update``). Existing policy is kept intact; only
    missing coverage is appended. ``bucket_obj`` injects a fake for offline tests.
    """
    name, wanted = handoff_lifecycle_rules(output_bucket)
    try:
        if bucket_obj is None:
            from google.cloud import storage  # lazy: only deploy needs it

            client = (storage.Client(credentials=credentials, project="_")
                      if credentials is not None else storage.Client())
            bucket_obj = client.bucket(name)
            bucket_obj.reload()
        rules = list(bucket_obj.lifecycle_rules or [])
        # Conditions in a GCS lifecycle rule are conjunctive: only a plain age+prefix
        # Delete rule counts as coverage.
        covered: dict[str, int] = {}
        for existing in rules:
            condition = existing.get("condition", {})
            age = condition.get("age")
            if (existing.get("action", {}).get("type") == "Delete"
                    and type(age) is int and age >= 0
                    and not (set(condition) - {"age", "matchesPrefix"})):
                for prefix in condition.get("matchesPrefix") or []:
                    covered[prefix] = min(age, covered.get(prefix, age))
        missing = [
            rule for rule in wanted
            if not all(covered.get(m, float("inf")) <= rule["condition"]["age"]
                       for m in rule["condition"]["matchesPrefix"])
        ]
        if not missing:
            return True
        bucket_obj.lifecycle_rules = [*rules, *missing]
        bucket_obj.patch()
        return True
    except Exception:  # noqa: BLE001 — best-effort backstop; never fail a deploy over it
        return False

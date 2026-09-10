"""Run-scoped GCS access for gemini workers.

Why this exists (found and verified 2026-09-02, see README "Secrets & security"): the
worker container runs the agent's shell as the same user as the worker, and that
container authenticates as the Agent Runtime service agent through the metadata server.
So a shell command inside the agent can mint that identity's token. With the identity
holding ``objectAdmin`` on the output bucket, one run could list and read every other
run's staged secrets, transcripts and configs, across every engine sharing the bucket.

The fix: the **client** mints one short-lived, downscoped token per turn (a Credential
Access Boundary: this bucket, only this run's object prefixes) and the **worker** does
all of the turn's GCS work with it instead of the runtime identity. The runtime identity
can then lose its bucket-wide role (the migration step in the README): a token minted
from the metadata server no longer opens the bucket, and the run's own token opens only
the run's own objects.

The token rides the invocation like the other pointers do (a directive line on the cold
path, a payload field on the warm path). It is a bearer token, so treat it as one: it is
worth this run's own objects for at most an hour, and nothing else.

Long turns: a downscoped token lives as long as its source token (at most an hour). The
client refreshes it while the run is live by overwriting one object under the run's own
secrets prefix, and the worker's credentials re-read that object shortly before expiry
(google-auth refreshes ~4 minutes early, while the old token still works).
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import urllib.parse
import urllib.request
from typing import Any, Callable

from ...checkpoint.session_store import _claude_session_id
from ...ports.blobstore import GcsBlobStore, parse_gcs_uri

logger = logging.getLogger(__name__)

# The object the client overwrites with a fresh token while the run is live. It sits under
# the run's own secrets prefix, so the run's current token can read it and nobody else's can.
TOKEN_KEY_SUFFIX = "gcs-token.json"

# The client re-mints this often. Source tokens last an hour; google-auth starts refreshing
# ~4 minutes before expiry, so a 25-minute cadence leaves a wide margin.
REFRESH_EVERY_S = 25 * 60

_CLOUD_PLATFORM = "https://www.googleapis.com/auth/cloud-platform"


def token_key(base: str, session_id: str) -> str:
    """Object key of the run's refresh token file (``base`` = bucket-relative prefix + '/')."""
    return f"{base}invocation-secrets/{session_id}-{TOKEN_KEY_SUFFIX}"


def run_object_prefixes(output_bucket: str, session_id: str) -> tuple[str, str, list[str]]:
    """``(bucket, base, prefixes)``: every object-name prefix one run touches in the bucket.

    ``output_bucket`` is the engine's ``gs://bucket[/prefix]``. The list is the whole GCS
    surface of a turn (handoff objects, the events mirror, checkpoints, artifacts); the
    checkpoint keys use the Claude session id the worker derives from the toolkit session
    id (``session_store._claude_session_id``), so the mapping is done here too. The
    control inbox is here because the worker polls it with these credentials: without
    its prefix every list would 403 and ``send()`` / ``interrupt()`` would never arrive.

    Every writer of run objects must appear here: ``tests/test_scoped_gcs.py`` drives the
    checkpoint code (Codex thread persist + resume, transcript store, workspace snapshot)
    through a recording store and fails on any key outside these prefixes. A prefix costs
    token budget (see :func:`access_boundary`): 10 rules is the google-auth maximum.
    """
    bucket, prefix = parse_gcs_uri(output_bucket)
    base = f"{prefix}/" if prefix else ""
    csid = _claude_session_id(session_id)
    prefixes = [
        f"{base}invocation-secrets/{session_id}-",        # staged secrets + the refresh token
        f"{base}session-config/{session_id}.json",        # the session's persisted config
        f"{base}turn-config/{session_id}-",               # this turn's config
        f"{base}events/{session_id}/",                    # the live mirror / durable history
        f"{base}checkpoints/sessions/{csid}/",            # transcript batches
        f"{base}checkpoints/workspace/{csid}.tar.gz",     # workspace snapshot
        f"{base}checkpoints/codex-threads/{csid}/",       # Codex conversation (harness/codex.py)
        f"{base}artifacts/{session_id}/",                 # produced files
        f"{base}control/{session_id}/",                   # the turn's control inbox (control.py)
        f"{base}control-delivered/{session_id}/",         # its delivered-message markers
    ]
    return bucket, base, prefixes


# Prefixes the worker LISTS with the run token: the control inbox (control.py polls it) and
# the checkpoint transcript directory (session_store.load enumerates batches on resume).
# Every other object of a turn is read or written by exact name. Kept to the minimum on
# purpose: STS caps the minted token at ~10.7k chars INCLUDING the caller's own token, and
# a list clause adds ~500 chars per rule. Nine list clauses fit under a ~250-char user token
# and overflow a ~1,100-char service-account token with "invalid_request" (measured
# 2026-09-08, zapi-workflow-bot); two fit either with room to spare.
_LISTED_PREFIXES = ("checkpoints/sessions/", "control/")


def access_boundary(bucket: str, prefixes: list[str], base: str = "") -> Any:
    """A Credential Access Boundary allowing objectAdmin on ``bucket`` for ``prefixes`` only.

    Each rule carries an object-name condition (get/create/delete/exists on objects under
    the prefix); only the prefixes in ``_LISTED_PREFIXES`` (relative to ``base``) also get
    the list condition GCS evaluates for ``objects.list`` (the request's ``prefix``
    parameter must start with the allowed prefix).
    """
    from google.auth import downscoped

    rules = []
    for p in prefixes:
        obj = f"projects/_/buckets/{bucket}/objects/{p}"
        expression = f'resource.name.startsWith("{obj}")'
        if p[len(base):].startswith(_LISTED_PREFIXES):
            expression += (
                ' || api.getAttribute("storage.googleapis.com/objectListPrefix", "")'
                f'.startsWith("{p}")'
            )
        rules.append(
            downscoped.AccessBoundaryRule(
                available_resource=f"//storage.googleapis.com/projects/_/buckets/{bucket}",
                available_permissions=["inRole:roles/storage.objectAdmin"],
                availability_condition=downscoped.AvailabilityCondition(
                    expression, title="run-scoped prefix"
                ),
            )
        )
    return downscoped.CredentialAccessBoundary(rules=rules)


def mint_run_token(
    source_credentials: Any | None, output_bucket: str, session_id: str
) -> tuple[str, _dt.datetime | None]:
    """Mint the run's downscoped token from the client's own credentials.

    Returns ``(token, expiry)``; ``expiry`` is a naive UTC datetime (google-auth's
    convention) or ``None`` when the token service did not report one. Raises when the
    source credentials cannot be resolved or the token service refuses.
    """
    import google.auth
    from google.auth import downscoped
    from google.auth.transport.requests import Request

    if source_credentials is None:
        source_credentials, _ = google.auth.default(scopes=[_CLOUD_PLATFORM])
    bucket, base, prefixes = run_object_prefixes(output_bucket, session_id)
    creds = downscoped.Credentials(
        source_credentials=source_credentials,
        credential_access_boundary=access_boundary(bucket, prefixes, base),
    )
    creds.refresh(Request())
    if not creds.token:
        raise RuntimeError("the token service returned no downscoped token")
    return creds.token, creds.expiry


def write_run_token(
    output_bucket: str, session_id: str, token: str, expiry: _dt.datetime | None,
    store: Any | None = None,
) -> None:
    """Client side: (over)write the run's refresh token object with a fresh token."""
    bucket, base, _prefixes = run_object_prefixes(output_bucket, session_id)
    blobs = store if store is not None else GcsBlobStore(bucket)
    payload = {"token": token, "expiry": expiry.isoformat() if expiry else None}
    blobs.put_bytes(token_key(base, session_id), json.dumps(payload).encode("utf-8"))


def delete_run_token(output_bucket: str, session_id: str, store: Any | None = None) -> None:
    """Client side: remove the refresh token object once the run is over (idempotent)."""
    bucket, base, _prefixes = run_object_prefixes(output_bucket, session_id)
    blobs = store if store is not None else GcsBlobStore(bucket)
    blobs.delete(token_key(base, session_id))


def _parse_expiry(value: Any) -> _dt.datetime | None:
    if not value:
        return None
    try:
        parsed = _dt.datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(_dt.timezone.utc).replace(tzinfo=None)
    return parsed


def _http_get_json(url: str, bearer: str) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {bearer}"})
    with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310 — fixed https host
        return json.loads(resp.read().decode("utf-8"))


def worker_credentials(
    token: str,
    output_bucket: str | None,
    session_id: str,
    *,
    expiry: _dt.datetime | None = None,
    fetch: Callable[[str, str], dict] | None = None,
) -> Any:
    """Worker side: google-auth credentials carrying the run's token, refreshing from GCS.

    The refresh handler reads the run's token object with the token that is about to
    expire (google-auth refreshes early, so it is still valid) and swaps in the new one.
    When there is no bucket to read from or the read fails, the current token is kept and
    the failure is logged; the turn then keeps working until the token really expires.
    ``fetch`` is injectable for tests (``(url, bearer) -> dict``).
    """
    from google.oauth2 import credentials as oauth2_credentials

    state = {"token": token}
    get_json = fetch or _http_get_json
    key = None
    if output_bucket:
        bucket, base, _prefixes = run_object_prefixes(output_bucket, session_id)
        key = (bucket, token_key(base, session_id))

    def retry_soon() -> _dt.datetime:
        # google-auth needs a datetime back; a few minutes ahead makes it ask again soon
        # while the current token, which may still be valid, keeps being used.
        return _dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None) + _dt.timedelta(minutes=5)

    def refresh_handler(_request: Any, scopes: Any = None) -> tuple[str, _dt.datetime]:
        if key is None:
            return state["token"], retry_soon()
        bucket, name = key
        url = (
            f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/"
            f"{urllib.parse.quote(name, safe='')}?alt=media"
        )
        try:
            data = get_json(url, state["token"])
            fresh = data.get("token")
            if fresh:
                state["token"] = fresh
                return fresh, _parse_expiry(data.get("expiry")) or retry_soon()
        except Exception:  # noqa: BLE001 — keep the current token; never crash the turn
            logger.warning("run-scoped GCS token refresh failed; keeping the current token")
        return state["token"], retry_soon()

    return oauth2_credentials.Credentials(
        # Legacy payloads without expiry get a conservative retry schedule, not
        # google-auth's expiry=None (non-expiring) semantics. This cannot recover
        # an already expired bearer; never fall back to ambient credentials.
        token=token, expiry=expiry or retry_soon(), refresh_handler=refresh_handler
    )


def output_bucket_from_env(events_uri: str | None) -> str | None:
    """The engine's ``gs://bucket[/prefix]`` from its ``AGENT_EVENTS_GCS`` (``.../events``)."""
    if not events_uri:
        return None
    if events_uri.endswith("/events"):
        return events_uri[: -len("/events")]
    return events_uri

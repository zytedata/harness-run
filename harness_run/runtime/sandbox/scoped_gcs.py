"""Run-scoped GCS access for sandbox workers.

The sandbox has no Google identity that could open the output bucket (DESIGN.md §13.1),
so the **client** mints one short-lived, downscoped token per turn (a Credential Access
Boundary: this bucket, only this run's object prefixes) and the **worker** does all of the
turn's GCS work with it — the event mirror, checkpoints, transcripts, artifacts. The
agent's shell can read the token (it is in the worker's process), which is why it opens
only the run's own objects: nothing another run wrote, nothing under another prefix.

The token rides the ``/turn`` body. It is a bearer token, so treat it as one: it is worth
this run's own objects for at most an hour, and nothing else.

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

# Worker side: never ride a token closer than this to its death. The worker fetches its
# replacement WITH the current token, so an expired one can never fetch anything — the
# refresh must happen while the current token still works. The client sends the token's
# real expiry in the turn body, so the horizon is a clamp on top of that, not the only
# schedule: it also covers a token object that reports an expiry far in the future (which
# would park the worker on one token past the client's next re-mint) and a payload that
# carries no expiry at all. Same constant as on the query-job runtime (#87).
REFRESH_HORIZON = _dt.timedelta(minutes=10)

# Consecutive refresh failures after which the token is presumed dead for good. Below this
# a failure is usually a transient blip and the current token still works; at it, the run
# has almost certainly lost GCS for the rest of the turn.
DEAD_TOKEN_AFTER = 3

_CLOUD_PLATFORM = "https://www.googleapis.com/auth/cloud-platform"


def _now() -> _dt.datetime:
    """Naive UTC now, the shape google-auth keeps credential expiries in (patched in tests)."""
    return _dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None)


def token_key(base: str, session_id: str) -> str:
    """Object key of the run's refresh token file (``base`` = bucket-relative prefix + '/')."""
    return f"{base}invocation-secrets/{session_id}-{TOKEN_KEY_SUFFIX}"


def run_object_prefixes(output_bucket: str, session_id: str) -> tuple[str, str, list[str]]:
    """``(bucket, base, prefixes)``: every object-name prefix one run touches in the bucket.

    ``output_bucket`` is the engine's ``gs://bucket[/prefix]``. The list is the whole GCS
    surface of a turn (handoff objects, the events mirror, checkpoints, artifacts); the
    checkpoint keys use the Claude session id the worker derives from the toolkit session
    id (``session_store._claude_session_id``), so the mapping is done here too.

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
    ]
    return bucket, base, prefixes


# Prefixes the worker LISTS with the run token: the checkpoint transcript directory
# (session_store.load enumerates batches on resume). Every other object of a turn is read
# or written by exact name. Kept to the minimum on purpose: STS caps the minted token at
# ~10.7k chars INCLUDING the caller's own token, and a list clause adds ~500 chars per
# rule. Nine list clauses overflowed a ~1,100-char service-account token with
# "invalid_request" (measured 2026-09-08); one fits with room to spare.
_LISTED_PREFIXES = ("checkpoints/sessions/",)


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
        # CEL accepts JSON string escapes. Keep caller-controlled names inside
        # literals without restoring list clauses for exact-name-only objects:
        # those extra clauses overflow STS tokens for service-account callers.
        expression = f'resource.name.startsWith({json.dumps(obj, ensure_ascii=False)})'
        if p[len(base):].startswith(_LISTED_PREFIXES):
            expression += (
                ' || api.getAttribute("storage.googleapis.com/objectListPrefix", "")'
                f'.startsWith({json.dumps(p, ensure_ascii=False)})'
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
    expire and swaps in the new one. That read only works while the current token is alive,
    so the credential is always given an expiry no further out than ``REFRESH_HORIZON`` —
    including when the token object reports one far in the future. google-auth refreshes a
    little before the expiry it is shown, so this keeps the swap happening on a live token.
    When there is no bucket to read from or the read fails, the current token is kept and
    the failure is logged once (and escalated once more, as an error, when it keeps failing
    and the token is presumed dead); the turn then keeps working until the token really
    expires. ``fetch`` is injectable for tests (``(url, bearer) -> dict``).
    """
    from google.oauth2 import credentials as oauth2_credentials

    state: dict[str, Any] = {"token": token, "failures": 0, "escalated": False}
    get_json = fetch or _http_get_json
    key = None
    if output_bucket:
        bucket, base, _prefixes = run_object_prefixes(output_bucket, session_id)
        key = (bucket, token_key(base, session_id))

    def retry_soon() -> _dt.datetime:
        # google-auth needs a datetime back; a few minutes ahead makes it ask again soon
        # while the current token, which may still be valid, keeps being used.
        return _now() + _dt.timedelta(minutes=5)

    def within_horizon(when: _dt.datetime | None) -> _dt.datetime:
        # Refresh no later than the horizon, whatever the token object claims. A real expiry
        # further out would park the worker on one token past the client's next re-mint.
        horizon = _now() + REFRESH_HORIZON
        return min(when, horizon) if when else horizon

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
                state["failures"], state["escalated"] = 0, False
                state["token"] = fresh
                return fresh, within_horizon(_parse_expiry(data.get("expiry")) or retry_soon())
        except Exception as exc:  # noqa: BLE001 — keep the current token; never crash the turn
            # Never raise: the turn keeps working and a client on the live ``/events`` channel
            # still gets its result, so a dead token must not take a healthy run down with it.
            # But say so ONCE, loudly, with the reason — logging the same line on every retry
            # (hundreds per run) buries it, and nothing ever said the token was gone for good.
            state["failures"] += 1
            if state["failures"] == 1:
                logger.warning(
                    "run-scoped GCS token refresh failed for %s; keeping the current token "
                    "(%s: %s)", session_id, type(exc).__name__, str(exc)[:200],
                )
            elif state["failures"] >= DEAD_TOKEN_AFTER and not state["escalated"]:
                state["escalated"] = True
                logger.error(
                    "run-scoped GCS token for %s could not be renewed %d times running and is "
                    "presumed dead; the replacement can only be fetched WITH a live token, so "
                    "every GCS write from this worker will now fail silently. The run itself "
                    "continues and a client on the live /events channel still gets its result, "
                    "but the durable mirror stops here: a session re-attached later will find "
                    "no record of the rest of this turn (%s: %s)",
                    session_id, state["failures"], type(exc).__name__, str(exc)[:200],
                )
        return state["token"], retry_soon()

    return oauth2_credentials.Credentials(
        # Payloads without expiry get a conservative retry schedule, not google-auth's
        # expiry=None (non-expiring) semantics; a known expiry is clamped to the horizon.
        # This cannot recover an already expired bearer; never fall back to ambient
        # credentials.
        token=token, expiry=within_horizon(expiry or retry_soon()), refresh_handler=refresh_handler
    )


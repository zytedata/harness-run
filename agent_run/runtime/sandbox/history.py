"""Past-session enumeration + event history for deployed engines (the GCS event mirror).

One durable record holds a session's events: the **mirror** — ``events/<sid>/*.jsonl``
under the output bucket, small JSONL batch files the worker streams AS THE TURN RUNS
(``stream.MirrorStream``, written with the run-scoped token). It is the client's fallback
live channel (``stream.tail_stream``) and the record ``Session.history()``,
``list_sessions()`` and a re-attached session's ``last_result`` read. Lexical name order ==
chronological. GCP imports are lazy; ``store`` params take any ``BlobStore`` for offline
tests.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from ...events import AgentEvent
from ...ports.blobstore import GcsBlobStore, parse_gcs_uri

_EVENTS_PREFIX = "events"


def _json_safe(value: Any) -> Any:
    """Coerce ``value`` to something JSON round-trips (stringify what does not)."""
    return json.loads(json.dumps(value, default=str))


# -- mirror format (write side runs in the worker; read side at the client) -----


def mirror_line(event: AgentEvent) -> dict:
    """One JSON-safe dict per event; the mirror file is one such line per event."""
    return {
        "kind": event.kind,
        "summary": event.summary,
        "raw": _json_safe(event.raw) if event.raw else None,
        "cost_usd": event.cost_usd,
        "usage": event.usage,
    }


def event_from_mirror(d: dict) -> AgentEvent:
    return AgentEvent(
        kind=d["kind"],
        summary=d.get("summary", ""),
        raw=d.get("raw"),
        cost_usd=d.get("cost_usd"),
        usage=d.get("usage"),
    )


def write_turn_mirror(
    events_uri: str, session_id: str, events: list[dict], *, now_ms: int, store: Any | None = None,
    turn_id: str | None = None,
) -> bool:
    """Write one batch of mirrored events as a file; best-effort (never raises).

    ``events_uri`` is ``<output_bucket>/events``; the file lands at
    ``<base>/<sid>/<epoch_ms>[-<turn>-<writer>-0000].jsonl`` so lexical order == chronological.
    Used for one-shot markers, pre-``MirrorStream`` failures and the client's own copy of a
    terminal result the worker could not persist. Returns whether the store took it.
    """
    if not events:
        return True
    try:
        bucket, prefix = parse_gcs_uri(events_uri)
        blobs = store if store is not None else GcsBlobStore(bucket)
        suffix = f"-{turn_id}-{uuid.uuid4().hex[:12]}-0000" if turn_id else ""
        key = f"{prefix + '/' if prefix else ''}{session_id}/{now_ms:015d}{suffix}.jsonl"
        data = "\n".join(json.dumps(line) for line in events).encode("utf-8")
        blobs.put_bytes(key, data)
    except Exception:  # noqa: BLE001 — mirroring is best-effort by design
        return False
    return True


def _parse_jsonl(data: bytes, parse) -> list[AgentEvent]:
    events: list[AgentEvent] = []
    for line in data.decode("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = parse(json.loads(line))
        except Exception:  # noqa: BLE001 — one bad line must not lose the rest
            continue
        if ev is not None:
            events.append(ev)
    return events


# -- read side -------------------------------------------------------------------


def read_history(
    output_bucket: str | None,
    session_id: str,
    *,
    credentials: Any | None = None,
    store: Any | None = None,
    turn_id: str | None = None,
) -> list[AgentEvent]:
    """All mirrored events of ``session_id``, oldest first (empty if nothing is found).

    ``turn_id`` restricts the read to that turn's files and events (a recovery read while
    a turn is live selects by identity, never by clock).
    """
    if not output_bucket:
        return []
    bucket, prefix = parse_gcs_uri(output_bucket)
    blobs = store if store is not None else GcsBlobStore(bucket, credentials=credentials)
    base = f"{prefix + '/' if prefix else ''}"
    events: list[AgentEvent] = []
    for key in blobs.list(f"{base}{_EVENTS_PREFIX}/{session_id}/"):
        if turn_id and f"-{turn_id}-" not in key.rsplit("/", 1)[-1]:
            continue
        events.extend(_parse_jsonl(blobs.get_bytes(key), event_from_mirror))
    if turn_id:
        events = [e for e in events if (e.raw or {}).get("turn_id") == turn_id]
    return events


def list_sessions(*, output_bucket: str | None, store: Any | None = None) -> list[dict]:
    """Enumerate the sessions recorded under ``output_bucket``, newest first.

    Returns dicts ``{"session_id", "sources": ["events"], "last_file": <key>}``. The mirror
    is bucket-wide (shared by every engine using that bucket), so with a shared bucket the
    sessions of other engines appear too.
    """
    if not output_bucket:
        return []
    bucket, prefix = parse_gcs_uri(output_bucket)
    blobs = store if store is not None else GcsBlobStore(bucket)
    base = f"{prefix + '/' if prefix else ''}{_EVENTS_PREFIX}/"
    sessions: dict[str, dict] = {}
    for key in blobs.list(base):
        sid = key[len(base):].split("/", 1)[0]
        if not sid:
            continue
        info = sessions.setdefault(sid, {"session_id": sid, "sources": ["events"], "last_file": None})
        if info["last_file"] is None or key > info["last_file"]:
            info["last_file"] = key
    # Newest first by the freshest file's name (an epoch-ms stamp), not its full key.
    return sorted(
        sessions.values(),
        key=lambda s: ((s["last_file"] or "").rsplit("/", 1)[-1], s["session_id"]),
        reverse=True,
    )

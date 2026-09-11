"""Past-session enumeration + event history for deployed engines (read side + mirror writer).

Three persistence layers hold a session's record, in decreasing fidelity:

1. **Mirrored events** — ``events/<sid>/*.jsonl`` under the output bucket: small JSONL batch
   files streamed by the worker AS THE TURN RUNS (``stream.MirrorStream``; the same mirror is
   the client's live channel — see ``stream.py``). Engines deployed before streaming wrote
   one file per turn at end-of-turn instead (:func:`write_turn_mirror` — kept for one-shot
   markers); both eras read identically here, lexical name order == chronological. The only
   session-keyed durable record for **warm** turns (their platform job output goes to a
   throwaway pool path), and the only layer that keeps ALL turns of a cold multi-turn session
   (the platform job output at ``jobs/<sid>.jsonl`` is per-job, so a resume overwrites it).
2. **Platform job output** — ``jobs/<sid>.jsonl``, the ADK event stream the platform writes
   for a cold job (covers engines deployed before mirroring existed).
3. **Cloud Logging** — the ``remote_agent_toolkit_steps`` per-step log (bounded by log
   retention, ~30 days by default).

:func:`read_history` tries them in that order; :func:`list_sessions` merges the GCS layers
with the engine's ADK sessions. All GCP imports are lazy; ``store`` params take any
``BlobStore`` for offline tests.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from ...events import AgentEvent
from ...ports.blobstore import GcsBlobStore, parse_gcs_uri

_EVENTS_PREFIX = "events"
_JOBS_PREFIX = "jobs"


# -- mirror format (write side runs in the worker; read side at the client) -----


def mirror_line(event: AgentEvent) -> dict:
    """One JSON-safe dict per event; the mirror file is one such line per event."""
    from ...ports.eventsink import _json_safe

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
) -> None:
    """Write one turn's mirrored events file; best-effort (a mirror failure never fails a run).

    ``events_uri`` is the ``AGENT_EVENTS_GCS`` base (``gs://bucket[/prefix]``); the file lands
    at ``<base>/<sid>/<epoch_ms>.jsonl`` so lexical order == chronological turn order.
    """
    if not events:
        return
    try:
        bucket, prefix = parse_gcs_uri(events_uri)
        blobs = store if store is not None else GcsBlobStore(bucket)
        suffix = f"-{turn_id}-{uuid.uuid4().hex[:12]}-0000" if turn_id else ""
        key = f"{prefix + '/' if prefix else ''}{session_id}/{now_ms:015d}{suffix}.jsonl"
        data = "\n".join(json.dumps(line) for line in events).encode("utf-8")
        blobs.put_bytes(key, data)
    except Exception:  # noqa: BLE001 — mirroring is best-effort by design
        pass


# -- platform job-output (ADK event) parsing ------------------------------------


def event_from_adk_line(d: dict) -> AgentEvent | None:
    """Parse one line of the platform's ``jobs/<sid>.jsonl`` back into an :class:`AgentEvent`.

    Toolkit events carry their identity in ``custom_metadata`` (kind + raw + cost/usage) with
    the summary as the first text part; lines without that metadata (foreign ADK events) are
    skipped by returning ``None``.
    """
    meta = d.get("custom_metadata") or {}
    kind = meta.get("kind")
    if not kind:
        return None
    parts = (d.get("content") or {}).get("parts") or []
    summary = next((p.get("text") for p in parts if p.get("text")), "") or ""
    return AgentEvent(
        kind=kind,
        summary=summary,
        raw=meta.get("raw"),
        cost_usd=meta.get("cost_usd"),
        usage=meta.get("usage"),
    )


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
    project: str | None = None,
    credentials: Any | None = None,
    store: Any | None = None,
    turn_id: str | None = None,
    worker: str | None = None,
) -> list[AgentEvent]:
    """All persisted events for ``session_id``, oldest first (empty if nothing is found).

    Tries the mirror (all turns), then the platform job output (last cold job), then Cloud
    Logging (retention-bounded). Layers are alternatives, not merged — the first hit is the
    most complete record available.

    ``turn_id`` restricts watchdog recovery across EVERY layer, including job output and
    logging fallbacks. ``worker`` additionally excludes an abandoned dispatch attempt.
    An incomplete mirror may fall through to a terminal record in another layer; absent
    a matching result it remains partial, never replaced by an earlier turn's success.
    """
    def select(events: list[AgentEvent]) -> list[AgentEvent]:
        if turn_id is None:
            return events
        return [e for e in events if (e.raw or {}).get("turn_id") == turn_id
                and (worker is None or (e.raw or {}).get("worker") == worker)]

    partial: list[AgentEvent] = []

    def complete(events: list[AgentEvent]) -> bool:
        return bool(events) and (turn_id is None or any(e.kind == "result" for e in events))

    if output_bucket:
        bucket, prefix = parse_gcs_uri(output_bucket)
        blobs = store if store is not None else GcsBlobStore(bucket, credentials=credentials)
        base = f"{prefix + '/' if prefix else ''}"

        # 1. Mirrored per-turn files (lexical order == chronological).
        events: list[AgentEvent] = []
        for key in blobs.list(f"{base}{_EVENTS_PREFIX}/{session_id}/"):
            if turn_id and f"-{turn_id}-" not in key.rsplit("/", 1)[-1]:
                continue
            events.extend(_parse_jsonl(blobs.get_bytes(key), event_from_mirror))
        events = select(events)
        if complete(events):
            return events
        partial = events

        # 2. Platform job output (cold path; the last job under this session).
        try:
            data = blobs.get_bytes(f"{base}{_JOBS_PREFIX}/{session_id}.jsonl")
            events = select(_parse_jsonl(data, event_from_adk_line))
            if complete(events):
                return events
            partial = partial or events
        except KeyError:
            pass

    # 3. Cloud Logging (one-shot; bounded by log retention).
    from ...ports.eventsink import CloudLoggingSink

    sink = CloudLoggingSink(project=project, credentials=credentials)
    try:
        events = select(sink.read(session_id))
        return events if complete(events) else (partial or events)
    except Exception:  # noqa: BLE001 — no logging access / no entries → empty history
        return partial


def list_sessions(
    *,
    output_bucket: str | None,
    resource: str | None = None,
    adk_sessions: Any | None = None,
    store: Any | None = None,
) -> list[dict]:
    """Enumerate this engine's known past sessions, newest first.

    Merges (a) mirrored ``events/<sid>/...`` files, (b) platform ``jobs/<sid>.jsonl`` outputs
    — both under the engine's output bucket — and (c) the engine's ADK sessions (cold-path
    sessions are real ADK sessions; warm ones are client-minted ids that only GCS knows).
    Returns dicts: ``{"session_id", "sources": [...], "last_file": <key or None>}``.

    NB: the bucket layers are bucket-wide (shared by every engine using that bucket), while
    the ADK layer is engine-scoped; with a shared bucket, sessions of other engines appear.
    """
    sessions: dict[str, dict] = {}

    def note(sid: str, source: str, last_file: str | None = None) -> None:
        info = sessions.setdefault(sid, {"session_id": sid, "sources": [], "last_file": None})
        if source not in info["sources"]:
            info["sources"].append(source)
        if last_file and (info["last_file"] is None or last_file > info["last_file"]):
            info["last_file"] = last_file

    if output_bucket:
        bucket, prefix = parse_gcs_uri(output_bucket)
        blobs = store if store is not None else GcsBlobStore(bucket)
        base = f"{prefix + '/' if prefix else ''}"

        for key in blobs.list(f"{base}{_EVENTS_PREFIX}/"):
            rest = key[len(f"{base}{_EVENTS_PREFIX}/"):]
            sid = rest.split("/", 1)[0]
            # `<name>-pool` entries are warm-pool readiness markers, not sessions.
            if sid and not sid.endswith("-pool"):
                note(sid, "events", key)

        for key in blobs.list(f"{base}{_JOBS_PREFIX}/"):
            name = key.rsplit("/", 1)[-1]
            if not name.endswith(".jsonl") or name.endswith("_input.jsonl"):
                continue
            note(name[: -len(".jsonl")], "jobs", key)

    if adk_sessions is not None and resource:
        try:
            for adk_session in adk_sessions.list(name=resource):
                sid = str(getattr(adk_session, "name", "")).rsplit("/", 1)[-1]
                if sid:
                    note(sid, "adk")
        except Exception:  # noqa: BLE001 — ADK listing is additive; GCS layers still answer
            pass

    # Newest first by the freshest file we saw (ADK-only sessions sort last, stably by id).
    return sorted(sessions.values(), key=lambda s: (s["last_file"] or "", s["session_id"]), reverse=True)

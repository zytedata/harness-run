"""Live event streaming over the GCS mirror — the quota-free channel (DESIGN §6).

The per-session mirror (``events/<sid>/*.jsonl`` under the output bucket) has always been
the *durable* history; this module makes it **incremental** so it doubles as the *live*
channel: the worker appends small JSONL batch objects as events happen
(:class:`MirrorStream`), and the client streams them by listing the session's prefix
(:func:`tail_stream`). Same prefix, same line format as the old per-turn flush, so
``history.read_history`` / ``list_sessions`` read both eras unchanged.

GCS list-after-write is strongly consistent, so an event is visible one flush + one poll
after it happened (~1-2 s), and a GCS list has no shared read budget.

With the sandbox runtime the client streams a turn **directly** from the worker
(``/events`` long-poll, ``backend.py``) and this tail is the fallback when the sandbox is
unreachable — and the mirror stays the durable record either way. Object names include
the worker clock, turn id, writer id and per-writer counter; turn readers select by
identity. Writes are batched (~0.5 s or 50 events, terminal ``result`` immediately) on a
background thread and are best-effort like every telemetry path in the worker: a storage
failure never fails a run.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
import uuid
from typing import Any, AsyncIterator, Callable

from ...events import AgentEvent
from ...ports.blobstore import GcsBlobStore, parse_gcs_uri

_FLUSH_INTERVAL_S = 0.5  # max time an event sits buffered before it is visible to tails
_FLUSH_MAX_EVENTS = 50  # size-based flush for bursty streams
_CLOSE_TIMEOUT_S = 10.0  # bound on draining the writer at turn end (close() is sync)
_TAIL_POLL_S = 0.5  # client list cadence (a GCS list is cheap and uncapped)
# Bound every call, ride out transient blips, surface persistent errors, never block forever.
_POLL_TIMEOUT_S = 30.0
_MAX_POLL_FAILURES = 10
_MAX_WAIT_S = 3600.0

_STOP = object()


def _blobs_and_prefix(events_uri: str, store: Any | None) -> tuple[Any, str]:
    """A BlobStore + key prefix for ``events_uri`` (``AGENT_EVENTS_GCS``-style base)."""
    bucket, prefix = parse_gcs_uri(events_uri)
    blobs = store if store is not None else GcsBlobStore(bucket)
    return blobs, f"{prefix}/" if prefix else ""


class MirrorStream:
    """Incremental mirror writer (worker side): buffered, batched, best-effort.

    ``append`` never blocks on storage — lines go to a background thread that flushes a
    batch object every ``_FLUSH_INTERVAL_S`` / ``_FLUSH_MAX_EVENTS`` (a terminal
    ``result`` event flushes immediately, so the tail's stop condition is never sitting
    in a buffer). ``close()`` drains synchronously (bounded) — safe in a ``finally`` that
    may run under ``GeneratorExit``, where awaiting is illegal. Storage errors are
    swallowed by design: the mirror is telemetry, the run must not fail for it.
    """

    def __init__(self, events_uri: str, session_id: str, *, store: Any | None = None,
                 turn_id: str | None = None) -> None:
        self._session_id = session_id
        self._turn_id = turn_id
        self._writer_id = uuid.uuid4().hex[:12]
        self._queue: queue.Queue = queue.Queue()
        self._seq = 0
        try:
            self._blobs, self._prefix = _blobs_and_prefix(events_uri, store)
        except Exception:  # noqa: BLE001 — a bad URI degrades to a no-op writer
            self._blobs = None
        self._thread = threading.Thread(
            target=self._drain, name=f"mirror-stream-{session_id[:8]}", daemon=True
        )
        self._thread.start()

    def append(self, event: AgentEvent) -> None:
        from .history import mirror_line

        try:
            self._queue.put((mirror_line(event), event.kind == "result"))
        except Exception:  # noqa: BLE001 — never raise into the worker loop
            pass

    def close(self, timeout: float = _CLOSE_TIMEOUT_S) -> None:
        """Flush the remainder and stop the writer thread (sync, bounded)."""
        try:
            self._queue.put(_STOP)
            self._thread.join(timeout)
        except Exception:  # noqa: BLE001
            pass

    def _drain(self) -> None:
        buffered: list[dict] = []
        while True:
            try:
                item = self._queue.get(timeout=_FLUSH_INTERVAL_S if buffered else None)
            except queue.Empty:
                self._flush(buffered)
                buffered = []
                continue
            if item is _STOP:
                self._flush(buffered)
                return
            line, urgent = item
            buffered.append(line)
            if urgent or len(buffered) >= _FLUSH_MAX_EVENTS:
                self._flush(buffered)
                buffered = []

    def _flush(self, lines: list[dict]) -> None:
        if not lines or self._blobs is None:
            return
        self._seq += 1
        key = (
            f"{self._prefix}{self._session_id}/"
            f"{int(time.time() * 1000):015d}-"
            f"{self._turn_id + '-' + self._writer_id + '-' if self._turn_id else ''}"
            f"{self._seq:04d}.jsonl"
        )
        try:
            data = "\n".join(json.dumps(line) for line in lines).encode("utf-8")
            self._blobs.put_bytes(key, data)
        except Exception:  # noqa: BLE001 — best-effort; the run must not fail for telemetry
            pass


async def tail_stream(
    events_uri: str,
    session_id: str,
    since: float | None = None,
    *,
    start_after: str | None = None,
    watermark: dict | None = None,
    store: Any | None = None,
    turn_id: str | None = None,
    accept_event: Callable[[AgentEvent], bool] | None = None,
    max_wait_s: float = _MAX_WAIT_S,
) -> AsyncIterator[AgentEvent]:
    """Stream a session's mirrored events live; stops after the terminal ``result``.

    ``max_wait_s`` bounds the tail as a whole (default an hour); a caller that already
    knows the writer is gone passes a short bound and synthesizes the end itself.

    With ``turn_id``, reads only that turn's objects and verifies each event's identity
    before yielding or terminating. Clock floors and a prior handle's watermark are
    unnecessary: a freshly reattached session is safe even on back-to-back sends. The
    optional predicate excludes events from abandoned workers after redispatch.

    Without ``turn_id``, polls the session's object listing from a floor, skipping
    previous turns' files without reading them. Two floors compose (the higher wins):

    * ``start_after`` — an exact object key; only strictly-later keys are consumed. THE
      turn-boundary floor: the previous turn's tail records its last consumed key (via
      ``watermark``), and the next turn starts after it. Clock-free, so it is immune to
      the back-to-back-turns hazard where a time floor with skew slack re-delivers the
      previous turn's just-written terminal result as the new turn's first event.
    * ``since`` (epoch seconds) — floors the names' leading epoch-ms stamp. The fallback
      for legacy readers; clock slack alone cannot separate back-to-back turns, including
      reattached sessions. Runtime turn submission uses the explicit identity instead.

    ``watermark`` (optional ``dict``) is updated in place — ``watermark["key"]`` is the
    last consumed object key — so the caller can hand it to the next turn's tail. Every
    storage call is bounded; transient errors ride out (up to ``_MAX_POLL_FAILURES``
    consecutive), and ``max_wait_s`` bounds the tail as a whole.
    """
    from .history import _parse_jsonl, event_from_mirror

    blobs, prefix = _blobs_and_prefix(events_uri, store)
    sid_prefix = f"{prefix}{session_id}/"
    # Names start with a 15-digit epoch-ms stamp, so the `since` floor string sorts below
    # every file written at/after it and above everything older.
    consumed = f"{sid_prefix}{int(since * 1000):015d}" if since else sid_prefix
    if start_after:
        consumed = max(consumed, start_after)
    start = time.monotonic()
    seen_keys: set[str] = set()
    failures = 0
    while time.monotonic() - start < max_wait_s:
        try:
            keys = await asyncio.wait_for(
                asyncio.to_thread(blobs.list, sid_prefix), timeout=_POLL_TIMEOUT_S
            )
            failures = 0
        except Exception:  # noqa: BLE001 — transient list failure / wedged call
            failures += 1
            if failures >= _MAX_POLL_FAILURES:
                raise
            await asyncio.sleep(_TAIL_POLL_S)
            continue
        for key in keys:
            if turn_id:
                # Names are only a read optimization; the payload is verified below.
                # Track exact keys so a later write with an earlier clock is not lost.
                if f"-{turn_id}-" not in key.rsplit("/", 1)[-1] or key in seen_keys:
                    continue
            elif key <= consumed:
                continue
            try:
                data = await asyncio.wait_for(
                    asyncio.to_thread(blobs.get_bytes, key), timeout=_POLL_TIMEOUT_S
                )
            except Exception:  # noqa: BLE001 — retry this object on the next poll
                failures += 1
                if failures >= _MAX_POLL_FAILURES:
                    raise
                break
            failures = 0
            consumed = key
            seen_keys.add(key)
            if watermark is not None:
                watermark["key"] = key
            for event in _parse_jsonl(data, event_from_mirror):
                if turn_id and (event.raw or {}).get("turn_id") != turn_id:
                    continue
                if accept_event is not None and not accept_event(event):
                    continue
                yield event
                if event.kind == "result":
                    return
        await asyncio.sleep(_TAIL_POLL_S)

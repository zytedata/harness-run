"""Live event streaming over the GCS mirror — the quota-free channel (DESIGN §6).

The per-session mirror (``events/<sid>/*.jsonl`` under the output bucket) has always been
the *durable* history; this module makes it **incremental** so it doubles as the *live*
channel: the worker appends small JSONL batch objects as events happen
(:class:`MirrorStream`), and the client streams them by listing the session's prefix
(:func:`tail_stream`). Same prefix, same line format as the old per-turn flush, so
``history.read_history`` / ``list_sessions`` read both eras unchanged.

Why this replaces tailing Cloud Logging (which stays as an emit-only ops/debug channel):

* **No shared read budget.** Cloud Logging caps ``entries.list`` at 60 reads/min for the
  whole project — one polling tail nearly exhausts it, and concurrent runs starve each
  other (they now back off; see ``ports/eventsink.py``). GCS has no comparable cap: tens
  of concurrent tails are a non-event.
* **No ingestion lag.** GCS list-after-write is strongly consistent, so an event is
  visible one flush + one poll after it happened (~1-2 s), vs Cloud Logging's variable
  write→queryable lag (seconds to minutes — it dominated warm first-event latency).

Object names are ``<epoch_ms:015d>-<seq:04d>.jsonl`` (worker clock, per-writer counter),
so lexical order == chronological order and a name encodes its time — the tail resumes
from a ``since`` watermark without reading anything. Writes are batched (~0.5 s or 50
events, terminal ``result`` immediately) on a background thread and are best-effort like
every telemetry path in the worker: a storage failure never fails a run.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
from typing import Any, AsyncIterator

from ...events import AgentEvent
from ...ports.blobstore import GcsBlobStore, parse_gcs_uri

_FLUSH_INTERVAL_S = 0.5  # max time an event sits buffered before it is visible to tails
_FLUSH_MAX_EVENTS = 50  # size-based flush for bursty streams
_CLOSE_TIMEOUT_S = 10.0  # bound on draining the writer at turn end (close() is sync)
_TAIL_POLL_S = 0.5  # client list cadence (a GCS list is cheap and uncapped)
# Same resilience envelope as the Cloud Logging tail (ports/eventsink.py): bound every
# call, ride out transient blips, surface persistent errors, never block forever.
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

    def __init__(self, events_uri: str, session_id: str, *, store: Any | None = None) -> None:
        self._session_id = session_id
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
            f"{int(time.time() * 1000):015d}-{self._seq:04d}.jsonl"
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
) -> AsyncIterator[AgentEvent]:
    """Stream a session's mirrored events live; stops after the terminal ``result``.

    Polls the session's object listing (lexical == chronological) from a floor, skipping
    previous turns' files without reading them. Two floors compose (the higher wins):

    * ``start_after`` — an exact object key; only strictly-later keys are consumed. THE
      turn-boundary floor: the previous turn's tail records its last consumed key (via
      ``watermark``), and the next turn starts after it. Clock-free, so it is immune to
      the back-to-back-turns hazard where a time floor with skew slack re-delivers the
      previous turn's just-written terminal result as the new turn's first event.
    * ``since`` (epoch seconds) — floors the names' leading epoch-ms stamp. The fallback
      for re-attached sessions (fresh process, no watermark yet); its 5 s submit slack is
      only safe when turns are NOT back-to-back, which re-attachment guarantees.

    ``watermark`` (optional ``dict``) is updated in place — ``watermark["key"]`` is the
    last consumed object key — so the caller can hand it to the next turn's tail. Every
    storage call is bounded; transient errors ride out (up to ``_MAX_POLL_FAILURES``
    consecutive), and ``_MAX_WAIT_S`` bounds the tail as a whole — same envelope as the
    Cloud Logging tail this replaces.
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
    failures = 0
    while time.monotonic() - start < _MAX_WAIT_S:
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
            if key <= consumed:
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
            if watermark is not None:
                watermark["key"] = key
            for event in _parse_jsonl(data, event_from_mirror):
                yield event
                if event.kind == "result":
                    return
        await asyncio.sleep(_TAIL_POLL_S)

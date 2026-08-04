"""``EventSink`` port + adapters (DESIGN.md §6, §7).

The only near-real-time async channel on Gemini Agent Runtime: the harness emits a
per-step structured log (write side), the client tails it filtered by ``session_id``
(read side). Protocol is stdlib-only; ``CloudLoggingSink`` imports
``google.cloud.logging`` lazily so this module stays importable with zero third-party
deps. An ``emit``/``tail`` pair MUST round-trip an :class:`AgentEvent`; ``tail`` stops at
the terminal ``"result"`` event so a client loop terminates without external signalling.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from datetime import datetime, timezone
from typing import AsyncIterator, Protocol, runtime_checkable

from ..events import AgentEvent

# Cloud Logging caps a single entry at 256KB; keep the (potentially huge) result text well
# under that. summary is the only free-form field that can grow without bound.
_SUMMARY_CAP = 60000
# tail poll cadence and overall safety ceiling: a client must not block forever if the
# terminal "result" event never arrives (crashed runtime, lost log entry, etc.). 1s keeps
# the observed stream snappy; the residual first-event latency is Cloud Logging's own
# write→queryable ingestion lag (a few seconds), inherent to a log-tail channel.
_POLL_INTERVAL_S = 1.0
_MAX_WAIT_S = 3600.0
# Per-poll bound + how many consecutive failed polls to ride out before surfacing the error.
_POLL_TIMEOUT_S = 30.0
_MAX_POLL_FAILURES = 10
# Read-quota rejections are NOT failures. Cloud Logging caps `entries.list` at 60 requests
# per minute PER PROJECT (fixed — cannot be raised), so concurrent tails in one project
# WILL trade 429s under contention: every agent run, live test, and colleague shares the
# same budget. A quota rejection proves the API is reachable and the per-minute window will
# refill, so the tail backs off (exponential + jitter, capped) instead of counting toward
# ``_MAX_POLL_FAILURES``; ``_MAX_WAIT_S`` still bounds the tail as a whole. The jittered cap
# also self-balances fleets: at the cap, each tail costs ~2-3 reads/min, so ~20-30 concurrent
# tails degrade to slower-but-steady delivery instead of starving each other out.
_QUOTA_BACKOFF_BASE_S = 2.0
_QUOTA_BACKOFF_CAP_S = 30.0
# RFC3339 with microseconds + trailing Z — the timestamp format Cloud Logging filters accept.
_RFC3339 = "%Y-%m-%dT%H:%M:%S.%fZ"


def _is_quota_error(exc: BaseException) -> bool:
    """True for a Cloud Logging read-quota rejection (HTTP 429 / gRPC RESOURCE_EXHAUSTED)."""
    try:
        from google.api_core import exceptions as gax  # ships with google-cloud-logging
    except Exception:  # pragma: no cover — only hit where the client itself can't exist
        return False
    return isinstance(exc, (gax.TooManyRequests, gax.ResourceExhausted))


def _quota_backoff_s(rejections: int) -> float:
    """Jittered exponential backoff for the ``rejections``-th consecutive quota rejection."""
    delay = min(_QUOTA_BACKOFF_CAP_S, _QUOTA_BACKOFF_BASE_S * 2 ** (rejections - 1))
    return random.uniform(delay / 2, delay)  # jitter de-synchronizes concurrent tails


def _json_safe(value):
    """Coerce ``value`` to something ``log_struct`` can serialize (round-trips via JSON).

    ``raw`` carries arbitrary harness/SDK payloads; anything not natively JSON-serializable
    is stringified rather than allowed to blow up the (best-effort) emit.
    """
    return json.loads(json.dumps(value, default=str))


@runtime_checkable
class EventSink(Protocol):
    """Per-step event channel: ``emit`` on the runtime side, ``tail`` on the client side."""

    def emit(self, event: AgentEvent) -> None:
        """Write ``event`` to the sink (runtime side)."""
        ...

    def tail(self, session_id: str, since: float | None = None) -> AsyncIterator[AgentEvent]:
        """Stream events for ``session_id`` (optionally from ``since``) (client side)."""
        ...


class CloudLoggingSink:
    """Cloud Logging-backed :class:`EventSink` — the near-real-time channel.

    ``emit`` writes one structured entry per step via ``log_struct`` (labelled with
    ``session_id`` + ``kind`` for filtering); ``tail`` polls ``list_entries`` filtered by
    ``session_id`` and reconstructs the :class:`AgentEvent`. ``google.cloud.logging`` is
    imported lazily inside the methods. ``session_id`` is required to ``emit`` but optional
    for a read-only tailing client (which passes the id to ``tail`` directly).
    """

    def __init__(
        self,
        session_id: str | None = None,
        log_name: str = "remote_agent_toolkit_steps",
        project: str | None = None,
        credentials=None,
    ) -> None:
        self.session_id = session_id
        self.log_name = log_name
        self.project = project
        self.credentials = credentials
        self._client = None  # cached google.cloud.logging.Client (built on first use)

    def _get_client(self):
        """Build/cache the Cloud Logging client once (project/credentials only if set)."""
        if self._client is None:
            import google.cloud.logging  # lazy: only available in the deployed engine

            if self.project or self.credentials is not None:
                self._client = google.cloud.logging.Client(
                    project=self.project, credentials=self.credentials
                )
            else:
                self._client = google.cloud.logging.Client()
        return self._client

    def emit(self, event: AgentEvent) -> None:
        if not self.session_id:
            raise ValueError("CloudLoggingSink.emit requires a session_id (construct with one)")
        # Best-effort, like the PoC: a logging failure must never crash a run. Everything
        # after the guard above is wrapped so a transient Logging API error is swallowed.
        try:
            logger = self._get_client().logger(self.log_name)
            payload = {
                "kind": event.kind,
                "summary": (event.summary or "")[:_SUMMARY_CAP],
                "raw": _json_safe(event.raw) if event.raw else None,
                "cost_usd": event.cost_usd,
                "usage": event.usage,
                "session_id": self.session_id,
            }
            severity = "ERROR" if (event.raw or {}).get("is_error") else "INFO"
            logger.log_struct(
                payload,
                labels={"session_id": self.session_id or "", "kind": event.kind},
                severity=severity,
            )
        except Exception:  # noqa: BLE001 — best-effort channel; never raise from emit
            pass

    def read(self, session_id: str, timeout: float = 60.0) -> list[AgentEvent]:
        """One-shot history read: all logged events for ``session_id``, oldest first.

        Unlike :meth:`tail` this never waits for new entries — it lists whatever Cloud Logging
        has right now (bounded by the log bucket's retention, ~30 days by default) and returns.
        The whole read is bounded by ``timeout``: the logging client has no per-call timeout of
        its own, and a dead connection would otherwise block forever (the wedged worker thread
        is abandoned on timeout). Read-quota rejections (the shared 60-reads/min project
        budget) are retried with backoff within the same ``timeout`` instead of raising.
        """
        import concurrent.futures

        import google.cloud.logging  # lazy

        client = self._get_client()
        project = self.project or client.project
        filter_str = (
            f'logName="projects/{project}/logs/{self.log_name}" '
            f'AND labels.session_id="{session_id}"'
        )

        def _list() -> list[AgentEvent]:
            events: list[AgentEvent] = []
            for entry in client.list_entries(filter_=filter_str, order_by=google.cloud.logging.ASCENDING):
                payload = entry.payload or {}
                if "kind" not in payload:
                    continue  # not a toolkit step entry
                events.append(AgentEvent(
                    kind=payload["kind"],
                    summary=payload.get("summary", ""),
                    raw=payload.get("raw"),
                    cost_usd=payload.get("cost_usd"),
                    usage=payload.get("usage"),
                ))
            return events

        # No `with`: the context manager's shutdown WAITS on the worker, which would block on
        # the very wedge the timeout exists to escape. shutdown(wait=False) abandons it.
        deadline = time.monotonic() + timeout
        quota_rejections = 0
        while True:
            ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            try:
                return ex.submit(_list).result(timeout=max(0.1, deadline - time.monotonic()))
            except Exception as exc:
                remaining = deadline - time.monotonic()
                if not (_is_quota_error(exc) and remaining > 0):
                    raise
                quota_rejections += 1
                time.sleep(min(remaining, _quota_backoff_s(quota_rejections)))
            finally:
                ex.shutdown(wait=False)

    async def tail(
        self, session_id: str, since: float | None = None
    ) -> AsyncIterator[AgentEvent]:
        import google.cloud.logging  # lazy: only available where the client is installed

        client = self._get_client()
        # The log filter needs a fully-qualified logName, which needs a project. Prefer the
        # explicit one, else fall back to the client's resolved project.
        project = self.project or client.project
        start = time.monotonic()
        # Watermark: only fetch entries at/after this RFC3339 timestamp. Start from `since`
        # (epoch seconds) if given, else a little before "now" to tolerate clock skew.
        if since is not None:
            watermark = datetime.fromtimestamp(since, tz=timezone.utc).strftime(_RFC3339)
        else:
            watermark = datetime.fromtimestamp(
                time.time() - 5.0, tz=timezone.utc
            ).strftime(_RFC3339)
        seen: set[str] = set()  # insert_ids already yielded, so repeated polls don't duplicate

        failures = 0  # consecutive failed/wedged polls; transient blips ride out, permanent raise
        quota_rejections = 0  # consecutive 429s; expected under contention, backed off, never fatal
        while time.monotonic() - start < _MAX_WAIT_S:
            filter_str = (
                f'logName="projects/{project}/logs/{self.log_name}" '
                f'AND labels.session_id="{session_id}" '
                f'AND timestamp>="{watermark}"'
            )
            # Bound every poll: the logging client exposes no per-call timeout, and a dead
            # connection otherwise blocks list_entries FOREVER (observed live — the tail hung
            # ~1 h past job completion on a dropped network). On timeout the (possibly wedged)
            # worker thread is abandoned and the next poll uses a fresh call.
            try:
                entries = await asyncio.wait_for(
                    asyncio.to_thread(
                        lambda f=filter_str: list(
                            client.list_entries(
                                filter_=f, order_by=google.cloud.logging.ASCENDING
                            )
                        )
                    ),
                    timeout=_POLL_TIMEOUT_S,
                )
                failures = 0
                quota_rejections = 0
            except Exception as exc:  # noqa: BLE001 — timeout / transient auth / network blip
                if _is_quota_error(exc):
                    # The 60-reads/min budget is shared by the whole project (see the
                    # constants above): exhaustion is an operating condition, not a fault.
                    # Back off so concurrent tails share the window; never raise for it.
                    quota_rejections += 1
                    await asyncio.sleep(_quota_backoff_s(quota_rejections))
                    continue
                failures += 1
                if failures >= _MAX_POLL_FAILURES:
                    raise  # permanent (bad filter, revoked perms): surface, don't spin forever
                await asyncio.sleep(_POLL_INTERVAL_S)
                continue
            stop = False
            for entry in entries:
                insert_id = getattr(entry, "insert_id", None)
                if insert_id and insert_id in seen:
                    continue
                if insert_id:
                    seen.add(insert_id)
                payload = entry.payload or {}
                event = AgentEvent(
                    kind=payload["kind"],
                    summary=payload.get("summary", ""),
                    raw=payload.get("raw"),
                    cost_usd=payload.get("cost_usd"),
                    usage=payload.get("usage"),
                )
                # Advance the watermark to this entry's timestamp so the next poll fetches
                # only newer entries (dedup by insert_id still guards the == boundary).
                ts = getattr(entry, "timestamp", None)
                if ts is not None:
                    watermark = ts.astimezone(timezone.utc).strftime(_RFC3339)
                yield event
                if event.kind == "result":
                    stop = True  # terminal event — let the client loop end naturally
                    break
            if stop:
                return
            await asyncio.sleep(_POLL_INTERVAL_S)


class InMemorySink:
    """In-process :class:`EventSink` for local dev / tests.

    A single instance shares ``self._events`` across ``emit`` and ``tail`` (same process),
    so a test can emit then tail and observe the round-trip. ``tail`` stops at the terminal
    ``"result"`` event and is bounded so a missing result never hangs.
    """

    # tail bound: stop after this long with no new event so a missing "result" can't hang.
    _IDLE_TIMEOUT_S = 5.0
    _POLL_INTERVAL_S = 0.01

    def __init__(self, session_id: str | None = None) -> None:
        self.session_id = session_id
        self._events: dict[str, list[AgentEvent]] = {}

    def emit(self, event: AgentEvent) -> None:
        if not self.session_id:
            raise ValueError("InMemorySink.emit requires a session_id (construct with one)")
        self._events.setdefault(self.session_id, []).append(event)

    async def tail(
        self, session_id: str, since: float | None = None
    ) -> AsyncIterator[AgentEvent]:
        idx = 0  # next index into the per-session buffer we have not yielded yet
        last_progress = time.monotonic()
        while time.monotonic() - last_progress < self._IDLE_TIMEOUT_S:
            buffer = self._events.get(session_id, [])
            while idx < len(buffer):
                event = buffer[idx]
                idx += 1
                last_progress = time.monotonic()  # new event seen — reset the idle bound
                yield event
                if event.kind == "result":
                    return  # terminal event — stop the stream
            await asyncio.sleep(self._POLL_INTERVAL_S)

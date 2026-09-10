"""The GCS control inbox: turn control for gemini turns (DESIGN.md §6).

A control message (steer / interrupt / stop — :class:`~remote_agent_toolkit.control.ControlMessage`)
must reach the worker while ``_run_turn`` is streaming, on both the cold and the warm path,
and from ANY process that holds the session id (a client that adopted the session after
the one that started the turn died). A query job has no inbound channel other than cancel,
and a pool worker stops pulling its subscription once it has claimed a turn — so the
inbox is a GCS prefix next to the other handoff objects:

* the client writes ``control/<sid>/<epoch_ms:015d>-<seq:04d>-<id>.json``
  (:func:`send_control`; ``seq`` is a per-process counter, so two messages sent within
  the same millisecond keep their order);
* the worker polls that prefix every ~1.5 s while the harness runs
  (:class:`GcsControlChannel`), hands each message to the harness in key order (lexical ==
  chronological) and DELETES the object, so the inbox only ever holds undelivered messages
  — a worker that dies and a platform retry of the same turn still find them;
* delivery is recorded as a marker at ``control-delivered/<sid>/<id>`` and a message whose
  id already has one is dropped: a client that retries a send (e.g. after adopting the
  session) cannot make the model see the message twice.

Same bucket and identity as the event mirror; ~2 s delivery latency, which is chat speed.
Reaped by a lifecycle rule (``handoff.py``). Messages are operator text — never secrets —
but they are still deleted on delivery and aged out, not kept as a record: the ``user``
event in the mirror is the durable record of what was said and when.

``google.cloud.storage`` is imported lazily through :class:`GcsBlobStore`; ``store=`` takes
any ``BlobStore`` for offline tests.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import re
import time
from typing import Any

from ...control import ControlMessage
from ...ports.blobstore import GcsBlobStore, parse_gcs_uri

# Under the output bucket: the inbox objects, and the delivered-id markers.
CONTROL_PREFIX = "control"
DELIVERED_PREFIX = "control-delivered"

_POLL_S = 1.5  # worker list cadence: a GCS list is cheap and uncapped
_POLL_TIMEOUT_S = 30.0  # bound on every storage call (same envelope as stream.py)
_SEQ = itertools.count(1)  # orders sends from one process that share a millisecond


def control_uri(output_bucket: str) -> str:
    """The ``AGENT_CONTROL_GCS``-style base for ``output_bucket``: ``gs://bucket/control``."""
    bucket, prefix = parse_gcs_uri(output_bucket)
    return f"gs://{bucket}/{prefix + '/' if prefix else ''}{CONTROL_PREFIX}"


def _blobs_and_prefix(base_uri: str, store: Any | None) -> tuple[Any, str]:
    bucket, prefix = parse_gcs_uri(base_uri)
    blobs = store if store is not None else GcsBlobStore(bucket)
    return blobs, f"{prefix}/" if prefix else ""


def _safe_id(message_id: str) -> str:
    """A message id as a key segment (the real id stays in the object's JSON)."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", message_id)[:80] or "msg"


def _delivered_prefix(inbox_prefix: str) -> str:
    """``control/`` → ``control-delivered/`` at the same depth (``prefix`` from ``_blobs_and_prefix``)."""
    head = inbox_prefix[: -len(CONTROL_PREFIX) - 1] if inbox_prefix.endswith(f"{CONTROL_PREFIX}/") else inbox_prefix
    return f"{head}{DELIVERED_PREFIX}/"


def send_control(
    base_uri: str, session_id: str, msg: ControlMessage, *, store: Any | None = None
) -> str:
    """Write ``msg`` to the session's inbox; return the object key (for a later delete)."""
    blobs, prefix = _blobs_and_prefix(base_uri, store)
    key = (
        f"{prefix}{session_id}/{int(time.time() * 1000):015d}-{next(_SEQ) % 10_000:04d}-"
        f"{_safe_id(msg.message_id)}.json"
    )
    payload = {**msg.to_dict(), "sent_at": time.time()}
    blobs.put_bytes(key, json.dumps(payload).encode("utf-8"))
    return key


def delete_control(base_uri: str, key: str, *, store: Any | None = None) -> None:
    """Best-effort delete of an inbox object (a ``stop`` that missed its turn)."""
    try:
        blobs, _ = _blobs_and_prefix(base_uri, store)
        blobs.delete(key)
    except Exception:  # noqa: BLE001 — cleanup must never fail a run
        pass


class GcsControlChannel:
    """Worker-side :class:`~remote_agent_toolkit.control.ControlChannel` over the GCS inbox.

    ``receive()`` polls the session's inbox prefix until a not-yet-delivered message is
    there, marks it delivered, deletes it and returns it. Unreadable objects are deleted
    too (a malformed message must not wedge the inbox). Storage errors are retried on the
    next poll; the harness bounds the whole turn, so this never needs its own cap.
    """

    def __init__(
        self,
        base_uri: str,
        session_id: str,
        *,
        store: Any | None = None,
        poll_s: float = _POLL_S,
    ) -> None:
        self._blobs, prefix = _blobs_and_prefix(base_uri, store)
        self._inbox = f"{prefix}{session_id}/"
        self._delivered = f"{_delivered_prefix(prefix)}{session_id}/"
        self._poll_s = poll_s
        self._seen: set[str] = set()  # message ids handed over by this worker

    @property
    def inbox(self) -> str:
        return self._inbox

    def _take_one(self) -> ControlMessage | None:
        """Sync: one pass over the inbox; the first deliverable message, or ``None``."""
        for key in self._blobs.list(self._inbox):
            try:
                msg = ControlMessage.from_dict(json.loads(self._blobs.get_bytes(key).decode()))
            except KeyError:
                continue  # taken by a concurrent poll (e.g. a platform retry racing us)
            except Exception:  # noqa: BLE001 — unreadable: drop it rather than wedge the inbox
                self._blobs.delete(key)
                continue
            marker = f"{self._delivered}{_safe_id(msg.message_id)}"
            if msg.message_id in self._seen or self._blobs.exists(marker):
                self._blobs.delete(key)  # a retry of a message the model already has
                continue
            self._seen.add(msg.message_id)
            self._blobs.put_bytes(marker, b"")
            self._blobs.delete(key)
            return msg
        return None

    async def receive(self) -> ControlMessage:
        while True:
            try:
                msg = await asyncio.wait_for(
                    asyncio.to_thread(self._take_one), timeout=_POLL_TIMEOUT_S
                )
            except Exception:  # noqa: BLE001 — transient storage error: poll again
                msg = None
            if msg is not None:
                return msg
            await asyncio.sleep(self._poll_s)

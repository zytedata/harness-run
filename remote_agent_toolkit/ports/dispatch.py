"""``DispatchTransport`` port + adapters (DESIGN.md §6, §7).

Warm-pool dispatch + atomic claim. A warm worker blocks on ``claim`` pulling a Pub/Sub
subscription; the control plane ``publish``es a turn payload. Two addressing modes share
the port:

* **Per-worker** (every deploy since per-worker dispatch): each worker has its own
  subscription on the pool's topic, filtered to messages carrying its ``worker``
  attribute (``runtime/gemini/pool.py``). ``publish(message, attributes=...)`` addresses
  one worker; the subscription's filter drops everything else before it can be pulled.
* **Shared** (legacy engines): every worker pulls one subscription, competing consumers.
  Kept only to address engines deployed before per-worker dispatch.

Protocol is stdlib-only; ``PubSubDispatch`` imports ``google.cloud.pubsub_v1`` /
``google.api_core`` lazily so this module stays importable with no third-party deps.
"""

from __future__ import annotations

import json
import queue
from typing import Protocol, runtime_checkable


@runtime_checkable
class DispatchTransport(Protocol):
    """Dispatch: ``publish`` a message (optionally addressed), ``claim`` one atomically."""

    def publish(self, message: dict, attributes: dict[str, str] | None = None) -> None:
        """Publish a dispatch ``message``; ``attributes`` address it (per-worker filters)."""
        ...

    def claim(self, timeout: float) -> dict | None:
        """Atomically claim one message, blocking up to ``timeout`` seconds.

        Returns the message, or ``None`` on timeout.
        """
        ...


class PubSubDispatch:
    """Pub/Sub-backed :class:`DispatchTransport`.

    ``google.cloud.pubsub_v1`` / ``google.api_core`` are imported lazily inside the
    methods so importing this module pulls in no third-party deps. A worker that only
    claims passes ``subscription`` (no ``topic``); the control plane passes ``topic`` (and,
    for the subscription admin methods, ``project``). ``topic`` / ``subscription`` are full
    resource paths (``projects/<p>/topics/<t>``, ``projects/<p>/subscriptions/<s>``).
    """

    def __init__(
        self,
        topic: str | None = None,
        subscription: str | None = None,
        project: str | None = None,
        credentials=None,
    ) -> None:
        self.topic = topic
        self.subscription = subscription
        self.project = project
        self.credentials = credentials
        # Clients are created on first use (kept None so construction stays import-light).
        self._publisher = None
        self._subscriber = None

    def _publisher_client(self):
        if self._publisher is None:
            from google.cloud import pubsub_v1

            kwargs = {"credentials": self.credentials} if self.credentials else {}
            self._publisher = pubsub_v1.PublisherClient(**kwargs)
        return self._publisher

    def _subscriber_client(self):
        if self._subscriber is None:
            from google.cloud import pubsub_v1

            kwargs = {"credentials": self.credentials} if self.credentials else {}
            self._subscriber = pubsub_v1.SubscriberClient(**kwargs)
        return self._subscriber

    def publish(self, message: dict, attributes: dict[str, str] | None = None) -> None:
        if self.topic is None:
            raise ValueError("PubSubDispatch.publish requires a topic (none configured).")
        client = self._publisher_client()
        data = json.dumps(message).encode("utf-8")
        client.publish(self.topic, data, **(attributes or {})).result(timeout=30)

    def claim(self, timeout: float) -> dict | None:
        if self.subscription is None:
            raise ValueError("PubSubDispatch.claim requires a subscription (none configured).")
        from google.api_core import exceptions as gax

        client = self._subscriber_client()
        try:
            resp = client.pull(
                request={"subscription": self.subscription, "max_messages": 1},
                timeout=timeout,
            )
        except (gax.DeadlineExceeded, gax.RetryError):
            return None  # long-poll window elapsed with no message
        if not resp.received_messages:
            return None
        msg = resp.received_messages[0]
        # Ack before returning so the message isn't redelivered to another worker.
        client.acknowledge(
            request={"subscription": self.subscription, "ack_ids": [msg.ack_id]}
        )
        return json.loads(msg.message.data.decode("utf-8"))

    # -- control-plane admin (deploy / fill_pool / retire) ---------------------------

    def ensure_topic(self) -> None:
        """Idempotently create the topic (control plane, at deploy)."""
        if self.topic is None:
            raise ValueError("PubSubDispatch.ensure_topic requires a topic.")
        from google.api_core import exceptions as gax

        try:
            self._publisher_client().create_topic(name=self.topic)
        except gax.AlreadyExists:
            pass

    def ensure(self) -> None:
        """Idempotently create the topic + the shared subscription (legacy pools).

        Each create is wrapped to ignore ``AlreadyExists`` so re-running deploy is a no-op.
        """
        if self.topic is None or self.subscription is None:
            raise ValueError("PubSubDispatch.ensure requires both topic and subscription.")
        self.ensure_topic()
        self.create_subscription(self.subscription)

    def create_subscription(
        self,
        subscription: str,
        *,
        filter: str | None = None,  # noqa: A002 — the Pub/Sub field is called filter
        ttl_s: int | None = None,
        retention_s: int | None = None,
    ) -> None:
        """Create ``subscription`` on this topic (idempotent).

        ``filter`` limits what the subscription can ever deliver (Pub/Sub drops the rest
        before delivery; immutable after creation). ``ttl_s`` is the expiration policy
        (deleted after that long without subscriber activity; minimum a day) and
        ``retention_s`` how long an unacked message is kept (minimum 10 minutes) — the
        per-worker housekeeping backstops (``pool.py``).
        """
        if self.topic is None:
            raise ValueError("PubSubDispatch.create_subscription requires a topic.")
        from google.api_core import exceptions as gax

        request: dict = {"name": subscription, "topic": self.topic}
        if filter:
            request["filter"] = filter
        if ttl_s:
            request["expiration_policy"] = {"ttl": {"seconds": int(ttl_s)}}
        if retention_s:
            request["message_retention_duration"] = {"seconds": int(retention_s)}
        try:
            self._subscriber_client().create_subscription(request=request)
        except gax.AlreadyExists:
            pass

    def delete_subscription(self, subscription: str) -> None:
        """Delete ``subscription`` (no error if already gone)."""
        from google.api_core import exceptions as gax

        try:
            self._subscriber_client().delete_subscription(subscription=subscription)
        except gax.NotFound:
            pass

    def list_subscriptions(self, prefix: str) -> list[str]:
        """Every subscription path in ``project`` starting with ``prefix`` (a pool sweep)."""
        if self.project is None:
            raise ValueError("PubSubDispatch.list_subscriptions requires a project.")
        client = self._subscriber_client()
        pager = client.list_subscriptions(request={"project": f"projects/{self.project}"})
        return sorted(s.name for s in pager if s.name.startswith(prefix))

    def delete_topic(self) -> None:
        """Delete the topic (no error if already gone)."""
        if self.topic is None:
            raise ValueError("PubSubDispatch.delete_topic requires a topic.")
        from google.api_core import exceptions as gax

        try:
            self._publisher_client().delete_topic(topic=self.topic)
        except gax.NotFound:
            pass


class InMemoryDispatch:
    """In-process :class:`DispatchTransport` for local dev / tests.

    Backed by thread-safe :class:`queue.Queue`\\ s: ``claim`` runs from
    ``asyncio.to_thread`` and several workers may claim concurrently, so we rely on
    ``Queue.get`` delivering each message to exactly one claimer (atomic claim). A message
    published with a ``worker`` attribute lands on that worker's own queue (the in-memory
    stand-in for a filtered per-worker subscription; claim it via :meth:`for_worker`);
    without one it lands on the shared queue :meth:`claim` reads (the legacy mode).
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[dict] = queue.Queue()
        self._worker_queues: dict[str, queue.Queue[dict]] = {}

    def _worker_queue(self, worker_id: str) -> queue.Queue[dict]:
        return self._worker_queues.setdefault(worker_id, queue.Queue())

    def publish(self, message: dict, attributes: dict[str, str] | None = None) -> None:
        worker = (attributes or {}).get("worker")
        target = self._worker_queue(worker) if worker else self._queue
        target.put(dict(message))  # shallow copy so callers can't mutate the queued msg

    def claim(self, timeout: float) -> dict | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def for_worker(self, worker_id: str) -> _WorkerView:
        """A claim-only view of ``worker_id``'s own queue (its per-worker subscription)."""
        return _WorkerView(self._worker_queue(worker_id))


class _WorkerView:
    """Claim side of one worker's in-memory queue."""

    def __init__(self, q: queue.Queue[dict]) -> None:
        self._queue = q

    def publish(self, message: dict, attributes: dict[str, str] | None = None) -> None:
        raise ValueError("a worker's view of its subscription is claim-only")

    def claim(self, timeout: float) -> dict | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

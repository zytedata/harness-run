"""``DispatchTransport`` port + adapters (DESIGN.md §6, §7).

Warm-pool dispatch + atomic claim. A warm worker blocks on ``claim`` pulling a shared
Pub/Sub subscription (competing-consumers = atomic claim); the control plane
``publish``es a (possibly resume-directive) message. Protocol is stdlib-only;
``PubSubDispatch`` imports ``google.cloud.pubsub_v1`` / ``google.api_core`` lazily so
this module stays importable with no third-party deps.
"""

from __future__ import annotations

import json
import queue
from typing import Protocol, runtime_checkable


@runtime_checkable
class DispatchTransport(Protocol):
    """Competing-consumers dispatch: ``publish`` a message, ``claim`` it atomically."""

    def publish(self, message: dict) -> None:
        """Publish a dispatch ``message`` (e.g. a kickoff or resume directive)."""
        ...

    def claim(self, timeout: float) -> dict | None:
        """Atomically claim one message, blocking up to ``timeout`` seconds.

        Returns the message, or ``None`` on timeout.
        """
        ...


class PubSubDispatch:
    """Pub/Sub-backed :class:`DispatchTransport` (competing-consumers).

    ``google.cloud.pubsub_v1`` / ``google.api_core`` are imported lazily inside the
    methods so importing this module pulls in no third-party deps. A worker that only
    claims passes ``subscription`` (no ``topic``); the control plane passes both.
    ``topic`` / ``subscription`` are full resource paths (``projects/<p>/topics/<t>``,
    ``projects/<p>/subscriptions/<s>``).
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

    def publish(self, message: dict) -> None:
        if self.topic is None:
            raise ValueError("PubSubDispatch.publish requires a topic (none configured).")
        client = self._publisher_client()
        data = json.dumps(message).encode("utf-8")
        client.publish(self.topic, data).result(timeout=30)

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

    def ensure(self) -> None:
        """Idempotently create the topic + subscription (control plane, at deploy).

        Each create is wrapped to ignore ``AlreadyExists`` so re-running deploy is a no-op.
        """
        if self.topic is None or self.subscription is None:
            raise ValueError("PubSubDispatch.ensure requires both topic and subscription.")
        from google.api_core import exceptions as gax

        publisher = self._publisher_client()
        try:
            publisher.create_topic(name=self.topic)
        except gax.AlreadyExists:
            pass

        subscriber = self._subscriber_client()
        try:
            subscriber.create_subscription(name=self.subscription, topic=self.topic)
        except gax.AlreadyExists:
            pass


class InMemoryDispatch:
    """In-process :class:`DispatchTransport` for local dev / tests.

    Backed by a thread-safe :class:`queue.Queue`: ``claim`` runs from
    ``asyncio.to_thread`` and several workers may claim concurrently, so we rely on
    ``Queue.get`` delivering each message to exactly one claimer (atomic claim).
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[dict] = queue.Queue()

    def publish(self, message: dict) -> None:
        self._queue.put(dict(message))  # shallow copy so callers can't mutate the queued msg

    def claim(self, timeout: float) -> dict | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

"""``DispatchTransport`` port + adapters (DESIGN.md §6, §7).

Warm-pool dispatch + atomic claim. A warm worker blocks on ``claim`` pulling a shared
Pub/Sub subscription (competing-consumers = atomic claim); the control plane
``publish``es a (possibly resume-directive) message. Protocol is stdlib-only;
``PubSubDispatch`` imports ``google.cloud.pubsub`` lazily. P0: protocol + stubs.
"""

from __future__ import annotations

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
    """Pub/Sub-backed :class:`DispatchTransport` (competing-consumers; P2).

    ``google.cloud.pubsub`` is imported lazily inside the methods.
    """

    def __init__(self, topic: str, subscription: str) -> None:
        self.topic = topic
        self.subscription = subscription

    def publish(self, message: dict) -> None:
        raise NotImplementedError("P2: publish the message to the Pub/Sub topic.")

    def claim(self, timeout: float) -> dict | None:
        raise NotImplementedError(
            "P2: pull one message from the shared subscription (atomic claim), ack it, "
            "and return it; None on timeout."
        )


class InMemoryDispatch:
    """In-process :class:`DispatchTransport` for local dev / tests (P1)."""

    def __init__(self) -> None:
        self._queue: list[dict] = []

    def publish(self, message: dict) -> None:
        raise NotImplementedError("P1: enqueue the message on an in-memory queue.")

    def claim(self, timeout: float) -> dict | None:
        raise NotImplementedError("P1: pop one message from the in-memory queue; None on timeout.")

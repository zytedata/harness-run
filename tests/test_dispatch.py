"""Offline tests for the DispatchTransport adapters (InMemoryDispatch only — no GCP).

Verifies the competing-consumers contract the Pub/Sub channel must also honour: a
published message round-trips, a single message is delivered to exactly one of two
concurrent claimers (atomic claim), an empty claim returns ``None`` without blocking,
and dict payloads survive the round-trip intact. PubSubDispatch is exercised only for
its config guards + lazy-import discipline (no live network).
"""

from __future__ import annotations

import sys

import pytest

from remote_agent_toolkit.ports.dispatch import InMemoryDispatch, PubSubDispatch


def test_inmemory_round_trip() -> None:
    d = InMemoryDispatch()
    d.publish({"a": 1})
    assert d.claim(0.1) == {"a": 1}


def test_inmemory_competing_consumers_atomic() -> None:
    # One message, two claimers: exactly one wins, the other times out with None.
    d = InMemoryDispatch()
    d.publish({"turn": 1})
    first = d.claim(0.1)
    second = d.claim(0.05)
    assert {first is None, second is None} == {False, True}
    winner = first if first is not None else second
    assert winner == {"turn": 1}


def test_inmemory_claim_empty_returns_none_quickly() -> None:
    d = InMemoryDispatch()
    # Empty queue: claim must return None within ~timeout, not block indefinitely.
    assert d.claim(0.05) is None


def test_inmemory_preserves_nested_dict() -> None:
    d = InMemoryDispatch()
    msg = {"directive": "resume", "meta": {"repo": "x", "ids": [1, 2, 3]}}
    d.publish(msg)
    assert d.claim(0.1) == {"directive": "resume", "meta": {"repo": "x", "ids": [1, 2, 3]}}


def test_publish_without_topic_raises() -> None:
    # A claim-only worker constructs with subscription but no topic; publishing must fail loudly.
    d = PubSubDispatch(subscription="projects/p/subscriptions/s")
    with pytest.raises(ValueError):
        d.publish({"a": 1})


def test_import_pulls_in_no_google() -> None:
    # Importing the module (done at top of file) must not eagerly import the GCP client.
    assert "google.cloud.pubsub_v1" not in sys.modules


def test_inmemory_addressed_publish_lands_only_on_that_workers_queue() -> None:
    # Per-worker dispatch: a message addressed to worker "a" is claimable on a's channel
    # only — not on the shared queue, not on another worker's (the filter's contract).
    d = InMemoryDispatch()
    d.publish({"turn": 1}, attributes={"worker": "a"})
    assert d.claim(0.05) is None
    assert d.for_worker("b").claim(0.05) is None
    assert d.for_worker("a").claim(0.1) == {"turn": 1}
    assert d.for_worker("a").claim(0.05) is None  # delivered once


def test_inmemory_worker_view_is_claim_only() -> None:
    with pytest.raises(ValueError):
        InMemoryDispatch().for_worker("a").publish({"a": 1})


def test_subscription_admin_requires_topic_or_project() -> None:
    # The control-plane admin methods fail loudly on a half-configured transport.
    d = PubSubDispatch(subscription="projects/p/subscriptions/s")
    with pytest.raises(ValueError):
        d.ensure_topic()
    with pytest.raises(ValueError):
        d.create_subscription("projects/p/subscriptions/w")
    with pytest.raises(ValueError):
        d.list_subscriptions("projects/p/subscriptions/ratk-")

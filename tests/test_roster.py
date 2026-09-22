"""Offline tests for the ready pool's idle-sandbox roster.

The roster is the shared, client-owned record of which sandboxes are ready and idle; a
turn goes to the sandbox whose entry the client managed to delete under a generation
precondition. These tests pin the ordering, expiry and atomic-claim contract over the
in-memory store, and the GCS store's precondition mapping over a fake storage client.
"""

from __future__ import annotations

from harness_run.runtime.sandbox.roster import (
    GcsRosterStore,
    InMemoryRosterStore,
    PoolRoster,
    SandboxEntry,
)

TOPIC = "projects/p/locations/l/reasoningEngines/h/sandboxEnvironmentTemplates/tpl1"


def _entry(i: int, submitted: float, expires: float) -> SandboxEntry:
    return SandboxEntry(
        sandbox=f"projects/p/locations/l/reasoningEngines/h/sandboxEnvironments/w{i}",
        template=TOPIC,
        created_at=submitted,
        expires_at=expires,
    )


def test_roster_keys_live_under_the_pool_prefix_and_round_trip():
    store = InMemoryRosterStore()
    roster = PoolRoster("gs://out/base", TOPIC, store=store)
    roster.add(_entry(1, 10.0, 100.0))
    # One object per idle sandbox under the template's pool prefix, keyed by the sandbox id.
    assert [name for name, _, _ in store.list("")] == ["base/pool/tpl1/idle/w1.json"]
    assert roster.entries() == [_entry(1, 10.0, 100.0)]
    roster.remove("w1")
    assert roster.entries() == []


def test_claim_is_oldest_first_and_skips_expired_entries():
    roster = PoolRoster("gs://out", TOPIC, store=InMemoryRosterStore())
    roster.add(_entry(2, 20.0, 200.0))
    roster.add(_entry(1, 10.0, 15.0))  # oldest, but its worker idled out at t=15
    roster.add(_entry(3, 30.0, 300.0))

    # Oldest LIVE sandbox; the expired one met on the way is pruned in the same listing and
    # handed back so it can be deleted.
    claimed, pruned = roster.claim(now=50.0)
    assert claimed.id == "w2" and pruned == [_entry(1, 10.0, 15.0)]
    # A margin treats an entry expiring within it as expired.
    roster.add(_entry(4, 40.0, 60.0))
    assert roster.claim(now=50.0, margin_s=15.0) == (_entry(3, 30.0, 300.0), [_entry(4, 40.0, 60.0)])
    roster.add(_entry(3, 30.0, 300.0))
    assert roster.prune_expired(now=50.0) == []
    assert roster.claim(now=50.0) == (_entry(3, 30.0, 300.0), [])
    assert roster.claim(now=50.0) == (None, [])  # empty: the caller creates a sandbox
    assert roster.entries() == []


def test_claim_is_atomic_across_clients():
    """Two clients racing for the same sandbox: the precondition delete lets exactly one
    win, and the loser moves on to the next entry instead of double-dispatching."""

    class Racy(InMemoryRosterStore):
        # Fires `hook` right after client B's listing: client A claims w1 in between B's
        # read of the roster and B's precondition delete (B holds a stale listing).
        hook = None

        def list(self, prefix):
            rows = super().list(prefix)
            if self.hook is not None:
                hook, self.hook = self.hook, None
                hook()
            return rows

    store = Racy()
    a = PoolRoster("gs://out", TOPIC, store=store)
    b = PoolRoster("gs://out", TOPIC, store=store)
    a.add(_entry(1, 10.0, 100.0))
    a.add(_entry(2, 20.0, 200.0))

    won: dict = {}
    store.hook = lambda: won.setdefault("a", a.claim(now=50.0)[0])
    got_b, _ = b.claim(now=50.0)
    assert won["a"].id == "w1"
    assert got_b.id == "w2"
    assert b.claim(now=50.0) == (None, [])


def test_clear_returns_what_was_recorded_and_corrupt_entries_are_ignored():
    store = InMemoryRosterStore()
    roster = PoolRoster("gs://out", TOPIC, store=store)
    roster.add(_entry(1, 10.0, 100.0))
    store.put("pool/tpl1/idle/junk.json", b"not json")  # never dispatch on it
    assert roster.entries() == [_entry(1, 10.0, 100.0)]
    assert roster.clear() == [_entry(1, 10.0, 100.0)]
    assert roster.entries() == []
    # Another pool's roster is untouched by prefix.
    other = PoolRoster("gs://out", TOPIC.replace("tpl1", "tpl2"), store=store)
    other.add(_entry(9, 1.0, 2.0))
    assert roster.entries() == [] and len(other.entries()) == 1


def test_gcs_store_maps_precondition_failures_to_a_lost_claim():
    """The atomic claim is a delete with ``if_generation_match``: a 412 (someone else
    deleted/rewrote it first) and a 404 (already gone) both mean "not ours"."""
    from google.api_core.exceptions import NotFound, PreconditionFailed

    class FakeBlob:
        def __init__(self, bucket, name):
            self.bucket, self.name = bucket, name
            self.metadata = None

        def download_as_bytes(self):
            try:
                return self.bucket.objects[self.name][0]
            except KeyError:
                raise NotFound(self.name) from None

        def upload_from_string(self, data, content_type=None):
            self.bucket.generation += 1
            self.bucket.objects[self.name] = (data, self.bucket.generation, self.metadata)

        def delete(self, if_generation_match=None):
            found = self.bucket.objects.get(self.name)
            if found is None:
                raise NotFound(self.name)
            if if_generation_match is not None and found[1] != if_generation_match:
                raise PreconditionFailed("generation mismatch")
            del self.bucket.objects[self.name]

    class FakeBucket:
        def __init__(self):
            self.objects: dict[str, tuple[bytes, int, dict | None]] = {}
            self.generation = 100

        def blob(self, name):
            return FakeBlob(self, name)

    class FakeClient:
        def __init__(self, bucket):
            self._bucket = bucket

        def bucket(self, name):
            return self._bucket

        def list_blobs(self, bucket, prefix=""):
            from types import SimpleNamespace as NS

            return [NS(name=n, generation=g, metadata=m)
                    for n, (_, g, m) in bucket.objects.items() if n.startswith(prefix)]

    bucket = FakeBucket()
    store = GcsRosterStore("out")
    store._client = FakeClient(bucket)  # skip the real storage.Client()

    store.put("pool/x/idle/w1.json", b"{}")
    (name, generation, inline), = store.list("pool/x/idle/")
    assert inline == "{}"  # the entry rides the listing (object metadata): no read on claim
    assert store.get(name) == b"{}"
    assert store.delete_if(name, generation + 1) is False  # someone rewrote it: lost
    assert store.delete_if(name, generation) is True  # ours
    assert store.delete_if(name, generation) is False  # already gone
    assert store.get(name) is None

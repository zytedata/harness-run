"""Per-invocation secrets handoff: staged single-use object, fetched + deleted by the worker.

SECURITY invariants under test: values never ride the invocation payload (only the gs://
pointer does), and the staged object is gone the moment it is read.
"""

from __future__ import annotations

from remote_agent_toolkit.ports.blobstore import LocalBlobStore
from remote_agent_toolkit.runtime.gemini import handoff


def test_stage_fetch_delete_round_trip(tmp_path):
    store = LocalBlobStore(str(tmp_path))
    secrets = {"SH_APIKEY": "k1", "GH_TOKEN": "t2"}

    uri = handoff.stage_secrets("gs://bkt", "sid-1", secrets, store=store)
    assert uri.startswith("gs://bkt/invocation-secrets/sid-1-") and uri.endswith(".json")
    key = uri[len("gs://bkt/"):]
    assert store.exists(key)  # staged

    # First fetch returns the values AND consumes the object (single-use).
    assert handoff.fetch_and_delete_secrets(uri, store=store) == secrets
    assert not store.exists(key)

    # Second fetch (already consumed) is empty, never an exception.
    assert handoff.fetch_and_delete_secrets(uri, store=store) == {}


def test_stage_secrets_unique_keys_and_bucket_prefix(tmp_path):
    store = LocalBlobStore(str(tmp_path))
    a = handoff.stage_secrets("gs://bkt/out", "sid", {"K": "v"}, store=store)
    b = handoff.stage_secrets("gs://bkt/out", "sid", {"K": "v"}, store=store)
    assert a != b  # nonce per staging: a retry never reuses/overwrites
    assert a.startswith("gs://bkt/out/invocation-secrets/sid-")


def test_delete_staged_secrets_best_effort(tmp_path):
    store = LocalBlobStore(str(tmp_path))
    uri = handoff.stage_secrets("gs://bkt", "sid", {"K": "v"}, store=store)
    handoff.delete_staged_secrets(uri, store=store)
    assert handoff.fetch_and_delete_secrets(uri, store=store) == {}
    handoff.delete_staged_secrets(uri, store=store)  # idempotent, no raise


def test_localblobstore_delete_idempotent(tmp_path):
    store = LocalBlobStore(str(tmp_path))
    store.put_bytes("a/b.txt", b"x")
    store.delete("a/b.txt")
    assert not store.exists("a/b.txt")
    store.delete("a/b.txt")  # absent -> no error

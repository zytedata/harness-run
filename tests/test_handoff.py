"""Per-invocation secrets handoff: staged object, retry-safe fetch, layered cleanup.

SECURITY invariants under test: values never ride the invocation payload (only the gs://
pointer does), and every staged object is reaped by at least one of the three cleanup
layers (worker at turn end / client on completion / bucket lifecycle rule).

RETRY invariant (regression for the 2026-07-28/29 live failures): the platform re-runs a
crashed job attempt with the SAME payload, so fetching must NOT consume the object — the
retry re-fetches the same pointer.
"""

from __future__ import annotations

from remote_agent_toolkit.ports.blobstore import LocalBlobStore
from remote_agent_toolkit.runtime.gemini import handoff


def test_fetch_is_retry_safe_and_delete_is_explicit(tmp_path):
    store = LocalBlobStore(str(tmp_path))
    secrets = {"SH_APIKEY": "k1", "GH_TOKEN": "t2"}

    uri = handoff.stage_secrets("gs://bkt", "sid-1", secrets, store=store)
    assert uri.startswith("gs://bkt/invocation-secrets/sid-1-") and uri.endswith(".json")
    key = uri[len("gs://bkt/"):]
    assert store.exists(key)  # staged

    # Fetch does NOT consume: a platform retry of a killed attempt re-fetches the same
    # pointer and must still get the values (the live failure mode when it couldn't).
    assert handoff.fetch_secrets(uri, store=store) == secrets
    assert store.exists(key)
    assert handoff.fetch_secrets(uri, store=store) == secrets  # attempt 2 gets them too

    # Deletion is explicit (worker at turn end / client on completion), then fetch is
    # empty — never an exception.
    handoff.delete_staged_secrets(uri, store=store)
    assert not store.exists(key)
    assert handoff.fetch_secrets(uri, store=store) == {}


def test_stage_secrets_unique_keys_and_bucket_prefix(tmp_path):
    store = LocalBlobStore(str(tmp_path))
    a = handoff.stage_secrets("gs://bkt/out", "sid", {"K": "v"}, store=store)
    b = handoff.stage_secrets("gs://bkt/out", "sid", {"K": "v"}, store=store)
    assert a != b  # nonce per staging: a resend never reuses/overwrites
    assert a.startswith("gs://bkt/out/invocation-secrets/sid-")


def test_delete_staged_secrets_best_effort(tmp_path):
    store = LocalBlobStore(str(tmp_path))
    uri = handoff.stage_secrets("gs://bkt", "sid", {"K": "v"}, store=store)
    handoff.delete_staged_secrets(uri, store=store)
    assert handoff.fetch_secrets(uri, store=store) == {}
    handoff.delete_staged_secrets(uri, store=store)  # idempotent, no raise


def test_stage_and_fetch_spec_round_trip(tmp_path):
    store = LocalBlobStore(str(tmp_path))
    spec_dict = {"name": "n", "model": "m", "repos": [{"url": "https://h/r.git"}]}
    uri = handoff.stage_spec("gs://bkt/out", "sid-9", spec_dict, store=store)
    assert uri.startswith("gs://bkt/out/invocation-spec/sid-9-") and uri.endswith(".json")

    # Same retry contract as secrets: fetch leaves the object (a platform retry re-fetches
    # the same pointer); deletion is explicit.
    assert handoff.fetch_spec(uri, store=store) == spec_dict
    assert handoff.fetch_spec(uri, store=store) == spec_dict
    handoff.delete_staged_secrets(uri, store=store)

    # A missing staging is None — the worker FAILS the turn on it (running the baked spec
    # instead of the requested one would be a silent wrong-configuration run), unlike
    # secrets where the run degrades gracefully.
    assert handoff.fetch_spec(uri, store=store) is None


def test_localblobstore_delete_idempotent(tmp_path):
    store = LocalBlobStore(str(tmp_path))
    store.put_bytes("a/b.txt", b"x")
    store.delete("a/b.txt")
    assert not store.exists("a/b.txt")
    store.delete("a/b.txt")  # absent -> no error


# -- lifecycle rule (the third cleanup layer, applied at deploy) ---------------------


def test_secrets_lifecycle_rule_scopes_to_prefix():
    name, rule = handoff.secrets_lifecycle_rule("gs://bkt/out")
    assert name == "bkt"
    assert rule == {
        "action": {"type": "Delete"},
        # Scoped under the output prefix: must never touch jobs/checkpoints/artifacts.
        # Covers BOTH staging prefixes (secrets + run-scoped specs).
        "condition": {
            "age": 1,
            "matchesPrefix": ["out/invocation-secrets/", "out/invocation-spec/"],
        },
    }
    _, bare = handoff.secrets_lifecycle_rule("gs://bkt")
    assert bare["condition"]["matchesPrefix"] == ["invocation-secrets/", "invocation-spec/"]


def test_ensure_lifecycle_upgrades_a_secrets_only_rule():
    # A bucket configured before the run-scoped-spec prefix existed has a secrets-only
    # delete rule; ensure() must notice the uncovered spec prefix and append the combined
    # rule (the doubled secrets coverage is harmless).
    old = {"action": {"type": "Delete"},
           "condition": {"age": 1, "matchesPrefix": ["invocation-secrets/"]}}
    bucket = _FakeBucket([old])
    assert handoff.ensure_secrets_lifecycle("gs://bkt", bucket_obj=bucket) is True
    assert bucket.patched == 1 and len(bucket.lifecycle_rules) == 2


class _FakeBucket:
    """The slice of google.cloud.storage.Bucket that ensure_secrets_lifecycle touches."""

    def __init__(self, rules=()):
        self.lifecycle_rules = list(rules)
        self.patched = 0

    def patch(self):
        self.patched += 1


def test_ensure_secrets_lifecycle_adds_rule_once():
    bucket = _FakeBucket()
    assert handoff.ensure_secrets_lifecycle("gs://bkt/out", bucket_obj=bucket) is True
    assert bucket.patched == 1
    assert bucket.lifecycle_rules == [handoff.secrets_lifecycle_rule("gs://bkt/out")[1]]

    # Idempotent: a covering rule (matched structurally, not by dict equality — GCS echoes
    # rules back with its own normalization) means no second patch.
    assert handoff.ensure_secrets_lifecycle("gs://bkt/out", bucket_obj=bucket) is True
    assert bucket.patched == 1


def test_ensure_secrets_lifecycle_preserves_foreign_rules_and_never_raises():
    foreign = {"action": {"type": "Delete"}, "condition": {"age": 30, "matchesPrefix": ["logs/"]}}
    bucket = _FakeBucket([foreign])
    assert handoff.ensure_secrets_lifecycle("gs://bkt", bucket_obj=bucket) is True
    assert foreign in bucket.lifecycle_rules and len(bucket.lifecycle_rules) == 2

    class Exploding:
        @property
        def lifecycle_rules(self):
            raise RuntimeError("403 storage.buckets.update denied")

    # Best-effort: a perms failure is a False (deploy warns), never an exception.
    assert handoff.ensure_secrets_lifecycle("gs://bkt", bucket_obj=Exploding()) is False

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


def test_config_persist_and_fetch(tmp_path):
    store = LocalBlobStore(str(tmp_path))
    cfg = {"model": "m-run", "repos": [{"url": "https://h/r.git"}]}

    # The session config lives at a STABLE per-session key (one config per session, bound
    # at open): every turn points here, and re-attach reads the same object back.
    uri = handoff.persist_session_config("gs://bkt/out", "sid-9", cfg, store=store)
    assert uri == "gs://bkt/out/session-config/sid-9.json"
    assert uri == handoff.session_config_uri("gs://bkt/out", "sid-9")
    assert handoff.fetch_config(uri, store=store) == cfg
    assert handoff.load_session_config("gs://bkt/out", "sid-9", store=store) == (uri, cfg)
    assert handoff.load_session_config("gs://bkt/out", "sid-none", store=store) is None

    # Turn configs are nonce-keyed (one per turn) and, like the session config, are KEPT
    # (post-mortem record) — only the lifecycle rule reaps them.
    turi = handoff.stage_turn_config("gs://bkt/out", "sid-9", {"model": "m-t"}, store=store)
    assert turi.startswith("gs://bkt/out/turn-config/sid-9-") and turi.endswith(".json")
    assert handoff.fetch_config(turi, store=store) == {"model": "m-t"}

    # A missing config is None — the worker FAILS the turn on it (running the baked spec
    # instead of the requested configuration would be a silent wrong-run), unlike secrets
    # where the run degrades gracefully.
    assert handoff.fetch_config("gs://bkt/out/turn-config/sid-9-gone.json", store=store) is None


def test_localblobstore_delete_idempotent(tmp_path):
    store = LocalBlobStore(str(tmp_path))
    store.put_bytes("a/b.txt", b"x")
    store.delete("a/b.txt")
    assert not store.exists("a/b.txt")
    store.delete("a/b.txt")  # absent -> no error


# -- lifecycle rule (the third cleanup layer, applied at deploy) ---------------------


def test_handoff_lifecycle_rules_scope_to_prefix():
    name, rules = handoff.handoff_lifecycle_rules("gs://bkt/out")
    assert name == "bkt"
    assert rules == [
        {
            # Secrets: tight reap (values; the rule is the backstop behind worker/client deletes).
            "action": {"type": "Delete"},
            "condition": {"age": 1, "matchesPrefix": ["out/invocation-secrets/"]},
        },
        {
            # Configs: kept for post-mortem debugging, aged out much later.
            "action": {"type": "Delete"},
            "condition": {
                "age": 30,
                "matchesPrefix": ["out/session-config/", "out/turn-config/"],
            },
        },
    ]
    _, bare = handoff.handoff_lifecycle_rules("gs://bkt")
    assert bare[0]["condition"]["matchesPrefix"] == ["invocation-secrets/"]
    assert bare[1]["condition"]["matchesPrefix"] == ["session-config/", "turn-config/"]


def test_ensure_lifecycle_appends_only_missing_rules():
    # A bucket configured by an older revision has a secrets-only delete rule; ensure()
    # must notice the uncovered config prefixes and append only the config rule (the
    # existing secrets coverage is respected, not duplicated).
    old = {"action": {"type": "Delete"},
           "condition": {"age": 1, "matchesPrefix": ["invocation-secrets/"]}}
    bucket = _FakeBucket([old])
    assert handoff.ensure_handoff_lifecycle("gs://bkt", bucket_obj=bucket) is True
    assert bucket.patched == 1 and len(bucket.lifecycle_rules) == 2
    assert bucket.lifecycle_rules[1]["condition"]["age"] == handoff.CONFIG_LIFECYCLE_DAYS


class _FakeBucket:
    """The slice of google.cloud.storage.Bucket that ensure_handoff_lifecycle touches."""

    def __init__(self, rules=()):
        self.lifecycle_rules = list(rules)
        self.patched = 0

    def patch(self):
        self.patched += 1


def test_ensure_handoff_lifecycle_adds_rules_once():
    bucket = _FakeBucket()
    assert handoff.ensure_handoff_lifecycle("gs://bkt/out", bucket_obj=bucket) is True
    assert bucket.patched == 1
    assert bucket.lifecycle_rules == handoff.handoff_lifecycle_rules("gs://bkt/out")[1]

    # Idempotent: covering rules (matched structurally, not by dict equality — GCS echoes
    # rules back with its own normalization) mean no second patch.
    assert handoff.ensure_handoff_lifecycle("gs://bkt/out", bucket_obj=bucket) is True
    assert bucket.patched == 1


def test_ensure_handoff_lifecycle_preserves_foreign_rules_and_never_raises():
    foreign = {"action": {"type": "Delete"}, "condition": {"age": 30, "matchesPrefix": ["logs/"]}}
    bucket = _FakeBucket([foreign])
    assert handoff.ensure_handoff_lifecycle("gs://bkt", bucket_obj=bucket) is True
    assert foreign in bucket.lifecycle_rules and len(bucket.lifecycle_rules) == 3

    class Exploding:
        @property
        def lifecycle_rules(self):
            raise RuntimeError("403 storage.buckets.update denied")

    # Best-effort: a perms failure is a False (deploy warns), never an exception.
    assert handoff.ensure_handoff_lifecycle("gs://bkt", bucket_obj=Exploding()) is False

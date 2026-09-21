"""The GCS records the client keeps next to a session: configs, deploy records, lifecycle rules."""

from __future__ import annotations

import pytest

from remote_agent_toolkit.ports.blobstore import LocalBlobStore
from remote_agent_toolkit.runtime.gemini import handoff


def test_session_config_is_stable_keyed_and_turn_configs_are_nonce_keyed(tmp_path):
    store = LocalBlobStore(str(tmp_path))
    cfg = {"model": "m-run", "repos": [{"url": "https://h/r.git"}]}
    uri = handoff.persist_session_config("gs://bkt/out", "sid-9", cfg, store=store)
    assert uri == "gs://bkt/out/session-config/sid-9.json" == handoff.session_config_uri("gs://bkt/out", "sid-9")
    assert handoff.load_session_config("gs://bkt/out", "sid-9", store=store) == (uri, cfg)
    assert handoff.load_session_config("gs://bkt/out", "sid-none", store=store) is None
    a = handoff.stage_turn_config("gs://bkt/out", "sid-9", {"model": "m-t"}, store=store)
    b = handoff.stage_turn_config("gs://bkt/out", "sid-9", {"model": "m-t"}, store=store)
    assert a != b and a.startswith("gs://bkt/out/turn-config/sid-9-") and a.endswith(".json")
    assert len(store.list("out/turn-config/")) == 2


def test_deploy_record_round_trip_and_best_effort_read(tmp_path):
    store = LocalBlobStore(str(tmp_path))
    uri = handoff.write_deploy_record("gs://bkt", "agent", "t1", {"spec": {"name": "agent"}}, store=store)
    assert uri == "gs://bkt/deploys/agent/t1.json" == handoff.deploy_record_uri("gs://bkt", "agent", "t1")
    assert handoff.load_deploy_record("gs://bkt", "agent", "t1", store=store) == {"spec": {"name": "agent"}}
    assert handoff.load_deploy_record("gs://bkt", "agent", "t2", store=store) is None

    class Denied:
        def get_bytes(self, key):
            raise PermissionError("403")

    assert handoff.load_deploy_record("gs://bkt", "agent", "t1", store=Denied()) is None  # never raises


def test_load_session_config_distinguishes_absence_from_failure():
    class Store:
        def __init__(self, outcome):
            self.outcome = outcome

        def get_bytes(self, key):
            if isinstance(self.outcome, Exception):
                raise self.outcome
            return self.outcome

    assert handoff.load_session_config("gs://b", "s", store=Store(KeyError("absent"))) is None
    for outcome in (PermissionError("secret-ish detail"), b"not json", b"[]"):
        with pytest.raises(RuntimeError, match="refusing fallback") as err:
            handoff.load_session_config("gs://b", "s", store=Store(outcome))
        assert "secret-ish" not in str(err.value)


def test_localblobstore_delete_idempotent(tmp_path):
    store = LocalBlobStore(str(tmp_path))
    store.put_bytes("a/b.txt", b"x")
    store.delete("a/b.txt")
    assert not store.exists("a/b.txt")
    store.delete("a/b.txt")


# -- lifecycle rules ---------------------------------------------------------------------


def test_handoff_lifecycle_rules_scope_to_prefix():
    name, rules = handoff.handoff_lifecycle_rules("gs://bkt/out")
    assert name == "bkt"
    assert rules == [
        {"action": {"type": "Delete"},
         "condition": {"age": 1, "matchesPrefix": ["out/invocation-secrets/"]}},
        {"action": {"type": "Delete"},
         "condition": {"age": 30, "matchesPrefix": ["out/session-config/", "out/turn-config/"]}},
    ]
    _, bare = handoff.handoff_lifecycle_rules("gs://bkt")
    assert bare[0]["condition"]["matchesPrefix"] == ["invocation-secrets/"]
    assert bare[1]["condition"]["matchesPrefix"] == ["session-config/", "turn-config/"]


class _FakeBucket:
    def __init__(self, rules=()):
        self.lifecycle_rules = list(rules)
        self.patched = 0

    def patch(self):
        self.patched += 1


def test_ensure_lifecycle_appends_only_missing_rules():
    old = {"action": {"type": "Delete"}, "condition": {"age": 1, "matchesPrefix": ["invocation-secrets/"]}}
    bucket = _FakeBucket([old])
    assert handoff.ensure_handoff_lifecycle("gs://bkt", bucket_obj=bucket) is True
    assert bucket.patched == 1 and len(bucket.lifecycle_rules) == 2
    assert bucket.lifecycle_rules[1]["condition"]["age"] == handoff.CONFIG_LIFECYCLE_DAYS


def test_ensure_handoff_lifecycle_adds_rules_once():
    bucket = _FakeBucket()
    assert handoff.ensure_handoff_lifecycle("gs://bkt/out", bucket_obj=bucket) is True
    assert bucket.patched == 1
    assert bucket.lifecycle_rules == handoff.handoff_lifecycle_rules("gs://bkt/out")[1]
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

    assert handoff.ensure_handoff_lifecycle("gs://bkt", bucket_obj=Exploding()) is False

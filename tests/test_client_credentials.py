"""Client-side GCS operations must honor the engine's explicit identity."""
import json
from types import SimpleNamespace

from sandbox_fakes import FakeSandboxProvider
from sandbox_fakes import make_engine as _make_engine

from agent_run import SessionConfig
from agent_run.ports import blobstore
from agent_run.runtime.sandbox import backend, handoff, history


def test_config_operations_use_explicit_credentials(monkeypatch):
    explicit = object()
    seen = []

    class CaptureStore:
        def __init__(self, bucket, prefix="", credentials=None):
            seen.append(credentials)

        def put_bytes(self, key, body):
            pass

        def get_bytes(self, key):
            return json.dumps({"model": "dummy"}).encode()

        def delete(self, key):
            pass

    monkeypatch.setattr(blobstore, "GcsBlobStore", CaptureStore)
    monkeypatch.setattr(handoff, "GcsBlobStore", CaptureStore)
    engine = _make_engine(FakeSandboxProvider(), output_bucket="gs://audit-bucket/prefix", credentials=explicit)
    session = backend.SandboxSession(engine, "audit")
    session._bind_config(SessionConfig(model="dummy"))
    engine.get_session("reattach")._client_spec()
    assert len(seen) == 2 and all(credential is explicit for credential in seen)


def test_history_uses_its_explicit_credentials(monkeypatch):
    explicit = object()
    seen = []

    class CaptureStore:
        def __init__(self, bucket, prefix="", credentials=None):
            seen.append(credentials)

        def list(self, prefix):
            return [prefix + "dummy.jsonl"]

        def get_bytes(self, key):
            return b'{"kind":"message","summary":"dummy"}\n'

    monkeypatch.setattr(history, "GcsBlobStore", CaptureStore)
    assert history.read_history("gs://audit-bucket", "audit", credentials=explicit)
    assert seen == [explicit]


def test_lifecycle_bucket_client_uses_explicit_credentials(monkeypatch):
    import google.cloud.storage

    explicit = object()
    seen = []
    bucket = SimpleNamespace(lifecycle_rules=handoff.handoff_lifecycle_rules("gs://audit-bucket")[1],
                             reload=lambda: None)

    def client(**kwargs):
        seen.append(kwargs)
        return SimpleNamespace(bucket=lambda name: bucket)

    monkeypatch.setattr(google.cloud.storage, "Client", client)
    assert handoff.ensure_handoff_lifecycle("gs://audit-bucket", credentials=explicit)
    assert seen == [{"credentials": explicit, "project": "_"}]


def test_roster_and_records_use_the_engine_credentials():
    explicit = object()
    engine = _make_engine(FakeSandboxProvider(), credentials=explicit, roster_store=None)
    assert engine._roster()._store._credentials is explicit
    assert engine._gcs_store_kwargs()["store"]._credentials is explicit

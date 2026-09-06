"""Scoped token expiry/renewal with fake clocks, stores and thread startup (B02)."""
import datetime as dt
from types import SimpleNamespace

import pytest
from google.auth import _helpers

from remote_agent_toolkit import AgentSpec
from remote_agent_toolkit.ports import blobstore
from remote_agent_toolkit.runtime.gemini import adk_agent, backend, scoped_gcs


def test_legacy_activation_never_binds_a_non_expiring_credential(monkeypatch):
    bound = []
    monkeypatch.setenv("AGENT_EVENTS_GCS", "gs://audit-bucket/events")
    monkeypatch.setattr(blobstore, "set_default_gcs_credentials", bound.append)
    assert adk_agent._use_run_gcs_token("AUDIT_FAKE_TOKEN", "audit")
    assert bound[0].expiry is not None


def test_client_preserves_expiry_and_writes_initial_refresh_before_thread(monkeypatch):
    expiry = dt.datetime(2099, 1, 1)
    seen = []
    monkeypatch.setattr(scoped_gcs, "mint_run_token", lambda *a: ("AUDIT_FAKE_TOKEN", expiry))
    monkeypatch.setattr(scoped_gcs, "write_run_token", lambda *a, **kw: seen.append(("write", a)))
    monkeypatch.setattr(scoped_gcs, "delete_run_token", lambda *a, **kw: None)
    monkeypatch.setattr(backend.threading, "Thread", lambda **kw: SimpleNamespace(
        start=lambda: seen.append(("thread", None))))
    engine = backend.GeminiEngine("dummy", AgentSpec(name="audit", model="dummy"), "dummy", "dummy",
                                  output_bucket="gs://audit-bucket")
    session = backend.GeminiSession(engine, "audit")
    assert session._mint_gcs_token() == "AUDIT_FAKE_TOKEN"
    assert seen[0] == ("write", ("gs://audit-bucket", "audit", "AUDIT_FAKE_TOKEN", expiry))
    assert seen[1][0] == "thread"
    assert session._gcs_token_expiry == expiry.isoformat()
    session._stop_gcs_token_refresh()


def test_worker_refreshes_before_transport_expiry(monkeypatch):
    initial = dt.datetime(2099, 1, 1, 1)
    bound, fetched = [], []
    original = scoped_gcs.worker_credentials

    def capture(*args, **kwargs):
        return original(*args, **kwargs, fetch=lambda url, bearer: fetched.append(bearer) or {
            "token": "AUDIT_FRESH", "expiry": "2099-01-01T02:00:00",
        })

    monkeypatch.setenv("AGENT_EVENTS_GCS", "gs://audit-bucket/events")
    monkeypatch.setattr(scoped_gcs, "worker_credentials", capture)
    monkeypatch.setattr(blobstore, "set_default_gcs_credentials", bound.append)
    assert adk_agent._use_run_gcs_token("AUDIT_OLD", "audit", expiry=initial.isoformat())
    assert bound[0].expiry == initial
    monkeypatch.setattr(_helpers, "utcnow", lambda: initial - dt.timedelta(minutes=1))
    headers = {}
    bound[0].before_request(None, "GET", "https://audit.invalid", headers)
    assert fetched == ["AUDIT_OLD"]
    assert headers["authorization"] == "Bearer AUDIT_FRESH"


def test_failed_initial_write_keeps_a_cleanup_handle_and_starts_no_thread(monkeypatch):
    deleted = []

    def failed_write(*a, **kw):
        raise TimeoutError("dummy write failure")

    monkeypatch.setattr(scoped_gcs, "write_run_token", failed_write)
    monkeypatch.setattr(scoped_gcs, "delete_run_token", lambda *a, **kw: deleted.append(a))
    monkeypatch.setattr(backend.threading, "Thread", lambda **kw: pytest.fail("unexpected thread"))
    engine = backend.GeminiEngine("dummy", AgentSpec(name="audit", model="dummy"), "dummy", "dummy",
                                  output_bucket="gs://audit-bucket")
    session = backend.GeminiSession(engine, "audit")
    with pytest.raises(TimeoutError):
        session._mint_gcs_token()
    assert session._gcs_token_stop is not None
    session._stop_gcs_token_refresh()
    assert deleted == [("gs://audit-bucket", "audit")]


def test_invalid_expiry_cannot_be_silently_treated_as_unknown(monkeypatch):
    monkeypatch.setenv("AGENT_EVENTS_GCS", "gs://audit-bucket/events")
    with pytest.raises(ValueError, match="expiry"):
        adk_agent._use_run_gcs_token("AUDIT_OLD", "audit", expiry="not-a-date")


def test_expiry_directive_and_warm_payload_roundtrip():
    from remote_agent_toolkit.runtime.gemini.pool import dispatch_payload

    expiry = "2099-01-01T01:00:00"
    token, rest = adk_agent._split_gcs_token_directive(
        f"AGENT_GCS_TOKEN=AUDIT_FAKE_TOKEN\nAGENT_GCS_TOKEN_EXPIRY={expiry}\nhello")
    assert token == "AUDIT_FAKE_TOKEN"  # old token parser still recognizes the credential
    parsed, prompt = adk_agent._split_gcs_token_expiry_directive(rest)
    assert parsed == expiry and prompt == "hello"
    payload = dispatch_payload("audit", "hello", False, gcs_token=token, gcs_token_expiry=expiry)
    assert payload["gcs_token_expiry"] == expiry

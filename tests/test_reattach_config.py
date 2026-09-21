"""A failed config read must not turn a reattached session into an unconfigured one."""
import json

import pytest
from sandbox_fakes import FakeSandboxProvider, make_engine

from agent_run.runtime.sandbox import handoff


class Store:
    def __init__(self, outcome):
        self.outcome = outcome

    def get_bytes(self, key):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


@pytest.mark.parametrize("outcome", [
    PermissionError("dummy secret must not be echoed"), TimeoutError("temporary"),
    b"invalid-json", b"[]", b"null",
])
def test_read_or_decode_failure_is_not_absence(outcome):
    with pytest.raises(RuntimeError, match="refusing fallback") as error:
        handoff.load_session_config("gs://audit-bucket", "audit", store=Store(outcome))
    assert "dummy secret" not in str(error.value)


def test_known_absence_still_supports_an_unconfigured_session():
    assert handoff.load_session_config("gs://audit-bucket", "audit",
                                       store=Store(KeyError("absent"))) is None


def test_failed_reattach_does_not_dispatch_or_cache_fallback(monkeypatch):
    store = Store(PermissionError("denied"))
    monkeypatch.setattr(handoff, "GcsBlobStore", lambda *a, **kw: store)
    provider = FakeSandboxProvider()
    engine = make_engine(provider, output_bucket="gs://audit-bucket")
    session = engine.get_session("audit")
    with pytest.raises(RuntimeError, match="refusing fallback"):
        session.run("dummy")
    assert not session._session_config_resolved
    assert session._session_config_uri is None
    assert provider.calls == []  # never reached the platform
    store.outcome = json.dumps({"permission_mode": "default"}).encode()
    assert session._client_spec().permission_mode == "default"
    assert session._session_config_resolved

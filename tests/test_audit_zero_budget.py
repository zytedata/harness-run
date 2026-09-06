"""Budget admission tests use a loopback server with a forbidden/fake HTTPS upstream."""
import http.client
from urllib.parse import urlsplit

import pytest

from remote_agent_toolkit.harness import _openrouter_proxy as module


def test_zero_budget_is_exhausted_before_any_cost_is_known():
    proxy = module.OpenRouterProxy("AUDIT_FAKE_KEY", max_budget_usd=0.0)
    assert proxy.exact_cost_usd is None
    assert proxy._over_budget()


@pytest.mark.parametrize("budget", [-1, float("nan"), float("inf"), -float("inf")])
def test_invalid_limits_are_rejected(budget):
    with pytest.raises(ValueError, match="max_budget_usd"):
        module.OpenRouterProxy("AUDIT_FAKE_KEY", max_budget_usd=budget)


def test_zero_budget_never_invokes_upstream(monkeypatch):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("Dummy upstream must not be reached")

    monkeypatch.setattr(module.http.client, "HTTPSConnection", forbidden)
    proxy = module.OpenRouterProxy("AUDIT_FAKE_KEY", max_budget_usd=0.0).start()
    address = urlsplit(proxy.base_url)
    client = http.client.HTTPConnection(address.hostname, address.port, timeout=3)
    try:
        client.request("POST", "/responses", body=b"{}", headers={
            "Authorization": "Bearer " + proxy.client_token,
        })
        response = client.getresponse()
        body = response.read()
        assert response.status == 402 and b"budget_exceeded" in body
        assert not calls
    finally:
        client.close()
        proxy.close()

"""OpenRouter response metadata parsing. No server or external request is used."""

from __future__ import annotations

import json
import http.client
from urllib.parse import urlsplit

import pytest

from remote_agent_toolkit.harness import _openrouter_proxy as proxy_module
from remote_agent_toolkit.harness._openrouter_proxy import (
    OpenRouterProxy,
    _append_capture,
    _capture_request,
    _model_matches,
    _provider_matches_request,
    _request_with_provider,
)


def test_captures_responses_stream_cost_and_selected_endpoint():
    response = {
        "type": "response.completed",
        "response": {
            "model": "deepseek/deepseek-v4-flash",
            "usage": {"cost": 0.0012},
            "openrouter_metadata": {
                "requested": "deepseek/deepseek-v4-flash",
                "region": "MAD",
                "attempt": 2,
                "summary": "available=2, attempts=2, selected=Novita",
                "endpoints": {
                    "available": [
                        {"provider": "Baidu", "model": "snapshot", "selected": False},
                        {"provider": "Novita", "model": "snapshot", "selected": True},
                    ]
                },
            },
        },
    }
    body = f"event: response.completed\ndata: {json.dumps(response)}\n\n".encode()

    found = _capture_request(body, "text/event-stream", None, "novita", 200)

    assert found is not None
    assert found.requested_model == "deepseek/deepseek-v4-flash"
    assert found.requested_provider == "novita"
    assert found.provider == "Novita"
    assert found.provider_model == "snapshot"
    assert found.cost_usd == 0.0012
    assert found.attempt == 2


def test_captures_anthropic_json_shape():
    body = json.dumps(
        {
            "model": "z-ai/glm-5.3",
            "provider": "Z.AI",
            "usage": {"cost": 0.0042},
            "openrouter_metadata": {
                "requested": "z-ai/glm-5.3",
                "region": "MAD",
                "endpoints": {"available": []},
            },
        }
    ).encode()

    found = _capture_request(body, "application/json", "asked", "z-ai", 200)

    assert found is not None
    assert found.provider == "Z.AI"
    assert found.cost_usd == 0.0042
    assert found.requested_model == "z-ai/glm-5.3"
    assert found.event().raw["provider_matches_request"] is True


def test_ignores_responses_without_openrouter_fields():
    assert _capture_request(b'{"ok":true}', "application/json", "m", None, 200) is None


def test_capture_buffer_keeps_final_stream_metadata():
    captured = bytearray()
    _append_capture(captured, b"old bytes\n", limit=16)
    _append_capture(captured, b"data: final\n\n", limit=16)
    assert bytes(captured).endswith(b"data: final\n\n")
    assert len(captured) == 16


def test_wait_until_idle_reports_timeout_and_completion():
    proxy = OpenRouterProxy("key")
    proxy._begin_request()
    assert proxy.wait_until_idle(0.001) is False
    proxy._end_request()
    assert proxy.wait_until_idle(0.001) is True


def test_model_scoped_relay_rejects_missing_or_different_model():
    assert _model_matches(None, None)
    assert _model_matches("vendor/model", "vendor/model")
    assert not _model_matches("vendor/model", "other/model")
    assert not _model_matches("vendor/model", None)


def test_adds_strict_provider_choice_to_request_body():
    body = json.dumps(
        {
            "model": "moonshotai/kimi-k3",
            "messages": [{"role": "user", "content": "hello"}],
            "provider": {"sort": "price"},
        }
    ).encode()

    changed = _request_with_provider(body, "moonshotai")

    assert changed is not None
    request = json.loads(changed)
    assert request["model"] == "moonshotai/kimi-k3"
    assert request["messages"] == [{"role": "user", "content": "hello"}]
    assert request["provider"] == {
        "only": ["moonshotai"],
        "allow_fallbacks": False,
    }


def test_provider_choice_requires_a_json_object():
    with pytest.raises(ValueError, match="JSON request body"):
        _request_with_provider(b"not json", "moonshotai")
    with pytest.raises(ValueError, match="JSON object"):
        _request_with_provider(b"[]", "moonshotai")


def test_provider_name_matching_accepts_openrouter_display_names():
    assert _provider_matches_request("moonshotai", "Moonshot AI") is True
    assert _provider_matches_request("z-ai", "Z.AI") is True
    assert _provider_matches_request("novita", "Together") is False
    assert _provider_matches_request(None, "Together") is None


def test_relay_sends_provider_choice_to_openrouter(monkeypatch):
    """The live relay path must send the provider rule, with no fallback."""
    upstream_requests = []

    class Response:
        status = 200
        reason = "OK"

        def __init__(self):
            self._body = json.dumps(
                {
                    "model": "moonshotai/kimi-k3",
                    "provider": "Moonshot AI",
                    "usage": {"cost": 0.001},
                }
            ).encode()

        def getheader(self, name, default=None):
            return "application/json" if name == "Content-Type" else default

        def getheaders(self):
            return [("Content-Type", "application/json")]

        def read(self, _size=-1):
            body, self._body = self._body, b""
            return body

    class Upstream:
        def __init__(self, host, timeout):
            assert host == "openrouter.ai"
            assert timeout == 600

        def request(self, method, path, body=None, headers=None):
            upstream_requests.append((method, path, body, headers))

        def getresponse(self):
            return Response()

        def close(self):
            pass

    monkeypatch.setattr(proxy_module.http.client, "HTTPSConnection", Upstream)

    with OpenRouterProxy(
        "real-key",
        expected_model="moonshotai/kimi-k3",
        provider="moonshotai",
    ) as proxy:
        target = urlsplit(proxy.base_url)
        conn = http.client.HTTPConnection(target.hostname, target.port, timeout=5)
        conn.request(
            "POST",
            "/api/v1/responses",
            body=json.dumps({"model": "moonshotai/kimi-k3", "input": "hello"}),
            headers={
                "Authorization": f"Bearer {proxy.client_token}",
                "Content-Type": "application/json",
            },
        )
        response = conn.getresponse()
        response.read()
        conn.close()
        assert response.status == 200
        assert proxy.wait_until_idle()

    assert len(upstream_requests) == 1
    method, path, body, headers = upstream_requests[0]
    assert (method, path) == ("POST", "/api/v1/responses")
    assert json.loads(body)["provider"] == {
        "only": ["moonshotai"],
        "allow_fallbacks": False,
    }
    assert headers["Authorization"] == "Bearer real-key"
    event = proxy.drain_events()[0]
    assert event.raw["requested_provider"] == "moonshotai"
    assert event.raw["provider"] == "Moonshot AI"
    assert event.raw["provider_matches_request"] is True

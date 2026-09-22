"""OpenRouter proxy: metadata parsing, and the localhost server with a faked upstream.

The server tests bind a real socket on 127.0.0.1 and monkeypatch ``HTTPSConnection``, so
nothing leaves the machine.
"""

from __future__ import annotations

import json
import http.client
from urllib.parse import urlsplit

import pytest

from harness_run.harness import _openrouter_proxy as proxy_module
from harness_run.harness._openrouter_proxy import (
    OpenRouterProxy,
    _append_capture,
    _capture_request,
    _model_matches,
    _provider_matches_request,
    _provider_matches_routing,
    _request_with_provider,
    _routing_allowed_providers,
)


def _pin(slug: str) -> dict:
    """The routing object a harness builds from ``spec.openrouter_provider``."""
    return {"only": [slug], "allow_fallbacks": False}


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

    found = _capture_request(body, "text/event-stream", None, _pin("novita"), 200)

    assert found is not None
    assert found.requested_model == "deepseek/deepseek-v4-flash"
    assert found.requested_routing == _pin("novita")
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

    found = _capture_request(body, "application/json", "asked", _pin("z-ai"), 200)

    assert found is not None
    assert found.provider == "Z.AI"
    assert found.cost_usd == 0.0042
    assert found.requested_model == "z-ai/glm-5.3"
    assert found.event().raw["provider_matches_request"] is True


def test_captures_anthropic_sse_shape():
    """Claude Code's wire: usage arrives on ``message_delta``, metadata on ``message_stop``.

    ``usage.cost`` is cumulative per response, so the last value is the charge — the same
    reason the reading is committed once per response rather than summed per delta.
    """
    events = [
        {"type": "message_start", "message": {"model": "moonshotai/kimi-k3", "usage": {}}},
        {"type": "content_block_delta", "delta": {"text": "hi"}},
        {"type": "message_delta", "usage": {"cost": 0.0009}},
        {"type": "message_delta", "usage": {"cost": 0.0021}},
        {
            "type": "message_stop",
            "openrouter_metadata": {
                "requested": "moonshotai/kimi-k3",
                "region": "MAD",
                "attempt": 1,
                "endpoints": {
                    "available": [
                        {"provider": "Baidu", "model": "kimi-k3", "selected": False},
                        {"provider": "Moonshot AI", "model": "kimi-k3-0821", "selected": True},
                    ]
                },
            },
        },
    ]
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()

    found = _capture_request(
        body, "text/event-stream", "moonshotai/kimi-k3", _pin("moonshotai"), 200
    )

    assert found is not None
    assert found.cost_usd == 0.0021
    assert found.provider == "Moonshot AI"
    assert found.provider_model == "kimi-k3-0821"
    assert found.requested_model == "moonshotai/kimi-k3"
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


def test_model_scoped_proxy_rejects_missing_or_different_model():
    assert _model_matches(None, None)
    assert _model_matches("vendor/model", "vendor/model")
    assert not _model_matches("vendor/model", "other/model")
    assert not _model_matches("vendor/model", None)


def test_provider_choice_requires_a_json_object():
    with pytest.raises(ValueError, match="JSON request body"):
        _request_with_provider(b"not json", _pin("moonshotai"))
    with pytest.raises(ValueError, match="JSON object"):
        _request_with_provider(b"[]", _pin("moonshotai"))


def test_sends_a_routing_object_verbatim():
    body = json.dumps(
        {
            "model": "moonshotai/kimi-k3",
            "messages": [{"role": "user", "content": "hello"}],
            "provider": {"sort": "price"},
        }
    ).encode()
    routing = {"order": ["moonshotai", "fireworks"], "allow_fallbacks": True}

    changed = _request_with_provider(body, routing)

    assert changed is not None
    request = json.loads(changed)
    assert request["model"] == "moonshotai/kimi-k3"
    assert request["messages"] == [{"role": "user", "content": "hello"}]
    assert request["provider"] == routing


def test_a_routing_object_leaves_the_body_alone_when_there_is_none():
    assert _request_with_provider(b'{"model":"m"}', None) == b'{"model":"m"}'


def test_routing_closes_the_provider_set_only_when_it_cannot_route_past_it():
    # ``only`` is closed: OpenRouter may fall back, but only inside the list.
    assert _routing_allowed_providers({"only": ["moonshotai", "fireworks"]}) == [
        "moonshotai",
        "fireworks",
    ]
    assert _routing_allowed_providers({"only": ["moonshotai"], "allow_fallbacks": True}) == [
        "moonshotai"
    ]
    # ``order`` is closed only with fallbacks off.
    assert _routing_allowed_providers({"order": ["moonshotai"], "allow_fallbacks": False}) == [
        "moonshotai"
    ]
    assert _routing_allowed_providers({"order": ["moonshotai"], "allow_fallbacks": True}) is None
    assert _routing_allowed_providers({"order": ["moonshotai"]}) is None
    # Nothing that names an allowed set: the set stays open.
    assert _routing_allowed_providers({"ignore": ["novita"]}) is None
    assert _routing_allowed_providers({"sort": "price"}) is None
    assert _routing_allowed_providers({}) is None
    assert _routing_allowed_providers(None) is None
    # A list that is not slugs proves nothing either.
    assert _routing_allowed_providers({"only": []}) is None
    assert _routing_allowed_providers({"only": "moonshotai"}) is None
    assert _routing_allowed_providers({"only": [{"slug": "moonshotai"}]}) is None


def test_routing_matching_accepts_any_allowed_provider():
    routing = {"only": ["moonshotai", "z-ai"]}
    assert _provider_matches_routing(routing, "Moonshot AI") is True
    assert _provider_matches_routing(routing, "Z.AI") is True
    assert _provider_matches_routing(routing, "Together") is False
    # An open set claims nothing, even about a provider that is in the list.
    assert _provider_matches_routing({"order": ["moonshotai"]}, "Moonshot AI") is None
    assert _provider_matches_routing(routing, None) is None


def test_provider_name_matching_accepts_openrouter_display_names():
    assert _provider_matches_request("moonshotai", "Moonshot AI") is True
    assert _provider_matches_request("z-ai", "Z.AI") is True
    assert _provider_matches_request("google-vertex/us-east5", "Google Vertex") is True
    assert _provider_matches_request("novita", "Together") is False
    assert _provider_matches_request(None, "Together") is None


def test_proxy_sends_provider_choice_to_openrouter(monkeypatch):
    """The live proxy path must send the provider rule, with no fallback."""
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
        routing=_pin("moonshotai"),
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
    assert json.loads(body)["provider"] == _pin("moonshotai")
    assert headers["Authorization"] == "Bearer real-key"
    event = proxy.drain_events()[0]
    assert event.raw["requested_routing"] == _pin("moonshotai")
    assert event.raw["provider"] == "Moonshot AI"
    assert event.raw["provider_matches_request"] is True


class _FakeResponse:
    """One canned upstream response, read once."""

    def __init__(self, status=200, body=b"{}", content_type="application/json"):
        self.status = status
        self.reason = "OK" if status == 200 else "ERR"
        self._body = body
        self._content_type = content_type

    def getheader(self, name, default=None):
        return self._content_type if name == "Content-Type" else default

    def getheaders(self):
        return [("Content-Type", self._content_type)]

    def read(self, _size=-1):
        body, self._body = self._body, b""
        return body


def _fake_upstream(monkeypatch, response=None, raises=None, calls=None):
    """Point the proxy's upstream at a canned response, or make connecting fail."""

    class Upstream:
        def __init__(self, host, timeout):
            assert host == "openrouter.ai"

        def request(self, method, path, body=None, headers=None):
            if calls is not None:
                calls.append((method, path, body, headers))
            if raises is not None:
                raise raises

        def getresponse(self):
            return response or _FakeResponse()

        def close(self):
            pass

    monkeypatch.setattr(proxy_module.http.client, "HTTPSConnection", Upstream)


def _proxy_post(proxy, body=None, headers=None, path="/api/v1/responses"):
    """POST to the running proxy and return (status, body-bytes)."""
    target = urlsplit(proxy.base_url)
    conn = http.client.HTTPConnection(target.hostname, target.port, timeout=5)
    sent = {"Authorization": f"Bearer {proxy.client_token}", "Content-Type": "application/json"}
    sent.update(headers or {})
    conn.request("POST", path, body=body, headers=sent)
    response = conn.getresponse()
    payload = response.read()
    status = response.status
    conn.close()
    return status, payload


@pytest.mark.parametrize(
    ("routing", "matches"),
    [
        ({"only": ["moonshotai", "fireworks"]}, True),
        ({"order": ["moonshotai", "fireworks"], "allow_fallbacks": True}, None),
    ],
    ids=["closed-set", "open-set"],
)
def test_proxy_sends_a_routing_object_and_judges_it(monkeypatch, routing, matches):
    """The live proxy path sends the routing object, and only claims a closed set."""
    calls = []
    _fake_upstream(
        monkeypatch,
        response=_FakeResponse(
            body=json.dumps(
                {
                    "model": "moonshotai/kimi-k3",
                    "provider": "Moonshot AI",
                    "usage": {"cost": 0.001},
                }
            ).encode()
        ),
        calls=calls,
    )

    with OpenRouterProxy(
        "real-key",
        expected_model="moonshotai/kimi-k3",
        routing=routing,
    ) as proxy:
        status, _ = _proxy_post(
            proxy, body=json.dumps({"model": "moonshotai/kimi-k3", "input": "hello"})
        )
        assert proxy.wait_until_idle()

    assert status == 200
    assert json.loads(calls[0][2])["provider"] == routing
    event = proxy.drain_events()[0]
    assert event.raw["requested_routing"] == routing
    assert event.raw["provider"] == "Moonshot AI"
    assert event.raw["provider_matches_request"] is matches


def test_upstream_failure_returns_502_json(monkeypatch):
    """A dead upstream must reach the CLI as a readable error, not a closed socket."""
    _fake_upstream(monkeypatch, raises=ConnectionRefusedError("no route"))

    with OpenRouterProxy("real-key", expected_model="moonshotai/kimi-k3") as proxy:
        status, payload = _proxy_post(
            proxy, body=json.dumps({"model": "moonshotai/kimi-k3", "input": "hi"})
        )
        assert proxy.wait_until_idle()

    assert status == 502
    assert json.loads(payload)["error"]["type"] == "proxy_upstream_error"
    assert "ConnectionRefusedError" in json.loads(payload)["error"]["message"]
    assert proxy.drain_events() == []


def test_chunked_request_body_is_rejected_with_411(monkeypatch):
    """Chunked framing would read as an empty body; say so instead of forwarding it."""
    calls = []
    _fake_upstream(monkeypatch, calls=calls)

    with OpenRouterProxy("real-key", expected_model="moonshotai/kimi-k3") as proxy:
        status, payload = _proxy_post(proxy, headers={"Transfer-Encoding": "chunked"})

    assert status == 411
    assert "chunked" in json.loads(payload)["error"]["message"]
    assert calls == []


def test_error_status_response_is_recorded(monkeypatch):
    """A 429 carries no metadata and no cost, and is only visible here."""
    _fake_upstream(
        monkeypatch,
        response=_FakeResponse(status=429, body=b'{"error":{"message":"rate limited"}}'),
    )

    with OpenRouterProxy(
        "real-key", expected_model="moonshotai/kimi-k3", routing=_pin("moonshotai")
    ) as proxy:
        status, _ = _proxy_post(
            proxy, body=json.dumps({"model": "moonshotai/kimi-k3", "input": "hi"})
        )
        assert proxy.wait_until_idle()

    assert status == 429
    event = proxy.drain_events()[0]
    assert event.raw["http_status"] == 429
    assert event.raw["provider"] is None
    assert event.raw["cost_usd"] is None
    assert event.raw["requested_routing"] == _pin("moonshotai")
    # The status says a request failed; only the message says what was wrong with it.
    assert event.raw["summary"] == "rate limited"
    assert proxy.exact_cost_usd is None


def test_error_message_wins_over_routing_metadata(monkeypatch):
    """A rejected request can still carry routing metadata; the message is what matters."""
    body = json.dumps(
        {
            "error": {"message": "provider does not support this request"},
            "openrouter_metadata": {"requested": "deepseek/deepseek-v4-flash", "summary": "av=1"},
        }
    ).encode()
    _fake_upstream(monkeypatch, response=_FakeResponse(status=400, body=body))

    with OpenRouterProxy(
        "real-key", expected_model="deepseek/deepseek-v4-flash", routing=_pin("novita")
    ) as proxy:
        _proxy_post(proxy, body=json.dumps({"model": "deepseek/deepseek-v4-flash"}))
        assert proxy.wait_until_idle()

    event = proxy.drain_events()[0]
    assert event.raw["http_status"] == 400
    assert event.raw["summary"] == "provider does not support this request"


@pytest.mark.parametrize(
    "headers",
    [{"Authorization": "Bearer wrong-token"}, {"Authorization": ""}],
    ids=["wrong-token", "no-token"],
)
def test_only_the_run_token_opens_the_proxy(monkeypatch, headers):
    """The token is what keeps the account key out of the CLI, so it must be checked."""
    calls = []
    _fake_upstream(monkeypatch, calls=calls)

    with OpenRouterProxy("real-key", expected_model="moonshotai/kimi-k3") as proxy:
        status, payload = _proxy_post(
            proxy,
            body=json.dumps({"model": "moonshotai/kimi-k3", "input": "hi"}),
            headers=headers,
        )

    assert status == 401
    assert json.loads(payload)["error"]["message"] == "invalid local proxy token"
    assert calls == []  # the real key never left the process
    assert proxy.drain_events() == []


def test_the_x_api_key_header_also_carries_the_run_token(monkeypatch):
    """Claude Code authenticates with ``X-Api-Key`` rather than a bearer header."""
    _fake_upstream(monkeypatch)

    with OpenRouterProxy("real-key", expected_model="moonshotai/kimi-k3") as proxy:
        status, _ = _proxy_post(
            proxy,
            body=json.dumps({"model": "moonshotai/kimi-k3", "input": "hi"}),
            headers={"Authorization": "", "X-Api-Key": proxy.client_token},
        )
        assert proxy.wait_until_idle()

    assert status == 200


def test_a_request_for_another_model_is_refused(monkeypatch):
    """The proxy serves one model for one run; a helper-model call must not slip through."""
    calls = []
    _fake_upstream(monkeypatch, calls=calls)

    with OpenRouterProxy("real-key", expected_model="moonshotai/kimi-k3") as proxy:
        status, payload = _proxy_post(
            proxy, body=json.dumps({"model": "anthropic/claude-haiku-4-5", "input": "hi"})
        )

    assert status == 400
    assert json.loads(payload)["error"]["message"] == "model does not match this proxy run"
    assert calls == []


def test_the_next_request_is_refused_once_the_budget_is_spent(monkeypatch):
    """Two charges cross the cap, and the third request never reaches OpenRouter."""
    charge = json.dumps(
        {"model": "moonshotai/kimi-k3", "provider": "Moonshot AI", "usage": {"cost": 0.004}}
    ).encode()
    calls = []

    class TwoCharges:
        """A fresh body per call: ``_FakeResponse`` can only be read once."""

        def __init__(self, host, timeout):
            assert host == "openrouter.ai"

        def request(self, method, path, body=None, headers=None):
            calls.append((method, path, body, headers))

        def getresponse(self):
            return _FakeResponse(body=charge)

        def close(self):
            pass

    monkeypatch.setattr(proxy_module.http.client, "HTTPSConnection", TwoCharges)
    body = json.dumps({"model": "moonshotai/kimi-k3", "input": "hi"})

    with OpenRouterProxy("real-key", 0.006, expected_model="moonshotai/kimi-k3") as proxy:
        first, _ = _proxy_post(proxy, body=body)
        assert proxy.wait_until_idle()
        assert not proxy.budget_blocked  # $0.004 of $0.006 spent
        second, _ = _proxy_post(proxy, body=body)
        assert proxy.wait_until_idle()
        third, payload = _proxy_post(proxy, body=body)

    assert (first, second, third) == (200, 200, 402)
    assert json.loads(payload)["error"]["type"] == "budget_exceeded"
    assert proxy.budget_blocked is True
    assert len(calls) == 2  # the refused request never went upstream
    assert proxy.exact_cost_usd == pytest.approx(0.008)  # both charges, summed
    assert len(proxy.drain_events()) == 2


def test_the_response_reaches_the_client_unchanged(monkeypatch):
    """The proxy reads the body for metadata; the CLI must still get every byte of it."""
    chunks = [
        b'data: {"type":"message_start","message":{"model":"moonshotai/kimi-k3"}}\n\n',
        b'data: {"type":"content_block_delta","delta":{"text":"hello"}}\n\n',
        b'data: {"type":"message_delta","usage":{"cost":0.0015}}\n\n',
        b'data: {"type":"message_stop","openrouter_metadata":'
        b'{"requested":"moonshotai/kimi-k3","endpoints":{"available":[]}}}\n\n',
    ]

    class Streamed:
        """Hands the body over in pieces, the way a real SSE response arrives."""

        def __init__(self, host, timeout):
            self._remaining = list(chunks)

        def request(self, method, path, body=None, headers=None):
            pass

        def getresponse(self):
            return self

        status = 200
        reason = "OK"

        def getheader(self, name, default=None):
            return "text/event-stream" if name == "Content-Type" else default

        def getheaders(self):
            return [("Content-Type", "text/event-stream")]

        def read(self, _size=-1):
            return self._remaining.pop(0) if self._remaining else b""

        def close(self):
            pass

    monkeypatch.setattr(proxy_module.http.client, "HTTPSConnection", Streamed)

    with OpenRouterProxy("real-key", expected_model="moonshotai/kimi-k3") as proxy:
        status, payload = _proxy_post(
            proxy, body=json.dumps({"model": "moonshotai/kimi-k3", "input": "hi"})
        )
        assert proxy.wait_until_idle()

    assert status == 200
    assert payload == b"".join(chunks)
    event = proxy.drain_events()[0]
    assert event.raw["cost_usd"] == 0.0015

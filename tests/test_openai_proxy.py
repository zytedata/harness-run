"""The OpenAI proxy: it holds the account key, the CLI holds a per-run token.

The server tests bind a real socket on 127.0.0.1 and replace the upstream (``HTTPSConnection``
for HTTP calls, ``_connect_upstream`` for the WebSocket tunnel), so nothing leaves the machine.
"""

from __future__ import annotations

import http.client
import json
import socket
import threading
from urllib.parse import urlsplit

import pytest

from harness_run.harness import _openai_proxy as proxy_module
from harness_run.harness._openai_proxy import OpenAIProxy


class _FakeResponse:
    def __init__(
        self, status=200, chunks=(b"{}",), headers=(("Content-Type", "application/json"),)
    ):
        self.status = status
        self.reason = "OK" if status == 200 else "ERR"
        self._chunks = list(chunks)
        self._headers = list(headers)

    def getheaders(self):
        return self._headers

    def read1(self, _size=-1):
        return self._chunks.pop(0) if self._chunks else b""


def _fake_upstream(monkeypatch, response=None, raises=None, calls=None):
    class Upstream:
        def __init__(self, host, timeout):
            assert host == "api.openai.com"

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


def _call(proxy, method="POST", path="/v1/responses", body=b"{}", headers=None, token=None):
    target = urlsplit(proxy.base_url)
    conn = http.client.HTTPConnection(target.hostname, target.port, timeout=5)
    sent = {"Authorization": f"Bearer {token or proxy.client_token}"}
    sent.update(headers or {})
    conn.request(method, path, body=body if method == "POST" else None, headers=sent)
    response = conn.getresponse()
    payload = response.read()
    conn.close()
    return response.status, payload, dict(response.getheaders())


def test_the_real_key_replaces_the_run_token(monkeypatch):
    calls = []
    _fake_upstream(monkeypatch, calls=calls)
    with OpenAIProxy("sk-real") as proxy:
        assert proxy.base_url.startswith("http://127.0.0.1:") and proxy.base_url.endswith("/v1")
        status, _, _ = _call(
            proxy,
            body=b'{"model":"m"}',
            headers={"OpenAI-Beta": "x", "Content-Type": "application/json"},
        )
    assert status == 200
    method, path, body, headers = calls[0]
    assert (method, path, body) == ("POST", "/v1/responses", b'{"model":"m"}')
    assert headers["Authorization"] == "Bearer sk-real"
    assert headers["OpenAI-Beta"] == "x"
    assert proxy.client_token not in json.dumps(headers)


@pytest.mark.parametrize(
    "headers",
    [{"Authorization": "Bearer wrong"}, {"Authorization": ""}, {"Authorization": "Bearer sk-real"}],
)
def test_only_the_run_token_opens_the_proxy(monkeypatch, headers):
    calls = []
    _fake_upstream(monkeypatch, calls=calls)
    with OpenAIProxy("sk-real") as proxy:
        status, payload, _ = _call(proxy, headers=headers)
    assert status == 401 and calls == []
    assert b"sk-real" not in payload


@pytest.mark.parametrize(
    "method, path",
    [
        ("GET", "/v1/files"),
        ("POST", "/v1/chat/completions"),
        ("GET", "/v1/responses/resp_123"),
        ("POST", "/v1/models"),
        ("GET", "/v1/organization/admin_api_keys"),
    ],
)
def test_other_endpoints_are_refused_and_remembered(monkeypatch, method, path):
    calls = []
    _fake_upstream(monkeypatch, calls=calls)
    with OpenAIProxy("sk-real") as proxy:
        status, _, _ = _call(proxy, method=method, path=path + "?limit=1")
        assert status == 403 and calls == []
        assert proxy.drain_refused() == [f"{method} {path}"]
        assert proxy.drain_refused() == []


@pytest.mark.parametrize(
    "method, path", [("GET", "/v1/models?client_version=1"), ("POST", "/v1/responses/compact")]
)
def test_the_endpoints_codex_uses_are_served(monkeypatch, method, path):
    calls = []
    _fake_upstream(monkeypatch, calls=calls)
    with OpenAIProxy("sk-real") as proxy:
        status, _, _ = _call(proxy, method=method, path=path)
        assert status == 200 and calls[0][:2] == (method, path)
        assert proxy.drain_refused() == []


def test_the_response_reaches_the_client_unchanged(monkeypatch):
    events = [b"event: a\ndata: 1\n\n", b"event: b\ndata: 2\n\n"]
    _fake_upstream(
        monkeypatch,
        response=_FakeResponse(
            status=429,
            chunks=events,
            headers=[
                ("Content-Type", "text/event-stream"),
                ("X-Request-Id", "req-1"),
                ("Transfer-Encoding", "chunked"),
            ],
        ),
    )
    with OpenAIProxy("sk-real") as proxy:
        status, payload, headers = _call(proxy)
    assert status == 429 and payload == b"".join(events)
    assert headers["X-Request-Id"] == "req-1" and "Transfer-Encoding" not in headers


def test_upstream_failure_returns_502_json(monkeypatch):
    _fake_upstream(monkeypatch, raises=OSError("network down"))
    with OpenAIProxy("sk-real") as proxy:
        status, payload, _ = _call(proxy)
    assert status == 502
    assert json.loads(payload)["error"]["type"] == "proxy_upstream_error"


def test_chunked_request_body_is_rejected_with_411(monkeypatch):
    calls = []
    _fake_upstream(monkeypatch, calls=calls)
    with OpenAIProxy("sk-real") as proxy:
        target = urlsplit(proxy.base_url)
        conn = http.client.HTTPConnection(target.hostname, target.port, timeout=5)
        conn.request(
            "POST",
            "/v1/responses",
            body=iter([b"{}"]),
            headers={
                "Authorization": f"Bearer {proxy.client_token}",
                "Transfer-Encoding": "chunked",
            },
        )
        status = conn.getresponse().status
        conn.close()
    assert status == 411 and calls == []


def _handshake(proxy, token=None):
    """Open a client socket to the proxy and send a WebSocket upgrade for /v1/responses."""
    target = urlsplit(proxy.base_url)
    client = socket.create_connection((target.hostname, target.port), timeout=5)
    client.sendall(
        (
            "GET /v1/responses HTTP/1.1\r\n"
            f"Host: {target.hostname}:{target.port}\r\n"
            f"Authorization: Bearer {token or proxy.client_token}\r\n"
            "Connection: Upgrade\r\nUpgrade: websocket\r\n"
            "Sec-WebSocket-Key: abc\r\nSec-WebSocket-Version: 13\r\n\r\n"
        ).encode()
    )
    return client


def test_a_websocket_is_tunnelled_with_the_real_key(monkeypatch):
    """The handshake goes upstream with the key swapped; bytes then pass both ways untouched."""
    listener = socket.create_server(("127.0.0.1", 0))
    seen = {}

    def upstream_server():
        conn, _ = listener.accept()
        with conn:
            head = b""
            while b"\r\n\r\n" not in head:
                head += conn.recv(4096)
            seen["handshake"] = head.decode()
            conn.sendall(b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n\r\n")
            seen["frame"] = conn.recv(4096)
            conn.sendall(b"server-frame")

    thread = threading.Thread(target=upstream_server, daemon=True)
    thread.start()
    monkeypatch.setattr(
        proxy_module,
        "_connect_upstream",
        lambda: socket.create_connection(listener.getsockname(), timeout=5),
    )
    with OpenAIProxy("sk-real") as proxy:
        client = _handshake(proxy)
        reply = b""
        while b"\r\n\r\n" not in reply:
            reply += client.recv(4096)
        assert reply.startswith(b"HTTP/1.1 101")
        client.sendall(b"client-frame")
        assert client.recv(4096) == b"server-frame"
        thread.join(timeout=5)
        assert client.recv(4096) == b""  # upstream closed, so the tunnel closes the client
        client.close()
    listener.close()
    handshake = seen["handshake"]
    assert handshake.startswith("GET /v1/responses HTTP/1.1\r\nHost: api.openai.com\r\n")
    assert "Authorization: Bearer sk-real\r\n" in handshake
    assert proxy.client_token not in handshake
    assert "Sec-WebSocket-Key: abc\r\n" in handshake and "Upgrade: websocket\r\n" in handshake
    assert handshake.count("Authorization:") == 1 and handshake.count("Host:") == 1
    assert seen["frame"] == b"client-frame"


def test_a_websocket_that_cannot_reach_upstream_gets_502(monkeypatch):
    def refuse():
        raise OSError("no route")

    monkeypatch.setattr(proxy_module, "_connect_upstream", refuse)
    with OpenAIProxy("sk-real") as proxy:
        client = _handshake(proxy)
        reply = client.recv(4096)
        client.close()
    assert reply.startswith(b"HTTP/1.1 502")


def test_a_websocket_needs_the_run_token(monkeypatch):
    monkeypatch.setattr(proxy_module, "_connect_upstream", lambda: pytest.fail("reached upstream"))
    with OpenAIProxy("sk-real") as proxy:
        client = _handshake(proxy, token="wrong")
        reply = client.recv(4096)
        client.close()
    assert reply.startswith(b"HTTP/1.1 401")


def test_closing_the_proxy_ends_an_open_tunnel(monkeypatch):
    listener = socket.create_server(("127.0.0.1", 0))
    accepted = []

    def upstream_server():
        conn, _ = listener.accept()
        accepted.append(conn)
        conn.recv(4096)
        conn.sendall(b"HTTP/1.1 101 Switching Protocols\r\n\r\n")
        conn.recv(4096)  # returns b"" when the proxy shuts the tunnel

    thread = threading.Thread(target=upstream_server, daemon=True)
    thread.start()
    monkeypatch.setattr(
        proxy_module,
        "_connect_upstream",
        lambda: socket.create_connection(listener.getsockname(), timeout=5),
    )
    proxy = OpenAIProxy("sk-real").start()
    client = _handshake(proxy)
    assert client.recv(4096).startswith(b"HTTP/1.1 101")
    proxy.close()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert client.recv(4096) == b""
    client.close()
    for conn in accepted:
        conn.close()
    listener.close()

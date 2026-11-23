"""Local HTTP proxy for the OpenAI API. It keeps the account key away from the agent.

Codex stores the key it is logged in with in ``CODEX_HOME/auth.json``, and the agent's
shell can read that file. So the CLI is logged in with a random per-run token instead and
pointed at this proxy (``openai_base_url``). The proxy swaps the token for the real key,
which stays in the calling process, and passes everything else through unchanged.

For each call it:

* checks the run token, and answers 401 without it;
* serves only the endpoints Codex uses for a turn (:data:`_ALLOWED_PATHS`) and answers 403
  for anything else, so the token is no general-purpose handle on the account. A refused
  call is kept for the harness to report (:meth:`OpenAIProxy.drain_refused`);
* tunnels a WebSocket upgrade. Codex opens ``/v1/responses`` as a WebSocket first and only
  falls back to HTTP when that fails. The key travels in the handshake headers alone, so
  after the handshake is forwarded the proxy copies bytes both ways without reading them;
* streams any other response back as it arrives.

The token dies with the proxy at the end of the run. Nothing is logged or kept.
"""

from __future__ import annotations

import http.client
import json
import secrets
import socket
import ssl
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

OPENAI_API_HOST = "api.openai.com"

# What Codex calls with an API key: the Responses endpoint (WebSocket or POST), its
# compaction endpoint and the model list.
_ALLOWED_PATHS = {
    ("GET", "/v1/models"),
    ("GET", "/v1/responses"),  # the WebSocket upgrade
    ("POST", "/v1/responses"),
    ("POST", "/v1/responses/compact"),
}
# Dropped from a forwarded HTTP request or response: the proxy sets its own framing and
# its own credentials.
_HOP_HEADERS = {
    "authorization",
    "connection",
    "content-length",
    "host",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
_CHUNK = 64 * 1024


def _connect_upstream() -> socket.socket:
    """A TLS connection to the OpenAI API, for a WebSocket tunnel."""
    raw = socket.create_connection((OPENAI_API_HOST, 443), timeout=30)
    try:
        return ssl.create_default_context().wrap_socket(raw, server_hostname=OPENAI_API_HOST)
    except BaseException:
        raw.close()
        raise


def _shutdown(sock: socket.socket) -> None:
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass


class OpenAIProxy:
    """Threaded localhost proxy with a random child-facing token."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self.client_token = secrets.token_urlsafe(32)
        self._lock = threading.Lock()
        self._refused: list[str] = []
        self._tunnels: set[socket.socket] = set()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        """The value for Codex's ``openai_base_url``."""
        if self._server is None:
            raise RuntimeError("OpenAI proxy is not started")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def drain_refused(self) -> list[str]:
        """The ``METHOD path`` of each call refused since the last drain."""
        with self._lock:
            refused, self._refused = self._refused, []
        return refused

    def start(self) -> OpenAIProxy:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, _format: str, *args: Any) -> None:
                return

            def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
                self._serve()

            def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
                self._serve()

            def _send_json(self, status: int, error_type: str, message: str) -> None:
                body = json.dumps({"error": {"type": error_type, "message": message}}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)
                self.close_connection = True

            def _serve(self) -> None:
                auth = self.headers.get("Authorization", "")
                if not secrets.compare_digest(
                    auth.encode(), f"Bearer {owner.client_token}".encode()
                ):
                    self._send_json(401, "invalid_proxy_token", "invalid local proxy token")
                    return
                path = self.path.split("?", 1)[0]
                if (self.command, path) not in _ALLOWED_PATHS:
                    with owner._lock:
                        owner._refused.append(f"{self.command} {path}")
                    self._send_json(
                        403,
                        "proxy_path_refused",
                        "the local OpenAI proxy only serves the endpoints Codex uses for a turn",
                    )
                    return
                if "websocket" in (self.headers.get("Upgrade") or "").lower():
                    self._tunnel()
                else:
                    self._forward()

            def _tunnel(self) -> None:
                """Forward the WebSocket handshake with the real key, then copy bytes."""
                lines = [
                    f"{self.command} {self.path} HTTP/1.1",
                    f"Host: {OPENAI_API_HOST}",
                    f"Authorization: Bearer {owner._api_key}",
                    *(
                        f"{key}: {value}"
                        for key, value in self.headers.items()
                        if key.lower() not in {"host", "authorization", "proxy-authorization"}
                    ),
                ]
                try:
                    upstream = _connect_upstream()
                    upstream.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1"))
                    upstream.settimeout(None)
                except OSError as exc:
                    # Codex falls back to plain HTTP when the WebSocket does not open.
                    self._send_json(502, "proxy_upstream_error", f"{type(exc).__name__}: {exc}")
                    return
                client = self.connection
                self.close_connection = True
                with owner._lock:
                    owner._tunnels.add(upstream)

                def downstream() -> None:
                    try:
                        while chunk := upstream.recv(_CHUNK):
                            client.sendall(chunk)
                    except OSError:
                        pass
                    _shutdown(client)

                pump = threading.Thread(target=downstream, daemon=True)
                pump.start()
                try:
                    # ``rfile`` and not the socket: it may already hold bytes the client
                    # sent right behind the handshake.
                    while chunk := self.rfile.read1(_CHUNK):
                        upstream.sendall(chunk)
                except OSError:
                    pass
                finally:
                    _shutdown(upstream)
                    pump.join(timeout=5)
                    upstream.close()
                    with owner._lock:
                        owner._tunnels.discard(upstream)

            def _forward(self) -> None:
                if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
                    # The read below goes by Content-Length, so a chunked body would be
                    # forwarded empty. Codex does not send one. Say so.
                    self._send_json(
                        411, "length_required", "the proxy does not accept chunked request bodies"
                    )
                    return
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else None
                headers = {
                    key: value
                    for key, value in self.headers.items()
                    if key.lower() not in _HOP_HEADERS
                }
                headers["Authorization"] = f"Bearer {owner._api_key}"
                upstream = http.client.HTTPSConnection(OPENAI_API_HOST, timeout=600)
                try:
                    try:
                        upstream.request(self.command, self.path, body=body, headers=headers)
                        response = upstream.getresponse()
                    except (OSError, http.client.HTTPException) as exc:
                        # Nothing has been sent yet, so the CLI can still get a readable
                        # error instead of a closed connection.
                        self._send_json(502, "proxy_upstream_error", f"{type(exc).__name__}: {exc}")
                        return
                    self.send_response(response.status, response.reason)
                    for key, value in response.getheaders():
                        if key.lower() not in _HOP_HEADERS:
                            self.send_header(key, value)
                    self.send_header("Connection", "close")
                    self.end_headers()
                    try:
                        # ``read1`` returns what has arrived, so a server-sent event is
                        # passed on when it comes and not when 64 KB have piled up.
                        while chunk := response.read1(_CHUNK):
                            self.wfile.write(chunk)
                            self.wfile.flush()
                    except OSError:
                        pass  # the CLI hung up, or the upstream stream broke mid-body
                finally:
                    upstream.close()
                    self.close_connection = True

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )
        self._thread.start()
        return self

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        with self._lock:
            tunnels = list(self._tunnels)
        for upstream in tunnels:
            _shutdown(upstream)
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> OpenAIProxy:
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.close()

"""Local OpenRouter relay used when a CLI drops response metadata.

The relay binds to localhost, forwards the CLI's HTTP stream unchanged, and records only
OpenRouter's routing metadata and billed cost. Response bytes are held in memory while the
stream is parsed and are discarded after the request.
"""

from __future__ import annotations

import http.client
import json
import secrets
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from ..events import AgentEvent

_CAPTURE_LIMIT = 8 * 1024 * 1024
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
    "x-api-key",
    "x-openrouter-metadata",
}


@dataclass(frozen=True)
class OpenRouterRequest:
    """Metadata retained for one OpenRouter HTTP response."""

    requested_model: str | None
    provider: str | None
    provider_model: str | None
    region: str | None
    attempt: int | None
    summary: str | None
    cost_usd: float | None
    status: int

    def event(self) -> AgentEvent:
        return AgentEvent(
            kind="status",
            summary=(
                "OpenRouter request: "
                f"provider={self.provider or 'unreported'} "
                f"model={self.provider_model or self.requested_model or 'unreported'}"
            ),
            raw={
                "event": "openrouter_request",
                "requested_model": self.requested_model,
                "provider": self.provider,
                "provider_model": self.provider_model,
                "region": self.region,
                "attempt": self.attempt,
                "summary": self.summary,
                "cost_usd": self.cost_usd,
                "http_status": self.status,
            },
        )


def _json_objects(body: bytes, content_type: str) -> list[dict[str, Any]]:
    """Decode a JSON response or the JSON payloads in an SSE response."""
    text = body.decode("utf-8", "replace")
    if "text/event-stream" not in content_type:
        try:
            value = json.loads(text)
            return [value] if isinstance(value, dict) else []
        except (TypeError, ValueError):
            return []
    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            value = json.loads(payload)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            out.append(value)
    return out


def _capture_request(
    body: bytes,
    content_type: str,
    requested_model: str | None,
    status: int,
) -> OpenRouterRequest | None:
    """Extract cost and selected endpoint from one buffered response."""
    metadata: dict[str, Any] = {}
    usage: dict[str, Any] = {}
    provider: str | None = None
    provider_model: str | None = None
    for value in _json_objects(body, content_type):
        candidates = [value]
        response = value.get("response")
        if isinstance(response, dict):
            candidates.append(response)
        for candidate in candidates:
            if isinstance(candidate.get("openrouter_metadata"), dict):
                metadata = candidate["openrouter_metadata"]
            if (
                isinstance(candidate.get("usage"), dict)
                and candidate["usage"].get("cost") is not None
            ):
                usage = candidate["usage"]
            if candidate.get("provider"):
                provider = str(candidate["provider"])
            if candidate.get("model"):
                provider_model = str(candidate["model"])

    selected = next(
        (
            endpoint
            for endpoint in (metadata.get("endpoints") or {}).get("available", [])
            if endpoint.get("selected")
        ),
        {},
    )
    provider = selected.get("provider") or provider
    provider_model = selected.get("model") or provider_model
    requested_model = metadata.get("requested") or requested_model
    cost = usage.get("cost")
    if not metadata and cost is None and provider is None:
        return None
    return OpenRouterRequest(
        requested_model=requested_model,
        provider=str(provider) if provider is not None else None,
        provider_model=str(provider_model) if provider_model is not None else None,
        region=metadata.get("region"),
        attempt=metadata.get("attempt"),
        summary=metadata.get("summary"),
        cost_usd=float(cost) if cost is not None else None,
        status=status,
    )


def _append_capture(buffer: bytearray, chunk: bytes, limit: int = _CAPTURE_LIMIT) -> None:
    """Keep the newest response bytes, where streaming APIs put final usage metadata."""
    buffer.extend(chunk)
    overflow = len(buffer) - limit
    if overflow > 0:
        del buffer[:overflow]


def _model_matches(expected: str | None, requested: Any) -> bool:
    """Whether a request may pass through a model-scoped relay."""
    return expected is None or (isinstance(requested, str) and requested == expected)


class OpenRouterProxy:
    """Threaded localhost relay with a random child-facing token."""

    def __init__(
        self,
        api_key: str,
        max_budget_usd: float | None = None,
        expected_model: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._max_budget_usd = max_budget_usd
        self._expected_model = expected_model
        self.client_token = secrets.token_urlsafe(32)
        self._requests: list[OpenRouterRequest] = []
        self._drained = 0
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)
        self._active_requests = 0
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.budget_blocked = False

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("OpenRouter proxy has not started")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    @property
    def exact_cost_usd(self) -> float | None:
        with self._lock:
            costs = [request.cost_usd for request in self._requests if request.cost_usd is not None]
        return sum(costs) if costs else None

    def drain_events(self) -> list[AgentEvent]:
        with self._lock:
            fresh = self._requests[self._drained :]
            self._drained = len(self._requests)
        return [request.event() for request in fresh]

    def _record(self, request: OpenRouterRequest) -> None:
        with self._lock:
            self._requests.append(request)

    def _begin_request(self) -> None:
        with self._idle:
            self._active_requests += 1

    def _end_request(self) -> None:
        with self._idle:
            self._active_requests -= 1
            self._idle.notify_all()

    def wait_until_idle(self, timeout: float = 5.0) -> bool:
        """Wait for request metadata to be recorded after the CLI sees the final bytes."""
        deadline = time.monotonic() + timeout
        with self._idle:
            while self._active_requests:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._idle.wait(remaining)
            return True

    def _over_budget(self) -> bool:
        cost = self.exact_cost_usd
        return (
            cost is not None and self._max_budget_usd is not None and cost >= self._max_budget_usd
        )

    def start(self) -> OpenRouterProxy:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, _format: str, *args: Any) -> None:
                return

            def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
                self._forward()

            def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
                self._forward()

            def _send_json(self, status: int, value: dict[str, Any]) -> None:
                body = json.dumps(value).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)
                self.close_connection = True

            def _forward(self) -> None:
                auth = self.headers.get("Authorization", "")
                x_api_key = self.headers.get("X-Api-Key", "")
                if owner.client_token not in {auth.removeprefix("Bearer "), x_api_key}:
                    self._send_json(401, {"error": {"message": "invalid local proxy token"}})
                    return
                if owner._over_budget():
                    owner.budget_blocked = True
                    self._send_json(
                        402,
                        {"error": {"type": "budget_exceeded", "message": "max_budget_usd reached"}},
                    )
                    return

                length = int(self.headers.get("Content-Length") or 0)
                request_body = self.rfile.read(length) if length else None
                requested_model = None
                if request_body:
                    try:
                        request_json = json.loads(request_body)
                        requested_model = request_json.get("model")
                    except (TypeError, ValueError):
                        pass
                if self.command == "POST" and not _model_matches(
                    owner._expected_model, requested_model
                ):
                    self._send_json(
                        400,
                        {"error": {"message": "model does not match this relay run"}},
                    )
                    return
                headers = {
                    key: value
                    for key, value in self.headers.items()
                    if key.lower() not in _HOP_HEADERS
                }
                headers.update(
                    {
                        "Authorization": f"Bearer {owner._api_key}",
                        "X-OpenRouter-Metadata": "enabled",
                        "Accept-Encoding": "identity",
                    }
                )
                upstream = http.client.HTTPSConnection("openrouter.ai", timeout=600)
                owner._begin_request()
                try:
                    upstream.request(self.command, self.path, body=request_body, headers=headers)
                    response = upstream.getresponse()
                    content_type = response.getheader("Content-Type", "")
                    self.send_response(response.status, response.reason)
                    for key, value in response.getheaders():
                        if key.lower() not in _HOP_HEADERS and key.lower() != "content-encoding":
                            self.send_header(key, value)
                    self.send_header("Connection", "close")
                    self.end_headers()
                    captured = bytearray()
                    while chunk := response.read(64 * 1024):
                        _append_capture(captured, chunk)
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    found = _capture_request(
                        bytes(captured), content_type, requested_model, response.status
                    )
                    if found is not None:
                        owner._record(found)
                finally:
                    upstream.close()
                    self.close_connection = True
                    owner._end_request()

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> OpenRouterProxy:
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.close()

"""OpenRouter response metadata parsing. No server or external request is used."""

from __future__ import annotations

import json

from remote_agent_toolkit.harness._openrouter_proxy import (
    OpenRouterProxy,
    _append_capture,
    _capture_request,
    _model_matches,
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

    found = _capture_request(body, "text/event-stream", None, 200)

    assert found is not None
    assert found.requested_model == "deepseek/deepseek-v4-flash"
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

    found = _capture_request(body, "application/json", "asked", 200)

    assert found is not None
    assert found.provider == "Z.AI"
    assert found.cost_usd == 0.0042
    assert found.requested_model == "z-ai/glm-5.3"


def test_ignores_responses_without_openrouter_fields():
    assert _capture_request(b'{"ok":true}', "application/json", "m", 200) is None


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

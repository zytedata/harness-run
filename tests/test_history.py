"""Session history: mirror round-trip, turn-scoped reads, enumeration, re-attach results.

Everything runs offline against LocalBlobStore; the live GCS path is validated separately.
"""

from __future__ import annotations

import json

from sandbox_fakes import FakeSandboxProvider, make_engine

from remote_agent_toolkit.events import AgentEvent, RunStatus, StopReason
from remote_agent_toolkit.ports.blobstore import LocalBlobStore
from remote_agent_toolkit.runtime.gemini import backend, history


def _result_event(text="done"):
    return AgentEvent(kind="result", summary=text, cost_usd=0.1,
                      raw={"subtype": "success", "is_error": False, "num_turns": 2})


def test_mirror_round_trip():
    ev = _result_event()
    line = history.mirror_line(ev)
    back = history.event_from_mirror(json.loads(json.dumps(line)))
    assert back == ev


def test_mirror_line_is_json_safe_for_odd_raw_payloads():
    ev = AgentEvent(kind="status", summary="s", raw={"when": object()})
    line = history.mirror_line(ev)
    json.dumps(line)  # no raise
    assert isinstance(line["raw"]["when"], str)


def test_read_history_keeps_turn_order_and_write_turn_mirror_is_best_effort(tmp_path):
    store = LocalBlobStore(str(tmp_path))
    history.write_turn_mirror("gs://bkt/events", "sid", [history.mirror_line(_result_event("first"))],
                              now_ms=1000, store=store)
    history.write_turn_mirror("gs://bkt/events", "sid", [history.mirror_line(_result_event("second"))],
                              now_ms=2000, turn_id="t" * 32, store=store)
    history.write_turn_mirror("gs://bkt/events", "sid", [], now_ms=3000, store=store)  # nothing written
    assert [e.summary for e in history.read_history("gs://bkt", "sid", store=store)] == ["first", "second"]
    assert history.read_history("gs://bkt", "other", store=store) == []
    assert history.read_history(None, "sid") == []

    class Broken:
        def put_bytes(self, key, data):
            raise OSError("bucket gone")

    history.write_turn_mirror("gs://bkt/events", "sid", [{"kind": "status"}], now_ms=1, store=Broken())


def test_list_sessions_reads_the_mirror_newest_first(tmp_path):
    store = LocalBlobStore(str(tmp_path))
    store.put_bytes("events/older/000000000001000.jsonl", b"{}")
    store.put_bytes("events/newer/000000000002000.jsonl", b"{}")
    store.put_bytes("events/newer/000000000003000-x.jsonl", b"{}")
    got = history.list_sessions(output_bucket="gs://bkt", store=store)
    assert [s["session_id"] for s in got] == ["newer", "older"]
    assert got[0] == {"session_id": "newer", "sources": ["events"],
                      "last_file": "events/newer/000000000003000-x.jsonl"}
    assert history.list_sessions(output_bucket=None) == []


def test_engine_list_sessions_and_history_use_the_mirror(tmp_path, monkeypatch):
    store = LocalBlobStore(str(tmp_path))
    store.put_bytes("events/s1/000000000001000.jsonl", json.dumps(history.mirror_line(_result_event("x"))).encode())
    monkeypatch.setattr(history, "GcsBlobStore", lambda bucket, *a, **kw: store)
    engine = make_engine(FakeSandboxProvider())
    assert [s["session_id"] for s in engine.list_sessions()] == ["s1"]
    assert [e.summary for e in engine.get_session("s1").history()] == ["x"]


def test_reattached_session_reconstructs_last_result(monkeypatch):
    engine = make_engine(FakeSandboxProvider())
    session = backend.GeminiSession(engine, "old-sid")
    monkeypatch.setattr(backend.GeminiSession, "history",
                        lambda self: [AgentEvent(kind="message", summary="hi"), _result_event("final answer")])
    result = session.last_result
    assert result is not None and result.text == "final answer" and result.num_turns == 2
    assert session.stop_reason == StopReason.END_TURN and session.status == RunStatus.IDLE


def test_reattached_session_without_history_has_no_result(monkeypatch):
    engine = make_engine(FakeSandboxProvider())
    session = backend.GeminiSession(engine, "old-sid")
    monkeypatch.setattr(backend.GeminiSession, "history", lambda self: [])
    assert session.last_result is None and session.status == RunStatus.PENDING

"""The worker's loopback metadata server: the model token Claude Code fetches and refreshes."""
import datetime as dt
import json
import urllib.error
import urllib.request

import pytest

from agent_run.runtime.gemini.worker import TurnRecord, Worker
from agent_run.spec import AgentSpec


def _worker(tmp_path) -> Worker:
    return Worker(workspace_root=str(tmp_path), baked=AgentSpec(name="w", model="m"), baked_skills=None,
                  metadata_port=0)


def _get(host: str, path: str, *, flavor: bool = True):
    req = urllib.request.Request(f"http://{host}{path}", headers={"Metadata-Flavor": "Google"} if flavor else {})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, resp.headers.get("Metadata-Flavor"), resp.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get("Metadata-Flavor"), exc.read().decode()


def _running_turn(worker: Worker, turn_id: str, token: dict | None) -> TurnRecord:
    record = TurnRecord(turn_id, worker._lock)
    record.model_token = token
    worker._turns[turn_id] = record
    return record


def test_metadata_server_binds_lazily_on_loopback_and_serves_the_running_turns_token(tmp_path):
    worker = _worker(tmp_path)
    assert worker.metadata_host is None and worker.health()["metadata_host"] is None
    expires = (dt.datetime.now(dt.UTC).replace(tzinfo=None) + dt.timedelta(minutes=30)).isoformat()
    _running_turn(worker, "t1", {"access_token": "TOK-1", "expires_at": expires})
    env = worker._metadata_env()
    host = env["GCE_METADATA_HOST"]
    assert host.startswith("127.0.0.1:") and env["METADATA_SERVER_DETECTION"] == "assume-present"
    assert worker.metadata_host == host == worker.health()["metadata_host"]

    status, flavor, body = _get(host, "/computeMetadata/v1/instance/service-accounts/default/token?scopes=x")
    assert status == 200 and flavor == "Google"
    answer = json.loads(body)
    assert answer["access_token"] == "TOK-1" and answer["token_type"] == "Bearer"
    assert 29 * 60 < answer["expires_in"] <= 30 * 60  # from the client's expires_at
    assert _get(host, "/computeMetadata/v1/universe/universe-domain") == (200, "Google", "googleapis.com")
    assert _get(host, "/computeMetadata/v1/instance/service-accounts/default/email")[0] == 200
    assert _get(host, "/computeMetadata/v1/instance")[0] == 200  # the library's presence probe
    assert _get(host, "/computeMetadata/v1/instance/hostname")[0] == 404


def test_metadata_server_demands_the_google_flavor_header(tmp_path):
    worker = _worker(tmp_path)
    _running_turn(worker, "t1", {"access_token": "TOK-1", "expires_at": None})
    host = worker._ensure_metadata_server()
    status, flavor, body = _get(host, "/computeMetadata/v1/instance/service-accounts/default/token", flavor=False)
    assert status == 403 and flavor == "Google" and "Metadata-Flavor" in body


def test_token_endpoint_replaces_the_served_token_and_a_finished_turn_serves_none(tmp_path):
    worker = _worker(tmp_path)
    record = _running_turn(worker, "t1", {"access_token": "TOK-1", "expires_at": None})
    host = worker._ensure_metadata_server()
    assert json.loads(_get(host, "/computeMetadata/v1/instance/service-accounts/default/token")[2])["expires_in"] == 3600

    assert worker.handle("/token", {"turn_id": "t1", "model_token": {"access_token": "TOK-2", "expires_at": "2099-01-01T00:00:00"}}) == {"ok": True}
    answer = json.loads(_get(host, "/computeMetadata/v1/instance/service-accounts/default/token")[2])
    assert answer["access_token"] == "TOK-2" and answer["expires_in"] > 365 * 24 * 3600
    # A timezone-aware expiry is honoured too; an unparsable one falls back to the default hour.
    worker.handle("/token", {"turn_id": "t1", "model_token": {"access_token": "TOK-3", "expires_at": "2099-01-01T00:00:00+00:00"}})
    assert json.loads(_get(host, "/computeMetadata/v1/instance/service-accounts/default/token")[2])["expires_in"] > 365 * 24 * 3600
    worker.handle("/token", {"turn_id": "t1", "model_token": {"access_token": "TOK-4", "expires_at": "soon"}})
    assert json.loads(_get(host, "/computeMetadata/v1/instance/service-accounts/default/token")[2])["expires_in"] == 3600

    assert worker.handle("/token", {"turn_id": "t1", "model_token": ""})["error"].startswith("model_token")
    assert worker.handle("/token", {"turn_id": "nope", "model_token": {"access_token": "x"}}) == {"ok": False, "error": "unknown turn_id"}

    record.finish()
    status, _flavor, body = _get(host, "/computeMetadata/v1/instance/service-accounts/default/token")
    assert status == 404 and "no model token" in body


def test_a_turn_without_a_model_token_refuses_token_pushes_and_serves_none(tmp_path):
    worker = _worker(tmp_path)
    _running_turn(worker, "t1", None)
    assert worker.handle("/token", {"turn_id": "t1", "model_token": {"access_token": "x"}}) == {
        "ok": False, "error": "this turn runs without a model token"}
    assert worker.served_model_token() is None


@pytest.mark.parametrize("value, expected", [
    ("legacy-bare-token", {"access_token": "legacy-bare-token", "expires_at": None}),
    ({"access_token": "t", "expires_at": "2099-01-01T00:00:00"}, {"access_token": "t", "expires_at": "2099-01-01T00:00:00"}),
    ({"access_token": "t", "expires_at": None}, {"access_token": "t", "expires_at": None}),
    ({"access_token": ""}, None), ({"expires_at": "x"}, None), (None, None), (42, None),
])
def test_token_record_normalization(value, expected):
    from agent_run.runtime.gemini.worker import _token_record

    assert _token_record(value) == expected

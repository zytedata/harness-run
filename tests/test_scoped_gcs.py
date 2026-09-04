"""Run-scoped GCS tokens (runtime/gemini/scoped_gcs.py): the worker's GCS work runs on a
token limited to the run's own objects, never on the runtime identity.

Everything here is offline: the access boundary is inspected as data, the worker
credentials refresh through an injected fetch, and the blob store is checked against a
fake storage client.
"""
from __future__ import annotations

import datetime as dt
import json

from remote_agent_toolkit.checkpoint.session_store import _claude_session_id
from remote_agent_toolkit.ports import blobstore as bs
from remote_agent_toolkit.runtime.gemini import adk_agent, pool, scoped_gcs


def test_run_object_prefixes_cover_every_gcs_surface_of_a_turn():
    bucket, base, prefixes = scoped_gcs.run_object_prefixes("gs://out", "sid-1")
    csid = _claude_session_id("sid-1")
    assert bucket == "out" and base == ""
    assert prefixes == [
        "invocation-secrets/sid-1-",
        "session-config/sid-1.json",
        "turn-config/sid-1-",
        "events/sid-1/",
        f"checkpoints/sessions/{csid}/",
        f"checkpoints/workspace/{csid}.tar.gz",
        "artifacts/sid-1/",
        "control/sid-1/",
        "control-delivered/sid-1/",
    ]
    # A bucket URI with a prefix puts that prefix in front of every key.
    bucket, base, prefixes = scoped_gcs.run_object_prefixes("gs://out/team-a", "sid-1")
    assert base == "team-a/"
    assert all(p.startswith("team-a/") for p in prefixes)
    # Another session shares nothing: no prefix of one is a prefix of the other.
    _b, _base, other = scoped_gcs.run_object_prefixes("gs://out", "sid-2")
    assert not any(o.startswith(p) or p.startswith(o) for p in prefixes for o in other)


def test_access_boundary_is_one_rule_per_prefix_with_get_and_list_conditions():
    boundary = scoped_gcs.access_boundary("out", ["events/sid-1/", "turn-config/sid-1-"])
    rules = boundary.rules
    assert len(rules) == 2
    for rule, prefix in zip(rules, ["events/sid-1/", "turn-config/sid-1-"]):
        assert rule.available_resource == "//storage.googleapis.com/projects/_/buckets/out"
        assert list(rule.available_permissions) == ["inRole:roles/storage.objectAdmin"]
        expr = rule.availability_condition.expression
        assert f'resource.name.startsWith("projects/_/buckets/out/objects/{prefix}")' in expr
        assert f'"storage.googleapis.com/objectListPrefix", "").startsWith("{prefix}")' in expr


def test_token_key_sits_under_the_run_secrets_prefix():
    _b, base, prefixes = scoped_gcs.run_object_prefixes("gs://out/pfx", "sid-9")
    key = scoped_gcs.token_key(base, "sid-9")
    assert key == "pfx/invocation-secrets/sid-9-gcs-token.json"
    assert key.startswith(prefixes[0])  # readable with the run's own token, nobody else's


def test_worker_credentials_refresh_from_the_run_token_object():
    calls = []

    def fetch(url, bearer):
        calls.append((url, bearer))
        return {"token": "tok-2", "expiry": "2099-01-01T12:00:00+00:00"}

    creds = scoped_gcs.worker_credentials("tok-1", "gs://out", "sid-1", fetch=fetch)
    assert creds.token == "tok-1"
    creds.refresh(None)
    assert creds.token == "tok-2"
    assert creds.expiry == dt.datetime(2099, 1, 1, 12, 0, 0)   # naive UTC, google-auth style
    url, bearer = calls[0]
    assert bearer == "tok-1"                                    # the old token reads the new one
    assert url.startswith("https://storage.googleapis.com/storage/v1/b/out/o/")
    assert "invocation-secrets%2Fsid-1-gcs-token.json" in url and url.endswith("?alt=media")


def test_worker_credentials_keep_the_current_token_when_refresh_fails():
    def fetch(url, bearer):
        raise OSError("bucket unreachable")

    creds = scoped_gcs.worker_credentials("tok-1", "gs://out", "sid-1", fetch=fetch)
    creds.refresh(None)
    assert creds.token == "tok-1"
    # No bucket known (a pre-stream engine): refresh is a no-op too.
    creds = scoped_gcs.worker_credentials("tok-1", None, "sid-1", fetch=fetch)
    creds.refresh(None)
    assert creds.token == "tok-1"


def test_output_bucket_from_env_strips_the_events_leaf():
    assert scoped_gcs.output_bucket_from_env("gs://out/events") == "gs://out"
    assert scoped_gcs.output_bucket_from_env("gs://out/pfx/events") == "gs://out/pfx"
    assert scoped_gcs.output_bucket_from_env(None) is None


def test_write_and_delete_run_token_use_the_store(tmp_path):
    store = bs.LocalBlobStore(str(tmp_path))
    scoped_gcs.write_run_token("gs://out", "sid-1", "tok-1", dt.datetime(2026, 9, 2, 12), store=store)
    data = json.loads(store.get_bytes("invocation-secrets/sid-1-gcs-token.json"))
    assert data == {"token": "tok-1", "expiry": "2026-09-02T12:00:00"}
    scoped_gcs.delete_run_token("gs://out", "sid-1", store=store)
    assert not store.exists("invocation-secrets/sid-1-gcs-token.json")


def test_gcs_token_directive_is_stripped_last():
    prompt = ("AGENT_SESSION=s1\nAGENT_SECRETS_GCS=gs://b/k.json\nAGENT_RESUME=s1\n"
              "AGENT_GCS_TOKEN=tok-abc\ndo the task")
    sid, prompt = adk_agent._split_session_directive(prompt)
    uri, prompt = adk_agent._split_secrets_directive(prompt)
    _s, _t, prompt = adk_agent._split_config_directives(prompt)
    resume, prompt = adk_agent._split_resume_directive(prompt)
    token, prompt = adk_agent._split_gcs_token_directive(prompt)
    assert (sid, uri, resume, token) == ("s1", "gs://b/k.json", "s1", "tok-abc")
    assert prompt == "do the task"                      # the model never sees the token
    assert adk_agent._split_gcs_token_directive("plain") == (None, "plain")


def test_dispatch_payload_carries_the_token_only_when_given():
    assert "gcs_token" not in pool.dispatch_payload("s", "m", False)
    assert pool.dispatch_payload("s", "m", False, gcs_token="tok")["gcs_token"] == "tok"


def test_use_run_gcs_token_sets_and_clears_the_default_credentials(monkeypatch):
    monkeypatch.setenv("AGENT_EVENTS_GCS", "gs://out/events")
    try:
        assert adk_agent._use_run_gcs_token("tok-1", "sid-1") is True
        creds = bs.default_gcs_credentials()
        assert creds is not None and creds.token == "tok-1"
        assert adk_agent._use_run_gcs_token(None, "sid-1") is False
        assert bs.default_gcs_credentials() is None
    finally:
        bs.set_default_gcs_credentials(None)


def test_gcs_blob_store_authorizes_with_the_default_credentials(monkeypatch):
    seen = {}

    class FakeClient:
        def __init__(self, project=None, credentials=None):
            seen["project"] = project
            seen["credentials"] = credentials

        def bucket(self, name):
            return ("bucket", name)

    import google.cloud.storage as storage

    monkeypatch.setattr(storage, "Client", FakeClient)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj-x")
    sentinel = object()
    bs.set_default_gcs_credentials(sentinel)
    try:
        assert bs.GcsBlobStore("out")._get_bucket() == ("bucket", "out")
        assert seen == {"project": "proj-x", "credentials": sentinel}
        # An explicit credentials argument wins over the process default.
        own = object()
        bs.GcsBlobStore("out", credentials=own)._get_bucket()
        assert seen["credentials"] is own
    finally:
        bs.set_default_gcs_credentials(None)
    # No credentials anywhere: plain ADC client (no kwargs).
    seen.clear()
    bs.GcsBlobStore("out")._get_bucket()
    assert seen == {"project": None, "credentials": None}


def test_cold_submit_carries_the_token_as_the_last_directive(monkeypatch):
    import asyncio
    import types

    from remote_agent_toolkit.runtime.gemini import backend
    from remote_agent_toolkit.spec import AgentSpec

    engine = backend.GeminiEngine(
        resource="r/reasoningEngines/1", spec=AgentSpec(name="c", model="m"),
        project=None, location=None, output_bucket="gs://out",
    )
    captured = {}

    class FakeAE:
        def run_query_job(self, name, config):
            captured["query"] = json.loads(config["query"])["input"]["message"]
            return types.SimpleNamespace(job_name=None)

    monkeypatch.setattr(engine, "_agent_engines", lambda: FakeAE())

    async def empty_tail(*a, **kw):
        if False:
            yield None

    monkeypatch.setattr(backend, "tail_stream", lambda *a, **kw: empty_tail())
    session = backend.GeminiSession(engine, "sid-7")

    async def go():
        run = session.run("hello")
        try:
            async for _ in run:
                pass
        except Exception:  # noqa: BLE001 — the empty tail ends the run without a result
            pass

    asyncio.run(go())
    lines = captured["query"].split("\n")
    assert lines[0] == "AGENT_SESSION=sid-7"
    assert lines[-2] == "AGENT_GCS_TOKEN=fake-run-token"     # last directive, then the message
    assert lines[-1] == "hello"

    # scoped_gcs=False (a pre-token engine during a migration): no token, no directive.
    engine._scoped_gcs = False
    session = backend.GeminiSession(engine, "sid-8")
    asyncio.run(go())
    assert "AGENT_GCS_TOKEN" not in captured["query"]

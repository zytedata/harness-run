"""Run-scoped GCS tokens (runtime/gemini/scoped_gcs.py): the worker's GCS work runs on a
token limited to the run's own objects (the sandbox has no identity of its own).

Everything here is offline: the access boundary is inspected as data, the worker
credentials refresh through an injected fetch, and the blob store is checked against a
fake storage client.
"""
from __future__ import annotations

import datetime as dt
import logging
import json

from agent_run import AgentSpec
from agent_run.checkpoint.session_store import _claude_session_id
from agent_run.ports import blobstore as bs
from agent_run.runtime.gemini import scoped_gcs


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
        f"checkpoints/codex-threads/{csid}/",
        "artifacts/sid-1/",
    ]
    # A bucket URI with a prefix puts that prefix in front of every key.
    bucket, base, prefixes = scoped_gcs.run_object_prefixes("gs://out/team-a", "sid-1")
    assert base == "team-a/"
    assert all(p.startswith("team-a/") for p in prefixes)
    # Another session shares nothing: no prefix of one is a prefix of the other.
    _b, _base, other = scoped_gcs.run_object_prefixes("gs://out", "sid-2")
    assert not any(o.startswith(p) or p.startswith(o) for p in prefixes for o in other)


def test_access_boundary_is_one_rule_per_prefix_and_lists_only_the_listed_ones():
    prefixes = ["events/sid-1/", "turn-config/sid-1-", "checkpoints/sessions/c1/"]
    boundary = scoped_gcs.access_boundary("out", prefixes)
    rules = boundary.rules
    assert len(rules) == 3
    for rule, prefix in zip(rules, prefixes):
        assert rule.available_resource == "//storage.googleapis.com/projects/_/buckets/out"
        assert list(rule.available_permissions) == ["inRole:roles/storage.objectAdmin"]
        expr = rule.availability_condition.expression
        assert f'resource.name.startsWith("projects/_/buckets/out/objects/{prefix}")' in expr
        listed = prefix.startswith("checkpoints/sessions/")
        assert (f'"storage.googleapis.com/objectListPrefix", "").startsWith("{prefix}")' in expr) is listed


def test_access_boundary_list_clauses_are_relative_to_the_bucket_prefix():
    boundary = scoped_gcs.access_boundary("out", ["pfx/checkpoints/sessions/c1/", "pfx/events/sid-1/"], "pfx/")
    exprs = [r.availability_condition.expression for r in boundary.rules]
    assert "objectListPrefix" in exprs[0] and "objectListPrefix" not in exprs[1]


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
    # The object's far-future expiry is CLAMPED to the refresh horizon: the replacement can
    # only be fetched with a live token, so the worker must never park on one that long.
    assert creds.expiry < dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) + scoped_gcs.REFRESH_HORIZON + dt.timedelta(seconds=5)
    assert creds.expiry > dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) + scoped_gcs.REFRESH_HORIZON - dt.timedelta(minutes=1)
    url, bearer = calls[0]
    assert bearer == "tok-1"                                    # the old token reads the new one
    assert url.startswith("https://storage.googleapis.com/storage/v1/b/out/o/")
    assert "invocation-secrets%2Fsid-1-gcs-token.json" in url and url.endswith("?alt=media")


def test_worker_credentials_expire_within_the_horizon_even_without_a_known_expiry():
    """The worker is handed a bare token string. With expiry=None google-auth would treat
    it as never expiring and only refresh after a call had already failed with 401 — by
    which point the token cannot fetch its own replacement."""
    creds = scoped_gcs.worker_credentials("tok-1", "gs://out", "sid-1", fetch=lambda u, b: {})
    assert creds.expiry is not None
    assert creds.expiry <= dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) + scoped_gcs.REFRESH_HORIZON + dt.timedelta(seconds=5)


def test_worker_credentials_keep_an_expiry_sooner_than_the_horizon():
    soon = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) + dt.timedelta(minutes=2)
    creds = scoped_gcs.worker_credentials(
        "tok-1", "gs://out", "sid-1", expiry=soon, fetch=lambda u, b: {}
    )
    assert creds.expiry == soon


def test_worker_credentials_report_a_dead_token_once(caplog):
    def fetch(url, bearer):
        raise OSError("401 expired")

    creds = scoped_gcs.worker_credentials("tok-1", "gs://out", "sid-1", fetch=fetch)
    with caplog.at_level(logging.ERROR, logger=scoped_gcs.logger.name):
        for _ in range(scoped_gcs.DEAD_TOKEN_AFTER + 4):
            creds.refresh(None)
    dead = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(dead) == 1                      # once, not once per retry
    assert "presumed dead" in dead[0].getMessage()
    assert creds.token == "tok-1"              # and the turn is never taken down


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


def test_write_and_delete_run_token_use_the_store(tmp_path):
    store = bs.LocalBlobStore(str(tmp_path))
    scoped_gcs.write_run_token("gs://out", "sid-1", "tok-1", dt.datetime(2026, 9, 2, 12), store=store)
    data = json.loads(store.get_bytes("invocation-secrets/sid-1-gcs-token.json"))
    assert data == {"token": "tok-1", "expiry": "2026-09-02T12:00:00"}
    scoped_gcs.delete_run_token("gs://out", "sid-1", store=store)
    assert not store.exists("invocation-secrets/sid-1-gcs-token.json")


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


def test_mint_run_token_sends_sts_a_boundary_that_lists_only_the_transcript_prefix(monkeypatch):
    """Drives the real google-auth exchange against a captured transport: the boundary STS
    receives must carry the objectListPrefix clause on exactly the checkpoint transcript
    dir, resolved under the bucket prefix. Every extra list clause costs ~500 chars of
    minted token, and STS caps the token including the caller's own."""
    import urllib.parse

    from google.auth.transport import requests as ga_requests

    class FakeSource:
        token = "src-token"
        expiry = None

        def refresh(self, request):
            pass

    class FakeResponse:
        status = 200
        data = b'{"access_token": "downscoped", "expires_in": 3600}'

    bodies = []

    def fake_request(url, method, headers, body):
        bodies.append(body)
        return FakeResponse()

    monkeypatch.undo()  # conftest stubs mint_run_token for the whole suite; here it runs for real
    monkeypatch.setattr(ga_requests, "Request", lambda: fake_request)

    token, _expiry = scoped_gcs.mint_run_token(FakeSource(), "gs://out/team-a", "sid-1")

    assert token == "downscoped"
    form = urllib.parse.parse_qs(bodies[0].decode())
    assert form["subject_token"] == ["src-token"]
    rules = json.loads(urllib.parse.unquote(form["options"][0]))["accessBoundary"]["accessBoundaryRules"]
    assert len(rules) == 8
    listed = [
        r["availabilityCondition"]["expression"].split('objects/')[1].split('"')[0]
        for r in rules
        if "objectListPrefix" in r["availabilityCondition"]["expression"]
    ]
    assert listed == [f"team-a/checkpoints/sessions/{_claude_session_id('sid-1')}/"]


# -- every checkpoint writer stays inside the run's prefixes ------------------------------


class _RecordingBlobStore:
    """A LocalBlobStore that records every key it is asked for and every prefix it lists."""

    def __init__(self, root):
        from agent_run.ports.blobstore import LocalBlobStore

        self._inner = LocalBlobStore(str(root))
        self.keys: set[str] = set()
        self.listed: set[str] = set()

    def put_bytes(self, key, data):
        self.keys.add(key)
        return self._inner.put_bytes(key, data)

    def get_bytes(self, key):
        self.keys.add(key)
        return self._inner.get_bytes(key)

    def exists(self, key):
        self.keys.add(key)
        return self._inner.exists(key)

    def delete(self, key):
        self.keys.add(key)
        return self._inner.delete(key)

    def list(self, prefix):
        self.listed.add(prefix)
        return self._inner.list(prefix)

    def put_tree(self, key, local_dir):  # the workspace snapshot writes with this
        self.keys.add(key)
        return self._inner.put_tree(key, local_dir)

    def get_tree(self, key, local_dir):  # ... and restores with this
        self.keys.add(key)
        return self._inner.get_tree(key, local_dir)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _assert_lists_are_allowed(listed, boundary):
    """Every prefix the worker listed must start with a prefix some rule allows listing.

    GCS evaluates the list clause as ``objectListPrefix.startsWith(<allowed>)``, so a list
    under a deeper sub-prefix (``sessions/<sid>/main/``) is covered by the allowed
    ``checkpoints/sessions/<sid>/``.
    """
    import re

    allowed = []
    for rule in boundary.rules:
        allowed += re.findall(
            r'objectListPrefix", ""\)\.startsWith\("([^"]+)"\)', rule.availability_condition.expression
        )
    for prefix in listed:
        full = "checkpoints/" + prefix
        assert any(full.startswith(a) for a in allowed), (
            f"the worker lists {full!r} but no rule allows listing it (allowed: {allowed})")


async def test_every_checkpoint_object_of_a_two_turn_codex_run_is_inside_the_run_prefixes(
    tmp_path, monkeypatch
):
    """The run token only opens ``run_object_prefixes``; a writer outside them fails silently
    (checkpoint persistence is best-effort) and the next turn starts without its history.
    Found live 2026-09-08: the Codex thread (``codex-threads/``) was missing, so every round
    two of a Codex session forgot round one. The store here is rooted where the engine roots
    it (``AGENT_CHECKPOINT_GCS = <bucket>/checkpoints``), so a key maps to the object
    ``checkpoints/<key>``."""
    from codex_fakes import agent_message, token_usage, turn_completed, turn_started
    from test_codex_harness import _events_of, _fake_rollout

    from agent_run.checkpoint.session_store import BlobSessionStore
    from agent_run.harness.context import RunContext

    sid = "5f2e3c1a-9b7d-4e6f-8a1b-2c3d4e5f6a7b"  # a UUID: the warm path's id, csid == sid
    assert _claude_session_id(sid) == sid
    blobs = _RecordingBlobStore(tmp_path / "blobs")
    spec = AgentSpec(name="a", model="gpt-5.6-luna", harness="codex", checkpoint=True)

    def ctx(job_root, **kw):
        c = RunContext(spec=spec, prompt="go", job_dir=job_root / "job", session_id=sid,
                       secrets={"OPENAI_API_KEY": "sk-test"}, blobs=blobs,
                       session_store=BlobSessionStore(blobs), **kw)
        c.workspace.mkdir(parents=True, exist_ok=True)
        return c

    ctx1 = ctx(tmp_path)

    def write_rollout(client):
        _fake_rollout(ctx1.job_dir / "codex_home")

    script1 = [turn_started(), write_rollout, agent_message("done"), token_usage(), turn_completed()]
    events1, _ = await _events_of(script1, tmp_path, monkeypatch, spec=spec, ctx=ctx1)
    assert any((e.raw or {}).get("event") == "checkpoint_saved" for e in events1)

    ctx2 = ctx(tmp_path / "w2", resume_sid=sid)
    script2 = [turn_started(), agent_message("again"), token_usage(), turn_completed()]
    events2, client2 = await _events_of(script2, tmp_path, monkeypatch, spec=spec, ctx=ctx2)
    assert any((e.raw or {}).get("event") == "thread_resumed" for e in events2)
    assert client2.thread_resumes  # the second turn really continued the first one

    bucket, _base, prefixes = scoped_gcs.run_object_prefixes("gs://out", sid)
    assert blobs.keys, "the run wrote nothing: the test drives no checkpoint"
    outside = sorted(k for k in blobs.keys if not any(("checkpoints/" + k).startswith(p) for p in prefixes))
    assert outside == [], f"objects the run token cannot reach: {outside}"
    _assert_lists_are_allowed(blobs.listed, scoped_gcs.access_boundary(bucket, prefixes))


def test_workspace_snapshot_and_restore_stay_inside_the_run_prefixes(tmp_path):
    """The Claude path's checkpoint writers (shared with Codex): transcript store + workspace tar."""
    import asyncio

    from agent_run.checkpoint import workspace
    from agent_run.checkpoint.session_store import BlobSessionStore

    sid = "5f2e3c1a-9b7d-4e6f-8a1b-2c3d4e5f6a7b"
    blobs = _RecordingBlobStore(tmp_path / "blobs")
    store = BlobSessionStore(blobs)
    asyncio.run(store.append({"session_id": sid}, [{"uuid": "u1", "type": "user"}]))
    asyncio.run(store.append({"session_id": sid, "subpath": "sub/agent-1"}, [{"uuid": "u2"}]))
    assert asyncio.run(store.load({"session_id": sid}))
    assert asyncio.run(store.list_subkeys({"session_id": sid})) == ["sub/agent-1"]
    workdir = tmp_path / "ws"
    workdir.mkdir()
    (workdir / "a.txt").write_text("x")
    workspace.snapshot(blobs, sid, str(workdir))
    assert workspace.restore(blobs, sid, str(tmp_path / "ws2"))

    bucket, _base, prefixes = scoped_gcs.run_object_prefixes("gs://out", sid)
    outside = sorted(k for k in blobs.keys if not any(("checkpoints/" + k).startswith(p) for p in prefixes))
    assert outside == [], f"objects the run token cannot reach: {outside}"
    _assert_lists_are_allowed(blobs.listed, scoped_gcs.access_boundary(bucket, prefixes))


def test_the_recording_store_sees_archive_writes_too(tmp_path):
    """Guards the guard: a tar written outside the run prefixes must show up as a stray key
    (the workspace snapshot is a put_tree, which a plain attribute pass-through would miss)."""
    blobs = _RecordingBlobStore(tmp_path / "blobs")
    src = tmp_path / "src"
    src.mkdir()
    (src / "f").write_text("x")
    blobs.put_tree("elsewhere/escape.tar.gz", str(src))
    blobs.get_tree("elsewhere/escape.tar.gz", str(tmp_path / "dst"))
    sid = "5f2e3c1a-9b7d-4e6f-8a1b-2c3d4e5f6a7b"
    _b, _base, prefixes = scoped_gcs.run_object_prefixes("gs://out", sid)
    outside = [k for k in blobs.keys if not any(("checkpoints/" + k).startswith(p) for p in prefixes)]
    assert outside == ["elsewhere/escape.tar.gz"]

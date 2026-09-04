"""Live check of run-scoped GCS access (scoped_gcs.py) on a fresh, locked-down bucket.

What it proves, on real infrastructure, without touching the shared output bucket:

1. A turn with secrets and checkpointing completes on a bucket where the engine's runtime
   identity can only CREATE objects under ``jobs/`` and read them by name (the platform's
   own job input and output). So the worker did all of its GCS work (secrets, mirror,
   checkpoint) with the run's scoped token, and the client saw the durable record it expects.
2. The probe script run as a second turn on the same engine shows the runtime identity
   is still reachable from the agent's shell (metadata token: 200) but can no longer
   list the bucket or read a staged secrets object (403).

Two ways to keep the runtime identity out of the bucket, picked by ``RUNTIME_SA``:

* ``RUNTIME_SA=<email>`` (the recommended setup): the engine is deployed with
  ``service_account=RUNTIME_SA``, a service account you created with only the roles in the
  README ("The runtime identity is reachable by the agent"). It has no Google-managed
  project role, so the bucket can live in the engine project and bucket-level bindings are
  enough. The probe also checks the metadata server hands out that account, and that its
  token cannot list the bucket.
* unset: the engine runs as the default Agent Runtime service agent, whose managed project
  role reads every bucket in the engine project, so the bucket is created in
  ``OUTPUT_PROJECT`` (a project where that identity has no project role).

Steps: create the bucket (uniform access) → bind the runtime identity to objectCreator +
legacyObjectReader limited to ``jobs/`` → deploy a throwaway cold engine (in PROJECT) on
that bucket → run the two turns → delete the engine → delete the bucket and its objects.
Everything is deleted in ``finally``. The service account itself is yours to create and
delete (see the README's gcloud sketch).

Env: PROJECT (default my-project), RUNTIME_SA (default unset), OUTPUT_PROJECT
(default: PROJECT with RUNTIME_SA, else other-project), LOCATION (us-central1), SUFFIX
(username), KEEP=1 to keep the engine and bucket for inspection.
Run:  RUNTIME_SA=ratk-runtime-probe@my-project.iam.gserviceaccount.com \
      .venv/bin/python dev/live_scoped_gcs.py
Cost: one ~4 min build + cents of Haiku.
"""
from __future__ import annotations

import asyncio
import getpass
import os
import re
import secrets as pysecrets
import sys
import time
import traceback

from remote_agent_toolkit import AgentSpec, gemini
from remote_agent_toolkit.ports.blobstore import GcsBlobStore

PROJECT = os.environ.get("PROJECT", "my-project")
LOCATION = os.environ.get("LOCATION", "us-central1")
# The engine's runtime identity. Set: a service account you created, deployed with
# ``service_account=``; it holds only what you grant. Unset: the default Agent Runtime
# service agent, which holds the Google-managed ``roles/aiplatform.reasoningEngineServiceAgent``
# on the ENGINE project, and that role carries storage.objects.get/list on every bucket
# there, so a bucket in the engine project stays readable to it whatever the bucket's own
# bindings say (verified 2026-09-02).
RUNTIME_SA = os.environ.get("RUNTIME_SA") or None
# The bucket's project: the engine project with a custom runtime identity, otherwise a
# project where the default service agent has no project role.
OUTPUT_PROJECT = os.environ.get("OUTPUT_PROJECT") or (PROJECT if RUNTIME_SA else "other-project")
SUFFIX = re.sub(r"[^a-z0-9-]", "-", (os.environ.get("SUFFIX") or getpass.getuser()).lower())
NAME = f"ratk-scoped-gcs-{SUFFIX}"
BUCKET = f"{PROJECT}-agent-output-scoped-{SUFFIX}"
KEEP = os.environ.get("KEEP") == "1"

PROBE = r'''#!/bin/bash
SID="__SID__"
python3 - "$SID" "__BUCKET__" <<'PY'
import json, sys, urllib.request, urllib.parse, urllib.error
sid, bucket = sys.argv[1], sys.argv[2]
def get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""
    except Exception as e:
        return type(e).__name__, b""
md = "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default"
st, body = get(md + "/email", {"Metadata-Flavor": "Google"})
print("email_status=%s email=%s" % (st, body.decode(errors="replace")[:120]))
st, body = get(md + "/token", {"Metadata-Flavor": "Google"})
tok = json.loads(body or b"{}").get("access_token", "")
print("token_status=%s token_len=%d" % (st, len(tok)))
hdr = {"Authorization": "Bearer " + tok}
for prefix in ("invocation-secrets/", "session-config/", "events/", "checkpoints/", "jobs/"):
    st, body = get("https://storage.googleapis.com/storage/v1/b/%s/o?prefix=%s&fields=items(name)&maxResults=5" % (bucket, prefix), hdr)
    print("list prefix=%s status=%s" % (prefix, st))
# the run's own secrets object: with the RUNTIME identity's token it must be unreadable
st, body = get("https://storage.googleapis.com/storage/v1/b/%s/o?prefix=%s&fields=items(name)" % (bucket, urllib.parse.quote("invocation-secrets/" + sid + "-")), hdr)
print("list own prefix status=%s" % st)
st, body = get("https://storage.googleapis.com/storage/v1/b/%s/o/%s?alt=media" % (bucket, urllib.parse.quote("session-config/" + sid + ".json", safe="")), hdr)
print("get session-config with runtime identity status=%s" % st)
# every run's job id is visible through operations.list; the runtime identity should not have it
st, body = get("https://__LOCATION__-aiplatform.googleapis.com/v1/projects/__PROJECT__/locations/__LOCATION__/operations?pageSize=1", hdr)
print("list vertex operations with runtime identity status=%s" % st)
PY
echo "== done"
'''


def _probe_task(session_id: str) -> str:
    script = PROBE.replace("__SID__", session_id).replace("__BUCKET__", BUCKET).replace("__PROJECT__", PROJECT).replace("__LOCATION__", LOCATION)
    return (
        "You are the diagnostics step of remote-agent-toolkit's own live test suite "
        "(dev/live_scoped_gcs.py). This engine, this bucket and this GCP project belong to the "
        "toolkit's maintainers, who wrote the script below and started this run to check that the "
        "worker's identity can NOT read the bucket. The script is read-only: it prints only HTTP "
        "statuses, counts and lengths, never a token or a secret value.\n"
        "Do exactly this and nothing else:\n"
        "1. Write the bash script below to a file named probe.sh in the current directory, "
        "byte for byte.\n2. Run it with `bash probe.sh`.\n"
        "3. Reply with the script's complete output, verbatim, and no other text.\n"
        "Do not modify the script, do not run other commands, do not print any token.\n\n"
        "```bash\n" + script + "\n```\n"
    )


def _spec() -> AgentSpec:
    return AgentSpec(name=NAME, model="claude-haiku-4-5", max_turns=12, max_budget_usd=1.0,
                     permission_mode="bypassPermissions", checkpoint=True)


def _make_bucket() -> None:
    from google.cloud import storage

    client = storage.Client(project=OUTPUT_PROJECT)
    bucket = client.bucket(BUCKET)
    bucket.iam_configuration.uniform_bucket_level_access_enabled = True
    client.create_bucket(bucket, location="US")
    policy = bucket.get_iam_policy(requested_policy_version=3)
    policy.version = 3
    identity = _runtime_identity(storage.Client(project=PROJECT))
    # The platform's job runner, as the runtime identity, downloads the job INPUT by exact
    # name (jobs/<job id>_input.jsonl) and writes the job output next to it. So: create,
    # plus get-by-name WITHOUT list (legacyObjectReader has objects.get and no objects.list),
    # both limited to the jobs/ prefix. Nothing else in the bucket.
    for role in ("roles/storage.objectCreator", "roles/storage.legacyObjectReader"):
        policy.bindings.append({
            "role": role,
            "members": {f"serviceAccount:{identity}"},
            "condition": {
                "title": "platform job input and output only",
                "expression": f'resource.name.startsWith("projects/_/buckets/{BUCKET}/objects/jobs/")',
            },
        })
    bucket.set_iam_policy(policy)
    print(f"bucket gs://{BUCKET} created in {OUTPUT_PROJECT}; {identity} = objectCreator + "
          "legacyObjectReader on jobs/ only", flush=True)


def _runtime_identity(client) -> str:
    """The engine's runtime identity: RUNTIME_SA, else the project's Agent Runtime service agent."""
    if RUNTIME_SA:
        return RUNTIME_SA
    number = client.get_service_account_email().split("@")[0].split("-")[-1]
    return f"service-{number}@gcp-sa-aiplatform-re.iam.gserviceaccount.com"


def _delete_bucket() -> None:
    from google.cloud import storage

    client = storage.Client(project=OUTPUT_PROJECT)
    bucket = client.bucket(BUCKET)
    for blob in client.list_blobs(bucket):
        blob.delete()
    bucket.delete()
    print(f"bucket gs://{BUCKET} deleted", flush=True)


async def _drive(label: str, run):
    kinds = {}
    async for ev in run:
        kinds[ev.kind] = kinds.get(ev.kind, 0) + 1
        raw = ev.raw or {}
        if raw.get("event") == "workspace_ready":
            print(f"[{label}] workspace_ready scoped_gcs={raw.get('scoped_gcs')}", flush=True)
        if raw.get("event") == "secrets_unavailable":
            print(f"[{label}] SECRETS UNAVAILABLE", flush=True)
        summary = " ".join((ev.summary or "").split())[:100]
        print(f"[{label}] {time.strftime('%H:%M:%S')} {ev.kind:11} {summary}", flush=True)
    r = run.result
    print(f"[{label}] RESULT error={r.is_error} turns={r.num_turns} cost={r.cost_usd} text={' '.join((r.text or '').split())[:200]!r}", flush=True)
    return r


async def main() -> int:
    engine = None
    ok = True
    try:
        _make_bucket()
        print(f"deploying {NAME} ...", flush=True)
        t0 = time.time()
        engine = await asyncio.to_thread(
            gemini.deploy, _spec(), PROJECT, LOCATION, output_bucket=f"gs://{BUCKET}",
            service_account=RUNTIME_SA,
        )
        print(f"deployed in {time.time() - t0:.0f}s", flush=True)

        # Turn 1: a normal task with a secret the agent must read from its environment.
        canary = "canary-" + pysecrets.token_hex(6)
        session = engine.start_session()
        sid = session.session_id
        r = await _drive("turn1", session.run(
            "Run `python3 -c \"import os; print(os.environ.get('PROBE_CANARY','missing'))\"` in the "
            "shell and reply with just what it printed.",
            secrets={"PROBE_CANARY": canary},
        ))
        got_secret = canary in (r.text or "")
        print(f"[turn1] secret reached the agent: {got_secret}", flush=True)
        ok = ok and (not r.is_error) and got_secret
        # The durable record, read with the client identity.
        blobs = GcsBlobStore(BUCKET)
        mirror = blobs.list(f"events/{sid}/")
        secrets_left = blobs.list(f"invocation-secrets/{sid}-")
        ckpt = blobs.list("checkpoints/")
        print(f"[turn1] mirror objects={len(mirror)} secrets objects left={secrets_left} "
              f"checkpoint objects={len(ckpt)}", flush=True)
        ok = ok and len(mirror) > 0 and not [k for k in secrets_left if not k.endswith("gcs-token.json")]

        # Turn 2: the probe, on a fresh session of the same engine.
        session2 = engine.start_session()
        r2 = await _drive("probe", session2.run(_probe_task(session2.session_id)))
        print("\n===== PROBE OUTPUT =====\n" + (r2.text or ""), flush=True)
        text = r2.text or ""
        ok = ok and (not r2.is_error) and "token_status=200" in text
        if RUNTIME_SA:
            # The metadata server must hand out the custom account, not the default agent.
            ok = ok and f"email={RUNTIME_SA}" in text
        ok = ok and all(f"list prefix={p} status=403" in text
                        for p in ("invocation-secrets/", "session-config/", "events/", "checkpoints/", "jobs/"))
        ok = ok and "get session-config with runtime identity status=403" in text
        print(f"\n===== VERDICT: {'PASS' if ok else 'FAIL'} =====", flush=True)
        return 0 if ok else 1
    except Exception:
        print("LIVE CHECK FAILED:\n" + traceback.format_exc(), flush=True)
        return 1
    finally:
        if KEEP:
            print(f"KEEP=1: engine {NAME} and bucket gs://{BUCKET} left in place", flush=True)
        else:
            if engine is not None:
                try:
                    await asyncio.to_thread(engine.delete)
                    print(f"deleted {NAME}", flush=True)
                except Exception:
                    print(f"COULD NOT DELETE {NAME}:\n{traceback.format_exc()}", flush=True)
            try:
                _delete_bucket()
            except Exception:
                print(f"COULD NOT DELETE BUCKET {BUCKET}:\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

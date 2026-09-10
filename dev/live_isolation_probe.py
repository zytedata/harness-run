"""Live probe: can the agent's shell reach the worker's own Google identity?

Deploys a throwaway cold engine from THIS checkout, runs ONE turn whose task is a fixed
shell script, prints the script's output, deletes the engine. Nothing is written to any
bucket, no object content is read except the run's OWN staged secrets object, which holds
a made-up canary value; the script prints only key NAMES, HTTP statuses, counts and
lengths. No token, no secret value, no object name is printed.

The deploy runs the engine as the project's ``ratk-runtime@`` service account (there is no
way to deploy as the default Agent Runtime service agent any more), so on a project set up
with ``ratk-gcp-setup`` this probe shows the FIXED state: the shell still gets a token
(step 3: 200) but step 4 and step 5 must return 403. Use it as the post-migration check
(README "Migrating an existing project"). The original finding (2026-09-02) was recorded
with the default identity, which this checkout can no longer reproduce.

What the script checks, in order:
  1. uid / user of the agent's shell
  2. environment variable NAMES (values only for *GCS* / *BUCKET* vars: bucket paths)
  3. metadata server: service-account email (identity name) + token endpoint status +
     access_token length + whether a GOOGLE_APPLICATION_CREDENTIALS file exists
  4. with that token: list the output bucket's invocation-secrets/ prefix (status + count)
  5. GET the run's own staged secrets object (status + sorted key names; expected
     ["PROBE_CANARY"])
  6. /proc/1/environ readability (status + count of variables)

Env: PROJECT (default my-project), LOCATION (us-central1), SUFFIX (username).
Run:  .venv/bin/python dev/live_isolation_probe.py
Cost: one ~4 min build + cents of Haiku. The engine is deleted in ``finally``.
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

PROJECT = os.environ.get("PROJECT", "my-project")
LOCATION = os.environ.get("LOCATION", "us-central1")
SUFFIX = re.sub(r"[^a-z0-9-]", "-", (os.environ.get("SUFFIX") or getpass.getuser()).lower())
NAME = f"ratk-isolation-probe-{SUFFIX}"
OUTPUT_BUCKET = f"{PROJECT}-agent-output"   # the deploy default (backend.py: f"gs://{project}-agent-output")

SCRIPT = r"""#!/bin/bash
# Prints only statuses, counts, lengths and key names. Never a token or a value.
echo "== 1 identity"; echo "uid=$(id -u) user=$(id -un) pid1_user=$(stat -c %U /proc/1 2>/dev/null)"
echo "== 2 bucket-ish vars"; env | grep -iE '^[A-Z_]*(GCS|BUCKET)[A-Z_]*=' | sort
echo "gac_var_set=$([ -n "$GOOGLE_APPLICATION_CREDENTIALS" ] && echo yes || echo no)"
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
print("== 3 metadata")
md = "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default"
st, body = get(md + "/email", {"Metadata-Flavor": "Google"})
print("email_status=%s email=%s" % (st, body.decode(errors="replace")[:120]))
st, body = get(md + "/token", {"Metadata-Flavor": "Google"})
tok = ""
try:
    d = json.loads(body or b"{}")
    tok = d.get("access_token", "")
    print("token_status=%s token_len=%d expires_in=%s" % (st, len(tok), d.get("expires_in")))
except Exception as e:
    print("token_status=%s token_parse_error=%s" % (st, type(e).__name__))
hdr = {"Authorization": "Bearer " + tok}
print("== 4 list invocation-secrets/ in", bucket)
st, body = get("https://storage.googleapis.com/storage/v1/b/%s/o?prefix=invocation-secrets/&fields=items(name)" % bucket, hdr)
names = []
try:
    names = [i["name"] for i in json.loads(body or b"{}").get("items", [])]
except Exception:
    pass
print("list_status=%s list_count=%d" % (st, len(names)))
print("== 5 read own staged secrets object")
mine = [n for n in names if sid in n]
if mine:
    st, body = get("https://storage.googleapis.com/storage/v1/b/%s/o/%s?alt=media" % (bucket, urllib.parse.quote(mine[0], safe="")), hdr)
    try:
        keys = sorted(json.loads(body or b"{}").keys())
    except Exception:
        keys = "unparseable"
    print("own_object_found=yes get_status=%s keys=%s" % (st, keys))
else:
    print("own_object_found=no (session id not in any listed name)")
print("== 6 other prefixes, count only")
for prefix in ("session-config/", "turn-config/", "events/", "artifacts/"):
    st, body = get("https://storage.googleapis.com/storage/v1/b/%s/o?prefix=%s&fields=items(name)&maxResults=50" % (bucket, prefix), hdr)
    try:
        n = len(json.loads(body or b"{}").get("items", []))
    except Exception:
        n = -1
    print("prefix=%s status=%s count_first_page=%d" % (prefix, st, n))
print("== 7 /proc/1/environ")
try:
    data = open("/proc/1/environ", "rb").read()
    print("proc1_environ_readable=yes var_count=%d" % len([x for x in data.split(b"\0") if x]))
except Exception as e:
    print("proc1_environ_readable=no", type(e).__name__)
PY
echo "== done"
""".replace("__BUCKET__", OUTPUT_BUCKET)

def _task(session_id: str) -> str:
    return (
    "You are running a read-only diagnostics script. Do exactly this and nothing else:\n"
    "1. Write the bash script below to a file named probe.sh in the current directory, "
    "byte for byte.\n"
    "2. Run it with `bash probe.sh`.\n"
    "3. Reply with the script's complete output, verbatim, and no other text.\n"
    "Do not modify the script, do not run other commands, do not print any token or "
    "secret value.\n\n"
    "```bash\n" + SCRIPT.replace("__SID__", session_id) + "\n```\n"
    )


def _spec() -> AgentSpec:
    return AgentSpec(name=NAME, model="claude-haiku-4-5", max_turns=12, max_budget_usd=1.0,
                     permission_mode="bypassPermissions")


async def _drive(run):
    async for ev in run:
        summary = " ".join((ev.summary or "").split())[:100]
        print(f"[{time.strftime('%H:%M:%S')}] {ev.kind:11} {summary}", flush=True)
    return run.result


async def main() -> int:
    engine = None
    try:
        print(f"deploying {NAME} in {PROJECT}/{LOCATION} ...", flush=True)
        t0 = time.time()
        engine = await asyncio.to_thread(gemini.deploy, _spec(), PROJECT, LOCATION)
        print(f"deployed in {time.time() - t0:.0f}s", flush=True)
        session = engine.start_session()
        print(f"session id length {len(session.session_id)}", flush=True)
        canary = {"PROBE_CANARY": "canary-" + pysecrets.token_hex(8)}
        r = await _drive(session.run(_task(session.session_id), secrets=canary))
        print("\n===== RESULT =====")
        print(f"error={r.is_error} turns={r.num_turns} cost={r.cost_usd}")
        print(r.text or "")
        return 0 if not r.is_error else 1
    except Exception:
        print("PROBE FAILED:\n" + traceback.format_exc(), flush=True)
        return 1
    finally:
        if engine is not None:
            try:
                await asyncio.to_thread(engine.delete)
                print(f"deleted {NAME}", flush=True)
            except Exception:
                print(f"COULD NOT DELETE {NAME} — delete it by hand:\n{traceback.format_exc()}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

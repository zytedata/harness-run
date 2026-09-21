"""Live smoke test — deploy a throwaway sandbox engine from THIS checkout, run turns, tear down.

The offline suite (`make test`) can't catch platform-contract breakage: the Agent Sandbox
side changes underneath us (see TESTING.md). This script is the standard live check for a
branch that touches any deploy/runtime contract — it validates, on real infrastructure:

  * ``gemini.deploy``: the image build + push (Docker), the template, the deploy record,
    a ready pool of one sandbox (``wait_until_warm``)
  * a **pool** turn: dispatch → first event / result latency on the ready sandbox, the
    refill, the sandbox's deletion at the terminal event
  * a **fresh** turn: ``get_engine`` (pure addressing) with a ``SessionConfig`` planting a
    marker — the reply must carry it, proving the worker ran the session's config on top of
    the baked spec; then a second turn on the SAME session (checkpoint resume on a new
    sandbox) with a ``TurnConfig(output_schema=...)`` — the client parses the JSON and the
    worker's ``effective_spec`` echo still carries the marker
  * **control**: a steer into a running turn is acknowledged by a ``user`` event; a steer
    sent before the turn's first event (the dispatch window) is queued and delivered too
  * **exec**: ``Session.exec()`` before dispatch waits for the worker and answers from the
    agent's cwd; mid-turn it sees the file the agent wrote; after the turn it raises
    ``ControlUnavailable``
  * **re-attach from another process**: a fresh interpreter that only knows the engine name
    and the session id (``get_engine(...).get_session(id)``) adopts the running turn through
    the event mirror — it probes the same workspace (and gets ``ControlUnavailable`` once the
    turn is over), its steer is acknowledged on the owner's stream and shapes the reply, and
    its ``interrupt()`` ends the owner's turn as ``INTERRUPTED``
  * **resources** (with pool-turn): the result carries the worker-sampled ``memory_peak_bytes`` /
    ``memory_limit_bytes``, ``session.resource_samples()`` has rows, and no sample reached
    the live stream
  * **isolation**: a fixed shell script run through the worker's ``/exec`` (no model) prints
    what the agent's shell can reach — the metadata server's identity must be a tenant one
    that is 403 on our project's storage and Vertex (the sandbox has no usable Google
    identity); only statuses and names are printed, never tokens
  * **model-token**: during a turn, ``session.exec()`` fetches the model token from the
    worker's loopback metadata server (the one the agent's CLI is pointed at) and tries it on
    non-predict calls — listing the project's reasoning engines, the output bucket and its
    service accounts must all be refused (401/403) — while the turn itself answering proves
    the token is good for the model; a token that can list any of those is a MODEL_SA with
    more than the predict-only role, and the check fails. Only statuses and lengths are
    printed, never the token
  * optional ``LONG_MINUTES=n``: a turn whose one Bash call sleeps that long (the model
    token, the sandbox TTL and the /events long-poll all have to hold). While it sleeps the
    check probes the container's ``/proc`` and cgroup files through ``session.exec()`` and
    logs what they say (input for the resource-metrics question).
  * optional ``TOKEN_LIFETIME_S=n`` (with ``LONG_MINUTES``): mint the model token for that
    many seconds instead of an hour and refresh it at 40 % of that, so a turn longer than the
    lifetime proves the refresh chain — client re-mint → ``/token`` → worker metadata server
    → Claude Code re-fetch — in minutes instead of an hour (``TOKEN_LIFETIME_S=360
    LONG_MINUTES=7``).

Pass criteria per check: terminal result with ``error=False``, ``turns > 0``, the expected
answer in the text (plus the check's own assertions). The engine (templates + sandboxes)
is deleted in ``finally``; exit code is non-zero if any check fails.

Configure via env (defaults are the shared my-project test setup):
  PROJECT, LOCATION, IMAGE_REPO, MODEL_SA (the account whose token the agent's model calls
  carry; default the toolkit's own ``agent-run-model@<project>``, the predict-only account
  ``agent-run-gcp-setup`` creates — never the operator account, whose token would hand the
  agent the project), SUFFIX (engine-name suffix; defaults to your username), CPU / MEMORY (the template's size, default the runtime's 4 / 4Gi — a 4 CPU
  template took the platform up to its 30-minute deadline on 2026-09-14/15; CPU=1 MEMORY=1Gi
  provisions in seconds), LONG_MINUTES, TOKEN_LIFETIME_S, KEEP=1 (skip teardown), IMPERSONATE=<service account email>
  (drive everything but the Docker push as that account — to prove a role is sufficient;
  your ADC needs roles/iam.serviceAccountTokenCreator on it).

Run:
  make live-smoke                       # or:
  .venv/bin/python dev/live_smoke.py

Needs Docker logged into the registry and ADC that can impersonate MODEL_SA. Costs a few
cents of Haiku and ~5-10 min (mostly the image build on a cold Docker cache).
"""

from __future__ import annotations

import asyncio
import getpass
import json
import os
import re
import sys
import time
import traceback

from agent_run import AgentSpec, SessionConfig, StopReason, SystemPrompt, TurnConfig, gemini
from agent_run.runtime.gemini.model_token import default_model_service_account

PROJECT = os.environ.get("PROJECT", "my-project")
LOCATION = os.environ.get("LOCATION", "us-central1")
IMAGE_REPO = os.environ.get("IMAGE_REPO") or f"{LOCATION}-docker.pkg.dev/{PROJECT}/agent-run-sandbox"
MODEL_SA = os.environ.get("MODEL_SA") or default_model_service_account(PROJECT)
OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET") or f"gs://{PROJECT}-agent-output"
SUFFIX = re.sub(r"[^a-z0-9-]", "-", (os.environ.get("SUFFIX") or getpass.getuser()).lower())
LONG_MINUTES = float(os.environ.get("LONG_MINUTES", "0") or 0)
TOKEN_LIFETIME_S = int(os.environ.get("TOKEN_LIFETIME_S", "0") or 0)
if TOKEN_LIFETIME_S:
    from agent_run.runtime.gemini import backend as _backend
    from agent_run.runtime.gemini import model_token as _model_token

    _real_mint = _model_token.mint_model_token
    _model_token.mint_model_token = lambda creds, sa, lifetime_s=TOKEN_LIFETIME_S: _real_mint(creds, sa, TOKEN_LIFETIME_S)
    _backend.TOKEN_REFRESH_S = TOKEN_LIFETIME_S * 0.4
NAME = f"agent-run-smoke-{SUFFIX}"
IMPERSONATE = os.environ.get("IMPERSONATE") or None
RESOURCE_LIMITS = (
    {"cpu": os.environ["CPU"], "memory": os.environ["MEMORY"]}
    if os.environ.get("CPU") or os.environ.get("MEMORY") else None
)

# A fresh interpreter re-attaching by engine name + session id (the adopting-worker case):
# ``exec <command>`` / ``steer <message> <message_id>`` / ``interrupt``; one JSON verdict line.
REATTACH_PROBE = r"""
import asyncio, json, os, sys
from agent_run import ControlUnavailable, gemini
name, project, location, sid, op = sys.argv[1:6]
args = sys.argv[6:]
creds = None
if os.environ.get("IMPERSONATE"):  # the same identity the smoke itself drives the control plane as
    import google.auth
    from google.auth import impersonated_credentials
    source, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds = impersonated_credentials.Credentials(
        source_credentials=source, target_principal=os.environ["IMPERSONATE"],
        target_scopes=["https://www.googleapis.com/auth/cloud-platform"], lifetime=600)
engine = gemini.get_engine(name, project=project, location=location, warm_pool=False, credentials=creds)
session = engine.get_session(sid)
async def main():
    try:
        if op == "exec":
            r = await session.exec(args[0], timeout=30)
            print(json.dumps({"ok": r.ok, "stdout": r.stdout, "rc": r.returncode, "ms": r.duration_ms}))
        elif op == "steer":
            session.send(args[0], message_id=args[1])  # adopts the running turn, posts the steer, exits
            print(json.dumps({"adopted": session._adopted, "busy": session.busy}))
        elif op == "interrupt":
            await session.interrupt(timeout=90)
            r = session.last_result
            print(json.dumps({"adopted": session._adopted is False, "stop_reason": getattr(session.stop_reason, "value", None),
                              "text": r.text if r else None, "error": r.is_error if r else None,
                              "warning": r.warning if r else None}))
    except ControlUnavailable as exc:
        print(json.dumps({"unavailable": str(exc)}))
asyncio.run(main())
"""


async def reattach_probe(session_id: str, op: str, *args: str) -> dict:
    """``exec`` / ``steer`` / ``interrupt`` on the session from a NEW process that re-attaches by id."""
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", REATTACH_PROBE, NAME, PROJECT, LOCATION, session_id, op, *args,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    out, err = await asyncio.wait_for(proc.communicate(), 120)
    line = next((ln for ln in out.decode().splitlines() if ln.startswith("{")), None)
    if line is None:
        raise RuntimeError(f"re-attach probe printed no verdict (rc={proc.returncode}): {err.decode()[-600:]}")
    return json.loads(line)


def _credentials():
    """ADC, or ADC impersonating ``IMPERSONATE`` (the control plane then runs as that account)."""
    if not IMPERSONATE:
        return None
    import google.auth
    from google.auth import impersonated_credentials

    source, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    return impersonated_credentials.Credentials(
        source_credentials=source, target_principal=IMPERSONATE,
        target_scopes=["https://www.googleapis.com/auth/cloud-platform"], lifetime=3600,
    )


CREDS = _credentials()

TASK = 'Run `python3 -c "print(6 * 7)"` in the shell and reply with just the number it prints.'
JSON_TASK = (
    'Run `python3 -c "print(6 * 7)"` in the shell, then reply with a JSON object of the '
    'form {"answer": <the number it printed>}.'
)
STEER_TASK = (
    "First run `sleep 25` in the shell (a single Bash call; do not skip it). Then reply with "
    "one line: the word FOLLOWUP followed by any extra instruction you received while waiting, "
    "or NONE if you received none."
)
ANSWER_SCHEMA = {"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"]}
# What a token can reach in OUR project: every one of these is outside the model account's
# predict-only role, so the answer must be 401/403 for the loopback model token and for
# whatever the platform's own metadata server hands out. Statuses only, never tokens.
_PROBE_PRELUDE = r"""
import json, os, urllib.request as u
def get(url, hdr=None):
    try:
        r = u.urlopen(u.Request(url, headers=hdr or {}), timeout=6)
        return r.status, r.read().decode()
    except Exception as e:
        return getattr(e, "code", type(e).__name__), ""
def reach(tok):
    h = {"Authorization": f"Bearer {tok}"}
    return {
        "gcs_list": str(get("https://storage.googleapis.com/storage/v1/b/%(bucket)s/o?maxResults=1", h)[0]),
        "vertex_engines": str(get("https://%(location)s-aiplatform.googleapis.com/v1/projects/%(project)s/locations/%(location)s/reasoningEngines", h)[0]),
        "iam_accounts": str(get("https://iam.googleapis.com/v1/projects/%(project)s/serviceAccounts", h)[0]),
    }
def token_from(host):
    mds = f"http://{host}/computeMetadata/v1/instance/service-accounts/default/"
    st, body = get(mds + "token", {"Metadata-Flavor": "Google"})
    tok = json.loads(body).get("access_token", "") if st == 200 and body.startswith("{") else ""
    return st, tok
"""

ISOLATION_SCRIPT = r"""
python3 - <<'PY'
%(prelude)s
out = {}
mds = "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/"
st, email = get(mds + "email", {"Metadata-Flavor": "Google"})
out["mds_identity"] = f"{st} {email.strip()[-40:]}" if st == 200 else str(st)
st, tok = token_from("169.254.169.254")
out["mds_token"] = f"{st} len={len(tok)}"
if tok:
    out.update(reach(tok))
out["env_token_names"] = sorted(k for k in os.environ if "TOKEN" in k or "KEY" in k)
print("ISOLATION " + json.dumps(out))
PY
""".strip()

# Run through ``session.exec()`` while a turn runs: the loopback metadata server's token is
# the one the agent's CLI uses for its model calls (``%(host)s`` from the worker's /health).
MODEL_TOKEN_SCRIPT = r"""
python3 - <<'PY'
%(prelude)s
out = {}
st, tok = token_from("%(host)s")
out["loopback_token"] = f"{st} len={len(tok)}"
if tok:
    out.update(reach(tok))
    st, body = get("https://oauth2.googleapis.com/tokeninfo?access_token=" + tok)
    info = json.loads(body) if st == 200 and body.startswith("{") else {}
    out["token_email"] = str(info.get("email") or "-")
    out["token_expires_in_s"] = str(info.get("expires_in") or "-")
print("MODEL-TOKEN " + json.dumps(out))
PY
""".strip()


def _probe_script(template: str, **fields: str) -> str:
    prelude = _PROBE_PRELUDE % {"bucket": OUTPUT_BUCKET.removeprefix("gs://"), "project": PROJECT, "location": LOCATION}
    return template % {"prelude": prelude, **fields}


def _refused(facts: dict) -> bool:
    """Every non-predict probe in ``facts`` was refused (or, for a probe that could not run, absent)."""
    return all(facts.get(k) in ("401", "403", None) for k in ("gcs_list", "vertex_engines", "iam_accounts"))
MARKER = "SESSION-CONFIG-OK"


def log(label: str, msg: str) -> None:
    print(f"[{label}] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def _spec() -> AgentSpec:
    return AgentSpec(name=NAME, model="claude-haiku-4-5", checkpoint=True, max_turns=8, max_budget_usd=1.0)


async def _drive(label: str, run, *, on_event=None):
    """Stream a run to completion; return (result, all events, first-event latency)."""
    t0 = time.time()
    events, first = [], None
    async for ev in run:
        first = first if first is not None else time.time() - t0
        events.append(ev)
        summary = " ".join((ev.summary or "").split())[:90]
        log(label, f"+{time.time() - t0:5.1f}s {ev.kind:11} {summary}")
        if on_event is not None:
            await on_event(ev)
    r = run.result
    text = " ".join((r.text or "").split())
    log(label, f"RESULT after {time.time() - t0:.1f}s (first event {first}): text={text[:100]!r} "
               f"turns={r.num_turns} cost={'?' if r.cost_usd is None else f'${r.cost_usd:.4f}'} "
               f"error={r.is_error} structured={r.structured_output!r}")
    assert sum(e.kind == "result" for e in events) == 1, "exactly one result event"
    assert len({(e.raw or {}).get("turn_id") for e in events}) == 1, "one turn id on every event"
    return r, events, first


def _ok(r, needle="42") -> bool:
    return (not r.is_error) and bool(r.num_turns) and needle in " ".join((r.text or "").split())


async def check_pool_turn(engine, verdicts) -> None:
    label = "pool-turn"
    try:
        session = engine.start_session()
        r, events, first = await _drive(label, session.run(TASK))
        started = next(e for e in events if (e.raw or {}).get("event") == "turn_started")
        verdicts[label] = _ok(r) and started.raw.get("warm") is True and first is not None and first < 4.0
        verdicts["pool-latency"] = first is not None and first < 4.0
        log(label, f"warm={started.raw.get('warm')} sandbox={started.raw.get('worker')}")
        # In-sandbox CPU/RAM sampling: the peak rides the result, the samples sit in the mirror only.
        rows = await asyncio.to_thread(session.resource_samples)
        streamed = [e for e in events if (e.raw or {}).get("event") == "resource_sample"]
        log(label, f"resources={r.resources} samples={len(rows)} first={rows[0] if rows else None} "
                   f"samples on the live stream={len(streamed)} (must be 0)")
        verdicts["resources"] = (bool(r.resources) and "memory_peak_bytes" in r.resources
                                 and "memory_limit_bytes" in r.resources and len(rows) >= 1 and not streamed
                                 and all(row["time"] is not None for row in rows))
    except Exception:
        log(label, "FAILED:\n" + traceback.format_exc())
        verdicts[label] = False


async def check_configs(verdicts) -> None:
    engine = await asyncio.to_thread(gemini.get_engine, NAME, PROJECT, LOCATION, warm_pool=False,
                                       credentials=CREDS)
    label = "session-config"
    try:
        session = engine.start_session(config=SessionConfig(
            system_prompt=SystemPrompt.inherit(append=f"CRITICAL: end your final reply with the exact token {MARKER}.")))
        r, events, _ = await _drive(label, session.run(TASK))
        echoes = [e.raw for e in events if (e.raw or {}).get("event") == "effective_spec"]
        verdicts[label] = _ok(r) and MARKER in (r.text or "") and any(
            e.get("session_config_gcs") and MARKER in str(e["spec"].get("system_prompt")) for e in echoes)
        first_turn = echoes[0]["turn_id"]
    except Exception:
        log(label, "FAILED:\n" + traceback.format_exc())
        verdicts[label] = False
        return
    label = "turn-config"
    try:
        session = engine.get_session(session.session_id)  # re-attach, as an app would
        r, events, _ = await _drive(label, session.send(
            JSON_TASK, config=TurnConfig(output_schema=ANSWER_SCHEMA, reasoning_effort="low")))
        echoes = [e.raw for e in events if (e.raw or {}).get("event") == "effective_spec"]
        ready = next(e.raw for e in events if (e.raw or {}).get("event") == "workspace_ready")
        verdicts[label] = ((not r.is_error) and bool(r.num_turns) and r.structured_output == {"answer": 42}
                           and bool(echoes) and echoes[0]["turn_id"] != first_turn
                           and any(e.get("turn_config_gcs") and e.get("session_config_gcs")
                                   and e["spec"].get("reasoning_effort") == "low"
                                   and MARKER in str(e["spec"].get("system_prompt")) for e in echoes))
        verdicts["resume-restored"] = ready.get("restored") is True
        log(label, f"workspace restored on resume: {ready.get('restored')}")
    except Exception:
        log(label, "FAILED:\n" + traceback.format_exc())
        verdicts[label] = False


async def check_steer(engine, verdicts) -> None:
    label = "steer"
    try:
        session = engine.start_session()
        sent = {"done": False}

        async def on_event(ev):
            if not sent["done"] and (ev.raw or {}).get("event") == "tool_use" or (
                    not sent["done"] and ev.kind == "tool_use"):
                await asyncio.sleep(2.0)
                session.send("EXTRA INSTRUCTION: mention the word PINEAPPLE in your reply.", message_id="steer-1")
                sent["done"] = True
                log(label, "steer sent")

        r, events, _ = await _drive(label, session.run(STEER_TASK), on_event=on_event)
        acked = any(e.kind == "user" and (e.raw or {}).get("message_id") == "steer-1" for e in events)
        verdicts[label] = (not r.is_error) and sent["done"] and acked and "PINEAPPLE" in (r.text or "").upper()
        log(label, f"acked={acked} sent={sent['done']}")
    except Exception:
        log(label, "FAILED:\n" + traceback.format_exc())
        verdicts[label] = False


async def check_early_steer(engine, verdicts) -> None:
    """A send() during the dispatch window (before any event) is queued and delivered."""
    label = "early-steer"
    try:
        session = engine.start_session()
        run = session.run(STEER_TASK)
        session.send("EXTRA INSTRUCTION: mention the word MANGO in your reply.", message_id="early-1")
        queued = bool(session._pending_control) and session._sandbox is None
        log(label, f"sent before the first event; queued={queued}")
        r, events, first = await _drive(label, run)
        acked = any(e.kind == "user" and (e.raw or {}).get("message_id") == "early-1" for e in events)
        ready_at = next((i for i, e in enumerate(events) if (e.raw or {}).get("event") == "control_ready"), None)
        ack_at = next((i for i, e in enumerate(events) if e.kind == "user"), None)
        verdicts[label] = (not r.is_error) and queued and acked and "MANGO" in (r.text or "").upper() \
            and r.warning is None
        log(label, f"acked={acked} control_ready_at={ready_at} ack_at={ack_at} warning={r.warning!r}")
    except Exception:
        log(label, "FAILED:\n" + traceback.format_exc())
        verdicts[label] = False


async def check_reattach_control(engine, verdicts) -> None:
    """send() and interrupt() from a fresh process that re-attaches by id: the steer is
    acknowledged on the owner's stream and shapes the reply; the interrupt ends the owner's
    turn as INTERRUPTED (through the worker, not the cancel fallback)."""
    label = "reattach-steer"
    try:
        session = engine.start_session()
        fired: dict[str, dict] = {}

        async def on_steer(ev):
            if "steer" not in fired and ev.kind == "tool_use":
                t0 = time.time()
                fired["steer"] = await reattach_probe(
                    session.session_id, "steer",
                    "EXTRA INSTRUCTION: mention the word KIWI in your reply.", "reattach-steer-1")
                log(label, f"steer from a fresh process after {time.time() - t0:.1f}s: {fired['steer']}")

        r, events, _ = await _drive(label, session.run(STEER_TASK), on_event=on_steer)
        acked = any(e.kind == "user" and (e.raw or {}).get("message_id") == "reattach-steer-1" for e in events)
        verdicts[label] = ((not r.is_error) and fired.get("steer", {}).get("adopted") is True and acked
                           and "KIWI" in (r.text or "").upper())
        log(label, f"acked={acked} adopted={fired.get('steer', {}).get('adopted')}")
    except Exception:
        log(label, "FAILED:\n" + traceback.format_exc())
        verdicts[label] = False
    label = "reattach-interrupt"
    try:
        session = engine.start_session()
        fired = {}

        async def on_interrupt(ev):
            if "stop" not in fired and ev.kind == "tool_use":
                t0 = time.time()
                fired["stop"] = await reattach_probe(session.session_id, "interrupt")
                log(label, f"interrupt from a fresh process returned after {time.time() - t0:.1f}s: {fired['stop']}")

        r, events, _ = await _drive(label, session.run(STEER_TASK), on_event=on_interrupt)
        probe = fired.get("stop", {})
        verdicts[label] = (session.stop_reason == StopReason.INTERRUPTED and not r.is_error
                           and probe.get("stop_reason") == "interrupted" and probe.get("warning") is None)
        log(label, f"owner stop_reason={session.stop_reason} probe={probe}")
    except Exception:
        log(label, "FAILED:\n" + traceback.format_exc())
        verdicts[label] = False


async def check_exec(engine, verdicts) -> None:
    """Session.exec(): a probe before dispatch waits for the worker; mid-turn it sees the
    agent's files in the agent's cwd; after the turn it raises ControlUnavailable."""
    label = "exec"
    try:
        from agent_run import ControlUnavailable

        session = engine.start_session()
        run = session.run("Create a file named probe.txt containing exactly the word PAPAYA in your "
                          "working directory, then wait 20 seconds (sleep 20), then reply with just 'done'.")
        t0 = time.time()
        early = await session.exec("pwd", timeout=30)  # no sandbox yet: waits for dispatch
        log(label, f"early exec after {time.time() - t0:.1f}s: rc={early.returncode} "
                   f"cwd={early.stdout.strip()!r} stderr={early.stderr[-100:]!r} {early.duration_ms}ms")
        seen = None
        while not run.done:
            r = await session.exec("cat probe.txt 2>/dev/null; ls", timeout=30)
            if "PAPAYA" in r.stdout:
                seen = time.time() - t0
                log(label, f"probe.txt seen mid-turn after {seen:.1f}s: {' '.join(r.stdout.split())[:80]!r} "
                           f"({r.duration_ms}ms)")
                break
            await asyncio.sleep(2)
        # A fresh process re-attaches by id while the turn runs: same sandbox, same file.
        t1 = time.time()
        other = await reattach_probe(session.session_id, "exec", "cat probe.txt 2>/dev/null; pwd")
        log(label, f"re-attached process probe after {time.time() - t1:.1f}s: {other}")
        r, _events, _ = await _drive(label, run)
        try:
            await session.exec("pwd")
            after = "answered (unexpected)"
        except ControlUnavailable as exc:
            after = f"ControlUnavailable: {exc}"
        log(label, f"after the turn: {after}")
        other_after = await reattach_probe(session.session_id, "exec", "pwd")
        log(label, f"re-attached process after the turn: {other_after}")
        verdicts[label] = (early.ok and early.stdout.strip().endswith("/workspace") and seen is not None
                           and not r.is_error and after.startswith("ControlUnavailable"))
        verdicts["exec-reattach"] = (other.get("ok") is True and "PAPAYA" in other["stdout"]
                                     and other["stdout"].strip().endswith("/workspace")
                                     and "unavailable" in other_after)
    except Exception:
        log(label, "FAILED:\n" + traceback.format_exc())
        verdicts[label] = False


async def check_isolation(engine, verdicts) -> None:
    """What the agent's shell can reach — asked of the worker's /exec directly (no model)."""
    label = "isolation"
    import json as _json

    sandbox = None
    try:
        handle = await asyncio.to_thread(engine._create_sandbox, 600)
        sandbox = handle.name
        out = await asyncio.to_thread(
            engine._provider().call, sandbox, "/exec",
            {"command": _probe_script(ISOLATION_SCRIPT), "timeout": 60}, timeout_s=90)
        line = next((ln for ln in (out.get("stdout") or "").splitlines() if ln.startswith("ISOLATION ")), "")
        facts = _json.loads(line[len("ISOLATION "):]) if line else {}
        log(label, f"rc={out.get('returncode')} facts={facts} stderr={(out.get('stderr') or '')[-120:]!r}")
        identity = str(facts.get("mds_identity", ""))
        verdicts[label] = (
            bool(facts)
            and "gserviceaccount.com" not in identity  # a tenant identity, not one of ours
            and _refused(facts)
            and "ANTHROPIC_AUTH_TOKEN" not in facts.get("env_token_names", [])  # nothing outside a turn
        )
    except Exception:
        log(label, "FAILED:\n" + traceback.format_exc())
        verdicts[label] = False
    finally:
        if sandbox is not None:
            await asyncio.to_thread(engine._release_sandbox, sandbox)


async def check_model_token(engine, verdicts) -> None:
    """The token the agent's model calls carry (the worker's loopback metadata server) is
    good for the model — the turn answers — and for nothing else in the project."""
    label = "model-token"
    import json as _json

    try:
        session = engine.start_session()
        run = session.run("Run `sleep 20` in the shell (a single Bash call; do not skip it), then run "
                          '`python3 -c "print(6 * 7)"` and reply with just the number it prints.')
        await session.exec("true", timeout=30)  # waits for the dispatch: the worker is reachable now
        host = None
        for _ in range(20):  # the worker binds its metadata server as the turn starts
            health = await asyncio.to_thread(engine._provider().call, session._sandbox, "/health", {}, timeout_s=30)
            host = health.get("metadata_host")
            if host or run.done:
                break
            await asyncio.sleep(0.5)
        log(label, f"worker metadata server at {host!r} (running turn {str(health.get('running_turn'))[:8]})")
        facts = {}
        if host:
            probe = await session.exec(_probe_script(MODEL_TOKEN_SCRIPT, host=host), timeout=60)
            line = next((ln for ln in probe.stdout.splitlines() if ln.startswith("MODEL-TOKEN ")), "")
            facts = _json.loads(line[len("MODEL-TOKEN "):]) if line else {}
            log(label, f"rc={probe.returncode} facts={facts} stderr={probe.stderr[-120:]!r}")
        r, _events, _ = await _drive(label, run)
        fetched = str(facts.get("loopback_token", "")).startswith("200 len=") and facts.get("loopback_token") != "200 len=0"
        verdicts[label] = _ok(r) and fetched and _refused(facts)
        if fetched and not _refused(facts):
            log(label, f"the model token reaches more than the model — {MODEL_SA} holds more than the "
                       "predict-only role (run `agent-run-gcp-setup --check`)")
    except Exception:
        log(label, "FAILED:\n" + traceback.format_exc())
        verdicts[label] = False


PROC_PROBE = (
    "echo '--- meminfo'; head -3 /proc/meminfo; echo '--- cpus'; nproc; grep -c ^processor /proc/cpuinfo; "
    "echo '--- cgroup'; cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.current /sys/fs/cgroup/cpu.max "
    "/sys/fs/cgroup/cpu.stat 2>&1 | head -12; ls /sys/fs/cgroup 2>&1 | head -5; "
    "echo '--- stat'; head -1 /proc/stat; cat /proc/loadavg; echo '--- self'; grep -E 'VmRSS|VmHWM' /proc/self/status; "
    "echo '--- ps'; ps -eo pid,rss,pcpu,comm --sort=-rss 2>&1 | head -6"
)


async def check_long(engine, verdicts) -> None:
    label = "long-turn"
    secs = int(LONG_MINUTES * 60)
    try:
        session = engine.start_session()
        if TOKEN_LIFETIME_S:
            log(label, f"model token lifetime {TOKEN_LIFETIME_S}s, refresh every {TOKEN_LIFETIME_S * 0.4:.0f}s; "
                       f"the turn sleeps {secs}s, so the CLI's second model call needs a refreshed token")
        started = time.time()
        driving = asyncio.ensure_future(_drive(label, session.run(
            f'Run `sleep {secs} && python3 -c "print(6 * 7)"` in the shell as ONE Bash call (it takes '
            f"{LONG_MINUTES:g} minutes; wait for it) and reply with just the number it prints.",
            config=TurnConfig(max_turns=6))))
        await asyncio.sleep(min(60, secs / 3))
        try:
            probe = await session.exec(PROC_PROBE, timeout=30)
            log(label, f"in-container resource files (rc={probe.returncode}):\n{probe.stdout[:3000]}"
                       + (f"\nstderr: {probe.stderr[:500]}" if probe.stderr else ""))
        except Exception as exc:  # noqa: BLE001 — the probe is informational
            log(label, f"resource probe failed: {exc!r}")
        r, events, _ = await driving
        elapsed = time.time() - started
        rows = await asyncio.to_thread(session.resource_samples)
        log(label, f"resources={r.resources}; {len(rows)} samples in the mirror"
                   + (f", last {rows[-1]}" if rows else ""))
        ok = _ok(r) and "42" in (r.text or "")
        if TOKEN_LIFETIME_S:
            ok = ok and elapsed > TOKEN_LIFETIME_S
            log(label, f"turn took {elapsed:.0f}s vs token lifetime {TOKEN_LIFETIME_S}s; "
                       f"model token expiry on the client now {session._model_token_expiry}")
        verdicts[label] = ok
    except Exception:
        log(label, "FAILED:\n" + traceback.format_exc())
        verdicts[label] = False


async def main() -> int:
    verdicts: dict[str, bool] = {}
    engine = None
    try:
        log("deploy", f"deploying {NAME} (warm_pool=True, pool_size=1) as "
                      f"{IMPERSONATE or 'the ADC principal'}; model token from {MODEL_SA} ...")
        t0 = time.time()
        spec = _spec()
        if LONG_MINUTES:
            spec = AgentSpec(**{**spec.to_dict(), "env": {"BASH_DEFAULT_TIMEOUT_MS": str(int(LONG_MINUTES * 60 * 1000) + 120_000),
                                                       "BASH_MAX_TIMEOUT_MS": str(int(LONG_MINUTES * 60 * 1000) + 120_000)}})
        engine = await asyncio.to_thread(
            gemini.deploy, spec, PROJECT, LOCATION, warm_pool=True, pool_size=1,
            image_repo=IMAGE_REPO, model_service_account=MODEL_SA, output_bucket=OUTPUT_BUCKET,
            log=lambda m: log("deploy", m), credentials=CREDS, resource_limits=RESOURCE_LIMITS,
        )
        log("deploy", f"deployed in {time.time() - t0:.0f}s: version={engine.version} image={engine.revisions()[0]['image']}")
        verdicts["deploy"] = True
        verdicts["warm"] = await asyncio.to_thread(engine.wait_until_warm, 60)
        log("deploy", f"warm={verdicts['warm']}")
        await check_pool_turn(engine, verdicts)
        await asyncio.gather(check_configs(verdicts), check_steer(engine, verdicts),
                             check_early_steer(engine, verdicts), check_exec(engine, verdicts),
                             check_reattach_control(engine, verdicts), check_isolation(engine, verdicts),
                             check_model_token(engine, verdicts))
        if LONG_MINUTES:
            await check_long(engine, verdicts)
    except Exception:
        log("main", "FAILED:\n" + traceback.format_exc())
        verdicts.setdefault("deploy", False)
    finally:
        if engine is not None and not os.environ.get("KEEP"):
            try:
                await asyncio.to_thread(engine.delete)
                log("teardown", "engine deleted (templates + sandboxes)")
            except Exception:
                log("teardown", "FAILED — delete by hand: gemini.get_engine(...).delete()\n" + traceback.format_exc())
    print("\n=== VERDICTS ===", flush=True)
    for check, ok in verdicts.items():
        print(f"  {check}: {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if verdicts and all(verdicts.values()) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

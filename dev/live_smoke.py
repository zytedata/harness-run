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
  * **isolation**: a fixed shell script run through the worker's ``/exec`` (no model) prints
    what the agent's shell can reach — the metadata server's identity must be a tenant one
    that is 403 on our project's storage and Vertex (the sandbox has no usable Google
    identity); only statuses and names are printed, never tokens
  * optional ``LONG_MINUTES=n``: a turn whose one Bash call sleeps that long (the model
    token, the sandbox TTL and the /events long-poll all have to hold)

Pass criteria per check: terminal result with ``error=False``, ``turns > 0``, the expected
answer in the text (plus the check's own assertions). The engine (templates + sandboxes)
is deleted in ``finally``; exit code is non-zero if any check fails.

Configure via env (defaults are the shared my-project test setup):
  PROJECT, LOCATION, IMAGE_REPO, MODEL_SA, SUFFIX (engine-name suffix; defaults to your
  username), LONG_MINUTES, KEEP=1 (skip teardown), IMPERSONATE=<service account email>
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
import os
import re
import sys
import time
import traceback

from remote_agent_toolkit import AgentSpec, SessionConfig, SystemPrompt, TurnConfig, gemini

PROJECT = os.environ.get("PROJECT", "my-project")
LOCATION = os.environ.get("LOCATION", "us-central1")
IMAGE_REPO = os.environ.get("IMAGE_REPO") or f"{LOCATION}-docker.pkg.dev/{PROJECT}/ratk-sandbox"
MODEL_SA = os.environ.get("MODEL_SA", "agent-runtime@my-project.iam.gserviceaccount.com")
SUFFIX = re.sub(r"[^a-z0-9-]", "-", (os.environ.get("SUFFIX") or getpass.getuser()).lower())
LONG_MINUTES = float(os.environ.get("LONG_MINUTES", "0") or 0)
NAME = f"ratk-smoke-{SUFFIX}"
IMPERSONATE = os.environ.get("IMPERSONATE") or None


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
ISOLATION_SCRIPT = r"""
python3 - <<'PY'
import json, urllib.request as u
def get(url, hdr=None):
    try:
        r = u.urlopen(u.Request(url, headers=hdr or {}), timeout=6)
        return r.status, r.read().decode()
    except Exception as e:
        return getattr(e, "code", type(e).__name__), ""
out = {}
mds = "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/"
st, email = get(mds + "email", {"Metadata-Flavor": "Google"})
out["mds_identity"] = f"{st} {email.strip()[-40:]}" if st == 200 else str(st)
st, body = get(mds + "token", {"Metadata-Flavor": "Google"})
tok = json.loads(body).get("access_token", "") if st == 200 and body.startswith("{") else ""
out["mds_token"] = f"{st} len={len(tok)}"
if tok:
    h = {"Authorization": f"Bearer {tok}"}
    out["gcs_list"] = str(get("https://storage.googleapis.com/storage/v1/b/my-project-agent-output/o?maxResults=1", h)[0])
    out["vertex"] = str(get("https://aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/reasoningEngines", h)[0])
import os
out["env_token_names"] = sorted(k for k in os.environ if "TOKEN" in k or "KEY" in k)
print("ISOLATION " + json.dumps(out))
PY
""".strip()
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


async def check_isolation(engine, verdicts) -> None:
    """What the agent's shell can reach — asked of the worker's /exec directly (no model)."""
    label = "isolation"
    import json as _json

    sandbox = None
    try:
        handle = await asyncio.to_thread(engine._create_sandbox, 600)
        sandbox = handle.name
        out = await asyncio.to_thread(
            engine._provider().call, sandbox, "/exec", {"command": ISOLATION_SCRIPT, "timeout": 60}, timeout_s=90)
        line = next((ln for ln in (out.get("stdout") or "").splitlines() if ln.startswith("ISOLATION ")), "")
        facts = _json.loads(line[len("ISOLATION "):]) if line else {}
        log(label, f"rc={out.get('returncode')} facts={facts} stderr={(out.get('stderr') or '')[-120:]!r}")
        identity = str(facts.get("mds_identity", ""))
        verdicts[label] = (
            bool(facts)
            and "gserviceaccount.com" not in identity  # a tenant identity, not one of ours
            and facts.get("gcs_list") in ("403", "401", None) and facts.get("vertex") in ("403", "401", None)
            and "ANTHROPIC_AUTH_TOKEN" not in facts.get("env_token_names", [])  # nothing outside a turn
        )
    except Exception:
        log(label, "FAILED:\n" + traceback.format_exc())
        verdicts[label] = False
    finally:
        if sandbox is not None:
            await asyncio.to_thread(engine._release_sandbox, sandbox)


async def check_long(engine, verdicts) -> None:
    label = "long-turn"
    secs = int(LONG_MINUTES * 60)
    try:
        session = engine.start_session()
        r, events, _ = await _drive(label, session.run(
            f'Run `sleep {secs} && python3 -c "print(6 * 7)"` in the shell as ONE Bash call (it takes '
            f"{LONG_MINUTES:g} minutes; wait for it) and reply with just the number it prints.",
            config=TurnConfig(max_turns=6)))
        verdicts[label] = _ok(r)
    except Exception:
        log(label, "FAILED:\n" + traceback.format_exc())
        verdicts[label] = False


async def main() -> int:
    verdicts: dict[str, bool] = {}
    engine = None
    try:
        log("deploy", f"deploying {NAME} (warm_pool=True, pool_size=1) as "
                      f"{IMPERSONATE or 'the ADC principal'} ...")
        t0 = time.time()
        spec = _spec()
        if LONG_MINUTES:
            spec = AgentSpec(**{**spec.to_dict(), "env": {"BASH_DEFAULT_TIMEOUT_MS": str(int(LONG_MINUTES * 60 * 1000) + 120_000),
                                                       "BASH_MAX_TIMEOUT_MS": str(int(LONG_MINUTES * 60 * 1000) + 120_000)}})
        engine = await asyncio.to_thread(
            gemini.deploy, spec, PROJECT, LOCATION, warm_pool=True, pool_size=1,
            image_repo=IMAGE_REPO, model_service_account=MODEL_SA, log=lambda m: log("deploy", m),
            credentials=CREDS,
        )
        log("deploy", f"deployed in {time.time() - t0:.0f}s: version={engine.version} image={engine.revisions()[0]['image']}")
        verdicts["deploy"] = True
        verdicts["warm"] = await asyncio.to_thread(engine.wait_until_warm, 60)
        log("deploy", f"warm={verdicts['warm']}")
        await check_pool_turn(engine, verdicts)
        await asyncio.gather(check_configs(verdicts), check_steer(engine, verdicts),
                             check_early_steer(engine, verdicts), check_isolation(engine, verdicts))
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

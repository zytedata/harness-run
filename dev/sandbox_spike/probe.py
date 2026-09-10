"""Live probe: the toolkit harness inside an Agent Sandbox custom container.

Answers the go/no-go questions of the sandbox spike against the numbers of the warm pool
(dev/live_warm_latency_probe.py: dispatch -> first event 4.2 s, -> result 14.5 s):

1. template create + sandbox create -> first successful HTTP answer, with OUR image
2. what the container sees (user, writable dirs, claude binary, cpu/mem, metadata reach)
3. a ready sandbox: dispatch -> first event / -> result for the warm probe's one-tool turn
4. a fresh sandbox: create -> result end to end (the no-ready-pool path)
5. refill: create -> ready again after the pool was used
6. optional: idle staleness (--idle-minutes) and whether exec resets the TTL (--ttl-check)

Needs google-cloud-agentplatform>=2.1 (the shell/custom-container surface; the toolkit
itself still pins aiplatform<2) — run it from a separate venv, see README.md. Model calls
run on a token minted for --model-sa (impersonation via ADC), handed to the container
through env; the sandbox itself has no Google identity. Prints statuses, names and
timings only. Everything created is deleted in `finally` unless --keep.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback

import agentplatform
import google.auth
import google.auth.transport.requests as tr
from agentplatform._genai import types

TASK = 'Run `python3 -c "print(6 * 7)"` in the shell and reply with just the number it prints.'
SPEC = {"name": "ratk-sandbox-spike", "model": "claude-haiku-4-5", "max_turns": 8, "max_budget_usd": 1.0}

ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True, help="Artifact Registry image URI")
ap.add_argument("--project", default="my-project")
ap.add_argument("--location", default="us-central1")
ap.add_argument("--vertex-region", default="global")
ap.add_argument("--model-sa", default="agent-runtime@my-project.iam.gserviceaccount.com")
ap.add_argument("--cpu", default="4")
ap.add_argument("--memory", default="8Gi")
ap.add_argument("--pool", type=int, default=2, help="ready sandboxes to pre-create")
ap.add_argument("--turns", type=int, default=2, help="turns to run on ready sandboxes")
ap.add_argument("--idle-minutes", type=float, default=0.0)
ap.add_argument("--ttl-check", action="store_true", help="~9 min: does exec reset the TTL?")
ap.add_argument("--instance", default=None, help="reuse an existing Agent Platform instance")
ap.add_argument("--template", default=None, help="reuse an existing template")
ap.add_argument("--keep", action="store_true")
args = ap.parse_args()

client = agentplatform.Client(project=args.project, location=args.location)
SB = client.sandboxes
T0 = time.time()


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} +{time.time() - T0:6.1f}s  {msg}", flush=True)


def call(sb: str, path: str, body: dict | None = None) -> dict:
    """POST `body` to our server inside the sandbox through the platform's execute proxy."""
    resp = SB._execute_code(
        name=sb,
        inputs=[
            types.Chunk(mime_type="application/x.sandbox-request-uri", data=path.encode()),
            types.Chunk(mime_type="application/x.sandbox-request-port", data=b"8080"),
            types.Chunk(mime_type="application/json", data=json.dumps(body or {}).encode()),
        ],
    )
    for out in resp.outputs or []:
        if out.data:
            return json.loads(out.data.decode("utf-8"))
    raise RuntimeError("empty response from sandbox")


def wait_ready(sb: str, timeout: float = 300.0) -> tuple[float, int, str]:
    """Poll /health until it answers: (seconds, attempts, last_error)."""
    t0, n, last = time.time(), 0, ""
    while time.time() - t0 < timeout:
        n += 1
        try:
            h = call(sb, "/health")
            if h.get("ok"):
                return time.time() - t0, n, f"uptime={h.get('uptime_s')}s uid={h.get('uid')} cwd_w={h.get('cwd_writable')} home_w={h.get('home_writable')}"
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {str(exc)[:90]}"
        time.sleep(1.0)
    raise TimeoutError(f"sandbox never answered /health: {last}")


def mint_model_token(lifetime_s: int = 3600) -> str:
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    s = tr.AuthorizedSession(creds)
    r = s.post(
        f"https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/{args.model_sa}:generateAccessToken",
        json={"scope": ["https://www.googleapis.com/auth/cloud-platform"], "lifetime": f"{lifetime_s}s"},
    )
    r.raise_for_status()
    return r.json()["accessToken"]


def run_turn(sb: str, label: str) -> dict:
    """Dispatch the warm probe's turn to `sb`; return timings (dispatch -> first event/result)."""
    env = {
        "CLAUDE_CODE_USE_VERTEX": "1",
        "CLAUDE_CODE_SKIP_VERTEX_AUTH": "1",
        "ANTHROPIC_AUTH_TOKEN": mint_model_token(),
        "ANTHROPIC_VERTEX_PROJECT_ID": args.project,
        "CLOUD_ML_REGION": args.vertex_region,
    }
    turn_id = f"t{int(time.time() * 1000)}"
    t0 = time.time()
    acc = call(sb, "/turn", {"turn_id": turn_id, "spec": SPEC, "prompt": TASK, "env": env})
    if "turn_id" not in acc:
        raise RuntimeError(f"turn not accepted: {acc}")
    dispatch_s = time.time() - t0
    first = None
    since = 0
    while True:
        ev = call(sb, "/events", {"turn_id": turn_id, "since": since})
        for e in ev["events"]:
            first = first or time.time() - t0
            log(f"    [{label}] +{time.time() - t0:5.1f}s {e['kind']:11} {e['summary'][:70]}")
        since = ev["next"]
        if ev["done"]:
            break
        time.sleep(0.1)
    total = time.time() - t0
    res = ev.get("result") or {}
    text = " ".join((res.get("text") or "").split())
    ok = (not res.get("is_error")) and "42" in text and not ev.get("error")
    if ev.get("error"):
        log(f"    [{label}] ERROR: {ev['error'][-600:]}")
    out = {"label": label, "dispatch_s": round(dispatch_s, 2), "first_event_s": None if first is None else round(first, 2),
           "result_s": round(total, 2), "ok": ok, "cost_usd": res.get("cost_usd"), "num_turns": res.get("num_turns")}
    log(f"  MEASURED [{label}] dispatch->first event {out['first_event_s']}s, dispatch->result {out['result_s']}s, ok={ok}")
    return out


created: dict = {"sbs": []}
measurements: list[dict] = []
try:
    instance = args.instance
    if not instance:
        instance = client.runtimes.create().api_resource.name
        created["instance"] = instance
        log(f"instance created: .../{instance.rsplit('/', 1)[-1]}")

    template = args.template
    if not template:
        t0 = time.time()
        op = SB.templates.create(
            name=instance, display_name="ratk-sandbox-spike",
            config={
                "custom_container_environment": {
                    "custom_container_spec": {"image_uri": args.image},
                    "ports": [{"port": 8080, "protocol": "TCP"}],
                    "resources": {"requests": {"cpu": args.cpu, "memory": args.memory},
                                  "limits": {"cpu": args.cpu, "memory": args.memory}},
                },
                "egress_control_config": {"internet_access": True},
                "wait_for_completion": True,
            },
        )
        template = op.response.name
        created["template"] = template
        log(f"template created in {time.time() - t0:.1f}s state={op.response.state}")

    def create_sandbox(label: str, ttl: str = "1800s") -> tuple[str, float, float, str]:
        t0 = time.time()
        op = SB.create(name=instance, config={"display_name": f"ratk-spike-{label}", "sandbox_environment_template": template,
                                              "wait_for_completion": True, "ttl": ttl})
        sb = op.response.name
        created["sbs"].append(sb)
        create_s = time.time() - t0
        ready_s, attempts, info = wait_ready(sb)
        log(f"sandbox {label}: create {create_s:.1f}s, first answer after {ready_s:.1f}s more ({attempts} polls); {info}")
        return sb, create_s, ready_s, info

    # 1+2: the ready pool, and what the container sees
    # Pool sandboxes must outlive an --idle-minutes wait (TTL counts from creation).
    pool_ttl = f"{int(max(1800, args.idle_minutes * 60 + 1800))}s"
    pool: list[str] = []
    for i in range(args.pool):
        sb, *_ = create_sandbox(f"pool{i}", ttl=pool_ttl)
        pool.append(sb)
    if pool:
        checks = {
            "identity": "id -un; id -u; ps -o user= -p 1",
            "claude": "python3 -c \"import claude_agent_sdk,os;print(os.path.join(os.path.dirname(claude_agent_sdk.__file__),'_bundled','claude'))\" | xargs -I{} sh -c '{} --version'",
            "tools": "git --version; uv --version; nproc; free -m | sed -n 2p; df -h /workspace | tail -1",
            "net": "python3 -c \"import urllib.request as u\nfor n,h in [('mds','http://169.254.169.254/computeMetadata/v1/project/project-id'),('vertex','https://us-central1-aiplatform.googleapis.com/'),('pypi','https://pypi.org/simple/pip/')]:\n  try: print(n, u.urlopen(u.Request(h,headers={'Metadata-Flavor':'Google'}),timeout=4).status)\n  except Exception as e: print(n, type(e).__name__, getattr(e,'code',''))\"",
        }
        for k, cmd in checks.items():
            r = call(pool[0], "/exec", {"command": cmd, "timeout": 30})
            log(f"  [{k}] rc={r.get('returncode')} {r.get('duration_ms')}ms :: {' | '.join((r.get('stdout') or '').strip().splitlines())[:300]}"
                + (f" || ERR {' | '.join((r.get('stderr') or '').strip().splitlines())[:160]}" if r.get('stderr') else ""))

    # 3: turns on ready sandboxes
    for i in range(min(args.turns, len(pool))):
        measurements.append(run_turn(pool[i], f"ready{i}"))

    # 5: refill after use
    if args.pool:
        create_sandbox("refill")

    # 4: fresh sandbox, end to end
    t0 = time.time()
    sb, create_s, ready_s, _ = create_sandbox("fresh")
    m = run_turn(sb, "fresh")
    m["create_to_result_s"] = round(time.time() - t0, 2)
    measurements.append(m)
    log(f"  MEASURED [fresh] create->result end to end {m['create_to_result_s']}s")

    # 6: TTL reset check (runs during the idle wait when both are requested) + idle staleness.
    ttl_sb = None
    ttl_t0 = 0.0
    if args.ttl_check:
        ttl_sb, *_ = create_sandbox("ttl", ttl="480s")
        ttl_t0 = time.time()
        log("ttl-check: exec at +420 s, then at +540 s check whether the 480 s-TTL sandbox still exists")

    def ttl_check_steps() -> None:
        """Blocks until the TTL check is done (sleeps in between); no-op without --ttl-check."""
        if ttl_sb is None:
            return
        time.sleep(max(0.0, ttl_t0 + 420 - time.time()))
        try:
            call(ttl_sb, "/health")
            log("ttl-check: exec at +420 s answered")
        except Exception as exc:  # noqa: BLE001
            log(f"ttl-check: exec at +420 s FAILED ({type(exc).__name__}) -> sandbox gone before its TTL?")
        time.sleep(max(0.0, ttl_t0 + 540 - time.time()))
        try:
            st = SB.get(name=ttl_sb).state
            log(f"ttl-check: still present at +540 s, state={st} -> exec RESET the TTL (or expiry is lazy)")
        except Exception as exc:  # noqa: BLE001
            log(f"ttl-check: get failed at +540 s ({type(exc).__name__}) -> exec did NOT reset the TTL")

    if args.idle_minutes > 0 and pool:
        idle_end = time.time() + args.idle_minutes * 60
        log(f"idling {args.idle_minutes} min on {pool[-1].rsplit('/', 1)[-1]} ...")
        ttl_check_steps()
        time.sleep(max(0.0, idle_end - time.time()))
        ready_s, attempts, info = wait_ready(pool[-1], 120)
        log(f"after idle: answered in {ready_s:.1f}s ({attempts} polls); {info}")
        measurements.append(run_turn(pool[-1], "after-idle"))
    else:
        ttl_check_steps()
except Exception:
    log("PROBE FAILED:\n" + traceback.format_exc())
finally:
    print("\nSUMMARY", json.dumps(measurements, indent=1))
    if args.keep:
        log(f"--keep: leaving {json.dumps({k: (len(v) if isinstance(v, list) else v.rsplit('/', 1)[-1]) for k, v in created.items()})}")
    else:
        for sb in created["sbs"]:
            try:
                SB.delete(name=sb)
            except Exception as exc:  # noqa: BLE001
                log(f"delete sandbox: {type(exc).__name__}: {str(exc)[:100]}")
        log(f"deleted {len(created['sbs'])} sandboxes")
        if created.get("template"):
            for attempt in range(8):
                try:
                    SB.templates.delete(name=created["template"])
                    log("deleted template")
                    break
                except Exception as exc:  # noqa: BLE001
                    log(f"delete template attempt {attempt}: {type(exc).__name__}: {str(exc)[:80]}")
                    time.sleep(15)
        if created.get("instance"):
            try:
                client.runtimes.delete(name=created["instance"], force=True)
                log("deleted instance")
            except Exception as exc:  # noqa: BLE001
                log(f"delete instance: {type(exc).__name__}: {str(exc)[:100]}")
    sys.exit(0 if measurements and all(m["ok"] for m in measurements) else 1)

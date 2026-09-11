"""Live probe: the undocumented limits of Agent Sandbox custom containers (DESIGN.md §13.4).

Runs against the spike image (see README.md) and answers, with numbers:

1. TTL ceiling: which ``ttl`` values a create accepts (1 h .. 30 d) and what comes back
2. resources: does an 8 CPU / 16 GiB template (and 16 / 32) provision, and is the memory real
3. disk: size of /workspace, /tmp, $HOME and the write speed of a 2 GiB file
4. execute body limits: request pad and response payload from 100 KB up to 32 MB
5. per-call proxy ceiling: ``/exec sleep N`` for growing N (this bounds the /events long-poll)
6. call rate: a sequential burst and a 10-thread burst of /health (any 429?)
7. concurrent creates: 10 sandboxes at once (quota?), then a listing under the instance
8. long turn (``--long-minutes``): an in-container ticker for N minutes with periodic execs,
   then a harness turn whose Bash tool sleeps 200 s (a turn longer than one proxy call), then a
   normal turn on the same, hour-old sandbox

Needs google-cloud-agentplatform>=2.1 in its own venv. Prints statuses, names and timings
only; every created resource is deleted in ``finally`` unless --keep. Independent groups run
in threads so the whole thing takes ~max(long-minutes, 15 min).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import agentplatform
import google.auth
import google.auth.transport.requests as tr
from agentplatform._genai import types

SPEC = {"name": "ratk-sandbox-spike", "model": "claude-haiku-4-5", "max_turns": 8, "max_budget_usd": 1.0}

ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True)
ap.add_argument("--project", default="my-project")
ap.add_argument("--location", default="us-central1")
ap.add_argument("--vertex-region", default="global")
ap.add_argument("--model-sa", default="agent-runtime@my-project.iam.gserviceaccount.com")
ap.add_argument("--long-minutes", type=float, default=0.0, help="ticker + long harness turn")
ap.add_argument("--sleep-steps", default="30,60,120,300,600", help="per-call ceiling probe")
ap.add_argument("--instance", default=None)
ap.add_argument("--keep", action="store_true")
args = ap.parse_args()

client = agentplatform.Client(project=args.project, location=args.location)
SB = client.sandboxes
T0 = time.time()
_LOG = threading.Lock()
findings: dict[str, object] = {}
created: dict = {"sbs": [], "templates": []}
_CREATED = threading.Lock()


def log(msg: str) -> None:
    with _LOG:
        print(f"{time.strftime('%H:%M:%S')} +{time.time() - T0:6.1f}s  {msg}", flush=True)


def note(key: str, value: object) -> None:
    findings[key] = value
    log(f"FINDING {key}: {value}")


def err_str(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {' '.join(str(exc).split())[:160]}"


def call(sb: str, path: str, body: dict | None = None, timeout_ms: int | None = None) -> dict:
    cfg = {"http_options": {"timeout": timeout_ms}} if timeout_ms else None
    resp = SB._execute_code(
        name=sb,
        inputs=[
            types.Chunk(mime_type="application/x.sandbox-request-uri", data=path.encode()),
            types.Chunk(mime_type="application/x.sandbox-request-port", data=b"8080"),
            types.Chunk(mime_type="application/json", data=json.dumps(body or {}).encode()),
        ],
        config=cfg,
    )
    for out in resp.outputs or []:
        if out.data:
            return json.loads(out.data.decode("utf-8"))
    raise RuntimeError("empty response from sandbox")


def wait_ready(sb: str, timeout: float = 300.0) -> float:
    t0, last = time.time(), ""
    while time.time() - t0 < timeout:
        try:
            if call(sb, "/health").get("ok"):
                return time.time() - t0
        except Exception as exc:  # noqa: BLE001
            last = err_str(exc)
        time.sleep(1.0)
    raise TimeoutError(f"sandbox never answered /health: {last}")


def create_template(label: str, cpu: str, memory: str) -> tuple[str | None, float, str]:
    t0 = time.time()
    try:
        op = SB.templates.create(
            name=INSTANCE, display_name=f"ratk-limits-{label}",
            config={
                "custom_container_environment": {
                    "custom_container_spec": {"image_uri": args.image},
                    "ports": [{"port": 8080, "protocol": "TCP"}],
                    "resources": {"requests": {"cpu": cpu, "memory": memory},
                                  "limits": {"cpu": cpu, "memory": memory}},
                },
                "egress_control_config": {"internet_access": True},
                "wait_for_completion": True,
            },
        )
        name = op.response.name
        with _CREATED:
            created["templates"].append(name)
        log(f"template {label} ({cpu} CPU / {memory}) created in {time.time() - t0:.1f}s state={op.response.state}")
        return name, time.time() - t0, str(op.response.state)
    except Exception as exc:  # noqa: BLE001
        log(f"template {label} ({cpu} CPU / {memory}) FAILED after {time.time() - t0:.1f}s: {err_str(exc)}")
        return None, time.time() - t0, err_str(exc)


def create_sandbox(template: str, label: str, ttl: str = "1800s", wait: bool = True) -> tuple[str, float, object]:
    t0 = time.time()
    op = SB.create(name=INSTANCE, config={"display_name": f"ratk-limits-{label}", "sandbox_environment_template": template,
                                          "wait_for_completion": True, "ttl": ttl})
    sb = op.response.name
    with _CREATED:
        created["sbs"].append(sb)
    create_s = time.time() - t0
    if wait:
        ready_s = wait_ready(sb)
        log(f"sandbox {label}: create {create_s:.1f}s, first answer +{ready_s:.1f}s")
    return sb, create_s, op.response


def delete_sandbox(sb: str) -> None:
    try:
        SB.delete(name=sb)
    except Exception as exc:  # noqa: BLE001
        log(f"delete {sb.rsplit('/', 1)[-1]}: {err_str(exc)}")
    with _CREATED:
        if sb in created["sbs"]:
            created["sbs"].remove(sb)


def mint_model_token(lifetime_s: int = 3600) -> str:
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    s = tr.AuthorizedSession(creds)
    r = s.post(
        f"https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/{args.model_sa}:generateAccessToken",
        json={"scope": ["https://www.googleapis.com/auth/cloud-platform"], "lifetime": f"{lifetime_s}s"},
    )
    r.raise_for_status()
    return r.json()["accessToken"]


def run_turn(sb: str, label: str, prompt: str, extra_env: dict | None = None, poll_s: float = 0.5) -> dict:
    env = {
        "CLAUDE_CODE_USE_VERTEX": "1",
        "CLAUDE_CODE_SKIP_VERTEX_AUTH": "1",
        "ANTHROPIC_AUTH_TOKEN": mint_model_token(),
        "ANTHROPIC_VERTEX_PROJECT_ID": args.project,
        "CLOUD_ML_REGION": args.vertex_region,
        **(extra_env or {}),
    }
    turn_id = f"t{int(time.time() * 1000)}"
    t0 = time.time()
    acc = call(sb, "/turn", {"turn_id": turn_id, "spec": SPEC, "prompt": prompt, "env": env})
    if "turn_id" not in acc:
        raise RuntimeError(f"turn not accepted: {acc}")
    first, since, calls = None, 0, 0
    while True:
        ev = call(sb, "/events", {"turn_id": turn_id, "since": since})
        calls += 1
        for e in ev["events"]:
            first = first or time.time() - t0
            log(f"    [{label}] +{time.time() - t0:6.1f}s {e['kind']:11} {e['summary'][:70]}")
        since = ev["next"]
        if ev["done"]:
            break
        time.sleep(poll_s)
    total = time.time() - t0
    res = ev.get("result") or {}
    text = " ".join((res.get("text") or "").split())
    ok = (not res.get("is_error")) and "42" in text and not ev.get("error")
    if ev.get("error"):
        log(f"    [{label}] ERROR: {ev['error'][-600:]}")
    out = {"first_event_s": None if first is None else round(first, 1), "result_s": round(total, 1), "ok": ok,
           "polls": calls, "cost_usd": res.get("cost_usd"), "text": text[:80]}
    log(f"  MEASURED [{label}] first event {out['first_event_s']}s, result {out['result_s']}s, ok={ok}")
    return out


# ---------------------------------------------------------------- probe groups

def probe_ttl(template: str) -> None:
    out = {}
    for label, ttl in [("1h", 3600), ("1d", 86400), ("7d", 7 * 86400), ("14d", 14 * 86400), ("30d", 30 * 86400)]:
        try:
            sb, create_s, resp = create_sandbox(template, f"ttl-{label}", ttl=f"{ttl}s", wait=False)
            got = SB.get(name=sb)
            out[label] = f"accepted ({create_s:.1f}s) ttl={got.ttl} expire_time={got.expire_time}"
            delete_sandbox(sb)
        except Exception as exc:  # noqa: BLE001
            out[label] = f"REJECTED {err_str(exc)}"
        log(f"  ttl {label}: {out[label]}")
    note("ttl", out)


def probe_resources() -> None:
    out = {}
    for label, cpu, mem in [("16g", "8", "16Gi"), ("32g", "16", "32Gi")]:
        tpl, secs, state = create_template(label, cpu, mem)
        if not tpl:
            out[label] = f"template FAILED: {state}"
            continue
        try:
            sb, create_s, _ = create_sandbox(tpl, f"res-{label}")
            r = call(sb, "/exec", {"command": "nproc; free -g | sed -n 2p; cat /sys/fs/cgroup/memory.max 2>/dev/null || echo no-cgroup", "timeout": 20})
            seen = " | ".join((r.get("stdout") or "").strip().splitlines())
            gib = int(mem[:-2])
            alloc = max(1, gib - 3)
            r2 = call(sb, "/exec", {"command": f"python3 -c \"b=bytearray({alloc}*2**30); print('alloc-ok', len(b)>>30)\"", "timeout": 120}, timeout_ms=200_000)
            out[label] = f"template {secs:.0f}s, sandbox {create_s:.1f}s; sees: {seen}; alloc {alloc} GiB -> rc={r2.get('returncode')} {(r2.get('stdout') or r2.get('stderr') or '').strip()[:80]} ({r2.get('duration_ms')} ms)"
            delete_sandbox(sb)
        except Exception as exc:  # noqa: BLE001
            out[label] = f"template ok ({secs:.0f}s) but sandbox FAILED: {err_str(exc)}"
        log(f"  resources {label}: {out[label]}")
    note("resources", out)


def probe_disk(sb: str) -> None:
    r = call(sb, "/exec", {"command": "df -h /workspace /tmp $HOME / | tail -n +2", "timeout": 20})
    lines = (r.get("stdout") or "").strip().splitlines()
    t0 = time.time()
    r2 = call(sb, "/exec", {"command": "dd if=/dev/zero of=/workspace/big.bin bs=64M count=32 2>&1 | tail -1; rm -f /workspace/big.bin", "timeout": 300}, timeout_ms=400_000)
    note("disk", {"df": lines, "write_2gib": f"rc={r2.get('returncode')} {(r2.get('stdout') or '').strip()[-100:]} ({time.time() - t0:.1f}s round trip)"})


def probe_body(sb: str) -> None:
    out = {}
    for kb in [100, 1024, 4096, 10240, 32768]:
        t0 = time.time()
        try:
            r = call(sb, "/exec", {"command": "echo ok", "pad": "x" * (kb * 1024)}, timeout_ms=300_000)
            out[f"request {kb} KB"] = f"ok rc={r.get('returncode')} ({time.time() - t0:.1f}s)"
        except Exception as exc:  # noqa: BLE001
            out[f"request {kb} KB"] = f"FAILED {err_str(exc)} ({time.time() - t0:.1f}s)"
        log(f"  body {out[f'request {kb} KB']}  [request {kb} KB]")
    for kb in [100, 1024, 4096, 10240, 32768]:
        t0 = time.time()
        try:
            r = call(sb, "/exec", {"command": f"head -c {kb * 1024} /dev/zero | tr '\\0' x", "timeout": 60}, timeout_ms=300_000)
            got = len(r.get("stdout") or "")
            out[f"response {kb} KB"] = f"ok got {got // 1024} KB ({time.time() - t0:.1f}s)"
        except Exception as exc:  # noqa: BLE001
            out[f"response {kb} KB"] = f"FAILED {err_str(exc)} ({time.time() - t0:.1f}s)"
        log(f"  body {out[f'response {kb} KB']}  [response {kb} KB]")
    note("body", out)


def probe_call_ceiling(sb: str) -> None:
    out = {}
    for n in [int(x) for x in args.sleep_steps.split(",") if x]:
        t0 = time.time()
        try:
            r = call(sb, "/exec", {"command": f"sleep {n}; echo done"}, timeout_ms=(n + 120) * 1000)
            out[f"sleep {n}"] = f"ok rc={r.get('returncode')} ({time.time() - t0:.1f}s)"
        except Exception as exc:  # noqa: BLE001
            out[f"sleep {n}"] = f"FAILED after {time.time() - t0:.1f}s: {err_str(exc)}"
            log(f"  call ceiling: {out[f'sleep {n}']}")
            break
        log(f"  call ceiling: {out[f'sleep {n}']}")
    # was the sandbox still fine after a failed long call?
    try:
        call(sb, "/health")
        out["health after"] = "ok"
    except Exception as exc:  # noqa: BLE001
        out["health after"] = f"FAILED {err_str(exc)}"
    note("call_ceiling", out)


def probe_rate(sb: str) -> None:
    def one() -> tuple[bool, str]:
        try:
            call(sb, "/health")
            return True, ""
        except Exception as exc:  # noqa: BLE001
            return False, err_str(exc)

    t0 = time.time()
    seq = [one() for _ in range(100)]
    seq_s = time.time() - t0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=10) as ex:
        par = [f.result() for f in as_completed([ex.submit(one) for _ in range(200)])]
    par_s = time.time() - t0
    errs = sorted({e for ok, e in seq + par if not ok})
    note("rate", {
        "sequential": f"100 calls in {seq_s:.1f}s = {100 / seq_s:.1f}/s, {sum(1 for ok, _ in seq if not ok)} failed",
        "parallel x10": f"200 calls in {par_s:.1f}s = {200 / par_s:.1f}/s, {sum(1 for ok, _ in par if not ok)} failed",
        "errors": errs[:3],
    })


def probe_concurrent_creates(template: str) -> None:
    t0 = time.time()
    results: list[str] = []

    def one(i: int) -> str:
        try:
            sb, create_s, _ = create_sandbox(template, f"burst{i}", wait=False)
            return f"ok {create_s:.1f}s"
        except Exception as exc:  # noqa: BLE001
            return f"FAILED {err_str(exc)}"

    with ThreadPoolExecutor(max_workers=10) as ex:
        results = [f.result() for f in as_completed([ex.submit(one, i) for i in range(10)])]
    burst_s = time.time() - t0
    t0 = time.time()
    try:
        listed = list(SB.list(name=INSTANCE))
        listing = f"{len(listed)} sandboxes listed in {time.time() - t0:.1f}s; states={sorted({str(s.state) for s in listed})}"
    except Exception as exc:  # noqa: BLE001
        listing = f"list FAILED {err_str(exc)}"
    # readiness of the burst: how long until all answer
    t0 = time.time()
    burst = [sb for sb in list(created["sbs"]) if "burst" in (SB.get(name=sb).display_name or "")]
    ready = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        for f in as_completed([ex.submit(wait_ready, sb, 240) for sb in burst]):
            try:
                ready.append(round(f.result(), 1))
            except Exception as exc:  # noqa: BLE001
                ready.append(err_str(exc))
    note("concurrent_creates", {"10 creates": f"{burst_s:.1f}s wall; " + ", ".join(sorted(results)), "listing": listing,
                                "first answer (s)": sorted(ready, key=str)})
    for sb in burst:
        delete_sandbox(sb)


def probe_long(sb: str, minutes: float) -> None:
    out = {}
    r = call(sb, "/exec", {"command": "nohup bash -c 'for i in $(seq 1 100000); do date +%s >> /workspace/tick; sleep 1; done' >/dev/null 2>&1 & echo started", "timeout": 10})
    out["ticker"] = (r.get("stdout") or "").strip()
    end = time.time() + minutes * 60
    while time.time() < end:
        time.sleep(min(300, max(1, end - time.time())))
        try:
            r = call(sb, "/exec", {"command": "wc -l < /workspace/tick; pgrep -c sleep", "timeout": 10})
            log(f"  long: +{minutes * 60 - (end - time.time()):.0f}s ticks/sleepers: {' '.join((r.get('stdout') or '').split())}")
        except Exception as exc:  # noqa: BLE001
            log(f"  long: exec FAILED {err_str(exc)}")
    r = call(sb, "/exec", {"command": "wc -l < /workspace/tick; uptime -s; cat /proc/uptime", "timeout": 10})
    out["after"] = " | ".join((r.get("stdout") or "").strip().splitlines())
    # a harness turn whose single Bash call outlives a proxy call, then a normal turn
    out["turn_sleep200"] = run_turn(sb, "long-turn", 'Run `sleep 200 && python3 -c "print(6 * 7)"` in the shell (it takes over three minutes, wait for it) and reply with just the number it prints.',
                                    extra_env={"BASH_DEFAULT_TIMEOUT_MS": "600000", "BASH_MAX_TIMEOUT_MS": "600000"}, poll_s=2.0)
    out["turn_after"] = run_turn(sb, "after-long", 'Run `python3 -c "print(6 * 7)"` in the shell and reply with just the number it prints.')
    note("long", out)


# ---------------------------------------------------------------- main

try:
    INSTANCE = args.instance or client.runtimes.create().api_resource.name
    if not args.instance:
        created["instance"] = INSTANCE
    log(f"instance .../{INSTANCE.rsplit('/', 1)[-1]}")

    base, _, _ = create_template("base", "4", "8Gi")
    if not base:
        raise RuntimeError("base template failed")

    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {}
        futs[ex.submit(probe_ttl, base)] = "ttl"
        futs[ex.submit(probe_resources)] = "resources"
        futs[ex.submit(probe_concurrent_creates, base)] = "concurrent_creates"

        def on(fn, label, *a):
            sb, *_ = create_sandbox(base, label)
            try:
                fn(sb, *a)
            finally:
                if fn is not probe_long:
                    delete_sandbox(sb)

        futs[ex.submit(on, lambda sb: (probe_disk(sb), probe_body(sb)), "disk-body")] = "disk+body"
        futs[ex.submit(on, probe_call_ceiling, "ceiling")] = "call_ceiling"
        futs[ex.submit(on, probe_rate, "rate")] = "rate"
        if args.long_minutes > 0:
            futs[ex.submit(on, probe_long, "long", args.long_minutes)] = "long"
        for f in as_completed(futs):
            try:
                f.result()
            except Exception:  # noqa: BLE001
                log(f"GROUP {futs[f]} FAILED:\n" + traceback.format_exc())
                findings[futs[f]] = "FAILED (see log)"
except Exception:
    log("PROBE FAILED:\n" + traceback.format_exc())
finally:
    print("\nFINDINGS", json.dumps(findings, indent=1, default=str))
    if args.keep:
        log(f"--keep: leaving {len(created['sbs'])} sandboxes, {len(created['templates'])} templates")
    else:
        for sb in list(created["sbs"]):
            delete_sandbox(sb)
        log("deleted sandboxes")
        for tpl in created["templates"]:
            for attempt in range(8):
                try:
                    SB.templates.delete(name=tpl)
                    log(f"deleted template .../{tpl.rsplit('/', 1)[-1]}")
                    break
                except Exception as exc:  # noqa: BLE001
                    log(f"delete template attempt {attempt}: {err_str(exc)[:100]}")
                    time.sleep(15)
        if created.get("instance"):
            try:
                client.runtimes.delete(name=created["instance"], force=True)
                log("deleted instance")
            except Exception as exc:  # noqa: BLE001
                log(f"delete instance: {err_str(exc)}")
    sys.exit(0)

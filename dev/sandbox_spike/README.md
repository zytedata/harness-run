# Sandbox spike: the harness inside an Agent Sandbox custom container

> **Outcome (2026-09-11): built.** The spike's answer was yes, and `runtime/gemini/` was rewritten on
> it the next day (DESIGN.md §13; `worker.py` is the productized `server.py`, `_image.py` generates
> the Dockerfile). This directory stays as the record of the measurements and the platform limits.

Question: can a Gemini Enterprise Agent Platform **sandbox** (gVisor, no ambient Google
identity, Google-managed pre-warmed pools) replace the Agent Runtime query-job worker plus
our own warm pool, at equal or better latency? Background and the shell-sandbox
measurements are in the 2026-09-10 evaluation.

Four files:

* `server.py` — the "runner": the toolkit harness (`local.deploy`) behind a stdlib HTTP
  server. The platform's `execute` API forwards a POST (URI + port + JSON) into the
  container, so this is the whole contract. `/exec` is compatible with Google's shell image
  (the SDK's `execute_bash` works against it), `/turn` + `/events` run and stream a turn.
* `Dockerfile` — Debian, Python 3.12, git, the toolkit with `[local]` (bundled Claude CLI),
  non-root `appuser`, port 8080.
* `probe.py` — creates a template from the pushed image and measures the go/no-go numbers
  (see its docstring); cleans up after itself.
* `limits_probe.py` — the platform's undocumented ceilings (TTL, resources, proxy body sizes, the
  per-call ceiling, call rate, concurrent creates, a long-running process); results below.

## One-time setup (operator; the classifier does not let the assistant run these)

```bash
PROJECT=my-project
REGION=us-central1
NUMBER=123456789012   # project number
REPO=ratk-sandbox

# 1. A Docker repo for the image (none exists in us-central1 today).
gcloud artifacts repositories create $REPO --repository-format=docker \
    --location=$REGION --project=$PROJECT

# 2. The Agent Sandbox service agent pulls the image (the sandbox itself has no identity).
gcloud artifacts repositories add-iam-policy-binding $REPO --location=$REGION --project=$PROJECT \
    --member="serviceAccount:service-$NUMBER@gcp-sa-vertex-sandbox.iam.gserviceaccount.com" \
    --role=roles/artifactregistry.reader

# 3. Build and push (from the repo root; the context is the repo, only pyproject/README/
#    remote_agent_toolkit/dev/sandbox_spike are copied in).
IMAGE=$REGION-docker.pkg.dev/$PROJECT/$REPO/ratk-sandbox-spike:$(git rev-parse --short HEAD)
docker build -f dev/sandbox_spike/Dockerfile -t $IMAGE .
gcloud auth configure-docker $REGION-docker.pkg.dev
docker push $IMAGE
```

## Run the probe

The probe needs the 2.x SDK (`google-cloud-agentplatform`), which the toolkit does not pin
yet, so give it its own venv:

```bash
uv venv /tmp/ratk-sbx && uv pip install --python /tmp/ratk-sbx/bin/python \
    "google-cloud-agentplatform>=2.1" google-auth
/tmp/ratk-sbx/bin/python dev/sandbox_spike/probe.py --image $IMAGE            # ~5 min
/tmp/ratk-sbx/bin/python dev/sandbox_spike/probe.py --image $IMAGE --idle-minutes 60 --ttl-check
```

Model calls: the probe mints a one-hour token for `--model-sa` (default
`agent-runtime@…`, which ADC can impersonate) and hands it to the container as
`ANTHROPIC_AUTH_TOKEN` with `CLAUDE_CODE_SKIP_VERTEX_AUTH=1`; verified locally against the
bundled CLI (2.4 s, one Haiku turn). A production design would refresh it (Claude Code's
`apiKeyHelper`) or use a predict-only account.

## Local smoke (no GCP)

```bash
docker run --rm -p 8080:8080 ratk-sandbox-spike &
curl -s localhost:8080/health
curl -s -XPOST localhost:8080/exec -d '{"command":"id -un; claude --version || true"}'
```

## Results (2026-09-10, my-project / us-central1, image 17993f8, 4 CPU / 8 GiB)

| Step | Measured |
|---|---|
| template create (pool provisioning) | 78 s |
| sandbox create from template | 1.8–4.8 s |
| create -> first HTTP answer | 12–31 s more (route propagation: the container's uptime was already 77 s at first answer, i.e. the pool had started it at template creation) |
| dispatch -> first event, ready sandbox | **1.0 s / 1.4 s** (warm pool: 4.2 s) |
| dispatch -> result, ready sandbox | **4.7 s / 4.0 s** (warm pool: 14.5 s) |
| fresh sandbox, create -> result end to end | 25.8 s (cold Runtime job: ~150 s) |
| proxy round trip per call | 0.2–0.3 s |
| inside | uid 1000 = PID 1, /workspace + HOME writable, claude 2.1.222, git, uv; metadata server answers with the tenant identity (403 everywhere); PyPI reachable |

All turns returned `42` with no error (7 turns over three runs; ready-sandbox dispatch -> first
event 0.8–1.4 s, -> result 3.6–5.7 s every time).

Second run (`--pool 1 --turns 1 --idle-minutes 60 --ttl-check`):

| Check | Result |
|---|---|
| sandbox idle for 60 min, then a turn | answered the first poll in 0.9 s (same process, uptime 3714 s); turn 1.1 s / 3.6 s |
| exec at 420 s into a 480 s TTL, present at 540 s? | **gone**: exec does NOT reset the TTL (unlike Code Execution's documented behaviour) |
| a 30 min-TTL sandbox left idle 60 min (first attempt) | gone, as expected: TTL is enforced |

Consequence for a ready pool: set each sandbox's TTL to its intended idle life at creation
(the roster's `expires_at` = create + TTL), the same model as today's day-long pool worker
wait. Nothing needs to keep idle sandboxes alive.

## Limits (2026-09-11, `limits_probe.py`, same project / image / 4 CPU / 8 GiB unless noted)

| Limit | Measured |
|---|---|
| sandbox TTL | 1 h, 1 d, 7 d, 14 d and 30 d all accepted (`expire_time` set accordingly) |
| resources | 8 CPU / 16 GiB template works (a 13 GiB allocation succeeds; template create 118 s); 16 CPU refused: "Request CPU exceeds maximum allowed: 8.0 vCPU" |
| disk | `/tmp` 63 GB; `/` and `/workspace` overlay; a 2 GiB write takes 1.2 s |
| proxied request body | 100 KB fine (0.5 s), 1 MB fine (4.2 s), 4 MB / 10 MB / 32 MB fail (`Execution Failed. Error: UNAVAILABLE`) |
| proxied response body | 100 KB and 1 MB fine (0.3 s); 4 MB+ fail: "Response size too large. Received at least 2016214 bytes" → the worker pages `/events` under 1 MB |
| one proxied call's duration | 30 / 60 / 120 / 300 s fine; 600 s → 502 Bad Gateway at 600 s (the sandbox stays healthy) → the client's long-poll holds 20 s |
| call rate | 100 sequential `/health` calls in 18.5 s (5.4/s); 200 calls on 10 threads in 4.0 s (50/s); no 429 |
| concurrent creates | 10 sandboxes created in parallel in 7.5 s wall (2.7–7.5 s each), all listed RUNNING; first answers 0.7–32 s later |
| long-running process | a background ticker ran 25 min untouched across 5-minute `/exec` polls (the probe's sandbox then hit its own 30 min TTL — a probe bug, not a platform limit) |

## What decides go/no-go

| Number | Warm pool today | Sandbox needs |
|---|---|---|
| dispatch -> first event (ready worker) | 4.2 s | <= 4 s |
| dispatch -> result, one-tool Haiku turn | 14.5 s | <= 15 s |
| create -> ready (refill, off the critical path) | 150-300 s cold boot | anything reasonable |
| idle sandbox still answers after an hour / TTL reset by exec | n/a | yes, or no ready pool |

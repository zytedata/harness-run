# Sandbox spike: the harness inside an Agent Sandbox custom container

Question: can a Gemini Enterprise Agent Platform **sandbox** (gVisor, no ambient Google
identity, Google-managed pre-warmed pools) replace the Agent Runtime query-job worker plus
our own warm pool, at equal or better latency? Background and the shell-sandbox
measurements are in the 2026-09-10 evaluation (DESIGN.md will get the outcome).

Three files:

* `server.py` — the "runner": the toolkit harness (`local.deploy`) behind a stdlib HTTP
  server. The platform's `execute` API forwards a POST (URI + port + JSON) into the
  container, so this is the whole contract. `/exec` is compatible with Google's shell image
  (the SDK's `execute_bash` works against it), `/turn` + `/events` run and stream a turn.
* `Dockerfile` — Debian, Python 3.12, git, the toolkit with `[local]` (bundled Claude CLI),
  non-root `appuser`, port 8080.
* `probe.py` — creates a template from the pushed image and measures the go/no-go numbers
  (see its docstring); cleans up after itself.

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

## What decides go/no-go

| Number | Warm pool today | Sandbox needs |
|---|---|---|
| dispatch -> first event (ready worker) | 4.2 s | <= 4 s |
| dispatch -> result, one-tool Haiku turn | 14.5 s | <= 15 s |
| create -> ready (refill, off the critical path) | 150-300 s cold boot | anything reasonable |
| idle sandbox still answers after an hour / TTL reset by exec | n/a | yes, or no ready pool |

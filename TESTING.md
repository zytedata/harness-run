# Testing — from unit tests to live validation

Cheapest first. Every change runs the offline suite; run the later rungs when your change
can only break in ways the earlier rungs can't see.

| Rung | Command | Catches | Cost |
|---|---|---|---|
| Offline tests | `make test` | logic, event plumbing, contracts we encode | ~40 s (parallel), free |
| Install parity | `make parity-build` / `-check` | dependency/install/glibc breakage | ~1 min, free |
| **Live validation** | `make live-smoke` | **platform-contract breakage** | ~5 min incl. the image build, ~$0.15 |
| Model-provider check | `make live-openrouter` | provider-contract breakage (OpenRouter) | ~4 min, ~$0.75 |
| Model-provider check, remote | `make live-openrouter-remote` | the same models + remote visibility on the sandbox runtime | ~5-10 min, ~$0.56 + build |
| Model attribution | `make live-attribution` | did the turn run the model we asked for — both harnesses | ~10 s, ~$0.06 |
| Usage accounting | `make live-usage` | usage/cost accounting drift — esp. the Codex subagent rollout recovery (non-public details) | ~5 min, well under $1 |
| Turn control | `make live-interactive` | steer / interrupt+continue / stop / resume on both harnesses, against the real CLIs (their mid-turn behavior is what the harness loops encode) | ~2 min, a few cents |
| Turn control, by hand | `make chat` | a local chat page ([`dev/chat.py`](dev/chat.py)) to steer, interrupt and stop a running turn yourself and watch the events; `HARNESS=codex` for Codex | whatever the model charges |

The two OpenRouter figures are measured (2026-08-22, all four models on both harnesses:
26/26 local checks for $0.74, 128/128 remote checks for $0.56 of model spend plus the engine
build). Most of `live-openrouter` is the big models: one DeepSeek v4 Pro basic turn on
claude-code cost $0.083 and one Kimi K3 $0.069, while DeepSeek v4 Flash on codex cost $0.002.
Set `MODELS=openrouter/deepseek/deepseek-v4-flash` to check the plumbing for about a cent.

`make live-usage` re-establishes the usage/cost contracts the normalization relies on
([`dev/live_usage_probe.py`](dev/live_usage_probe.py)): Claude's `model_usage` being
subagent-inclusive and segment-cumulative, and — the fragile one — Codex's subagent rollout
layout (`session_meta` `parent_thread_id`, `turn_context.model`, cumulative `token_count`),
which is non-public and has no wire alternative (openai/codex#14642 closed as not-planned).
Run it after bumping the pinned Codex CLI or `openai-codex` SDK, or when touching
`harness/_usage.py` / the subagent recovery in `harness/codex.py`. `SCENARIO=codex-subagent`
runs the core recovery check alone.

`make live-interactive` also prints a latency table. Measured 2026-09-04 on the local
runtime (Haiku 4.5 on claude-code, gpt-5.6-luna on codex, both PASS), from the
`send()` / `interrupt()` call:

| Step | claude-code | codex |
|---|---|---|
| steer: `user` ack (the model has the message) | < 0.1 s | < 0.1 s |
| steer: the model's reply (`result`) | 8.7 s | 7.1 s |
| interrupt + continue: `interrupt_requested` and `user` ack | < 0.1 s | < 0.1 s |
| interrupt + continue: the model's reply (`result`) | 0.9 s | 0.8 s |
| `interrupt()` returns (turn ended, checkpoint saved) | 0.8 s | < 0.1 s |

`python dev/live_resume_probe.py` additionally forces a Bash background task, stops only
after `awaiting_tasks` reports it pending, and verifies that the next prompt is answered
from the saved workspace. This catches a resumed CLI's stale notification and empty
zero-turn result, which an ordinary foreground-command stop does not exercise. It is
a paid manual test (Haiku, a few cents). `ENGINE=<engine name>` targets a deployed sandbox
engine with a ready pool; the caller must arrange its deployment and teardown.

The control channel adds nothing measurable on `local`; the time is the model's. A steer
waits for the running tool call to finish (here a `sleep 8`), because the model reads the
message at its next step. On `gemini` a steer is one HTTPS call into the sandbox (~0.3 s);
`make live-smoke` exercises it on every run.

## 1. Offline tests (`make test`)

`make test` runs with `-n auto` (pytest-xdist). Each worker is its own process, so the
autouse fixtures, the pricing cache and the signal-handler test stay isolated; `make
test-serial` gives readable output when debugging. The wall clock is bounded by a few
deliberate poll-cadence tests; the slowest takes about 15 seconds. Expect about 40 seconds
overall.

The pytest suite makes **no network calls**: no GCP, no model, no subprocesses talking to
real services. Harnesses are exercised against the fakes in [`tests/fakes.py`](tests/fakes.py) /
[`tests/codex_fakes.py`](tests/codex_fakes.py), storage against `LocalBlobStore`, and the sandbox
platform against [`tests/sandbox_fakes.py`](tests/sandbox_fakes.py): `FakeSandboxProvider` keeps
templates and sandboxes in memory and routes every call to a worker — the REAL `Worker` class
driven in-process with its harness faked (a client turn then runs client → provider → worker →
harness fake → events → client), or a `ScriptedWorker` a test feeds events to. `make_engine`
builds a `GeminiEngine` over it with an in-memory roster. Useful conventions when adding tests:

- **Patch the seam, not the internals** — e.g. monkeypatch `handoff.GcsBlobStore` to
  `LocalBlobStore`, inject a `FakeSandboxProvider`, and drive the real code path above it.
- **Re-attach sessions by id** — `GeminiSession(engine, "sid")` constructs a handle without
  any platform call.
- The client's `/events` long-poll is shortened by an autouse fixture (`asyncio.run` waits for
  executor threads at shutdown, so a cancelled run would otherwise hold a test for the hold time).
- Ruff must pass too: `make lint`.

## 2. Install parity (`dev/` image)

Catches install-time breakage (wheels, glibc, uv resolution) locally, with a shell at hand,
before a deploy builds the real image. See [`dev/README.md`](dev/README.md).

## 3. Live validation on the sandbox platform

### Why offline green isn't enough

The platform's behavior changes **server-side, with zero client changes** — the offline
suite stays green through it. Agent Sandbox is a v1beta1 surface whose SDK renamed
`agent_engines` → `runtimes` between 1.x and 2.x and whose proxy limits (call ceiling,
body caps) are undocumented and were measured, not read. The §6/§13 contracts in
[`DESIGN.md`](DESIGN.md) exist because of such findings; live validation is how we keep
them true.

**Run a live check when your change touches:** the image (`_image.py`), the provider
adapter (`provider.py`), the worker (`worker.py`) or the turn body it reads, the dispatch and
streaming paths in `backend.py`, the run-scoped token or the model token, or the
`google-cloud-agentplatform` pin. Prose-only / pure-logic changes with solid offline coverage
don't need it.

### The standard smoke test

```bash
make live-smoke
PROJECT=my-proj LOCATION=us-central1 make live-smoke   # non-default project
LONG_MINUTES=70 make live-smoke                        # + a turn whose one Bash call sleeps 70 min
```

[`dev/live_smoke.py`](dev/live_smoke.py) deploys **a throwaway engine from your checkout**
(named `agent-run-smoke-<you>`: image build + push with your Docker, a template, a ready pool of
one), runs Haiku turns against it, and tears everything down in `finally`:

- **pool-turn** — a turn on the ready sandbox: `turn_started` must say `warm=True`, the first
  event must arrive within 4 s (measured 1.3 s), the result must be `42`.
- **session-config / turn-config** — `get_engine` (pure addressing) + a `SessionConfig`
  planting a marker in the system prompt: the reply must carry it (the worker ran the
  session's config over the baked spec), then a second turn on the SAME session — a checkpoint
  resume on a fresh sandbox (`workspace_ready.restored` must be true) — with a
  `TurnConfig(output_schema=...)`: the client parses `structured_output`, and the worker's
  `effective_spec` echo must still carry the marker. The second turn re-attaches by session id
  the way an application does.
- **steer** — `session.send()` into a running turn (the model is inside a `sleep 25`): the
  `user` event must acknowledge the message id and the reply must reflect it.
- **isolation** — a fixed read-only script run through the worker's `/exec` on a fresh
  sandbox (no model involved): the metadata server's identity must not be one of ours, its
  token must be 403 on the project's storage and Vertex, and no credential-like env var is
  set outside a turn. Prints statuses and names, never a token.
- **long-turn** (`LONG_MINUTES`) — one Bash call that long; what it exercises is the model
  token's lifetime (an hour unless the org policy extends it), the sandbox TTL and the
  `/events` long-poll over hours.

Typical numbers (2026-09-11): deploy 81 s (build 32 s on a warm cache, push 13 s, template
18 s, pool fill 15 s), pool turn 6.6 s to the result, fresh-sandbox turns 22–28 s, a few cents
of model spend. Exit code is non-zero on any FAIL, so you can gate on it.

Prerequisites: the GCP setup from the
[README "GCP setup & required permissions"](README.md#gcp-setup--required-permissions)
(ADC that can impersonate the model service account, Haiku enabled in Vertex Model Garden)
and the Docker CLI logged into the registry (`docker login -u oauth2accesstoken
--password-stdin us-central1-docker.pkg.dev` with an access token). Defaults target the shared
`my-project` test project and its `agent-run-sandbox` repo; `MODEL_SA` overrides the model
service account (default: the toolkit's `agent-run-model@<project>`, the predict-only account
`agent-run-gcp-setup` creates — the operator account would hand the agent the whole project, and
the smoke's **model-token** check fails on any account whose token reaches more than the
model).

### The limits probe (`dev/live_limits_probe.py`)

Not part of the regular ladder: it measures the platform's undocumented ceilings (TTL, CPU and
memory, disk, proxy body sizes, the per-call ceiling, call rate, concurrent creates) and takes
~15 min plus the optional long-running-process check. It runs against any image `gemini.deploy`
built (`--image`, from a deploy record or `engine.revisions()`), needs only the worker's `/health`
and `/exec`, creates its own templates and sandboxes and deletes them in `finally`. Re-run it when
the platform announces changes to Agent Sandbox or when a limit in `backend.py` / `worker.py`
(`EVENTS_WAIT_S`, `EVENTS_PAGE_BYTES`, the `exec()` timeout cap) needs re-grounding.

```bash
.venv/bin/python dev/live_limits_probe.py --image us-central1-docker.pkg.dev/<project>/agent-run-sandbox/<image>:<tag> [--long-minutes 25]
```

Findings of 2026-09-11 (my-project / us-central1, 4 CPU / 8 GiB unless noted):

| Limit | Measured |
|---|---|
| sandbox TTL | 1 h, 1 d, 7 d, 14 d and 30 d all accepted (`expire_time` set accordingly) |
| resources | 8 CPU / 16 GiB template works (a 13 GiB allocation succeeds; template create 118 s); 16 CPU refused: "Request CPU exceeds maximum allowed: 8.0 vCPU" |
| disk | `/tmp` 63 GB; `/` and `/workspace` overlay; a 2 GiB write takes 1.2 s |
| proxied request body | 100 KB fine (0.5 s), 1 MB fine (4.2 s), 4 MB / 10 MB / 32 MB fail (`Execution Failed. Error: UNAVAILABLE`) |
| proxied response body | 100 KB and 1 MB fine (0.3 s); 4 MB+ fail: "Response size too large. Received at least 2016214 bytes" → the worker pages `/events` under 1 MB |
| one proxied call's duration | 30 / 60 / 120 / 300 s fine; 600 s → 502 Bad Gateway at 600 s (the sandbox stays healthy) → the client's long-poll holds 20 s, `exec()` is capped at 240 s |
| call rate | 100 sequential `/health` calls in 18.5 s (5.4/s); 200 calls on 10 threads in 4.0 s (50/s); no 429 |
| concurrent creates | 10 sandboxes created in parallel in 7.5 s wall (2.7–7.5 s each), all listed RUNNING; first answers 0.7–32 s later |
| long-running process | a background ticker ran 25 min untouched across 5-minute `/exec` polls |
| exec resets the TTL? | **no**: a 480 s-TTL sandbox was gone at 540 s despite an exec at 420 s (so a pool sandbox's TTL is set at creation to its whole intended life) |
| idle sandbox after 60 min | answers the first call in 0.9 s, same process |

### The OpenRouter model check

```bash
OPENROUTER_API_KEY=... make live-openrouter
MODELS=openrouter/z-ai/glm-5.3 make live-openrouter  # one model, both harnesses
```

This paid local test runs every model on both harnesses. Each model must call a shell tool,
return its output, produce schema-valid structured output, hide provider credentials from the
tool, report its selected upstream, and use OpenRouter's exact cost. It also checks resume, a
strict provider choice, whole routing objects, and a tiny budget cap on both harnesses.

Every model call goes through the local proxy (`harness/_openrouter_proxy.py`), so the test
also requires an `http_status` on every `openrouter_request` event, which only the proxy
reports. That shows the proxy saw the calls the run made. What keeps a call from going
straight to OpenRouter is the environment: the test also asserts that the account key is
absent from the tool environment, and the CLI only ever gets the proxy's per-run token. One
row per harness runs with no provider pinned, because that case used to skip the proxy
entirely.

Three checks retry once on a soft miss: the basic turn, resume, and structured output. Models
occasionally answer without running the command they were asked to run. Measured on DeepSeek
v4 Pro under codex: one run answered 6 for `print(6 * 7)`, two immediate re-runs answered 42.

Most model requests select one known provider and disable fallbacks. Kimi uses Moonshot AI,
GLM uses Z.AI, and both DeepSeek models use Novita because the shared account's ZDR policy
excludes DeepSeek's own endpoint. Those rows fail if OpenRouter reports a different provider.
Three kinds of row do something else on purpose: the two rows named "unpinned" above, the two
DeepSeek structured-output checks on Codex (the README footnote explains that exception), and
the routing rows below, which send a whole provider object. Set `OPENROUTER_PROVIDER` to test
every feature against one specific provider. To test one model with another provider:

```bash
MODELS=openrouter/moonshotai/kimi-k3 OPENROUTER_PROVIDER=fireworks \
make live-openrouter
```

Set `SERIAL=1` for ordered output while debugging.

Two rows per harness send a whole routing object instead of a single slug, both on Kimi K3
because it has many providers (GLM-5.3 has one endpoint, which would prove nothing). The
"closed" row sends `{"only": [moonshotai, fireworks]}` and requires that OpenRouter reports
one of those two — either counts, because a closed set bounds who may serve a turn without
keeping the turn on one of them — and that `provider_matches_request` is true. The "open" row
sends `{"order": [fireworks, moonshotai], "allow_fallbacks": true}` and requires the opposite:
the turn completes and `provider_matches_request` is null. The README explains when that verdict
can be claimed. Both rows judge the reported provider name themselves rather than trusting the
toolkit's own verdict. The second provider never has to be reachable — the primary is in both
objects — so `OPENROUTER_ALTERNATE_PROVIDER` only needs changing to test a different pair.

Other knobs: `HARNESSES=codex` (or `claude-code`) runs one harness instead of both, which
roughly halves a pass, and `PROBE_REASONING_EFFORT` sets the effort every turn asks for.

**It costs real money** (the summary table above has the current figure) and needs a key, so
run it **by hand, sparingly, locally**. It must never run in CI: `pytest -q` stays free and credential-less
(see §1 and `.github/workflows/ci.yml`) — the offline tests pin the config the harness
emits, and that is what CI checks.

### The OpenRouter model check on the sandbox runtime

```bash
OPENROUTER_API_KEY=... make live-openrouter-remote
```

This paid remote test repeats the local checks on the sandbox runtime. One engine contains both
CLIs and serves every model through per-turn overrides. Every model runs a structured-output turn
on both harnesses. Resume, direct provider selection, routing objects, and budget caps also run on
both harnesses. The two routing rows are the same closed/open pair the local test runs, which is
where a routing object is proven to survive the trip to a deployed worker.

It also checks the remote-only surface: the worker's `effective_spec` echo names the model and
`session.history()` replays the events with a terminal result.

The engine is deleted in `finally`; a failed teardown prints loudly, because a ready pool's
sandboxes bill while they exist. `KEEP=1` leaves it up for debugging and hands you the cleanup.

`MODELS=` narrows the model list here too. The deployment knobs are `PROJECT`, `LOCATION`,
`SUFFIX` (the engine name's suffix) and `IMPERSONATE_SA`; each falls back to the same default
the other live probes use.

**Costs real money** (the summary table above has the current figures). Checks run
concurrently. Teardown also runs on SIGTERM/SIGINT, because a `timeout` that fires mid-run
would otherwise leave sandboxes billing.

DeepSeek v4 sometimes returns no final message under Claude Code (the README has the details).
The test retries one soft failure and reports when the retry was used. It stays failed when the
retry also has no answer; in the 2026-08-22 validation no retry was needed.

### Model attribution: did we run what we asked for?

```bash
OPENROUTER_API_KEY=... OPENAI_API_KEY=... make live-attribution
```

`dev/live_model_attribution.py` checks evidence returned by the CLI, app-server, and
OpenRouter:

- **claude-code** — the CLI's `system/init` message names the model it resolved.
- **codex** — the app-server's `thread.read()` names the thread's bound `model_provider`,
  surfaced as a `model_routing` event. This is what proves an OpenRouter override took
  effect; a mismatch is reported as `matches_request: false`.
- **OpenRouter on both harnesses** — `openrouter_request` names the selected upstream and
  reports exact cost for each model response.

Checks run concurrently and cost a few cents. `SERIAL=1` gives ordered output. The claude-code
check always runs, so it needs whatever Claude auth your shell already uses — an
`ANTHROPIC_API_KEY` or a logged-in `claude` CLI. `CLAUDE_MODEL` and `OPENAI_MODEL` change the
native models it checks alongside the OpenRouter ones.

### Isolation and the run-scoped GCS token

The sandbox has no Google identity of its own, so the two questions the earlier runtime needed
separate probes for — what can the agent's shell reach, and did the worker do its GCS work on the
run-scoped token — are answered by `make live-smoke` on every run: the **isolation** check prints
what the shell reaches (a tenant identity that is 403 on the project), the **model-token** check
fetches the token the agent's CLI actually uses — from the worker's loopback metadata server,
mid-turn, through `session.exec()` — and shows it refused (401/403) on listing the project's
reasoning engines, the output bucket and its service accounts, and every turn's
`workspace_ready` event carries `scoped_gcs: true` while the mirror it streams to was written with
that token (the worker has no other credential that could open the bucket). Run the smoke for any
change to `scoped_gcs.py`, `GcsBlobStore`, the worker's turn body or the model-token path.

### Writing a bespoke live probe

When the smoke test doesn't cover your change (e.g. validating crash/retry behavior, or a
new event field), follow the same pattern — it's what keeps live testing safe and cheap.
Worked examples in [`dev/`](dev): `live_two_turn_probe.py` (two turns on one session over
the event stream — the probe that caught the stale-result replay bug) and
`live_limits_probe.py` (the platform's ceilings, measured):

- **Throwaway, named engines**: suffix with something identifying (`-itest`, your name) so
  leftovers are attributable; never point a probe at someone's standing engine.
- **Cheap model, tight caps**: Haiku, low `max_turns` / `max_budget_usd` — the platform
  round-trip is the subject, not the model's work.
- **Teardown in `finally`** with `engine.delete()` (a ready pool's sandboxes bill while
  they exist; the templates go with them).
- **Check for leftovers** after any failed run — sandboxes bill while they exist:

  ```python
  from agent_run import gemini
  from agent_run.runtime.gemini.provider import AgentSandboxProvider
  for e in gemini.list_engines("my-project", "us-central1"):
      print(e)
  for sb in AgentSandboxProvider("my-project", "us-central1").list():
      print(sb["display_name"], sb["state"], sb["expire_time"])
  ```

### Debugging a live run

- **The events are the truth**, live and after the fact: `session.history()` reads the same
  mirror objects the client streamed (plus whatever a dying worker flushed before the client's
  last poll). There is no Cloud Logging step log any more.
- **A turn that ends with `sandbox_unreachable`** means the sandbox went away mid-turn (an OOM
  is the usual reason — raise `resource_limits`) or the proxy stopped routing to it; the last
  events before it say where the agent was. `dispatch_failed` means no sandbox took the turn
  (the message names the platform error); `dispatch_uncertain` means a `/turn` call lost its
  answer and the sandbox could not be asked whether it had started the turn — the sandbox was
  deleted and the turn was not replayed (the message names the sandbox).
- **Poke a live sandbox by hand**: `AgentSandboxProvider(...).call(sandbox, "/exec",
  {"command": "..."})` runs a shell command in it (the same contract as Google's shell image);
  `/health` reports uptime and the running turn. Sandbox ids appear in `turn_started.worker`.
- **The container's own logs** (the worker's stderr) are not exposed by the platform; the
  worker therefore surfaces every failure it can as an event (`harness_error`, `config_error`,
  `late_harness_error`) rather than logging it.

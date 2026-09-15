# remote-agent-toolkit — Design

A Python library for **defining and running remote/background AI agents** at Zyte. It distills the
experience of two proofs-of-concept (self-healing spiders, interactive spider creation) built on
`gemini-agent-runtime` into reusable building blocks, so any team can stand up a Claude-Code-based
agent — with custom skills, GitHub access, structured outputs, checkpoint/resume and warm starts —
without re-learning the platform's sharp edges.

> Status: **design**. This document is the agreed architecture and roadmap. Code is being scaffolded
> against it. The README (team-facing onboarding) grows alongside the code, phase by phase (see §11).

---

## 1. Goals & non-goals

**Goals**
- Let a team define an agent **declaratively** and run it both **locally** (dev) and **remotely** on
  **Gemini Agent Runtime** (prod) through *one* API — local deploy is first-class, not an afterthought.
- Cover the use-cases in the requirements doc: background long-running jobs (self-heal, QA, analysis)
  *and* interactive, human-in-the-loop jobs (requirement-building, prompt-driven setup).
- Bake in the hard-won platform knowledge (latency, checkpointing, warm pool, IAM, build constraints)
  so consumers inherit it for free.
- Be a **source of knowledge**, not just code: the README + design docs explain *why*.

**Non-goals (for now)**
- Custom-LLM harnesses beyond the two that ship. We design the **seam** (a `Harness` protocol) and ship
  the **Claude Code (Agent SDK)** binding (default) and the **Codex (openai-codex SDK)** binding
  (`spec.harness="codex"`). Both can run `openrouter/` model ids: Claude Code uses OpenRouter's
  Anthropic-compatible endpoint, and Codex uses `model_providers.*` config overrides. Providers beyond
  OpenRouter are a later step. `openrouter_provider` selects one OpenRouter provider and
  `openrouter_routing` carries OpenRouter's whole `provider` object, either one on the spec, a
  session or a single turn.
  The prefix keeps model selection per turn and preserves compatibility with existing engine configs.
- Non-GCP backends. We design the **ports** (storage, events, dispatch, secrets) as protocols, but ship
  GCP adapters (GCS, Cloud Logging, Pub/Sub, Secret Manager) plus local/in-memory adapters for dev.
- Replacing Scrapy-Cloud / monitoring logic — that stays in `gemini-agent-runtime` behind the seam.

---

## 2. Background: which "Claude agent" do we build on?

"Claude Managed Agents" is two different Anthropic surfaces; the distinction drives the architecture.

| | **Claude Managed Agents (CMA)** | **Claude Agent SDK** |
|---|---|---|
| What | Server-hosted agent harness exposed as REST (`/v1/agents`, `/v1/sessions`) | Client library (`claude_agent_sdk`) that drives the Claude Code loop in-process |
| Runs where | Anthropic infra (loop + sandbox container) | **Your** process / container |
| On our platform? | ❌ Not offered on Vertex/Bedrock/Foundry | ✅ via `AnthropicVertex` model access on Gemini Agent Runtime |
| Object model | Versioned `Agent` → `Environment` → `Session` → events | `query()` / `ClaudeAgentOptions` / `SessionStore` |

**Decision:** build on the **Claude Agent SDK** (the only Claude harness runnable on Gemini Agent
Runtime, via `AnthropicVertex` model access), and **borrow CMA's API shape** as the conceptual template.
We cannot host *on* CMA.

> **Naming.** We call the platform **Gemini Agent Runtime** — Google's recent rename of what was
> "Vertex AI Agent Engine". The public namespace is **`gemini.*`** (`gemini.deploy`, `gemini.get_engine`).
> The underlying Google Python SDK is imported as `agentplatform` / `google-cloud-aiplatform` — the
> `vertexai` namespace was renamed to `agentplatform` in aiplatform 1.154 and now emits a
> `FutureWarning` on `Client()`; that's an internal detail, not part of our surface.

What we borrow from CMA / the SDK (things our PoC did implicitly or not at all):
- **Versioned agent-definition vs. run** split → control plane / data plane.
- **Session state machine** where `idle` carries a `stop_reason` and is **not** "done" — this is exactly
  our checkpoint-and-end-turn interactive pattern, formalized.
- **Event-stream-first** interface with reconnect/dedupe (open the stream before sending the kickoff).
- **Three-verb session model**: continue / resume / fork.
- **Pluggable `SessionStore`** + a **conformance test suite** other teams run against their own backend.
- **System-prompt inherit + append** instead of "string or nothing".
- **Structured outputs** via the SDK's `ResultMessage.structured_output` / constrained decoding.

What we deliberately avoid: server-hosted-loop lock-in, single-model-family assumptions baked into the
API, Claude-Code-specific assumptions leaking past the harness seam, cwd-keyed local session files
(fatal for remote/serverless — our `SessionStore` keys by `session_id`), and date-stamped beta headers
in the public surface.

---

## 3. Design principles

1. **Declarative, serializable agent definition.** An `AgentSpec` is data; it can live in code or YAML,
   be version-controlled, diffed in CI, and deployed reproducibly.
2. **Local / remote parity.** `local.deploy(spec)` (in-process) and `gemini.deploy(spec)` /
   `gemini.get_engine(name)` return the *same* `Engine` + `Session` surface. Develop and test locally,
   ship remotely, **zero code change** — swap `local` ↔ `gemini`. Local uses filesystem/in-memory port
   adapters, so it needs no GCP beyond model access.
3. **Deploy is rare; lookup-and-run is the hot path.** App code addresses a deployed engine **by name**
   (optionally a version) via `gemini.get_engine(...)` and never triggers a deploy. `gemini.deploy(...)`
   is an ops/CI action.
4. **Ports & adapters.** Every platform dependency is a protocol with a concrete adapter. Storage, event
   sink, dispatch transport, secret resolver, session store, harness — all swappable.
5. **Secrets are per-invocation; never bake secrets/paths.** The spec is baked into the sandbox image, so
   `AgentSpec` carries no secrets at all — credentials are passed as a `name → value` map to `run`/`send`,
   scoped to that one turn (nothing baked, nothing shared across tenants). Since §13 the values travel in
   the turn's HTTPS body to the worker and nowhere else (nothing persisted by the platform); before, they
   were staged at a **per-invocation GCS object** the worker fetched (and deleted only at the end of a
   completed turn — the platform re-ran a crashed attempt with the same payload, so the object had to survive for the
   retry; a client completion-cleanup and a 1-day lifecycle rule back it up); only the pointer travels. The
   harness routes each secret to its consumer — a repo's `auth` token into git `origin`, a GitHub MCP token
   into headers, the rest into the agent's env — and checkpoint snapshots are credential-scrubbed.
6. **Encode the platform contracts as defaults, document the why.** The deploy path applies the known-good
   settings automatically (see §6) and every one is explained.
7. **Interactive = checkpoint-and-resume, not a blocking call.** No long-lived `ask_human` tool. The agent
   ends a turn with a question → the run checkpoints and releases its worker → the human's answer resumes it
   on a warm worker. One mechanism, stateless workers, survives worker death, doesn't fight the >10-min limit.

---

## 4. Architecture

Three planes over a set of pluggable ports:

```
 DEFINITION plane        DEPLOY plane (ops/CI)         RUN plane (app code)
 ────────────────        ─────────────────────        ──────────────────────────────
  AgentSpec        ──▶    gemini.deploy(spec)          gemini.get_engine(name[, ver]) ─┐
  (declarative,           → versioned engine,          local.deploy(spec)              │
   serializable;            encodes platform           → Engine                        ▼
   code or YAML)            contracts §6                  · start_session()  → Session
                                                          · get_session(id)  → Session  (re-attach/poll)
                         local.deploy(spec) ─────────▶    Session
                         (in-process,                       · run(msg) → Run :
                          same Engine/Session API)              await Run      → RunResult   (wait)
                                                                async for ev   → AgentEvent  (stream)
                                                                Run.done/status              (poll)
                                                            · send(msg) / interrupt()
                                                            · status / stop_reason
                                                            · resume / fork

 PORTS (protocols; concrete adapters shipped — GCP for prod, local/in-memory for dev)
 ────────────────────────────────────────────────────────────────────────────────────
  Harness            ClaudeCodeHarness                  (the Agent SDK binding)
  BlobStore          GcsBlobStore   | LocalBlobStore    (checkpoint, workspace tar, artifacts)
  EventSink          CloudLoggingSink | InMemorySink    (the only near-real-time async channel; tailed)
  DispatchTransport  PubSubDispatch | InMemoryDispatch  (warm-pool dispatch + atomic claim)
  SessionStore       GcsSessionStore                    (Claude SDK protocol, keyed by session_id)
  SecretResolver     GcpSecretResolver | EnvSecretResolver  (secret name → value, at runtime)
```

- **Harness** is the abstraction over "a coding agent loop." `ClaudeCodeHarness` wraps the Agent SDK
  `query()`, builds `ClaudeAgentOptions` from the `AgentSpec`, translates SDK messages → generic
  `AgentEvent`s; the sandbox worker (`runtime/gemini/worker.py`) drives it inside the container exactly as
  `local` does in-process. This is the generalized `ClaudeCodeAgent` from the PoC.
- **Engine** is the deployed (or local) agent handle. `gemini.deploy(spec)` registers an engine under
  `spec.name` and returns a `GeminiEngine`; `gemini.get_engine(name[, version])` looks one up without
  deploying; `local.deploy(spec)` returns a `LocalEngine`. All expose `start_session()` and
  `get_session(id)`.
- **Session** is the run-plane object with the CMA-style lifecycle:
  `pending → running ↔ idle(stop_reason) → terminated`. `idle` + `stop_reason="needs_input"` is the
  interactive pause; `send()` resumes via checkpoint. A `Session` is addressable by id, so another process
  can re-attach via `engine.get_session(id)` and poll `status` / fetch `result`. While a turn is
  **running**, `send()` talks to it instead of starting a second turn — `interrupt=False` steers (the
  model sees the message at its next step), `interrupt=True` interrupts first and continues from the
  message — and `interrupt()` stops it: the turn ends through its normal end-of-turn path with
  `stop_reason="interrupted"`, checkpoint taken, so the session is resumable (`control.py`; issue #44).
  Every delivered message is a `user` event, the acknowledgement that the model has it.
- **Run** is the handle returned by `session.run(msg)` — awaitable (→ `RunResult`), async-iterable
  (→ `AgentEvent`s), and pollable (`done` / `status` / `result`). For `gemini`, "stream" = tail the
  `EventSink`; "await" = wait for the terminal result event; "poll" = check the latest logged status.

---

## 5. Public API

```python
from remote_agent_toolkit import AgentSpec, SystemPrompt, SkillSource, McpServer, gemini, local

spec = AgentSpec(
    name="spider-builder",
    model="claude-sonnet-4-6",
    system_prompt=SystemPrompt.inherit(append="Prefer the Zyte web-scraping skills."),
    skills=[SkillSource.git("https://github.com/zytedata/claude-skills", ref="0.2.0")],
    mcp_servers=[McpServer.github()],        # token supplied per-invocation (run/send secrets=), not here
    permission_mode="bypassPermissions",
    max_turns=120,
    max_budget_usd=10.0,
    checkpoint=True,                          # enable resume / interactive pauses
    output_schema=BookSchema,                 # optional: structured output (pydantic / JSON schema)
)
```

```python
# ── DEV: run locally, in-process — same Engine + Session API as prod, no GCP beyond model access ──
engine = local.deploy(spec)
session = engine.start_session()
result = await session.run("scrape https://books.toscrape.com for title, price")   # wait for completion
print(result.text, result.structured_output, result.cost_usd)
```

```python
# ── PROD ──────────────────────────────────────────────────────────────────────────────────────────
# Ops / CI deploys once (rare):
gemini.deploy(spec, project="my-project", location="us-central1", warm_pool=True)

# App code looks the engine up by name (optionally a version) and runs — it never deploys:
engine = gemini.get_engine("spider-builder")               # latest deployed version
session = engine.start_session()
result = await session.run("/scrape https://books.toscrape.com title, price")   # default: wait for the result
```

**Three ways to consume a run** (`session.run(msg)` returns a `Run` handle):

```python
# (a) wait for completion (simplest — the headline default)
result = await session.run("/scrape ...")

# (b) stream events as they happen (UI / live logs)
async for ev in session.run("/scrape ..."):
    print(ev.kind, ev.summary)               # thinking | tool_use | tool_result | message | status | result
result = session.last_result                 # populated once the run completes

# (c) poll — e.g. a web handler that can't hold the connection, or another process
run = session.run("/scrape ...")             # returns immediately
while not run.done:
    await asyncio.sleep(2)
    print(run.status)                        # running | idle | terminated
result = run.result

# re-attach later / elsewhere by session id, then poll or continue
session = engine.get_session(session_id)
if session.status == "idle" and session.stop_reason == "needs_input":
    await session.send("yes, that schema looks right")     # resumes via checkpoint on a warm worker

# enumerate past sessions and read a finished one's record (gemini)
for info in engine.list_sessions():                        # newest first
    past = engine.get_session(info["session_id"])
    events = past.history()                                # persisted AgentEvents, oldest first
    result = past.last_result                              # reconstructed from the terminal event
```

**Managing deployed engines** (control plane):

```python
gemini.deploy(spec, project=..., location=...)   # create / update; mints a new version
gemini.get_engine("spider-builder")              # serving version (app code default)
gemini.get_engine("spider-builder", version=3)   # pin a version (an assertion — see below)
gemini.list_engines(project=..., location=...)   # discover what's deployed
engine.versions()                                # list versions of one engine
engine.set_traffic(3) / engine.set_traffic()     # roll back to a version / back to always-latest
engine.name, engine.version, engine.resource     # identity / underlying resource name
```

Engine identity maps `spec.name` → a stable engine (Agent Engine `display_name`); each `deploy` mints a
new version. App code pins or takes latest; it does not deploy.

A **version is a sandbox template** (`…/reasoningEngines/{host}/sandboxEnvironmentTemplates/{id}`, displayed
under `spec.name`): `deploy` creates a new immutable template whenever the image or resources differ from
the newest one, and `get_engine(name)` resolves the newest, `get_engine(name, version=)` a specific one.
A pin **routes** — any template can be dispatched to — so a pinned handle keeps running its version after a
newer deploy; "serving" simply means "newest", and there is no traffic configuration. (Until §13 a version
was an Agent Runtime runtime revision, and a pin could only assert on the serving one.)

**Key types** (sketch — finalized in `spec.py` / `runtime/base.py`):

- `AgentSpec` — the declarative definition. Serializable (`to_dict` / `AgentSpec.from_yaml`).
- `SystemPrompt.inherit(append=...)` — inherit the harness's built-in
  system prompt and append. A plain `str` replaces it entirely.
- `SkillSource` — `.git(url, ref=)`, `.local(path)`, `.builtin(name)`; a list allows base + extra sources
  (multi-source skills, §12). Resolved & staged at deploy/run time.
- `McpServer` — `.github()`, `.remote(name, url)`, `.stdio(...)`; credentials come from the per-invocation
  `secrets`, not here.
- `RepoSource` — `.git(url, ref=, auth=, auth_user=)`; cloned into the cwd before the agent runs, push-ready
  when the `auth`-named per-invocation secret is supplied (host-aware token injection).
- `Engine` — `start_session()`, `get_session(id)`, `list_sessions()`, `versions()`,
  `name`/`version`/`resource`.
- `Session` — `run(msg, secrets=)`, `send(msg, secrets=, interrupt=, message_id=)` (idle: resume;
  running: steer or interrupt-and-continue, same `Run`), `interrupt()` (interrupt and stop, resumable),
  `status`, `stop_reason`, `last_result`, `history()`, `fork()`.
- `Run` — `__await__` (→ `RunResult`), `__aiter__` (→ `AgentEvent`s), `done`, `status`, `result`.
- `AgentEvent` — `kind`, `summary`, `raw`; cost/usage carried on the terminal event.
- `RunResult` — `text`, `structured_output`, `is_error`, `num_turns`, `cost_usd` (`float | None`;
  `None` means the spend is unknown, `0.0` means the run was free), `usage` (the normalized
  whole-turn token record — one shape on every harness, subagents and re-invocation segments
  included, `None` for a number the harness doesn't report; `harness/_usage.py`), `session_id`,
  `artifacts`, `warning`. Error results keep their accounting (`cost_usd`/`num_turns`/`usage`): an
  `error_max_turns` run spends right up to its limit, so zeroing them under-reports exactly the most
  expensive runs (eval feedback).
- **The terminal result event is authoritative — even if the harness dies afterwards.** The claude CLI
  can exit non-zero *after* the final `ResultMessage` (it does so on error results, and transiently even
  on finished runs — observed: a completed ~$10 run voided by an exit-1 during result delivery). Both
  runtimes keep the already-received result and record the late death as `RunResult.warning` + a
  `late_harness_error` status event, instead of overwriting a finished run with an empty error result.
- **Claude CLI stderr is captured** (`jobs/<sid>/stderr.log`, capped, beside — not inside — the agent
  workspace). The SDK pipes stderr only when a callback is registered; without one, `ProcessError`'s
  "Check stderr output for details" promises output nobody captured. On a CLI-exit failure the tail is
  surfaced as a `claude_stderr` status event (reaches Cloud Logging on gemini) and embedded in the error.

---

## 6. Execution model & platform contracts (the hard-won knowledge)

These are facts measured during the PoC. The library encodes them so consumers inherit them for free.

**Execution paths (2026-09-11: the sandbox runtime replaced the query-job runtime; §13)**
- **Agent Sandbox custom container** (the current runtime) — the harness in a gVisor sandbox behind
  Google's authenticated HTTP proxy: dispatch→first event ~1.3 s, →result ~4–7 s on a ready sandbox
  (measured live 2026-09-11 by `make live-smoke`), ~20 s create→result without one, and no Google
  identity in the container. One sandbox per turn, deleted at the terminal event; multi-turn continuity
  rides the checkpoint. Limits measured 2026-09-11: 8 vCPU max, ~5 min per proxied call, ~1 MB request /
  ~2 MB response bodies, TTL up to 30 d (`dev/sandbox_spike/README.md`).
- Kept for the record — the Agent Runtime paths this replaced: **async `run_query_job`** (long-running,
  validated 64.5 min, up to 7 days; **~2.5 min per-job worker provisioning** that no config knob
  reduced), **sync `stream_query`** (~3–9 s but a ~600 s per-request ceiling), and our **warm pool**
  (pre-warmed jobs blocked on a Pub/Sub channel; 4.2 s / 14.5 s to first event / result with per-worker
  dispatch).

**The ready pool (`roster.py`; replaces the warm pool)**
- `deploy(warm_pool=True)` creates N sandboxes from the new template, waits until each answers `/health`
  (12–33 s: route propagation, the container is already running) and records them in the **roster**:
  client-owned GCS objects under `pool/<template>/idle/`, claimed atomically (generation-precondition
  delete) so several client processes share one pool. A turn takes the oldest ready sandbox, POSTs the
  turn to it and refills the pool in the background; the roster is empty → the turn creates a sandbox for
  itself (~20 s), never a stranded turn.
- A sandbox's **TTL is set at creation and cannot be extended** (an exec does not reset it — measured), so
  pool sandboxes are created with `pool_max_wait_s + max_turn_s` of life and the roster entry expires at
  `pool_max_wait_s`: one claimed at the end of its idle life still has `max_turn_s` for the turn. An
  expired entry is pruned and its sandbox deleted; nothing is replaced automatically except after a dispatch.
- The roster is a coordination device, not a security boundary: there is no per-sandbox IAM, so whoever can
  execute on the host instance can drive any sandbox, and the client identity is the trust boundary.
- Historical (the warm pool): a job submitted with a `__POOL_WAIT__` sentinel blocked pulling **its own**
  Pub/Sub subscription (per-worker dispatch, `pool.py`); the control plane published the turn addressed to
  one idle worker off the roster; idle workers heartbeated, pre-warmed, expired after `pool_max_wait_s`;
  a pickup watchdog re-dispatched when a worker never started its turn. All of it is gone with the
  query-job worker.

**Checkpoint / resume**
- Conversation: the Claude SDK `SessionStore` (append/load) + `resume=session_id`. Our `GcsSessionStore`
  keys by **`session_id` alone** (ignores `project_key`) so **any worker, any cwd** can resume — the
  default cwd-keyed local store is fatal for serverless.
- Workspace: tar the agent workspace to GCS; restore on resume. Pin the Claude session id up front via
  `options.session_id` (the cloud binary doesn't surface it on messages otherwise).
  The restore extracts with `tarfile`'s `data` filter relaxed for symlinks only (`blobstore.extract_tree`):
  a snapshot always carries `.venv/bin/python -> /usr/local/bin/python3`, which `data` refuses (it aborted
  every cross-worker resume of a coding run, #74), while a symlink is only followed at use — member paths
  still cannot leave the workspace, not even through a restored symlink. Other refused members (FIFOs,
  device nodes, hard links outside) are skipped and listed in `workspace_ready.skipped`. A restore that
  fails removes the directory it created, so the fresh-workspace fallback never clones into leftovers,
  and reports why in `workspace_ready.restore_error`.
- **The agent cwd is a leaf literally named `workspace`** (`.../jobs/<session-id>/workspace/`), on every
  backend. An anonymous `jobs/<uuid>` cwd — typically under `/tmp` — reads as a disposable temp location,
  and weaker models act on that hint: observed with claude-haiku-4.5, whose *first* command was `cd /tmp`;
  it built a perfectly working deliverable in `/tmp/spider_project` and the run was scored `no_deliverable`
  because callers collect artifacts from the job dir. Prompt disclaimers ("stay in your current directory")
  fight the path instead of fixing it. The leaf also keeps `jobs/<sid>/` itself free for session
  bookkeeping the agent shouldn't see. Derived in ONE place (`RunContext.workspace`); callers use the
  `Session.workspace` accessor (local: host `Path`, created on access, seed-before/collect-after; gemini:
  raises — the filesystem is remote) instead of hand-building `workdir/jobs/<sid>`. `local.deploy(spec,
  workspace=...)` swaps that leaf for a caller-owned directory: a per-session path is a per-session system
  prompt, so suites of short sessions pay prompt-cache creation on every one of them (measured 2x on a
  111-attempt trigger suite); one shared cwd is an identical prefix and a cache read. Isolation is the
  caller's to give up — but only isolation: everything that would have the toolkit *write* the shared
  directory is off there, because a caller-owned dir is already durable and holds files no session owns.
  `repos` is rejected (a fixed clone destination admits one session), and a checkpoint snapshots the
  conversation alone — no workspace tar, no credential scrub over the caller's `.git/config`s, no restore
  rolling files back to what one session last saw. `jobs/<sid>/` is still created per session, for the
  bookkeeping (`stderr.log`, the codex home, `list_sessions()`) that never was agent-visible.
- **Finalize inline at the terminal `result` event.** The async executor **stops draining the generator
  after the result event**, so post-loop checkpoint/artifact code is dead in-cloud — it must run as a
  side-effect at the terminal event (the worker's `/events` record and the mirror are the channels).

**Observability (§13: one record — the events)**
- **The worker's `/events` is the live channel**: the client long-polls it (held until new events exist,
  paged under ~1 MB because the proxy caps a response at ~2 MB — measured), so an event is seen within one
  proxy round trip (~0.2 s) of its emission. **The GCS event mirror is the durable record** (`stream.py`):
  the worker streams every surfaced event into small JSONL batch objects under `events/<sid>/` with the
  run-scoped token (~0.5 s cadence; a terminal `result` flushes immediately) — what `Session.history()`
  reads and the client's **fallback stream** when the sandbox stops answering (delivered events are
  skipped; a sandbox that is gone and left no terminal record ends the run with an explained
  `sandbox_unreachable` result after a 30 s grace).
- **Cloud Logging, Cloud Trace and CPU/RAM self-sampling are gone** (decided 2026-09-11): the sandbox has
  no logging identity and gVisor exposes no cgroup files. The history below records what they were and
  what the platform taught us while they existed; observability of sandbox resources from outside the
  container is an open investigation (§12).
- History — **Cloud Logging was emit-only** (per-step `remote_agent_toolkit_steps` log; its READ path is
  capped at 60 `entries.list` requests/min per project, which is why it had been dropped as a data plane
  before this move; ingestion lag exceeded minutes). **Cloud Trace spans per turn** were rebuilt from the
  `AgentEvent` stream (`TurnTracer`; root `invoke_agent` + `execute_tool` children) and needed a bounded
  `force_flush` at turn end because the platform never flushed OTel on the job path. INCIDENT LOG
  (2026-08-03..06): every worker span/metrics batch 403'd on engines deployed from environments whose
  gcloud/ADC default project wasn't the deploy target — `AdkApp.__init__` snapshotted the global config's
  project into its pickle and the worker routed telemetry there; fixed by `build_adk_app` pinning the
  deploy target. **CPU/RAM self-sampling** (`resources.py`) read the worker's cgroup and shipped samples
  to a side log; an OOM now shows up as `sandbox_unreachable`.

**Failure semantics (the client owns liveness; there are no platform retries)**
- **A sandbox that stops answering** — the proxy fails `DIRECT_MAX_FAILURES` polls in a row, or the
  platform says it is gone (TTL, OOM, deleted elsewhere) — switches the stream to the mirror tail; a gone
  sandbox that left no terminal record ends the run with an explained `sandbox_unreachable` error result
  after a 30 s grace. **A dispatch that no sandbox takes** (three attempts, each deleting the sandbox that
  refused) ends the run with `dispatch_failed`. Neither hangs, neither replays the turn silently (history:
  Agent Runtime's job runner re-ran a killed attempt from scratch with the same payload).
- **`interrupt()`** posts `stop` to `/control` and waits for the harness's normal end of turn (checkpoint,
  `INTERRUPTED` result); the fallback — no `control_ready` yet, or no answer within the timeout — cancels the
  driver and **deletes the sandbox**, which is the cancel (and stops the billing).
- **The client dying** leaves the sandbox to its TTL (`max_turn_s` for an on-demand sandbox,
  `pool_max_wait_s + max_turn_s` for a pool one): the platform's backstop.
- The in-sandbox **Bash tool default timeout is 120 s** (Claude Code's own) — long commands need an explicit
  timeout (`BASH_DEFAULT_TIMEOUT_MS` in `spec.env`) or backgrounding.

**Turn control (steer / interrupt / probe a running turn; `control.py`, the worker's `/control` and `/exec`)**
- The client POSTs `{"op": "steer"|"interrupt"|"stop", "message", "message_id"}` to the sandbox worker's
  `/control` through the platform's proxy (~0.3 s); the worker feeds it to the harness's `ControlChannel`
  (an asyncio queue on the turn's loop) and dedupes on `message_id`. The worker announces the channel with a
  `control_ready` event; until then the client **queues** the message in the session (it does not yet know
  which sandbox took the turn, or the worker has not built its channel) and a housekeeping thread posts the
  queue in order the moment `control_ready` is observed on the stream — so `send()` never refuses a message
  during the dispatch window (~1 s from the pool, 15–25 s on a fresh sandbox; agentic-scraping's inbox
  flushes pending user messages exactly then). Messages still queued when the turn ends (dispatch failed,
  sandbox died) are dropped and counted on `RunResult.warning`; `interrupt()` drops the queue and posts its
  `stop` directly. `ControlUnavailable` is left for a *ready* worker that did not take the message.
  History: before §13 the channel was a **GCS inbox**
  (`control/<sid>/…`, polled by the worker every ~1.5 s, delivered markers under `control-delivered/`) because
  a query job had no inbound channel other than cancel; the transport sat behind the `ControlChannel`
  protocol, which is what let it be swapped.
- **What the harnesses do (verified live, 2026-09-04).** Claude Code: a `query()` on the streaming
  client mid-turn is injected into the model's next call and the turn ends with one result; the CLI
  never echoes it, so the harness emits the `user` event and proves delivery the way it tracks
  background-task notifications (a `tool_result`/`init` boundary followed by model output), holding a
  result that arrives with an unproven message open for a short settle window in case the CLI starts a
  new invocation for it. `interrupt()` makes the CLI answer with an `error_during_execution` result —
  the end of the turn for a stop (re-stamped `interrupted`, not an error) or a segment boundary before a
  follow-up `query()`. Codex: `turn/steer` is native (the model sees it after its current step);
  `interrupt()` yields `turn/completed status=interrupted`, after which a new `thread.turn()` on the same
  thread continues the run.
- **`exec()` is the read-only probe** (`Session.exec(command, cwd=, timeout=) -> ExecResult`, #85): the
  client POSTs `{"turn_id", "command", "cwd", "timeout"}` to the worker's `/exec`, which runs it under
  `/bin/bash -c` in the running turn's workspace (the agent's cwd) through `control.run_shell` — the same
  function `local` runs on the host, so the output cap (500 000 characters per stream, `truncated` flagged)
  and the timeout semantics (killed, `returncode` 124) are identical. Only a running turn can be probed
  (the sandbox is deleted at the terminal event; `local` keeps the rule for parity): before the sandbox is
  known `exec()` waits for dispatch, afterwards it raises `ControlUnavailable`. A session re-attached in
  another process holds no run; `exec()` then recovers the running turn's `(sandbox, turn_id)` from the
  mirror — the last `turn_started` marker without a `result` after it; every event is stamped with both
  (#80) — in one read, reused until the worker stops answering. The timeout is capped at 240 s on
  `gemini` because the proxy cuts a call at ~300 s. Motivation: agentic-scraping renders the
  workspace's `git diff` every ~10 s while the coding agent works — exact and harness-independent, which
  the events (paths only on Codex, context-free old/new strings on Claude Code) are not.
- **`interrupt()` is a stop, not a cancel.** The turn's normal end-of-turn path runs (snapshot,
  transcript, terminal result with `StopReason.INTERRUPTED`, accounting kept), so the interrupted turn's
  workspace changes are in the checkpoint the next `send()` restores. `cancel_query_job` remains the
  fallback when the worker never announced the inbox or does not stop within the timeout; the result's
  `warning` says so.

**Persistence / job history (what survives a run, and where)**
- **Events mirror** — `events/<sid>/*.jsonl` under the output bucket, written by the worker as the turn runs
  (the durable record and the fallback live channel); `Session.history()` reads it, `list_sessions()`
  enumerates it (bucket-wide). Every event carries its `turn_id`, so a re-attached session's back-to-back
  turns never replay each other's results (#80).
- **Configs** — the session's `SessionConfig` at a stable key (`session-config/<sid>.json`, what re-attach
  reads back; a re-attacher cannot substitute one) and each turn's `TurnConfig` nonce-keyed; both also ride
  the turn body, so the worker never reads them from GCS. Kept 30 days.
- **Deploy record** — `deploys/<name>/<template>.json`: the spec and settings a template was deployed with,
  so a `get_engine` handle knows the baked spec and the model identity. Kept.
- **Token objects** — `invocation-secrets/<sid>-gcs-token.json`: the run-scoped GCS token's refresh object
  (a value; deleted at turn end, reaped after a day).
- **Checkpoints** — transcript + workspace tar under `checkpoints/`, keyed by (mapped) session id.
- Session ids are client-minted canonical UUIDs (the Claude CLI rejects non-UUID session ids; a non-UUID id
  from an older record maps through `uuid5`). Nothing is persisted by the platform any more: no job input,
  no job output, no ADK session. History: before §13 `history()` fell back to the platform's job output and
  to Cloud Logging, and cold turns had numeric ADK session ids.

**Deploy contracts (applied automatically by `gemini.deploy`; `_image.py`)**
- **The image**: `python:3.12-slim-bookworm` (glibc — the `claude-agent-sdk` wheel ships a self-contained
  glibc ELF `claude` binary; no Alpine/musl), `git`, `curl`, build tools; the base requirements
  (`claude-agent-sdk` pin, `openai-codex` when the Codex harness is baked, `google-cloud-storage`, `uv`, …)
  merged with `spec.packages` (an unsatisfiable merge fails before the build); the toolkit package copied
  from the deploying venv (the image runs THIS toolkit revision — client and image are one wire contract);
  the resolved skills flat under `/opt/toolkit/skills`; the spec at `/opt/toolkit/spec.json`; a non-root
  `agent` user (uid 1000; the sandbox requires non-root); `/workspace`; the worker as `CMD` on port 8080.
- **The tag is a content digest** of the whole build context, so an unchanged spec from an unchanged
  toolkit skips the build (`docker manifest inspect`) and a deploy whose image and resources match the
  newest template is a no-op. Docker CLI + a registry login are the deploy's prerequisites; the platform no
  longer builds for us (history: the Agent Runtime build took ~4 min, pinned a pickle-coupled
  aiplatform/cloudpickle/pydantic layer via `constraints.txt`, and needed `NUM_WORKERS=1` to keep the
  platform's uvicorn from spawning `cpu_count + 1` processes; all gone).
- **Template**: the image, `ports=[8080]`, `resources` requests = limits = `resource_limits` (default
  default 4 CPU / 4 GiB; 4 / 8 GiB run throughout the spike; the platform refuses more than 8 vCPU; 16 GiB verified), `egress_control_config.internet_access`.
  `customContainerSpec` carries only `imageUri` — no command, args or env — so everything per run arrives
  over HTTP after creation.

**Identity / IAM (§13: two identities, and the sandbox has none)**
- **The client identity** (an impersonated operator SA, or the caller's own) is the whole control plane:
  `roles/aiplatform.user` on the project (sandbox + template permissions, the host `reasoningEngine`),
  `roles/storage.admin` on the output bucket (records, roster, minting the run-scoped tokens),
  `artifactregistry.writer` on the image repo, `iam.serviceAccountTokenCreator` on the model SA. There is
  **no per-sandbox IAM**: `aiplatform.sandboxEnvironments.execute` covers every sandbox under the host
  instance, so the client identity is the trust boundary and the roster is only a coordination device.
- **The model identity** (`ratk-model@`, `model_token.py`): a service account with only the
  `ratkRuntimePredict` custom role (`aiplatform.endpoints.predict`). The client mints its access token per
  turn (an hour by default; up to `max_turn_s` ≤ 12 h when the org policy
  `constraints/iam.allowServiceAccountCredentialLifetimeExtension` lists the account) and hands it to the
  worker as `ANTHROPIC_AUTH_TOKEN` with `CLAUDE_CODE_USE_VERTEX=1` / `CLAUDE_CODE_SKIP_VERTEX_AUTH=1`
  (verified live). Claude Code reads it once and does not consult `apiKeyHelper` in this mode (checked
  2026-09-11), so the token's lifetime bounds the turn's model access.
- **The sandbox has no usable Google identity** (verified live 2026-09-10 and by every `live-smoke` run):
  the metadata server answers with a tenant-project workload identity that is 403 on the project's storage,
  Vertex and resource manager. So what the agent's shell can reach is exactly what the turn brought: the
  run-scoped GCS token (its own prefixes), the model token (predict only), the turn's own secrets.
- **The platform side**: the Agent Sandbox service agent (`service-<number>@gcp-sa-vertex-sandbox`) needs
  `artifactregistry.reader` on the image repo to pull the image. Nothing else is granted to Google.
- **Run-scoped GCS tokens** (`scoped_gcs.py`, kept from PR #41): the client mints one downscoped token per
  turn (a Credential Access Boundary: this bucket, only the run's own prefixes — configs, events mirror,
  checkpoints, artifacts; one list clause, the transcript dir), sends it in the turn body, refreshes it
  while the run lives by overwriting one object under the run's prefix, and the worker does all GCS work
  with it via the `GcsBlobStore` process default. STS caps a minted token at ~10.7k chars including the
  caller's own, so list clauses are kept to one.
- History (PR #41, 2026-09-02..04): on Agent Runtime the worker container ran as the engine's **runtime
  identity**, reachable from the agent's shell through the metadata server, and that identity shared the
  output bucket with every other run — so the fixes were run-scoped tokens, per-worker Pub/Sub dispatch and
  a custom runtime service account with a predict-only role. The runtime service account, its roles, the
  conditional bucket bindings and the migration are gone with the query-job worker; §13 explains why the
  sandbox makes them unnecessary.

---

## 7. Ports & adapters

Each is a `typing.Protocol`; concrete adapters ship for prod (GCP) and dev (local/in-memory).

- **`Harness`** — `build_options(spec, ctx)` + an async run loop yielding `AgentEvent`s. Adapters:
  `ClaudeCodeHarness` (default) and `CodexHarness`, selected by `spec.harness` via `resolve_harness`
  (the single seam both runtimes use). Shared policy — secret routing, agent-env layering, the
  interactive suffix, the inline workspace checkpoint — lives in `harness/_shared.py` so the bindings
  can't drift where the spec doesn't distinguish them. Both read `ctx.control` (a `ControlChannel`)
  alongside their stream through `control.ControlledStream` and act on steer / interrupt / stop
  messages the same way (§6 "Turn control").
- **`ControlChannel`** — the harness side of turn control: `receive()` blocks for the next operator
  message. Adapters: `LocalControlChannel` (in-process queue, `local`) and `GcsControlChannel` (the
  worker polling the GCS inbox, `gemini`). A **conformance suite**
  (`run_harness_conformance`) pins what embedders read off `AgentEvent.raw` across every backend: an
  init status event and the terminal result's spend/turn count. `run_claude_code_harness_conformance`
  layers Claude Code's stricter promise on top — the init payload's session id — which Codex's init
  event does not carry.
- **`CodexHarness`** (OpenAI Codex binding, `openai-codex` SDK) — one `AsyncCodex` app-server per run,
  against a per-job `CODEX_HOME` (auth.json, rollouts — bookkeeping level, beside the workspace).
  OpenAI models are called **directly** (no Vertex path): the `OPENAI_API_KEY` travels as a
  per-invocation secret, is routed to `codex login` and excluded from the agent's env/shell. Spec
  translation: `permission_mode` → sandbox+approval (bypassPermissions = full access + never ask;
  default = workspace-write + Codex's auto-reviewer so headless runs never block); `system_prompt`
  str → `base_instructions`, `SystemPrompt.append` → `developer_instructions`; skills → staged into
  `<cwd>/.agents/skills` (same SKILL.md format, Codex's repo-level discovery — verified live: the
  model reads a staged SKILL.md unprompted); MCP servers → `--config mcp_servers.*` overrides (a
  github server's token rides `bearer_token_env_var` + process env, never argv); `output_schema` →
  per-turn `output_schema`; `reasoning_effort` → per-turn `effort` (re-applied every turn, so resume
  keeps it; the Claude-only `max` maps to `xhigh` with a status warning). Codex's shell-env policy default (filter `*KEY*`/`*TOKEN*` names from the
  shell) is the opposite of the toolkit's contract, so the default excludes are lifted with the two
  harness-consumed names explicitly re-excluded. **Parity gaps handled adapter-side**: Codex reports
  token counts but no USD and enforces no caps — the harness prices tokens via LiteLLM's live
  pricing dataset (`harness/pricing.py`; the same upstream `ccusage` uses, fetched once per process,
  baked 5.6-family fallback for offline; models unknown to both: cost `None` + a `cost_unknown`
  status, budget unenforceable; the result raw carries `price_source`). The harness also counts
  `thread/tokenUsage/updated` notifications as model calls (= `num_turns`; verified live: one per
  call), and interrupts the turn at `max_turns` / `max_budget_usd` (`error_max_turns` /
  `error_budget_exceeded` result subtypes, accounting kept). **Checkpoint/resume**: the workspace
  snapshot is the shared machinery; the conversation is Codex's local rollout file
  (`CODEX_HOME/sessions/**/rollout-*-<thread_id>.jsonl`) — worthless across serverless workers — so
  the harness copies it to the BlobStore keyed by OUR session id at the terminal result and restores
  it before `thread_resume` on any worker. No Codex equivalent of Claude Code's background-task
  re-invocation exists (`spec.background_task_timeout` is inert); `allowed_tools`/`disallowed_tools`
  have no mapping and are ignored with a status warning.
- **`openrouter/` models go through a per-run localhost proxy** (`harness/_openrouter_proxy.py`), on
  **both** bindings, Claude Code and Codex. Two things force this. The CLIs cannot send OpenRouter's
  `provider` request field, and they cannot report what OpenRouter charged. The proxy puts the
  caller's routing object in the request body verbatim. `spec.openrouter_provider` is the shorthand
  for pinning one provider and is resolved to `{"only": [slug], "allow_fallbacks": false}` before
  the proxy starts, so only one form travels below the spec. It passes the response through
  unchanged and reads the routing metadata and the charge out of it. Each response becomes one
  `openrouter_request` event, failed responses included — nothing else can see a retried 429. The
  CLI holds a random per-run token, never the account key. The summed charges become the result's
  `cost_usd` (`price_source="openrouter"`). The Claude Code binding also keeps the CLI's own figure
  as `cli_reported_cost_usd`; the Codex CLI reports no USD at all, so it has none to keep. These
  models are never priced from the table, and a turn OpenRouter reports no charge for reports none.
  The budget is measured against that total, and the proxy answers 402 once the cap is spent.
- **Background-task semantics are honored** (eval feedback: a model armed the Monitor tool and ended its
  turn — correct, trained behavior — and the one-shot `query()` tore the CLI down, firing the advertised
  notification into the void; the run was scored no-deliverable). The CLI itself re-invokes the model when
  a background task (Bash `run_in_background`, Monitor) reaches a terminal state — **but only while the
  message stream stays open** (proven live: model ends turn → `task_notification` arrives → CLI re-invokes,
  zero messages injected by us). The harness therefore drives a `ClaudeSDKClient` stream instead of
  `query()`, tracks the task lifecycle (`task_started` / terminal-via-`task_notification`-OR-`task_updated`
  — the SDK warns terminal state can arrive as an update patch only), and treats a result event with
  pending or just-terminal-undelivered tasks as a *segment boundary* (demoted to an `awaiting_tasks`
  status), ending the turn only at a result with no pending work. `spec.background_task_timeout` (default
  1 h) bounds each pending wait. Delivery is proven by stream ordering (verified live, CLI 2.1.223): the
  CLI injects a terminal notification into the next model call it *assembles*, so a notification followed
  by a tool-result boundary and then model output is delivered (the result finalizes immediately — this is
  the common case, since the CLI auto-backgrounds long foreground commands too); a notification that lands
  while the final call is already in flight is NOT — the CLI re-invokes moments after the result, and a
  20 s grace holds the turn open to catch that re-invocation (or, if none comes, expires and finalizes the
  result already produced). Accounting: the CLI's `total_cost_usd` is cumulative across re-invocations (take the last);
  `num_turns` resets per re-invocation (the harness sums — and `spec.max_turns` therefore caps each
  invocation, not the whole run; `max_budget_usd` stays a global cap). An error result finalizes
  immediately — never wait on tasks a dead conversation can't consume. This also kills the poll tax:
  a well-behaved agent no longer burns a turn per `sleep 60; tail crawl.log` poll during a long crawl.
- **`BlobStore`** — `put_bytes / get_bytes / put_tree / get_tree / exists / list / delete`. Adapters:
  `GcsBlobStore`, `LocalBlobStore`. Backs checkpoint, workspace, event mirrors, the secrets handoff,
  artifacts.
- **`EventSink`** — `emit(event)` (write side, runtime) + `tail(session_id, since)` (live read side) +
  `read(session_id)` (one-shot history read). Adapters: `CloudLoggingSink`, `InMemorySink`.
- **`DispatchTransport`** — `publish(message, attributes)` + `claim(timeout)` (an addressed per-worker
  pull; plain competing-consumers only for legacy shared-subscription pools). Adapters:
  `PubSubDispatch`, `InMemoryDispatch`.
- **`SecretResolver`** — `resolve(name) -> value`, a *control-plane* helper for callers who keep secret
  values in env/Secret Manager and need to build the per-invocation `secrets` dict (the runtime itself
  receives values via the staged GCS handoff, §3.5). Adapters: `EnvSecretResolver` (implemented),
  `GcpSecretResolver` (stub).
- **`SessionStore`** — the Claude SDK protocol (`append` / `load` / `list_subkeys`). Adapter:
  `GcsSessionStore` (keyed by `session_id`). A **conformance suite** (`run_session_store_conformance`)
  validates any implementation.

---

## 8. What lifts from the PoC

| Capability | PoC source | Disposition |
|---|---|---|
| ADK wrapper + run loop | `agent.py` (`ClaudeCodeAgent`) | → `harness/claude_code.py` (generalized) |
| Event translation | `event_mapping.py` | → `harness/translate.py` (GENERIC) |
| Checkpoint / resume | `checkpoint/` | → `checkpoint/` over `BlobStore` |
| Warm pool | `pool.py` + worker loop | → the ready pool (`runtime/gemini/roster.py`; the Pub/Sub warm pool was retired by §13) |
| Skills staging | `skills.py` | → `skills.py` (configurable sources) |
| Git / GitHub MCP | `git_repo.py`, `_mcp_servers` | → `integrations/` |
| Artifacts | `artifacts.py` | → over `BlobStore` |
| Cost tracking | result metadata passthrough | → `RunResult` |
| Config / runtime resolution | `config.py` | → `AgentSpec` + resolvers |
| Deploy | `deploy/deploy_agent_engine.py` | → `runtime/gemini/_image.py` + `backend.deploy` (contracts encoded) |
| Control-plane dispatch + log tail | `scrape_cli.py` | → `Session` / the worker's `/events` long-poll |
| **Scrapy Cloud, monitoring, scrape prompts** | `selfheal/`, `_INTERACTIVE_SUFFIX` | **stay in gemini-agent-runtime** behind the seam |
| Probes, scratch, `gh_pat.txt` | `scratch_minimal/`, probes | **dropped** |

---

## 9. Decided design choices (2026-06-29)

1. **Harness scope** — seams now, Claude-first implementation. Harness-agnostic `AgentSpec` + a
   `Harness` protocol. (Superseded 2026-07: the seam paid off — a second binding, `CodexHarness`,
   now ships behind `spec.harness`; see §7.)
2. **Definition API** — **declarative `AgentSpec` + `deploy()`** (control-plane / data-plane split), not an
   imperative builder or thin functions.
3. **First milestone** — a **minimal generic (non-Zyte) example agent** run local + on Gemini Agent Runtime,
   proving the core API + deploy + warm pool with the least surface, before porting a real PoC.
4. **Local deploy is first-class** — `local.deploy(spec)` mirrors `gemini` exactly (same Engine/Session
   API), so the dev loop and the prod loop are the same code.
5. **App code looks up engines, never deploys** — `gemini.get_engine(name[, version])` is the hot path;
   `gemini.deploy(...)` is ops/CI.

---

## 10. Package layout

(No `src/` layer — the package sits at the repo root.)

```
remote-agent-toolkit/
├── pyproject.toml                 # all deps in core (no extras); dev tooling in a dependency group
├── README.md                      # team onboarding (grows each phase, §11); the only user-facing doc
├── DESIGN.md                      # this file (internal design record)
├── remote_agent_toolkit/
│   ├── __init__.py                # AgentSpec, SystemPrompt, SkillSource, McpServer, gemini, local
│   ├── spec.py                    # AgentSpec + value types (serializable)
│   ├── events.py                  # AgentEvent, RunResult, RunStatus, StopReason, Run handle
│   ├── control.py                 # ControlMessage, ControlChannel, LocalControlChannel, ControlledStream
│   ├── harness/
│   │   ├── base.py                # Harness protocol (+ resolve_harness in __init__)
│   │   ├── claude_code.py         # ClaudeCodeHarness (drives a ClaudeSDKClient stream; ADK-free)
│   │   ├── codex.py               # CodexHarness (drives an openai-codex AsyncCodex app-server)
│   │   ├── _shared.py             # policy shared by the bindings (secret routing, env, checkpoint)
│   │   ├── _openrouter_proxy.py   # per-run localhost proxy for openrouter/ models (both bindings)
│   │   ├── pricing.py             # LiteLLM price lookup + baked OpenAI fallbacks, context windows
│   │   └── translate.py           # Claude SDK message → AgentEvent (codex's lives in codex.py)
│   ├── runtime/
│   │   ├── base.py                # Engine + Session protocol + state machine + Run handle
│   │   ├── local.py               # local.deploy / local.run -> LocalEngine
│   │   ├── venv.py                # per-engine uv venv for spec.packages on local
│   │   └── gemini/
│   │       ├── backend.py         # deploy(), get_engine(), list_engines(), GeminiEngine, GeminiSession
│   │       ├── provider.py        # SandboxProvider seam + AgentSandboxProvider (the platform adapter)
│   │       ├── worker.py          # the in-container worker: /health /turn /events /control /token /exec
│   │       ├── _image.py          # Dockerfile generation, content-digest tag, docker build/push
│   │       │                      #   (underscored: `deploy.py` would shadow gemini.deploy)
│   │       ├── model_token.py     # the predict-only model identity's per-turn token
│   │       ├── scoped_gcs.py      # the run-scoped GCS token (mint / refresh / worker credentials)
│   │       ├── roster.py          # the ready pool's idle-sandbox roster (GCS, atomic claim)
│   │       ├── handoff.py         # the GCS records: configs, deploy record, lifecycle rules
│   │       ├── stream.py          # the GCS event mirror (writer) + its tail (the fallback stream)
│   │       ├── history.py         # history / list_sessions readers over the mirror
│   │       └── project_setup.py   # ratk-gcp-setup
│   ├── ports/
│   │   ├── blobstore.py           # BlobStore + GcsBlobStore + LocalBlobStore
│   │   └── secrets.py             # SecretResolver + GcpSecretResolver + EnvSecretResolver
│   ├── checkpoint/
│   │   ├── session_store.py       # GcsSessionStore (Claude SDK protocol)
│   │   └── workspace.py           # snapshot / restore over BlobStore
│   ├── integrations/
│   │   ├── git.py                 # clone + PAT injection
│   │   └── github.py              # GitHub MCP attach + (optional) REST helpers
│   ├── skills.py                  # discover + provision from SkillSource[]
│   └── conformance.py             # SessionStore/BlobStore conformance test
├── examples/
│   └── minimal/                   # the first-milestone generic example agent
└── tests/
```

---

## 11. Roadmap

The **README grows with the code** — each phase leaves the README runnable for what's shipped, rather than
saving all onboarding docs for the end.

- **P0 — Scaffold.** Repo skeleton (no `src/`), `pyproject` (all deps in core; dev tooling in a dependency
  group), the port & harness protocols as real signatures, the conformance harness, CI.
  *README:* skeleton + install + the headline example as the north-star target.
- **P1 — Generic core + local.** `ClaudeCodeHarness`, checkpoint/SessionStore, skills, git/github,
  artifacts, `AgentSpec`, `local.deploy`/`local.run` with filesystem/in-memory adapters.
  *Gate:* the generic example runs in-process. *README:* the **local quickstart** works end-to-end
  (define → `local.deploy` → `run` / stream / poll).
- **P2 — Gemini backend + warm pool + deploy.** `gemini.deploy` (contracts encoded) + `gemini.get_engine`
  / `list_engines`; the engine handle & `Session` state machine; `WarmPool` over Pub/Sub; Cloud-Logging
  tailing; structured outputs. Permissions runbook. (Done; re-based on Agent Sandbox in §13, 2026-09-11.)
  *First-milestone gate:* **the minimal generic example runs locally *and* remotely with a warm start.**
  *README:* the **deploy quickstart** + engine management.
- **P3 — Onboarding polish.** Examples, the platform-notes appendix, conformance docs, README polish
  (troubleshooting, IAM runbook link, the "why" sections).
- **P4 — Migrate the PoC.** `gemini-agent-runtime` depends on the toolkit; re-implement self-heal +
  interactive spider on top; delete the lifted code; `scrapy` becomes an extra there.

---

## 12. Open questions / future work

- **Multi-source skills** — `SkillSource[]` is designed for base + extra sources; the resolver/merge
  (precedence on name collision, per-source ref pinning) is specified in P1 but the merge policy needs a
  decision. (P1 ships **last-source-wins**.)
- **Baked engine dependencies** — `AgentSpec.packages` means "the agent's starting Python packages" on
  BOTH backends: `gemini.deploy` encodes them via the uv-via-requirements contract; `local.deploy`
  resolves them into a per-engine venv (uv, engine-contract Python 3.12) activated in the agent env.
  (Decided after eval-harness feedback: a spec field silently meaning different things per backend broke
  local/remote parity; a Docker-based local mode was rejected — it would trade away the in-process fast
  dev loop that `local` exists for, and uv's cache makes the venv path near-instant.)
- **Local parity with the engine image** — *Python package* parity is handled by the venv above; full
  OS-level parity (base OS, glibc, system tools) is the `dev/` parity image's job (install/dependency
  issues debugged on the laptop instead of through ~10-min cloud rebuilds).
- **Quota-free event streaming — BUILT (2026-08, `stream.py`; the channel is described in §6).** The
  Cloud Logging tail shared one fixed 60-reads/min budget per project, so live-event latency degraded
  as concurrent runs grew (tens of agents ⇒ ~30 s batches); the incremental GCS mirror removed that
  wall. Alternatives considered and rejected, for the record: Cloud Logging's streaming `tail` API
  (hard cap of 10 concurrent tail sessions per project, also fixed — a wall below "tens of agents");
  a log sink → Pub/Sub fan-out (works, but new infra + per-client subscription lifecycle + every
  client receives all sessions' traffic); per-process tail multiplexing (helps one client with many
  sessions, not many clients); the platform's native streams — sync `streamQuery` and A2A
  `tasks/subscribe` — which live on the serving path (~10-min ceiling, standing-container billing),
  not the async-job path the run plane is built on; and managed `sessions.events`, which would lean on
  the platform session machinery that broke under us on 2026-07-28 (and our events are deliberately
  `partial`, i.e. not appended). Worth raising in the existing Google query-job telemetry thread: a
  native progress-stream for `asyncQuery` would let us delete this channel entirely.
- **Structured outputs — BUILT (2026-08).** Both harnesses send the schema through the CLI's own
  structured-output option and keep "parse the last JSON block" only as a fallback. OpenRouter turns
  also get the schema in the prompt, because OpenRouter accepts the schema but leaves enforcement to
  the model. Still open: confirming Vertex model support per model.
- **Outcomes / rubrics** — CMA's iterate-until-graded "definition of done" is attractive for autonomous
  background work; candidate post-P4 capability (a `verify=Rubric(...)` on `AgentSpec`).
- **Second harness** — codex (subprocess) or a custom LLMNL harness behind the `Harness` protocol; the
  seam exists from P0, the binding is future.
- **Non-GCP backends** — the ports are protocols; an AWS/Azure adapter set is possible but unscheduled.
- **Scheduled runs** — CMA-style cron `deployments` (each firing = a session) could wrap deploy+session
  for recurring jobs (e.g. self-heal monitoring); future.
- **Sandbox observability** — the sandbox runtime (§13) dropped Cloud Logging, Cloud Trace and in-sandbox
  resource sampling; what the platform exposes about a sandbox's CPU/memory from outside the container is
  to be investigated. Until then an OOM shows up as a `sandbox_unreachable` result.
- **Long turns and the model token** — Claude Code reads its Vertex token once; turns longer than the
  token lose model access. The org-policy lifetime extension (≤ 12 h) is the current answer; a per-sandbox
  proxy that injects a fresh token (the OpenRouter proxy is the pattern) would lift the ceiling.
- **Other sandbox providers** — the platform surface the runtime needs is small (§13.6); no plan, but the
  `SandboxProvider` seam keeps the door open.

## 13. Sandbox runtime (spike 2026-09-10; built 2026-09-11)

**Status: built on branch `sandbox-spike` (PR #84), live smoke passing; in-team adoption next.** The
`gemini` backend, re-based on a Gemini Enterprise Agent Platform **Agent Sandbox custom container** instead
of an Agent Runtime query job — **replaced in place**, not added beside: same module, same `Engine` /
`Session` / `Run` API, everything the query-job path needed and the sandbox path does not is deleted
(`_deploy.py`, `adk_agent.py`, `pool.py`, `tracing.py`, `resources.py`, `revisions.py`, `translate.py`, the
GCS control inbox, `constraints.txt`, the `DispatchTransport` and `EventSink` ports, the ADK/aiplatform/
Pub/Sub/Logging dependencies). New: `provider.py` (the seam, §13.6), `worker.py`, `_image.py`,
`model_token.py`; `roster.py` keyed by template; `history.py` mirror-only. Delivery: a long-lived branch,
adopted first by the in-team consumers (self-healing, agentic-scraping) until battle-tested, then merged
for the other teams with the CHANGELOG's backwards-incompatible notes (§13.3). Spike code and raw numbers:
`dev/sandbox_spike/` (image, runner, probe, limits probe, README with the result and limits tables).

**Measured on the built runtime (2026-09-11, `make live-smoke`, 4 CPU / 8 GiB, Haiku, `checkpoint=True`)**:
deploy 81 s (image build 32 s on a warm Docker cache, push 13 s, template 18 s, pool fill 15 s); pool turn
**1.3 s to the first event, 6.6 s to the result**; fresh-sandbox turns 15–23 s to the first event, 22–28 s to
the result; a steer delivered into a running turn and acknowledged; checkpoint resume restored on a new
sandbox. The spike's table below is the pre-build comparison against the warm pool.

**Why.** PR #41 closed the cross-run reach of the runtime identity with tokens, IAM and per-worker dispatch,
but every one of those is our machinery on top of a platform that hands the agent's shell a Google identity
and a 2.5 min per-job boot. Agent Sandbox inverts both: the container has **no usable Google identity** (the
metadata server answers with a tenant-project workload identity that is 403 on everything in our project —
verified), and Google runs a **pre-warmed pool per template**, so a sandbox is assigned in ~2 s. The spike
put the whole harness (Claude Code via `local.deploy`) behind a stdlib HTTP server in such a container and
measured it against the warm pool with the identical one-tool Haiku turn (`dev/live_warm_latency_probe.py`).

**Measured (my-project / us-central1, 4 CPU / 8 GiB template, egress on; 3 runs, 7 turns, all `42`)**

| | Sandbox | Agent Runtime today |
|---|---|---|
| dispatch → first event, ready worker | **0.8–1.4 s** | 4.2 s (warm pool) |
| dispatch → result, one-tool Haiku turn | **3.6–5.7 s** | 14.5 s (warm pool) |
| no ready worker: create → result | **18–26 s** | ~150 s (cold job) |
| refill (create → first proxied answer) | 2 s + 12–31 s route propagation | 150–300 s boot |
| template create (Google's pool starts our image) | 78 s | ~4 min image build + revision |
| idle 60 min, then a turn | answers the first poll in 0.9 s | n/a |
| exec resets the TTL? | **no** (480 s TTL gone at 540 s despite exec at 420 s) | n/a |
| proxy round trip per HTTP call | 0.2–0.3 s | — |
| in the container | gVisor, uid 1000 = PID 1, `/workspace` + `$HOME` writable, claude 2.1.222, git, uv; PyPI reachable | root-equivalent single user, metadata server = runtime identity |

Container facts that shape the design: the readiness lag is **route propagation, not boot** (at first
answer the container had 77 s uptime — the pool started it at template creation), so it only affects
refill; the template's declared `ports` are the only ones the proxy forwards; `customContainerSpec` has
**only `imageUri`** (no command, args or env) so everything per-run arrives over HTTP after creation;
pause/resume broke the sandbox in the shell probe and is not used; egress is an all-or-nothing
`internet_access` flag; there is **no per-sandbox IAM** (`aiplatform.sandboxEnvironments.execute` covers
every sandbox under the parent instance); the shell/custom-container surface is **v1beta1** and needs the
2.x `agentplatform` SDK (`client.sandboxes`, `client.runtimes`) while the toolkit pins aiplatform `<2`.

### 13.1 Architecture

```
 DEPLOY (ops/CI)                                RUN (app code; the client IS the control plane)
 ───────────────                                ────────────────────────────────────────────────
 gemini.deploy(spec)                            gemini.get_engine(name) → Engine
   docker build (Dockerfile = toolkit + harness    start_session() → Session
     CLIs + baked skills + spec.packages)          run(msg):
   push → Artifact Registry                          1. claim a READY sandbox off the roster (GCS,
   create SandboxEnvironmentTemplate                    precondition delete — roster.py unchanged)
     (image, ports=[8080], resources, egress)           or create one (~20 s) when the roster is empty
   pre-create N sandboxes, wait /health,             2. POST /turn {spec overlay, prompt, secrets,
     record them in the roster                          run-scoped GCS token, model token}
   template name  ⇔  engine version                  3. stream: tail the GCS event mirror (as today)
                                                        or poll /events directly (0.2 s/call)
                                                     4. background: refill the roster (create + warm)
 IN THE CONTAINER (server.py → worker)               5. at the terminal event: delete the sandbox
   HTTP on :8080 behind Google's authenticated          (or snapshot+delete later, §13.6)
   proxy: /health /turn /events /control /exec
   one turn per sandbox process; the harness
   runs exactly as under adk_agent._run_turn
   minus ADK; events → GCS mirror with the
   run-scoped token; checkpoint tar → GCS
```

- **Deploy = image + template.** `gemini.deploy` builds the image from a generated Dockerfile (the
  `dev/Dockerfile` contract: Debian, Python 3.12, git, uv, the toolkit with the baked harness CLIs, resolved
  skills, `spec.packages`), pushes it to an Artifact Registry repo in the project, and creates a template
  named `<spec.name>` with the image, `ports=[8080]`, `resources` (today's `resource_limits`) and
  `egress_control_config.internet_access=True`. Templates are immutable, so **a template is a version** (the
  image tag is the toolkit+spec digest); `get_engine(name)` resolves the newest template of that display
  name, `version=` pins one. Cutover = create the new template, fill its pool, then drain and delete the old
  one (mirrors today's generation cutover, minus Pub/Sub). Needs Docker (or Cloud Build) on the deploying
  machine — the platform no longer builds for us.
- **The client is the whole control plane.** No Agent Runtime engine, no query jobs, no runtime service
  account, no Pub/Sub, no per-worker subscriptions, no pickup watchdog: `pool.py`'s dispatch half and the
  `DispatchTransport` port go; `roster.py` stays as the ready-sandbox roster (its `expires_at` = sandbox
  create time + TTL, which the platform enforces — exec does not extend it). The parent "instance"
  (`reasoningEngine`) is an empty resource created once per project/location (sandboxes must hang off one).
- **Turn protocol.** One `POST /turn` carries what today rides three GCS objects and a job input: the
  effective spec overlay (deploy-baked ← session ← turn merged client-side, echoed back as `effective_spec`
  as now), the prompt, the per-invocation **secrets**, the **run-scoped GCS token** (`scoped_gcs.py`
  unchanged: same boundary rules, same refresh-by-object; the worker refreshes from the object as today),
  and the **model token**. Nothing is persisted by the platform: no `jobs/<sid>_input.jsonl`, no
  `invocation-secrets/` object, no 1-day reaper (the persisted-input exposure class from PR #41 disappears).
  Session/turn configs are still written to GCS by the client for the 30-day post-mortem record, but the
  worker no longer reads them from there.
- **Model credentials.** The sandbox has no Google identity, so the model token is ours to provide: Claude
  Code accepts a Vertex OAuth bearer via `CLAUDE_CODE_USE_VERTEX=1 CLAUDE_CODE_SKIP_VERTEX_AUTH=1
  ANTHROPIC_AUTH_TOKEN=<token>` (verified locally and in the spike). The client mints it by impersonating a
  **predict-only service account** (the `ratkRuntimePredict` custom role is exactly right; the client needs
  `serviceAccountTokenCreator` on it) with a 1 h lifetime and refreshes it for long turns the same way the
  GCS token is refreshed (a token object under the run's prefix, read by a Claude Code `apiKeyHelper` in the
  image, or re-injected over `/turn`). The agent's shell can read it — as it can read the metadata-server
  token today — but it is predict-only, run-scoped in time, and there is no metadata server behind it.
  `use_vertex=False` (API-key mode) and OpenRouter work unchanged (the key rides `secrets`).
- **Events: direct by default, mirror as the record.** The live stream a `Run` consumes comes straight from
  the worker: `POST /events {since, wait}` **long-polls** — the worker holds the request until new events
  exist or `wait` seconds pass (the per-call proxy budget allows well over a minute), so a streaming client
  issues one call per event batch, not a fixed-cadence poll, and sees an event within one proxy round trip
  (~0.2 s) of its emission. This is what the spike measured (first event ~1 s; a mirror tail adds the
  ~0.5 s batch flush plus a listing poll). The worker still writes the **GCS event mirror** with the
  run-scoped token (`stream.py` unchanged) as the **durable record**: `history()`, `list_sessions()`,
  `last_result` of a finished session, and the fallback stream when the sandbox is unreachable (turn
  finished and sandbox deleted; `execute` quota or a proxy error — the client switches to the mirror tail
  and back). The sandbox name is recorded in the session's GCS record at dispatch so a client that
  re-attaches mid-turn (`get_session(sid)`) can long-poll directly too. Open item for the limit probes: the
  `execute` call quota (undocumented) — long-polling keeps the call rate at roughly one per event batch per
  streaming client, which is far below any plausible per-minute limit, and the fallback exists regardless.
- **Control (steer / interrupt / stop).** `/control` on the worker replaces the GCS inbox: sub-second, no
  polling loop in the worker. `GcsControlChannel` can stay as the fallback for a client that re-attached
  without the sandbox name; the `ControlChannel` protocol already abstracts this.
- **Checkpoint / resume / interactive.** Unchanged: transcript + workspace tar to GCS at the terminal event,
  restore on the next turn's sandbox (#74's symlink fix included). A sandbox is deleted at turn end — it
  bills while it exists, whereas a Runtime job between turns costs nothing — so multi-turn continuity still
  rides the checkpoint. Native sandbox **snapshots** (disk + memory) were considered and set aside on
  purpose, not only for maturity: a snapshot restores **only onto the identical image**, and we update
  agents (new template) far more often than we retire old sessions — a job checkpointed on one revision
  must resume on the next one, which the tar does and a snapshot cannot. Keep the note; revisit only if the
  platform lifts the same-image restriction.
- **Liveness.** No platform retries and no job handle: the client owns it. `/health` polls during a turn,
  a turn-level deadline, and the sandbox TTL (set at creation to the max turn length + margin) as the
  backstop for a client that dies. Deleting a sandbox is the cancel. A startup sweep deletes stale
  sandboxes of our display-name prefix that no roster entry or live run claims (as `delete` sweeps
  subscriptions today).
- **IAM (one identity fewer).** Client: `aiplatform.sandboxEnvironments.{create,get,list,execute,delete}`,
  `sandboxEnvironmentTemplates.*`, `reasoningEngines.get` on the project (a custom role; today these live only
  in `roles/aiplatform.user` and broader), bucket rights it already has, `serviceAccountTokenCreator` on the
  predict-only account, Artifact Registry writer for deploys. Platform: the Agent Sandbox service agent
  (`service-<number>@gcp-sa-vertex-sandbox`) needs `artifactregistry.reader` on the image repo. Gone: the
  runtime service account and its six roles, the conditional `jobs/` / `events/ratk-` bucket bindings,
  `pubsub.subscriber`, the "never grant `pool/`" rule (the roster is now purely a coordination device — with
  no per-sandbox IAM, whoever can execute on the instance can drive any sandbox, so the trust boundary is the
  client identity, as it already is). `ratk-gcp-setup` shrinks accordingly.
- **Cost.** Same rates ($0.085/vCPU-h + $0.009/GiB-h). A ready sandbox bills like an idle warm worker; a
  sandbox is billed while it exists, so delete at turn end and size the ready pool + TTL to the dispatch
  pattern exactly as `pool_size` / `pool_max_wait_s` are sized today. The template's Google-side pre-warmed
  pool: billing not documented (open question).

### 13.2 Feature survey: what carries over, what changes, what is lost

| Feature (today) | Sandbox runtime | Notes |
|---|---|---|
| `AgentSpec`, `SessionConfig`, `TurnConfig`, three scopes, `effective_spec` echo | **unchanged** | merged client-side, executed by the same harness code |
| Harnesses (Claude Code, Codex), `harnesses=` baking, OpenRouter proxy, MCP servers, skills (git/local/builtin), `repos`, structured output, usage/cost accounting, stderr capture, late-result handling | **unchanged** | all harness-level; the image bakes the same CLIs and skills |
| `spec.packages` | unchanged semantics | installed at image build with uv instead of the platform build |
| Warm pool (`warm_pool`, `pool_size`, `pool_max_wait_s`, `fill_pool`, `wait_until_warm`) | **replaced** by the ready pool | same knobs, same roster; no Pub/Sub, no watchdog; `wait_until_warm` = a roster entry whose `/health` answered |
| Cold path (no pool) | **replaced**: create-on-demand ~20 s | there is no 150 s path any more |
| Checkpoint / resume / interactive `send()` | unchanged | same tar + transcript in GCS |
| Steer / interrupt / stop | **adapted**: direct `/control` | GCS inbox kept only as fallback |
| Event streaming (`async for ev in run`), `history()`, `list_sessions()`, `last_result` | unchanged (mirror) + optional direct poll | `list_sessions` loses its ADK-sessions layer (GCS only) |
| Per-invocation secrets, run-scoped GCS tokens | **improved**: no staged secrets object, no persisted job input | the GCS token boundary is unchanged |
| Vertex model access without a key in the env | **adapted**: a predict-only OAuth token in the env | refresh needed for turns > 1 h |
| Versions / revisions (`versions()`, `revisions()`, `set_traffic`, `delete_version`, `get_engine(version=)`) | **replaced**: template = version; no traffic config | a pin is real routing now (any template can be dispatched to), which is *better* than today's assertion; `set_traffic` has no equivalent — "serving" = newest template |
| `resource_limits` | unchanged shape | template `resources`; 4 CPU / 8 GiB verified, ceiling unknown |
| Resource sampling (`resource_samples()`, memory-pressure event, peak in result) | **dropped** (decided 2026-09-11) | in-sandbox sampling goes (gVisor exposes no cgroup files, and the sandbox has no logging identity to ship samples); observability of sandbox resources is to be investigated *outside* the container (platform metrics for sandboxes, if any) — `resources.py` is deleted |
| Cloud Logging per-step log (`remote_agent_toolkit_steps`) | **dropped** (decided 2026-09-11) | the sandbox has no logging identity; the event mirror is the record; `eventsink.CloudLoggingSink` and the `EventSink` port go |
| Cloud Trace spans per turn (`tracing.py`) | **lost** | no exporter identity in the sandbox; acceptable (the console never showed the job path properly anyway) |
| Platform job retries on OOM / crash | **lost** | the client re-dispatches or fails the turn explicitly; arguably better than a silent replay |
| Platform 7-day job ceiling | replaced by the sandbox TTL | max TTL for custom containers undocumented (Code Execution: 14 d) — verify |
| `IS_SANDBOX=1`, `NUM_WORKERS=1`, glibc/uv/`/tmp` contracts, pickle/aiplatform pin coupling, `build_adk_app`, `verify_deploy_env` | **gone** | no ADK, no pickle, no platform build |
| `service_account=`, `scoped_gcs=`, `min_instances`, `max_instances`, `staging_bucket`, `new_engine` | **gone** | see 13.3 |
| `ratk-gcp-setup` | rewritten | client role + sandbox service agent grant + AR repo; no runtime SA |
| `local` runtime, `dev/` parity image | unchanged | the sandbox image *is* the parity image |

### 13.3 Public API: backwards-incompatible changes (`gemini` replaced in place)

`gemini` is replaced **in place** (decided 2026-09-11): the module name, `deploy` / `get_engine` /
`list_engines` and the `Engine` / `Session` / `Run` API stay, so app code that only looks up an engine and
runs turns keeps working; the deploy-side and ops-side surface changes. The branch lands as one minor
release with these notes in the CHANGELOG; other teams update when it merges. What breaks for callers:

1. **`gemini.deploy` kwargs.** Dropped:
   `service_account`, `scoped_gcs`, `min_instances`, `max_instances`, `staging_bucket`, `new_engine`,
   `use_vertex` (replaced by `model_service_account=` for the predict token; API-key mode stays via
   `secrets`). Kept: `project`, `location`, `output_bucket`, `warm_pool`/`pool_size`/`pool_max_wait_s`
   (renamed `ready_pool`? — decide), `resource_limits`, `credentials`. New: `image_repo` (Artifact Registry),
   `model_service_account`.
2. **Versions.** `engine.set_traffic()` and `engine.delete_version()` go; `engine.versions()` lists templates
   (newest first); `get_engine(name, version=)` **routes** to that template instead of asserting on the
   serving revision; `engine.resource` names a template, not a `reasoningEngine`. `engine.revisions()`
   returns template metadata (no `serving` flag).
3. **Session ids** are always client-minted UUIDs (today cold turns get numeric ADK ids). `list_sessions()`
   is GCS-only. `history()` loses layers 2 and 3 (job output, Cloud Logging) — the mirror is the record.
4. **`session.resource_samples()` is removed**, along with the memory-pressure status event and the
   `memory_peak_bytes` / `cpu_usec` keys on the result's `raw`. `session.transcripts()`, `history()`,
   `last_result`, `workspace` (still raises remotely) unchanged.
5. **Observability side channels**: no Cloud Trace spans; the `remote_agent_toolkit_steps` and
   `remote_agent_toolkit_resources` Cloud Logging logs stop. The event mirror (`history()`) is the record;
   the TESTING.md debugging recipes that query Cloud Logging are rewritten against it.
6. **Deploy prerequisites**: Docker (or Cloud Build) available where `deploy` runs; an Artifact Registry
   repo; the 2.x `google-cloud-agentplatform` client (the toolkit's aiplatform `<2` pin goes, which is a
   dependency break for consumers pinning alongside).
7. **IAM setup**: the runtime service account, its custom role and bucket conditions are no longer needed;
   the client identity needs the sandbox permissions; `ratk-gcp-setup` output changes.
8. **Events**: `workspace_ready.scoped_gcs` is always true; `turn_started` carries a sandbox name instead of
   a worker id; `control_ready` reflects the HTTP channel.

### 13.4 Risks and open questions

- **Preview surface** (v1beta1). Readiness lag before the proxy routes (12–31 s, first exec could 502);
  pause/resume unusable; API names moved between SDK 1.x and 2.x. Mitigation: the ready pool hides the
  lag; we never pause; pin the SDK.
- **Undocumented limits**: max sandbox TTL for custom containers; max resources (4 CPU / 8 GiB works; we
  deploy up to 16 GiB today); disk size; `execute` request/response body limits (Code Execution says 100 MB;
  our payloads are KB); quotas on sandboxes per project and executes per minute (the quotas page 404s).
  Verify each with a probe before building.
- **Long turns**: model token refresh (1 h) and the per-call `execute` timeout (a `/events` poll is short;
  the turn itself runs in the background thread — verified for 70 s+ commands, not for hours). Run a 60-min
  turn probe.
- **No per-sandbox IAM**: the instance is the trust boundary. One instance per project/location is fine
  while all callers are one tenant; multi-tenant callers would need one instance (and client identity)
  each.
- **Egress**: all-or-nothing internet; the Kubernetes API server answers (401) from inside the sandbox.
  Report to Google; no impact on us.
- **Google-side pool** for a template: size, minimum instances and whether it bills are not exposed. If a
  large image (the Codex CLI adds tens of MB, `packages` can add GBs) slows pool refill, refill latency
  grows but stays off the critical path.
- **Observability**: Cloud Logging, Cloud Trace and in-sandbox resource sampling are dropped (accepted).
  What the platform exposes about a sandbox's CPU/memory from outside is an open investigation; until then
  an OOM shows up as the worker going unreachable mid-turn, which the client reports as an explained error.
- **Snapshot restore** (memory + disk): set aside — same-image restriction (checkpoint bullet above).

### 13.5 Plan

Steps 1 and 2 are done (2026-09-11: limits in `dev/sandbox_spike/README.md`; the rewrite is on the branch).

1. Probes for the open limits (TTL max, 16 GiB, 60-min turn, execute body size and call quota), ~a day.
2. On a long-lived branch, rewrite `runtime/gemini/` in place: `_image.py` (Dockerfile generation, build,
   push), `_template.py`, `backend.py` (`deploy`/`get_engine`/`list_engines`, `GeminiEngine`/`GeminiSession`
   over the existing `Run`; roster, mirror, checkpoint, control), `worker.py` (the spike's `server.py`,
   productized: one turn per process, `/events` long-poll, `/control`, token refresh), and a
   `SandboxProvider` seam (§13.6). Delete what the query-job path alone needed: `_deploy.py`,
   `adk_agent.py`, `pool.py`'s dispatch half and the `DispatchTransport` port, `tracing.py`, `resources.py`,
   `eventsink.py` / the `EventSink` port, `revisions.py`, the runtime-SA parts of `project_setup.py`, the
   ADK/aiplatform/cloudpickle dependencies (the `<2` pin with them). Tests: a fake provider behind the
   seam (the probe's `call` is the shape).
3. Adopt in the in-team consumers (self-healing, agentic-scraping) from the branch; run them for real until
   battle-tested, comparing cost and latency against the query-job engines they replace.
4. Merge as a minor release with the §13.3 notes in the CHANGELOG; other teams update on their schedule
   (their existing query-job engines keep serving until they redeploy).

### 13.6 Consequence: the platform is now a small, replaceable surface

The query-job design leaned on Agent Runtime for a great deal: the image build, the job runner and its
retries, ADK sessions, Cloud Logging and Trace identities, Pub/Sub for dispatch, IAM for the worker. The
sandbox design asks the platform for four things only: *create a container from an image with N CPU / M
GiB and internet egress, reach an HTTP port on it through an authenticated proxy, delete it, and enforce a
TTL*. Everything else — dispatch, events, checkpoints, control, secrets — is ours over HTTP and an object
store. That is the shape of every hosted code-sandbox product (Cloud Run jobs, E2B, Modal, Daytona, Fly
machines, …), so the runtime should keep the provider-specific part behind one small port,
`SandboxProvider` (`create(template, ttl) -> handle`, `call(handle, path, body, timeout)`, `delete`,
`list`, plus template create/delete at deploy), with the Agent Sandbox adapter as the first and only
implementation. No other provider is planned; the point is to not pay later for coupling we do not need
now, and to keep the worker image provider-agnostic (it already is: a Dockerfile and an HTTP server).

# agent-run — Design

A Python library for **defining and running remote/background AI agents** at Zyte. It distills the
experience of two proofs-of-concept (self-healing spiders, interactive spider creation) built on
`sandbox-agent-runtime` into reusable building blocks, so any team can stand up a Claude-Code-based
agent — with custom skills, GitHub access, structured outputs, checkpoint/resume and ready sandboxes —
without re-learning the platform's sharp edges.

> Status: **the design as built.** This document describes the current architecture and the platform
> contracts it encodes; the README is the team-facing onboarding doc and the CHANGELOG records what changed
> between releases.

---

## 1. Goals & non-goals

**Goals**
- Let a team define an agent **declaratively** and run it both **locally** (dev) and **remotely** in a
  **Google Agent Sandbox** (prod; §13) through *one* API — local deploy is first-class, not an afterthought.
- Cover the use-cases in the requirements doc: background long-running jobs (self-heal, QA, analysis)
  *and* interactive, human-in-the-loop jobs (requirement-building, prompt-driven setup).
- Bake in the hard-won platform knowledge (latency, checkpointing, the ready pool, IAM, image contracts)
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
- Non-GCP backends. We design the **ports** (storage, secrets, the sandbox provider) as protocols, but ship
  GCP adapters (GCS, Agent Sandbox) plus local/in-memory adapters for dev.
- Replacing Scrapy-Cloud / monitoring logic — that stays in `sandbox-agent-runtime` behind the seam.

---

## 2. Background: which "Claude agent" do we build on?

"Claude Managed Agents" is two different Anthropic surfaces; the distinction drives the architecture.

| | **Claude Managed Agents (CMA)** | **Claude Agent SDK** |
|---|---|---|
| What | Server-hosted agent harness exposed as REST (`/v1/agents`, `/v1/sessions`) | Client library (`claude_agent_sdk`) that drives the Claude Code loop in-process |
| Runs where | Anthropic infra (loop + sandbox container) | **Your** process / container |
| On our platform? | ❌ Not offered on Vertex/Bedrock/Foundry | ✅ via `AnthropicVertex` model access from our own container (Agent Sandbox) |
| Object model | Versioned `Agent` → `Environment` → `Session` → events | `query()` / `ClaudeAgentOptions` / `SessionStore` |

**Decision:** build on the **Claude Agent SDK** (the only Claude harness runnable in our own container on
Google's platform, via `AnthropicVertex` model access), and **borrow CMA's API shape** as the conceptual template.
We cannot host *on* CMA.

> **Naming.** The remote backend runs in **Agent Sandbox**, the sandbox-environment feature of Google's
> Gemini Enterprise Agent Platform (§13). The public namespace is **`sandbox.*`** (`sandbox.deploy`,
> `sandbox.get_engine`). The underlying Google Python SDK is `google-cloud-agentplatform` 2.x
> (`client.sandboxes` / `client.runtimes`); that's an internal detail, not part of our surface.

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
2. **Local / remote parity.** `local.deploy(spec)` (in-process) and `sandbox.deploy(spec)` /
   `sandbox.get_engine(name)` return the *same* `Engine` + `Session` surface. Develop and test locally,
   ship remotely, **zero code change** — swap `local` ↔ `sandbox`. Local uses filesystem/in-memory port
   adapters, so it needs no GCP beyond model access.
3. **Deploy is rare; lookup-and-run is the hot path.** App code addresses a deployed engine **by name**
   (optionally a version) via `sandbox.get_engine(...)` and never triggers a deploy. `sandbox.deploy(...)`
   is an ops/CI action.
4. **Ports & adapters.** Every platform dependency is a protocol with a concrete adapter. Storage, secret
   resolver, session store, harness, sandbox provider — all swappable.
5. **Secrets are per-invocation; never bake secrets/paths.** The spec is baked into the sandbox image, so
   `AgentSpec` carries no secrets at all — credentials are passed as a `name → value` map to `run`/`send`,
   scoped to that one turn (nothing baked, nothing shared across tenants). The values travel in the turn's
   HTTPS body to the worker and nowhere else: nothing is persisted by the platform or written to GCS. The
   harness routes each secret to its consumer — a repo's `auth` token into git `origin`, a GitHub MCP token
   into headers, the rest into the agent's env — and checkpoint snapshots are credential-scrubbed.
6. **Encode the platform contracts as defaults, document the why.** The deploy path applies the known-good
   settings automatically (see §6) and every one is explained.
7. **Interactive = checkpoint-and-resume, not a blocking call.** No long-lived `ask_human` tool. The agent
   ends a turn with a question → the run checkpoints and releases its sandbox → the human's answer resumes it
   on a ready sandbox. One mechanism, stateless workers, survives worker death, and no sandbox bills while
   a human thinks.

---

## 4. Architecture

Three planes over a set of pluggable ports:

```
 DEFINITION plane        DEPLOY plane (ops/CI)         RUN plane (app code)
 ────────────────        ─────────────────────        ──────────────────────────────
  AgentSpec        ──▶    sandbox.deploy(spec)          sandbox.get_engine(name[, ver]) ─┐
  (declarative,           → versioned engine,          local.deploy(spec)              │
   serializable;            encodes platform           → Engine                        ▼
   code or YAML)            contracts §6                  · start_session()  → Session
                                                          · get_session(id)  → Session  (re-attach/poll)
                         local.deploy(spec) ─────────▶    Session
                         (in-process,                       · run(msg) → Run :
                          same Engine/Session API)              await Run      → RunResult   (wait)
                                                                async for ev   → AgentEvent  (stream)
                                                                Run.done/status              (poll)
                                                            · send(msg) / interrupt() / exec()
                                                            · status / stop_reason
                                                            · resume / fork

 PORTS (protocols; concrete adapters shipped — GCP for prod, local/in-memory for dev)
 ────────────────────────────────────────────────────────────────────────────────────
  Harness            ClaudeCodeHarness | CodexHarness       (the coding-agent loop)
  BlobStore          GcsBlobStore | LocalBlobStore          (checkpoint, workspace tar, event mirror, records)
  SandboxProvider    AgentSandboxProvider | fakes (tests)   (create / call / delete a container; templates)
  ControlChannel     an in-process queue on the turn's loop (fed by send() locally, by /control in the sandbox)
  SessionStore       BlobSessionStore                       (Claude SDK protocol, keyed by session_id)
  SecretResolver     GcpSecretResolver | EnvSecretResolver  (secret name → value, at runtime)
```

- **Harness** is the abstraction over "a coding agent loop." `ClaudeCodeHarness` wraps the Agent SDK
  `query()`, builds `ClaudeAgentOptions` from the `AgentSpec`, translates SDK messages → generic
  `AgentEvent`s; the sandbox worker (`runtime/sandbox/worker.py`) drives it inside the container exactly as
  `local` does in-process. This is the generalized `ClaudeCodeAgent` from the PoC.
- **Engine** is the deployed (or local) agent handle. `sandbox.deploy(spec)` registers an engine under
  `spec.name` and returns a `SandboxEngine`; `sandbox.get_engine(name[, version])` looks one up without
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
  (→ `AgentEvent`s), and pollable (`done` / `status` / `result`). For `sandbox`, "stream" = long-poll the
  worker's `/events` (the GCS event mirror is the fallback); "await" = wait for the terminal result event;
  "poll" = check the run's status.

---

## 5. Public API

```python
from agent_run import AgentSpec, SystemPrompt, SkillSource, McpServer, sandbox, local

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
sandbox.deploy(spec, project="my-project", location="us-central1", warm_pool=True)

# App code looks the engine up by name (optionally a version) and runs — it never deploys:
engine = sandbox.get_engine("spider-builder")               # latest deployed version
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
    await session.send("yes, that schema looks right")     # resumes via checkpoint on a ready sandbox

# enumerate past sessions and read a finished one's record (sandbox)
for info in engine.list_sessions():                        # newest first
    past = engine.get_session(info["session_id"])
    events = past.history()                                # persisted AgentEvents, oldest first
    result = past.last_result                              # reconstructed from the terminal event
```

**Managing deployed engines** (control plane):

```python
sandbox.deploy(spec, project=..., location=...)   # create / update; mints a new version when the image or resources change
sandbox.get_engine("spider-builder")              # newest version (app code default)
sandbox.get_engine("spider-builder", version=3)   # pin a version (routes to that template)
sandbox.list_engines(project=..., location=...)   # discover what's deployed
engine.versions() / engine.revisions()           # list versions of one engine (newest first) / their metadata
engine.delete_version(3) / engine.delete()       # retire one version / the whole engine (its ready pool included)
engine.name, engine.version, engine.resource     # identity / underlying resource name
```

Engine identity maps `spec.name` → a stable engine (the template `display_name`); each `deploy` that changes
the image or the resources mints a new version. App code pins or takes latest; it does not deploy.

A **version is a sandbox template** (`…/reasoningEngines/{host}/sandboxEnvironmentTemplates/{id}`, displayed
under `spec.name`): `deploy` creates a new immutable template whenever the image or resources differ from
the newest one, and `get_engine(name)` resolves the newest, `get_engine(name, version=)` a specific one.
A pin **routes** — any template can be dispatched to — so a pinned handle keeps running its version after a
newer deploy; "serving" simply means "newest", and there is no traffic configuration.

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
  `exec(command)` (a read-only shell probe into the running turn's workspace), `status`, `stop_reason`,
  `last_result`, `history()`, `fork()`.
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
  surfaced as a `claude_stderr` status event (in the event mirror on sandbox) and embedded in the error.

---

## 6. Execution model & platform contracts (the hard-won knowledge)

These are facts measured live. The library encodes them so consumers inherit them for free.

**The execution path**
- **An Agent Sandbox custom container per turn** — the harness in a gVisor sandbox behind Google's
  authenticated HTTP proxy: dispatch→first event ~1.3 s, →result ~4–7 s on a ready sandbox (measured live
  2026-09-11 by `make live-smoke`), ~20 s create→result without one, and no Google identity in the
  container. One sandbox per turn, deleted at the terminal event; multi-turn continuity rides the
  checkpoint. The platform's ceilings (8 vCPU max, ~5 min per proxied call, ~1 MB request / ~2 MB
  response bodies, TTL up to 30 d) are measured, not documented — see TESTING.md, "The limits probe".

**The ready pool (`roster.py`)**
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

**Checkpoint / resume**
- Conversation: the Claude SDK `SessionStore` (append/load) + `resume=session_id`. Our `BlobSessionStore`
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
  `Session.workspace` accessor (local: host `Path`, created on access, seed-before/collect-after; sandbox:
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

**Observability (one record — the events)**
- **The worker's `/events` is the live channel**: the client long-polls it (held until new events exist,
  paged under ~1 MB because the proxy caps a response at ~2 MB — measured), so an event is seen within one
  proxy round trip (~0.2 s) of its emission. **The GCS event mirror is the durable record** (`stream.py`):
  the worker streams every surfaced event into small JSONL batch objects under `events/<sid>/` with the
  run-scoped token (~0.5 s cadence; a terminal `result` flushes immediately) — what `Session.history()`
  reads and the client's **fallback stream** when the sandbox stops answering (delivered events are
  skipped; a sandbox that is gone and left no terminal record ends the run with an explained
  `sandbox_unreachable` result after a 30 s grace).
- **No Cloud Logging, no Cloud Trace.** The sandbox has no logging identity, and one record is enough: the
  mirror holds every event of every turn, and the TESTING.md debugging recipes read it.
- **CPU / memory come from inside the sandbox** (`resources.py`): the platform exposes nothing about a
  sandbox's resources from outside (no Monitoring metric type, no usage fields on the resource; checked
  2026-09-16), but gVisor mounts cgroup v1 accounting inside and reports the template's memory limit as
  `MemTotal`. The worker samples both every 20 s into the turn's **event mirror only** (`resource_sample`
  rows; `Session.resource_samples()` reads them back, `history()` skips them unless asked), emits one
  visible **memory-pressure status event** the first time usage crosses the pressure fraction, and stamps
  the peak, the limit and the CPU time into the terminal result (`RunResult.resources`). An OOM shows up
  as the worker going unreachable mid-turn, with the last sample at most one interval before death.

**Failure semantics (the client owns liveness; there are no platform retries)**
- **A sandbox that stops answering** — the proxy fails `DIRECT_MAX_FAILURES` polls in a row, or the
  platform says it is gone (TTL, OOM, deleted elsewhere) — switches the stream to the mirror tail; a gone
  sandbox that left no terminal record ends the run with an explained `sandbox_unreachable` error result
  after a 30 s grace. **A dispatch that no sandbox takes** (three attempts, moving on only from a sandbox
  that definitely never accepted the turn — gone before the call reached it, or refusing — and deleting it)
  ends the run with `dispatch_failed`. **A `/turn` call whose answer was lost** (proxy timeout, 5xx) may
  hide an acceptance — the worker answers as soon as the turn's thread runs — so it is settled with the
  *same* sandbox first: acceptance is idempotent per turn id, the worker re-acknowledges a turn it knows,
  and only a definitive refusal moves on. When that sandbox cannot be asked any more (vanished, or silent
  for `DISPATCH_RECONCILE_S`), the run ends with `dispatch_uncertain` naming the sandbox, which is deleted
  (stopping whatever it started). Neither hangs, and a turn is never replayed on a second sandbox while the
  first may be running it. **The terminal result is durable before it is visible**: the worker writes it
  to the mirror (awaited, bounded) before serving it on `/events`, so the client deletes the sandbox only
  once the record holds the result; when that write fails (dead run-scoped token, storage outage) the
  live copy carries `mirror: "failed"` and the client writes the line itself, before the delete.
- **`interrupt()`** posts `stop` to `/control` and waits for the harness's normal end of turn (checkpoint,
  `INTERRUPTED` result); the fallback — no `control_ready` yet, or no answer within the timeout — cancels the
  driver and **deletes the sandbox**, which is the cancel (and stops the billing). A hand-over still in
  flight cannot be stopped (the dispatch thread claims, then POSTs): the run is marked abandoned and the
  sandbox the dispatch lands on — claimed, or already running the turn — is deleted the moment it lands.
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
  known `exec()` waits for dispatch, afterwards it raises `ControlUnavailable`. The timeout is capped at
  240 s on `sandbox` because the proxy cuts a call at ~300 s. Motivation: agentic-scraping renders the
  workspace's `git diff` every ~10 s while the coding agent works — exact and harness-independent, which
  the events (paths only on Codex, context-free old/new strings on Claude Code) are not.
- **A re-attached session adopts its turn running under another process** (`SandboxSession._attach`).
  `get_session(id)` in a fresh process holds no run; the first `send()` / `interrupt()` / `exec()` /
  `run()` / `current_run` / `busy` / `last_result` reads the mirror once — the last `turn_started` marker without a `result` after it names the
  turn and its sandbox (every event is stamped with both, #80; the sandbox name is rebuilt from the
  engine template's host instance) and whether `control_ready` was announced — and builds a `DrivenRun`
  over the same `_stream_turn` the owner uses (the worker replays the whole turn from `since=0`, then
  streams; the mirror tail is the fallback). From there the session behaves as the owner's: steer,
  stop, probe, `last_result` — and `Session.current_run` (protocol) hands the adopter the `Run`
  itself, so it consumes the events instead of polling. Ownership rules: the adopter runs its own run-scoped token refresher (the
  owner may be dead; the worker reads the record back), deletes the sandbox `ADOPTED_RELEASE_DELAY_S`
  (10 s) after the result so a still-alive owner can drain its last events first (its `/events` call is
  served from memory the moment the result lands, and the mirror holds the result anyway), and an
  adopter whose event loop shuts down without an explicit `cancel()` (a poller exiting) releases
  nothing — the turn goes on under its owner. The check is one-shot per session object: a session
  started in the process never has a foreign turn, and a `get_session` one that found none resumes as
  before.
- **`interrupt()` is a stop, not a cancel.** The turn's normal end-of-turn path runs (snapshot,
  transcript, terminal result with `StopReason.INTERRUPTED`, accounting kept), so the interrupted turn's
  workspace changes are in the checkpoint the next `send()` restores. Deleting the sandbox is the
  fallback when the worker never announced its channel or does not stop within the timeout; the result's
  `warning` says so.

**Persistence / job history (what survives a run, and where)**
- **Events mirror** — `events/<sid>/*.jsonl` under the output bucket, written by the worker as the turn runs
  (the durable record and the fallback live channel); `Session.history()` reads it, `list_sessions()`
  enumerates it (bucket-wide). Every event carries its `turn_id`, so a re-attached session's back-to-back
  turns never replay each other's results (#80).
- **Configs** — the session's `SessionConfig` at a stable key (`session-config/<sid>.json`, what re-attach
  reads back; a re-attacher cannot substitute one) and each turn's `TurnConfig` nonce-keyed; both also ride
  the turn body, so the worker never reads them from GCS. Kept 30 days. Only a definite "no such object"
  means the session has no config: a permission error, a timeout or a malformed record fails the re-attach
  instead of silently running the baked spec, and the next attempt re-reads.
- **Deploy record** — `deploys/<name>/<template>.json`: the spec and settings a template was deployed with,
  so a `get_engine` handle knows the baked spec and the model identity. Kept.
- **Token objects** — `invocation-secrets/<sid>-gcs-token.json`: the run-scoped GCS token's refresh object
  (a value; deleted at turn end, reaped after a day).
- **Checkpoints** — transcript + workspace tar under `checkpoints/`, keyed by (mapped) session id.
- Session ids are client-minted canonical UUIDs (the Claude CLI rejects non-UUID session ids; a non-UUID id
  maps through `uuid5`). Nothing is persisted by the platform: no job input, no job output, no platform
  session.

**Deploy contracts (applied automatically by `sandbox.deploy`; `_image.py`)**
- **The image**: `python:3.12-slim-bookworm` (glibc — the `claude-agent-sdk` wheel ships a self-contained
  glibc ELF `claude` binary; no Alpine/musl), `git`, `curl`, build tools; the base requirements
  (`claude-agent-sdk` pin, `openai-codex` when the Codex harness is baked, `google-cloud-storage`, `uv`, …)
  merged with `spec.packages` (an unsatisfiable merge fails before the build); the toolkit package copied
  from the deploying venv (the image runs THIS toolkit revision — client and image are one wire contract);
  the resolved skills flat under `/opt/toolkit/skills`; the spec at `/opt/toolkit/spec.json`; a non-root
  `agent` user (uid 1000; the sandbox requires non-root); `/workspace`; the worker as `CMD` on port 8080.
- **The tag is a content digest** of the whole build context, so an unchanged spec from an unchanged
  toolkit skips the build (`docker manifest inspect`) and a deploy whose image and resources match the
  newest template is a no-op. Docker CLI + a registry login are the deploy's prerequisites; the platform
  does not build for us.
- **Template**: the image, `ports=[8080]`, `resources` requests = limits = `resource_limits` (default
  4 CPU / 4 GiB; the platform refuses more than 8 vCPU; 16 GiB verified), `egress_control_config.internet_access`.
  `customContainerSpec` carries only `imageUri` — no command, args or env — so everything per run arrives
  over HTTP after creation.

**Identity / IAM (two identities, and the sandbox has none)**
- **The client identity** (an impersonated operator SA, or the caller's own) is the whole control plane:
  `roles/aiplatform.user` on the project (sandbox + template permissions, the host `reasoningEngine`),
  `roles/storage.admin` on the output bucket (records, roster, minting the run-scoped tokens),
  `artifactregistry.writer` on the image repo, `iam.serviceAccountTokenCreator` on the model SA. There is
  **no per-sandbox IAM**: `aiplatform.sandboxEnvironments.execute` covers every sandbox under the host
  instance, so the client identity is the trust boundary and the roster is only a coordination device.
- **The model identity** (`agent-run-model@`, `model_token.py`): a service account with only the
  `agentRunPredict` custom role (`aiplatform.endpoints.predict`). The client mints its access token
  (an hour, IAM's default ceiling) per turn and every 25 minutes after that, pushing each one to the worker
  (`/token`). The worker never puts it in the agent's environment: it runs a loopback **metadata server**
  that speaks the GCE metadata protocol for exactly one thing, the running turn's token, and sets
  `GCE_METADATA_HOST` / `CLAUDE_CODE_USE_VERTEX=1` for the CLI — whose Google auth then fetches the token as
  it would on a VM and re-fetches it when less than five minutes remain (verified with the CLI 2026-09-16:
  a two-step turn crossing that threshold re-read the token and finished). So a turn's model access lasts as
  long as some client process holds the session — the starter or an adopter (`_attach`) — with no org
  policy and no long-lived credential.
- **The sandbox has no usable Google identity** (verified live 2026-09-10 and by every `live-smoke` run):
  the metadata server answers with a tenant-project workload identity that is 403 on the project's storage,
  Vertex and resource manager. So what the agent's shell can reach is exactly what the turn brought: the
  run-scoped GCS token (its own prefixes), the model token (predict only), the turn's own secrets.
- **The platform side**: the Agent Sandbox service agent (`service-<number>@gcp-sa-vertex-sandbox`) needs
  `artifactregistry.reader` on the image repo to pull the image. Nothing else is granted to Google.
- **Run-scoped GCS tokens** (`scoped_gcs.py`): the client mints one downscoped token per turn (a Credential
  Access Boundary: this bucket, only the run's own prefixes — configs, events mirror, checkpoints,
  artifacts; one list clause, the transcript dir), sends it with its expiry in the turn body, refreshes it
  while the run lives by overwriting one object under the run's prefix, and the worker does all GCS work
  with it via the `GcsBlobStore` process default. The worker fetches each replacement WITH its current
  token, so it refreshes no later than a ten-minute horizon before the expiry it knows, and reports a token
  it can no longer renew once (three consecutive failures), not on every retry. STS caps a minted token at
  ~10.7k chars including the caller's own, so list clauses are kept to one.

---

## 7. Ports & adapters

Each is a `typing.Protocol`; concrete adapters ship for prod (GCP) and dev (local/in-memory).

- **`Harness`** — `build_options(spec, ctx)` + an async run loop yielding `AgentEvent`s. Adapters:
  `ClaudeCodeHarness` and `CodexHarness`, selected by `spec.harness` via `resolve_harness`
  (the single seam both runtimes use). Shared policy — secret routing, agent-env layering, the
  interactive suffix, the inline workspace checkpoint — lives in `harness/_shared.py` so the bindings
  can't drift where the spec doesn't distinguish them. Both read `ctx.control` (a `ControlChannel`)
  alongside their stream through `control.ControlledStream` and act on steer / interrupt / stop
  messages the same way (§6 "Turn control").
- **`ControlChannel`** — the harness side of turn control: `receive()` blocks for the next operator
  message. One adapter, `LocalControlChannel` (an in-process queue on the turn's loop): `local` feeds it
  from `send()` directly, the sandbox worker from its `/control` endpoint. A **conformance suite**
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
  The budget is measured against that total, and the proxy answers 402 once the cap is spent; a zero
  budget admits no request at all. This is admission between responses, not a hard spending ceiling:
  concurrent requests can pass it together, and a response without a known cost adds nothing to the
  total.
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
  `GcsBlobStore`, `LocalBlobStore`. Backs checkpoint, workspace, the event mirror, the records,
  artifacts.
- **`SandboxProvider`** — the one platform seam of the remote runtime (§13.5): `create(template, ttl)`,
  `call(handle, path, body, timeout)`, `delete`, `list`, plus template create/list/delete at deploy.
  Adapters: `AgentSandboxProvider` (Agent Sandbox over the 2.x SDK), fakes in the test suite.
- **`SecretResolver`** — `resolve(name) -> value`, a *control-plane* helper for callers who keep secret
  values in env/Secret Manager and need to build the per-invocation `secrets` dict (the runtime itself
  receives values in the turn body, §3.5). Adapters: `EnvSecretResolver` (implemented),
  `GcpSecretResolver` (stub).
- **`SessionStore`** — the Claude SDK protocol (`append` / `load` / `list_subkeys`). Adapter:
  `BlobSessionStore` (keyed by `session_id`). A **conformance suite** (`run_session_store_conformance`)
  validates any implementation.

---

## 8. Origins

The toolkit distilled the `sandbox-agent-runtime` proofs-of-concept. What was lifted and generalized: the
Claude Code run loop (`harness/claude_code.py`), the event translation (`harness/translate.py`),
checkpoint/resume (`checkpoint/` over `BlobStore`), skills staging (`skills.py`, configurable sources), git
clone + token injection and the GitHub MCP attach (`integrations/`), artifacts, cost tracking on the result,
and the config/runtime resolution that became `AgentSpec`. Scrapy Cloud, monitoring and the scrape prompts
stayed in `sandbox-agent-runtime` behind the seam.

---

## 9. Decided design choices (2026-06-29)

1. **Harness scope** — seams now, Claude-first implementation. Harness-agnostic `AgentSpec` + a
   `Harness` protocol. (Superseded 2026-07: the seam paid off — a second binding, `CodexHarness`,
   now ships behind `spec.harness`; see §7.)
2. **Definition API** — **declarative `AgentSpec` + `deploy()`** (control-plane / data-plane split), not an
   imperative builder or thin functions.
3. **First milestone** — a **minimal generic (non-Zyte) example agent** run local + remote, proving the core
   API + deploy + the ready pool with the least surface, before porting a real PoC.
4. **Local deploy is first-class** — `local.deploy(spec)` mirrors `sandbox` exactly (same Engine/Session
   API), so the dev loop and the prod loop are the same code.
5. **App code looks up engines, never deploys** — `sandbox.get_engine(name[, version])` is the hot path;
   `sandbox.deploy(...)` is ops/CI.

---

## 10. Package layout

(No `src/` layer — the package sits at the repo root.)

```
agent-run/
├── pyproject.toml                 # runtime deps in core; the harness SDKs behind the `local` extra; dev tooling in a group
├── README.md                      # team onboarding; the only user-facing doc
├── DESIGN.md                      # this file (internal design record)
├── TESTING.md                     # the test ladder: offline suite, parity image, live probes
├── CHANGELOG.md                   # releases and the Unreleased section
├── agent_run/
│   ├── __init__.py                # AgentSpec, SystemPrompt, SkillSource, McpServer, sandbox, local
│   ├── spec.py                    # AgentSpec + value types (serializable)
│   ├── events.py                  # AgentEvent, RunResult, RunStatus, StopReason, Run handle
│   ├── control.py                 # ControlMessage, ControlChannel, LocalControlChannel, ControlledStream, run_shell
│   ├── structured.py              # structured-output parsing fallback
│   ├── harness/
│   │   ├── base.py                # Harness protocol (+ resolve_harness in __init__)
│   │   ├── context.py             # RunContext — the per-run inputs a runtime hands the harness
│   │   ├── claude_code.py         # ClaudeCodeHarness (drives a ClaudeSDKClient stream)
│   │   ├── codex.py               # CodexHarness (drives an openai-codex AsyncCodex app-server)
│   │   ├── _shared.py             # policy shared by the bindings (secret routing, env, checkpoint)
│   │   ├── _openrouter_proxy.py   # per-run localhost proxy for openrouter/ models (both bindings)
│   │   ├── _usage.py              # the normalized whole-turn usage record
│   │   ├── pricing.py             # LiteLLM price lookup + baked OpenAI fallbacks, context windows
│   │   └── translate.py           # Claude SDK message → AgentEvent (codex's lives in codex.py)
│   ├── runtime/
│   │   ├── base.py                # Engine + Session protocol + state machine + Run handle
│   │   ├── _run.py                # DrivenRun — the shared, backend-agnostic Run handle
│   │   ├── local.py               # local.deploy / local.run -> LocalEngine
│   │   ├── venv.py                # per-engine uv venv for spec.packages on local
│   │   └── sandbox/
│   │       ├── backend.py         # deploy(), get_engine(), list_engines(), SandboxEngine, SandboxSession
│   │       ├── provider.py        # SandboxProvider seam + AgentSandboxProvider (the platform adapter)
│   │       ├── worker.py          # the in-container worker: /health /turn /events /control /token /exec
│   │       ├── _image.py          # Dockerfile generation, content-digest tag, docker build/push
│   │       │                      #   (underscored: `deploy.py` would shadow sandbox.deploy)
│   │       ├── model_token.py     # the predict-only model identity's per-turn token
│   │       ├── scoped_gcs.py      # the run-scoped GCS token (mint / refresh / worker credentials)
│   │       ├── roster.py          # the ready pool's idle-sandbox roster (GCS, atomic claim)
│   │       ├── handoff.py         # the GCS records: configs, deploy record, lifecycle rules
│   │       ├── stream.py          # the GCS event mirror (writer) + its tail (the fallback stream)
│   │       ├── history.py         # history / list_sessions readers over the mirror
│   │       ├── resources.py       # in-sandbox CPU/RAM sampling (cgroup) → mirror, pressure event, result
│   │       └── project_setup.py   # agent-run-gcp-setup
│   ├── ports/
│   │   ├── blobstore.py           # BlobStore + GcsBlobStore + LocalBlobStore
│   │   └── secrets.py             # SecretResolver + GcpSecretResolver + EnvSecretResolver
│   ├── checkpoint/
│   │   ├── session_store.py       # BlobSessionStore (Claude SDK protocol)
│   │   └── workspace.py           # snapshot / restore over BlobStore
│   ├── integrations/
│   │   ├── git.py                 # clone + PAT injection
│   │   └── github.py              # GitHub MCP attach + (optional) REST helpers
│   ├── skills.py                  # discover + provision from SkillSource[]
│   └── conformance.py             # SessionStore / Harness conformance suites
├── examples/
│   ├── minimal/                   # the generic example agent, local
│   └── sandbox/                    # deploy + run on the sandbox runtime
├── dev/                           # the parity image and the paid live probes (dev/README.md)
├── docs/                          # notes referenced from the README
└── tests/
```

---

## 11. Roadmap

The **README grows with the code** — each phase leaves the README runnable for what's shipped, rather than
saving all onboarding docs for the end.

- **P0 — Scaffold.** Repo skeleton (no `src/`), `pyproject`, the port & harness protocols as real signatures,
  the conformance harness, CI. *README:* skeleton + install + the headline example as the north-star target.
  (Done.)
- **P1 — Generic core + local.** `ClaudeCodeHarness`, checkpoint/SessionStore, skills, git/github,
  artifacts, `AgentSpec`, `local.deploy`/`local.run` with filesystem/in-memory adapters.
  *Gate:* the generic example runs in-process. *README:* the **local quickstart** works end-to-end
  (define → `local.deploy` → `run` / stream / poll). (Done.)
- **P2 — Sandbox backend + ready pool + deploy.** `sandbox.deploy` (contracts encoded) + `sandbox.get_engine`
  / `list_engines`; the engine handle & `Session` state machine; the ready pool; the event channel;
  structured outputs. Permissions runbook. (Done, on Agent Sandbox — §13.)
  *First-milestone gate:* **the minimal generic example runs locally *and* remotely with a warm start.**
  *README:* the **deploy quickstart** + engine management.
- **P3 — Onboarding polish.** Examples, the platform-notes appendix, conformance docs, README polish
  (troubleshooting, IAM runbook link, the "why" sections).
- **P4 — Migrate the PoC.** `sandbox-agent-runtime` depends on the toolkit; re-implement self-heal +
  interactive spider on top; delete the lifted code; `scrapy` becomes an extra there.

---

## 12. Open questions / future work

- **Multi-source skills** — `SkillSource[]` is designed for base + extra sources; the resolver/merge
  (precedence on name collision, per-source ref pinning) is specified in P1 but the merge policy needs a
  decision. (P1 ships **last-source-wins**.)
- **Baked engine dependencies** — `AgentSpec.packages` means "the agent's starting Python packages" on
  BOTH backends: `sandbox.deploy` bakes them into the image with uv; `local.deploy` resolves them into a
  per-engine venv (uv, engine-contract Python 3.12) activated in the agent env. (Decided after eval-harness
  feedback: a spec field silently meaning different things per backend broke local/remote parity; a
  Docker-based local mode was rejected — it would trade away the in-process fast dev loop that `local`
  exists for, and uv's cache makes the venv path near-instant.)
- **Local parity with the engine image** — *Python package* parity is handled by the venv above; full
  OS-level parity (base OS, glibc, system tools) is the `dev/` parity image's job (install/dependency
  issues debugged on the laptop instead of through a deploy).
- **Structured outputs — BUILT (2026-08).** Both harnesses send the schema through the CLI's own
  structured-output option and keep "parse the last JSON block" only as a fallback. OpenRouter turns
  also get the schema in the prompt, because OpenRouter accepts the schema but leaves enforcement to
  the model. Still open: confirming Vertex model support per model.
- **Outcomes / rubrics** — CMA's iterate-until-graded "definition of done" is attractive for autonomous
  background work; candidate post-P4 capability (a `verify=Rubric(...)` on `AgentSpec`).
- **Non-GCP backends** — the ports are protocols; an AWS/Azure adapter set is possible but unscheduled.
- **Scheduled runs** — CMA-style cron `deployments` (each firing = a session) could wrap deploy+session
  for recurring jobs (e.g. self-heal monitoring); future.
- **Other sandbox providers** — the platform surface the runtime needs is small (§13.5); no plan, but the
  `SandboxProvider` seam keeps the door open.
- **A platform-wide spending ceiling** — `max_budget_usd` is enforced per harness and per run (between
  model responses; the OpenRouter proxy refuses requests once the cap is spent). A hard ceiling across
  concurrent runs, subagents and every provider would need provider-side prepaid limits or a budget
  authority in front of them; not designed.

---

## 13. Sandbox runtime

The `sandbox` backend runs each turn in a **Gemini Enterprise Agent Platform Agent Sandbox custom
container**: our image, behind Google's authenticated HTTP proxy, on gVisor, with **no usable Google
identity** inside (the metadata server answers with a tenant-project workload identity that is 403 on
everything in our project — verified), and created from a **pre-warmed pool Google keeps per template**, so
a sandbox is assigned in ~2 s. The platform's part is small and the toolkit owns the rest — dispatch,
events, checkpoints, control, secrets — over HTTP and an object store (§13.5). `runtime/sandbox/` is the
implementation; `provider.py` is the only module that talks to the platform.

**Measured (2026-09-11, `make live-smoke`, 4 CPU / 8 GiB, Haiku, `checkpoint=True`)**: deploy 81 s (image
build 32 s on a warm Docker cache, push 13 s, template 18 s, pool fill 15 s); pool turn **1.3 s to the first
event, 6.6 s to the result**; fresh-sandbox turns 15–23 s to the first event, 22–28 s to the result; a steer
delivered into a running turn and acknowledged; checkpoint resume restored on a new sandbox. Provisioning a
4 CPU template takes anywhere from 8 s to ~4 min depending on the platform's capacity at that minute
(ten sequential creates measured 78–241 s with seven other templates active; one 30-minute stall observed);
it is off the hot path, and `deploy` does not block on it.

### 13.1 Architecture

```
 DEPLOY (ops/CI)                                RUN (app code; the client IS the control plane)
 ───────────────                                ────────────────────────────────────────────────
 sandbox.deploy(spec)                            sandbox.get_engine(name) → Engine
   docker build (Dockerfile = toolkit + harness    start_session() → Session
     CLIs + baked skills + spec.packages)          run(msg):
   push → Artifact Registry                          1. claim a READY sandbox off the roster (GCS,
   create SandboxEnvironmentTemplate                    precondition delete — roster.py)
     (image, ports=[8080], resources, egress)           or create one (~20 s) when the roster is empty
   pre-create N sandboxes, wait /health,             2. POST /turn {spec overlay, prompt, secrets,
     record them in the roster                          run-scoped GCS token, model token}
   template name  ⇔  engine version                  3. stream: long-poll /events (0.2 s/call);
                                                        the GCS event mirror is the fallback
 IN THE CONTAINER (worker.py)                        4. background: refill the roster (create + warm)
   HTTP on :8080 behind Google's authenticated       5. at the terminal event: delete the sandbox
   proxy: /health /turn /events /control /token /exec
   one turn per sandbox process; the harness
   runs exactly as under local.deploy;
   events → GCS mirror with the run-scoped
   token; checkpoint tar → GCS
```

- **Deploy = image + template.** `sandbox.deploy` builds the image from a generated Dockerfile (the
  `dev/Dockerfile` contract: Debian, Python 3.12, git, uv, the toolkit with the baked harness CLIs, resolved
  skills, `spec.packages`), pushes it to an Artifact Registry repo in the project, and creates a template
  named `<spec.name>` with the image, `ports=[8080]`, `resources` (`resource_limits`) and
  `egress_control_config.internet_access=True`. Templates are immutable, so **a template is a version** (the
  image tag is the toolkit+spec digest); `get_engine(name)` resolves the newest template of that display
  name, `version=` pins one. Cutover = create the new template, fill its pool, then drain and delete the old
  one. Needs Docker on the deploying machine — the platform does not build for us.
- **The client is the whole control plane.** There is no server-side component of ours: the client claims
  or creates the sandbox, hands it the turn, streams its events, refreshes its tokens and deletes it.
  `roster.py` is the ready-sandbox roster (its `expires_at` = sandbox create time + TTL, which the platform
  enforces — exec does not extend it). The parent "instance" (`reasoningEngine`) is an empty resource
  created once per project/location (sandboxes must hang off one).
- **Turn protocol.** One `POST /turn` carries everything: the effective spec overlay (deploy-baked ←
  session ← turn merged client-side, echoed back as `effective_spec`), the prompt, the per-invocation
  **secrets**, the **run-scoped GCS token** with its expiry (`scoped_gcs.py`: the worker refreshes from
  the token object under the run's prefix), and the **model token**. Nothing is persisted by the platform.
  Session/turn configs are written to GCS by the client for the 30-day post-mortem record, but the worker
  never reads them from there.
- **Model credentials.** The sandbox has no Google identity, so the model token is ours to provide. The
  client mints it by impersonating a **predict-only service account** (the `agentRunPredict` custom
  role; the client needs `serviceAccountTokenCreator` on it) with a 1 h lifetime and re-pushes a fresh one
  over `/token` every 25 minutes for as long as it holds the running turn. The worker serves it from a
  loopback metadata server behind `GCE_METADATA_HOST`, so the CLI refreshes on its own and the token never
  sits in the agent's environment (§6 "Identity"). The agent's shell can fetch it from that loopback
  server, as the CLI does — but it is predict-only and hourly. `use_vertex=False` (API-key mode) and
  OpenRouter work unchanged (the key rides `secrets`).
- **Events: direct by default, mirror as the record.** The live stream a `Run` consumes comes straight from
  the worker: `POST /events {since, wait}` **long-polls** — the worker holds the request until new events
  exist or `wait` seconds pass, so a streaming client issues one call per event batch, not a fixed-cadence
  poll, and sees an event within one proxy round trip (~0.2 s) of its emission. The worker also writes the
  **GCS event mirror** with the run-scoped token (`stream.py`) as the **durable record**: `history()`,
  `list_sessions()`, `last_result` of a finished session, and the fallback stream when the sandbox is
  unreachable (turn finished and sandbox deleted; a proxy error — the client switches to the mirror tail
  and back). Every event is stamped with the turn and the sandbox, so a client that re-attaches mid-turn
  (`get_session(sid)`) finds the sandbox and long-polls directly too (§6 "Turn control").
- **Control (steer / interrupt / stop / probe).** `/control` and `/exec` on the worker, sub-second, no
  polling loop in the worker (§6 "Turn control").
- **Checkpoint / resume / interactive.** Transcript + workspace tar to GCS at the terminal event, restored
  on the next turn's sandbox. A sandbox is deleted at turn end — it bills while it exists — so multi-turn
  continuity rides the checkpoint. Native sandbox **snapshots** (disk + memory) were considered and set
  aside on purpose, not only for maturity: a snapshot restores **only onto the identical image**, and we
  update agents (new template) far more often than we retire old sessions — a job checkpointed on one
  revision must resume on the next one, which the tar does and a snapshot cannot. Revisit only if the
  platform lifts the same-image restriction.
- **Liveness.** No platform retries and no job handle: the client owns it. The `/events` long-poll doubles
  as the health check during a turn, a turn-level deadline bounds it, and the sandbox TTL (set at creation
  to the max turn length + margin) is the backstop for a client that dies. Deleting a sandbox is the
  cancel. A startup sweep deletes stale sandboxes of our display-name prefix that no roster entry or live
  run claims.
- **IAM.** Client: `roles/aiplatform.user` on the project (the sandbox and template permissions), bucket
  rights, `serviceAccountTokenCreator` on the predict-only account, Artifact Registry writer for deploys.
  Platform: the Agent Sandbox service agent (`service-<number>@gcp-sa-vertex-sandbox`) needs
  `artifactregistry.reader` on the image repo. There is no per-sandbox IAM: whoever can execute on the
  instance can drive any sandbox, so the trust boundary is the client identity, and the roster is purely a
  coordination device. `agent-run-gcp-setup` applies exactly this (§6 "Identity").

### 13.2 Platform facts and limits

Container facts that shape the design: the readiness lag after a create is **route propagation, not boot**
(at first answer a fresh sandbox's PID 1 had been running since template creation — the pool started it),
so it only affects refill; the template's declared `ports` are the only ones the proxy forwards;
`customContainerSpec` has **only `imageUri`** (no command, args or env) so everything per-run arrives over
HTTP after creation; pause/resume broke the sandbox when probed and is not used; egress is an
all-or-nothing `internet_access` flag; the Kubernetes API server answers (401) from inside the sandbox
(reported to Google; no impact on us); there is **no per-sandbox IAM** (`aiplatform.sandboxEnvironments.execute`
covers every sandbox under the parent instance); the shell/custom-container surface is **v1beta1** on the
2.x `agentplatform` SDK (`client.sandboxes`, `client.runtimes`).

The platform's ceilings are undocumented and were measured (2026-09-11, `dev/live_limits_probe.py`;
the table is in TESTING.md, "The limits probe"): TTL 1 h to 30 d accepted; 8 vCPU / 16 GiB is the largest
template (16 CPU refused); ~1 MB request and ~2 MB response bodies through the proxy (the worker pages
`/events` under 1 MB); one proxied call is cut at ~600 s (the client's long-poll holds 20 s, `exec()` is
capped at 240 s); 50 calls/s on ten threads without a 429; ten concurrent creates in 7.5 s wall. Re-run the
probe when the platform announces changes or a constant in `backend.py` / `worker.py` needs re-grounding.

### 13.3 Cost

$0.085/vCPU-h + $0.009/GiB-h, billed per sandbox while it exists. A ready sandbox bills like any other, so
delete at turn end (the runtime does) and size the ready pool + `pool_max_wait_s` to the dispatch pattern.
The template's Google-side pre-warmed pool is not documented and **verified unbilled (2026-09-17)**: ten
idle 4 CPU / 4 GiB templates with no sandbox ever created from them were kept for a full billing day and
the project's Agent Platform Compute / Memory lines went *down* that day (54.71 vCPU-h / 144.64 GiB-h,
below the allocation of the ready-pool sandboxes alive that day), where billed pools would have added
~960–1900 GiB-h. So a cold engine (image + template, no sandbox) bills only for image storage, however
many engines exist. Should Google ever start billing the pool, the fallback is to create the template on
first use and reap it when idle. Sandbox billing itself comes out at or below allocation. Still delete
engine versions nobody runs: each ACTIVE template competes for provisioning capacity.

### 13.4 Risks and open items

- **Preview surface** (v1beta1). Readiness lag before the proxy routes (12–31 s; the first call can 502);
  pause/resume unusable; API names moved between SDK 1.x and 2.x. Mitigation: the ready pool hides the
  lag; we never pause; the SDK is pinned `>=2.1`.
- **No per-sandbox IAM**: the instance is the trust boundary. One instance per project/location is fine
  while all callers are one tenant; multi-tenant callers would need one instance (and client identity)
  each.
- **Google-side pool.** Persistent (two sandboxes created from a 34-hour-old template both had a PID 1
  that was 34 hours old: Google keeps at least two template-sized containers running per ACTIVE template
  from creation on) and unbilled (§13.3). Its size is not exposed or tunable — worth asking Google. Inside,
  an idle pool container is a gVisor pod with no CFS quota and `MemTotal` = the template's memory, using
  37 MB RSS and 7 CPU-seconds in 34 h, so the risk is a pricing decision, not physics.
- **Provisioning capacity.** Template creation varies from seconds to minutes with the platform's capacity
  and the number of ACTIVE templates (§13 intro); hundreds of templates would need parallel creates
  measured. `deploy` does not block on the template; `get_engine` skips FAILED and PROVISIONING ones.
- **Resource visibility from outside** — none (§6 "Observability"); an OOM is the worker going unreachable,
  reported as an explained error with the last in-sandbox sample at most 20 s before death.
- **Snapshot restore** (memory + disk): set aside — same-image restriction (§13.1, checkpoint bullet).

### 13.5 The provider seam

The runtime asks the platform for four things only: *create a container from an image with N CPU / M GiB
and internet egress, reach an HTTP port on it through an authenticated proxy, delete it, and enforce a TTL*.
Everything else — dispatch, events, checkpoints, control, secrets — is ours over HTTP and an object store.
That is the shape of every hosted code-sandbox product (Cloud Run jobs, E2B, Modal, Daytona, Fly machines,
…), so the runtime keeps the provider-specific part behind one small port, `SandboxProvider`
(`create(template, ttl) -> handle`, `call(handle, path, body, timeout)`, `delete`, `list`, plus template
create/list/delete at deploy), with the Agent Sandbox adapter (`AgentSandboxProvider`) as the first and only
implementation and in-memory fakes behind it in the test suite. No other provider is planned; the point is
to not pay later for coupling we do not need now, and to keep the worker image provider-agnostic (it already
is: a Dockerfile and an HTTP server).

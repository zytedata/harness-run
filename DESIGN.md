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
2. **Local / remote parity.** `local.deploy(spec)` (in-process ADK) and `gemini.deploy(spec)` /
   `gemini.get_engine(name)` return the *same* `Engine` + `Session` surface. Develop and test locally,
   ship remotely, **zero code change** — swap `local` ↔ `gemini`. Local uses filesystem/in-memory port
   adapters, so it needs no GCP beyond model access.
3. **Deploy is rare; lookup-and-run is the hot path.** App code addresses a deployed engine **by name**
   (optionally a version) via `gemini.get_engine(...)` and never triggers a deploy. `gemini.deploy(...)`
   is an ops/CI action.
4. **Ports & adapters.** Every platform dependency is a protocol with a concrete adapter. Storage, event
   sink, dispatch transport, secret resolver, session store, harness — all swappable.
5. **Secrets are per-invocation; never pickle secrets/paths.** Agent Engine cloudpickles the agent, so
   `AgentSpec` carries no secrets at all — credentials are passed as a `name → value` map to `run`/`send`,
   scoped to that one turn (nothing baked, nothing shared across tenants). Values never ride the invocation
   payload either (the platform persists a job's input verbatim; Pub/Sub retains unacked messages): they are
   staged at a **per-invocation GCS object** the worker fetches (and deletes only at the end of a completed
   turn — the platform re-runs a crashed attempt with the same payload, so the object must survive for the
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
                         (in-process ADK,                   · run(msg) → Run :
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
  `AgentEvent`s, and runs as an ADK `BaseAgent` so it can deploy to Agent Engine. This is the generalized
  `ClaudeCodeAgent` from the PoC.
- **Engine** is the deployed (or local) agent handle. `gemini.deploy(spec)` registers an engine under
  `spec.name` and returns a `GeminiEngine`; `gemini.get_engine(name[, version])` looks one up without
  deploying; `local.deploy(spec)` returns a `LocalEngine`. All expose `start_session()` and
  `get_session(id)`.
- **Session** is the run-plane object with the CMA-style lifecycle:
  `pending → running ↔ idle(stop_reason) → terminated`. `idle` + `stop_reason="needs_input"` is the
  interactive pause; `send()` resumes via checkpoint. A `Session` is addressable by id, so another process
  can re-attach via `engine.get_session(id)` and poll `status` / fetch `result`.
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

A **version is an Agent Runtime runtime revision** (`…/reasoningEngines/{id}/runtimeRevisions/{rev}`):
`deploy` updates the engine of that display name and the platform mints an immutable revision, with the
engine's traffic config deciding which one serves. One platform constraint shapes the API: `asyncQuery` —
the toolkit's entire run plane — exists on the **engine** only (a revision has `query`/`streamQuery` but no
async form), so a run always lands on the serving revision. Version pinning is therefore an *assertion*
(`get_engine(version=…)` fails unless that revision is the one serving) and moving traffic is an explicit
ops action (`engine.set_traffic`), not a per-caller routing choice.

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
- `Session` — `run(msg, secrets=)`, `send(msg, secrets=)`, `interrupt()`, `status`, `stop_reason`,
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
  surfaced as a `claude_stderr` status event (reaches Cloud Logging on gemini) and embedded in the error.

---

## 6. Execution model & platform contracts (the hard-won knowledge)

These are facts measured during the PoC. The library encodes them so consumers inherit them for free.

**Execution paths on Gemini Agent Runtime**
- **Async `run_query_job`** — long-running (validated 64.5 min; supports up to 7 days). But **~2.5 min
  per-job worker provisioning**, and it is *per job*, not a one-time cold start. **No config knob reduces
  it** — image size, `container_concurrency`, `min_instances` were all ruled out. The genai
  `Client().agent_engines` surface exposes only `run_query_job` / `check_query_job` / `cancel_query_job`.
- **Sync `stream_query`** — a warm instance returns in **~3–9 s**, but has a **~600 s per-request ceiling**
  and lives only on the classic `agentplatform.agent_engines.get(name)` surface.
- **Warm pool** — the way to get fast *and* long: pre-warmed jobs blocked on an inbound channel.
  Dispatch→result measured **~5.4 s** with pre-warm. See below.

**Warm pool (the `DispatchTransport` + worker loop)**
- A job submitted with a `__POOL_WAIT__` sentinel blocks pulling **its own** Pub/Sub subscription: a
  random, filtered subscription the client created in `fill_pool` and named only in that worker's job
  input (per-worker dispatch; `runtime/gemini/pool.py`). The control plane claims one idle worker off
  the pool's roster (client-owned GCS objects under `pool/`, `roster.py`; atomic via a
  generation-precondition delete) and publishes the turn addressed to that worker alone. The
  dispatched message may carry a resume directive and is processed as a normal turn. Why per worker:
  `roles/pubsub.subscriber` is consume-by-name only, so a shell in one worker cannot reach another
  worker's channel, and a channel only ever carries the one turn its worker was addressed (README,
  "The runtime identity is reachable by the agent"). Engines deployed before this (a shared
  `AGENT_POOL_SUBSCRIPTION` in their env) are still driven the old competing-consumers way, with a warning.
- The worker emits **heartbeats** while idle (the async executor kills silent jobs) and **pre-warms**
  during the wait (the Logging/GCS channels) so post-assignment setup is small. Its first event on a
  turn is `turn_started`; the client's pickup watchdog re-dispatches to another worker if it never comes.
- On dispatch, the control plane refills the pool so a warm worker is ready for the next turn; a turn
  that finds no idle worker spawns one for itself (cold latency, never a stranded turn).
- An idle worker **expires** after `AGENT_POOL_MAX_WAIT_S` (deploy-configurable via
  `pool_max_wait_s`; default a day) and is not replaced — refill is dispatch-driven only.

**Checkpoint / resume**
- Conversation: the Claude SDK `SessionStore` (append/load) + `resume=session_id`. Our `GcsSessionStore`
  keys by **`session_id` alone** (ignores `project_key`) so **any worker, any cwd** can resume — the
  default cwd-keyed local store is fatal for serverless.
- Workspace: tar the agent workspace to GCS; restore on resume. Pin the Claude session id up front via
  `options.session_id` (the cloud binary doesn't surface it on messages otherwise).
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
  side-effect at the terminal event, with Cloud Logging as the reliable emit channel.

**Observability**
- Async `run_query_job` surfaces **no** stderr, **no** stack traces, and only coarse (~12 min) GCS flushes.
  **The GCS event mirror is the live channel** (`stream.py`): the worker streams every surfaced event
  into small JSONL batch objects under `events/<sid>/` (~0.5 s cadence; a terminal `result` flushes
  immediately), and the client tails the object listing — lexical name order == chronological,
  list-after-write is strongly consistent (no ingestion lag), and GCS has no restrictive read cap, so
  tens of concurrent streamed runs are a non-event. The same objects ARE the durable history
  (`Session.history()` reads them) — one record, two roles. It is the ONLY live channel: engines
  deployed before streaming wrote the mirror at end-of-turn, so against them a turn's events all
  arrive in one batch with the terminal result (correct, just not live) and `wait_until_warm`
  times out soft — redeploy them.
- **Cloud Logging is emit-only**: every worker still writes the per-step `remote_agent_toolkit_steps`
  log (labelled by `session_id`) because debugging and alerting need an indexed, queryable,
  cross-session store — but no client run depends on it. Its READ path is capped at
  **60 `entries.list` requests/min PER PROJECT** (fixed; Google says not raisable), which is why
  it was dropped as a data plane (`CloudLoggingSink` has no `tail`). The only remaining log read is
  the bounded one-shot history backstop (`CloudLoggingSink.read`, `read_history` layer 3 — sessions
  older than mirroring); it retries 429s with backoff inside its timeout.
- **Cloud Trace spans per turn** (the console's Agent Platform *Traces* tab). The platform's ADK
  auto-instrumentation can't see inside the Claude subprocess, so the worker rebuilds the structure from
  the `AgentEvent` stream (`gemini/tracing.TurnTracer`): a root `invoke_agent` span per turn (parented
  under ADK's `invocation` span; result status/usage/cost, `gen_ai.conversation.id` = session id — what
  groups turns in the console's session view) + one `execute_tool` child per tool call with real
  durations (opened on `tool_use`, closed on the paired `tool_result`); messages/thinking are span
  events. The load-bearing platform fact (pinned by a standalone repro,
  `~/zyte/agent-runtime-tracing-repro/`, shared with Google): **the platform initializes OTel in
  query-job workers — a real `TracerProvider` with exporter and `cloud.resource_id` resource — but never
  flushes it on the job path**; the sync serving path force-flushes per query stream, while a job worker
  is torn down with the batch buffer unexported (identical stock-ADK engine: sync span exports, job span
  never). So `TurnTracer.close()` does a bounded `force_flush` at the end of every turn — that flush is
  the whole workaround; without it no toolkit span ever reaches Cloud Trace. History: an initial
  `ensure_export_pipe` (self-installed OTLP provider for a no-op ambient provider) was REMOVED after the
  probe evidence — exported spans carry the platform's `service.instance.id=<hex>-<pid>` resource stamp,
  not ours, so every cloud worker already has the platform's provider and the pipe never fired;
  resurrect from git history only if the platform ever ships job workers without one. The default RE
  service-agent role suffices for the export. INCIDENT LOG (2026-08-03..06, root-caused 08-06): every
  worker span/metrics batch 403'd (`Failed to export span batch code: 403`) on engines deployed from
  environments whose gcloud/ADC default project wasn't the deploy target. ROOT CAUSE: `AdkApp.__init__`
  snapshots `initializer.global_config.project` into its pickled `_tmpl_attrs`; the worker sets
  `GOOGLE_CLOUD_PROJECT` from the pickle and telemetry export routes to THAT project — a pickled
  `other-project` (the org-wide gcloud default) made the RE service agent attempt cross-project writes,
  hence 403 Forbidden, persistently, on every engine deployed from such a machine (and only those —
  which made it look like server-side per-engine state keyed on creation time for three days).
  Settled by an artifact-bisect ladder, each step one variable: byte-identical redeploy of a broken
  engine's staged artifacts under a different creator → still 403 (artifact-borne, not
  creation-context); model swap both directions → no effect; installed-version matrix across
  broken/clean builds → identical versions on both sides (packages ruled out); `pickletools.dis` diff
  of the two pickles → `project: other-project` vs `project: my-project`; byte-patching ONLY that
  string in the broken pickle → clean. Earlier theories disproven en route: IAM grants (telemetry
  writer roles — inert), exporter version pins (release-timing coincidence), server-side incident
  windows (the "windows" were just who deployed from which laptop). FIX: `build_adk_app` pins the
  deploy target via `aiplatform.init(project=..., location=...)` before constructing the app and hard-
  fails if the pickle captured anything else. Debugging surfaces that settled it: engine build logs
  land under the engine's `reasoning_engine_id` in Cloud Logging (diff `Successfully installed` sets;
  compare assembly-image digests), staged artifacts are readable in the staging bucket (the pickle's
  strings are cleartext — `pickletools.dis`), and beware time-confounded canaries — always re-run the
  BROKEN configuration (ideally the broken ARTIFACT, ideally byte-identically) before declaring a fix
  causal.
  Strictly best-effort: `TurnTracer` never
  raises — a tracing failure must not take a run down. Span values are truncated summaries (same text as
  the log/mirror; no new exposure surface). Traces are diagnostics; `history()` is the record.
  Live-validated facts: **Cloud Trace ingestion lag ~5–10 min** (poll patiently before declaring spans
  lost; since 2026-08-04 the v1 read API returns NOTHING for new spans — they live in the new
  telemetry-backed store, console-only — and it never exposed span exception records);
  **one trace per turn** — every turn runs in its own query job (cold submits one; a warm pool
  worker claims exactly one turn, processes it, exits), so a turn's spans nest under that job's ADK
  wrapper spans and a multi-turn session spans several traces, stitched by `gen_ai.conversation.id`
  (what keys the console's session view).
- **Content capture: tried and REMOVED (2026-07-17)** — don't re-add without the platform gap closing.
  A `deploy(capture_content=True)` default briefly baked the console's "prompt-response collection" env
  vars (`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=EVENT_ONLY` +
  `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` — a per-ENGINE setting, patchable on a live
  engine via REST `updateMask=spec.deployment_spec.env`) and had `TurnTracer` put `rat.prompt` /
  `rat.final_text` on the turn span (that part worked, live-verified). Removed because the headline
  purpose failed: the console's *session conversation* panel is fed by the platform capture
  instrumentation's content logs, which never materialize for query-job executions (the job-path
  telemetry gap above), so it stayed "No chat conversation data" and the feature over-promised. Security review stands for a future revisit: no new
  exposure class (secret values never ride prompts/payloads; the text already persists to the
  log/mirror), but Google's setting also logs `user.id` (consent caveat).
- **Known residual Traces-UI gaps (2026-07-17, platform-side; parked — revisit with Google support).**
  (1) *"(Missing span ID …)" placeholder node*: ADK's outermost runner span is never exported — it only
  ends after our turn-end `force_flush` has already run (a flush exports ended spans only), and the
  platform tears the worker down before any later export. Cosmetic: our turn/tool spans are complete
  underneath.
  (2) *Session conversation tab shows "No chat conversation data"*: see the content-capture entry above
  — the platform's content logs never materialize on the job path; for warm turns the ADK session's
  user event is the `__POOL_WAIT__` sentinel anyway (the real prompt rides the dispatch payload). Emitting their
  undocumented content-log format ourselves was judged too brittle; full prompts/outputs live in
  `session.history()`.
- **ADK events must carry `invocation_id`** (`ctx.invocation_id`, threaded through `to_adk_event`): the
  runner appends every non-partial event (our terminal result) to the platform session service, whose
  API rejects events without it (400, surfaced as an exception on the job's trace — invisible before
  tracing existed; it fired on every turn ever, after the client already had its result).
- **Every tail poll is bounded** (the logging client has no per-call timeout; a dead connection
  otherwise wedges `list_entries` forever — observed live). Transient failures ride out; a run of
  consecutive failures surfaces the error.
- **Cold runs get a job watchdog**: after ~60 s of stream silence the client probes
  `check_query_job(job_name)`; once the job is terminal (+~120 s grace) with the tail still silent, the
  client first tries to **recover the real result from the GCS event mirror** (written by the worker at
  turn end, no ingestion lag — this rescued a live run whose log entries took >8 min to become
  queryable), and only synthesizes an explained error result when no durable terminal record exists.
  Warm turns have no per-turn job handle (the turn runs in whichever pool worker claimed it) — bounded
  polls only.
- **Cloud Logging ingestion lag can exceed several minutes** (observed live) — the mirror, not the log,
  is the reliable record; the log is the low-latency *usually*-fast channel.
- **A young query-job operation is not cancellable** (`FAILED_PRECONDITION` for roughly its first
  minutes); external cancels/interrupts may need retries. The in-cloud **Bash tool default timeout is
  120 s** — long commands need an explicit timeout or backgrounding.

**Persistence / job history (what survives a run, and where)**
- **Mirrored events** — the worker buffers every surfaced `AgentEvent` and flushes one
  `events/<sid>/<epoch_ms>.jsonl` per turn to the output bucket. The canonical durable history: the only
  session-keyed record for **warm** turns (their platform job output goes to a throwaway pool path) and the
  only layer keeping **all** turns of a multi-turn session. Read via `session.history()`.
- **Platform job output** — `jobs/<sid>.jsonl` (+ `<sid>_input.jsonl`), written by the platform for each
  cold job. Per-job: a resume under the same session **overwrites** it. ⚠️ the input file persists the
  query verbatim — which is why per-invocation secrets never ride the invocation payload (see §3.5): they
  are staged at a per-invocation `invocation-secrets/<sid>-<nonce>.json` object the worker fetches and
  deletes at the end of a completed turn (retry-safe; a 1-day lifecycle rule reaps orphans).
- **Session/turn configs** — run-time configuration is typed by the lifetime of what it configures
  (README "Deploy / session / turn"): a `SessionConfig` binds ONCE at `start_session(config=)` and is
  persisted at the session's stable `session-config/<sid>.json` key (every turn's payload points at
  it; `get_session` re-attach reads it back and accepts no substitute — the session's world is
  created on turn 1 and snapshot-restored after, so a mid-conversation change could not be honored);
  a `TurnConfig` rides one `run()`/`send()` as a nonce-keyed `turn-config/<sid>-<nonce>.json` (cold
  directives `AGENT_SESSION_CONFIG_GCS=` / `AGENT_TURN_CONFIG_GCS=`). The worker merges deploy-baked
  ← session ← turn, validates the harness choice against the image's baked CLIs, executes the
  result, and echoes it as an `effective_spec` event — the ground-truth record for post-mortem and
  replay. Config objects are non-secret by construction (inline `user:token@` repo URLs are refused
  at config construction) and are KEPT (30-day lifecycle rule) rather than deleted like secrets. A
  missing config, an unknown config field, or an unbaked harness FAILS the turn — never a silent
  fall-back to the baked spec. With no configs, nothing is staged and the turn runs the deploy-baked
  spec exactly as before. Session/turn parameters (target repo/ref, model, prompt variant, budgets)
  thus need no engine redeploy; deploy-time-only fields (`packages`, harness availability) still
  come from the deployment, and baked skills serve as a staging fast path only while the effective
  `skills` match the baked declaration.
- **Cloud Logging** — the per-step log, bounded by log-bucket retention (~30 days default).
- **Checkpoints** — transcript + workspace tar under `checkpoints/`, keyed by (mapped) session id.
- `engine.list_sessions()` merges the GCS layers (bucket-wide) with the engine's ADK sessions;
  `session.history()` reads mirror → job output → Cloud Logging, first hit wins;
  `session.last_result` reconstructs from the terminal history event for re-attached sessions.
- Session ids: cold = ADK `sessions.create` (numeric); warm = client-minted UUID. The Claude/checkpoint
  side always uses a canonical UUID (`uuid5` of a non-UUID sid — the CLI rejects non-UUID session ids);
  the raw sid keys Cloud Logging and the GCS records.

**Deploy contracts (applied automatically by `gemini.deploy`)**
- **uv ships as a Python requirement** + its bin dir prepended to PATH — **build-script filesystem changes
  do NOT persist** into the runtime container (only `git` survives from the base image).
- **glibc base, Python 3.12** — the `claude-agent-sdk` wheel ships a self-contained glibc ELF `claude`
  binary; no Alpine/musl; SDK supports Python ≤3.13.
- `a2a-sdk>=0.3.4,<0.4` (1.x incompatible with ADK 2.3.0).
- **Platform-critical pins live in-tree** (`runtime/gemini/constraints.txt`, pip `-c` semantics) —
  engines resolve requirements at BUILD time, so unpinned deps mean two deploys of one commit can differ.
  `build_requirements` merges the constraints client-side (the build offers no hook for a real `-c` file:
  extra_packages extract only at container runtime, after pip has run) and fails fast on an unsatisfiable
  merge with `spec.packages`. The pickle-coupled packages (aiplatform/cloudpickle/pydantic — the engine
  build unpickles the AdkApp the deploy venv pickles) are additionally checked against the deploy venv by
  `verify_deploy_env()`. Refresh = bump pin(s) + canary deploy + live turn + clean telemetry, one
  reviewed diff.
- Only `/tmp` is writable → agent cwd under `/tmp/agent-jobs/<session-id>/workspace` (see the
  workspace-leaf contract below).
- `IS_SANDBOX=1` to allow `bypassPermissions` under root.
- `min_instances=0` (the default): the toolkit is async-only, where every job provisions its own worker —
  a standing container serves only the unused sync path while billing continuously. (`>=1` would matter
  only for real long *sync* jobs; min=0 SIGTERM-recycles a sync container ~2.5 min.)
- `resource_limits` (deploy kwarg, optional): container CPU/memory; the platform default is
  `{"cpu": "4", "memory": "4Gi"}` — that 4Gi is shared by the harness CLI, the ADK app, subagents, and
  everything the agent's tools spawn, and memory-heavy turn work (dependency builds, whole-project
  imports) can OOM-kill the worker mid-turn (observed live 2026-07-28/29; the job runner then replays
  the turn from scratch). Deploy memory-heavy agents with e.g. `{"cpu": "4", "memory": "16Gi"}`
  (memory max `32Gi`; cpu one of 1/2/4/6/8).

**Identity / IAM (two identities — documented in the runbook)**
- Deploy/operator = the **impersonated SA** (publish, read logs, submit jobs).
- **Runtime identity = the RE service agent** `service-<n>@gcp-sa-aiplatform-re.iam.gserviceaccount.com` —
  *all* runtime resource access (Secret Manager, GCS, Pub/Sub, Logging) authorizes against **it**, not the
  operator SA. Granting the operator SA a role does nothing for the running job. The library ships a
  permissions runbook listing every grant each port needs and on which identity.

---

## 7. Ports & adapters

Each is a `typing.Protocol`; concrete adapters ship for prod (GCP) and dev (local/in-memory).

- **`Harness`** — `build_options(spec, ctx)` + an async run loop yielding `AgentEvent`s. Adapters:
  `ClaudeCodeHarness` (default) and `CodexHarness`, selected by `spec.harness` via `resolve_harness`
  (the single seam both runtimes use). Shared policy — secret routing, agent-env layering, the
  interactive suffix, the inline workspace checkpoint — lives in `harness/_shared.py` so the bindings
  can't drift where the spec doesn't distinguish them. A **conformance suite**
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
| Warm pool | `pool.py` + worker loop | → `runtime/gemini/pool.py` + `DispatchTransport` |
| Skills staging | `skills.py` | → `skills.py` (configurable sources) |
| Git / GitHub MCP | `git_repo.py`, `_mcp_servers` | → `integrations/` |
| Artifacts | `artifacts.py` | → over `BlobStore` |
| Cost tracking | result metadata passthrough | → `RunResult` |
| Config / runtime resolution | `config.py` | → `AgentSpec` + resolvers |
| Deploy | `deploy/deploy_agent_engine.py` | → `runtime/gemini/_deploy.py` (contracts encoded) |
| Control-plane dispatch + log tail | `scrape_cli.py` | → `Session` / `EventSink.tail` |
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
│   │       ├── backend.py         # deploy(), get_engine(), list_engines(), GeminiEngine
│   │       ├── _deploy.py         # packaging + contracts (uv/glibc/IS_SANDBOX/...)
│   │       │                      #   (underscored: `deploy.py` would shadow gemini.deploy)
│   │       ├── adk_agent.py       # the deployed ADK BaseAgent wrapping the harness
│   │       ├── translate.py       # AgentEvent → ADK Event
│   │       ├── handoff.py         # GCS staging of per-invocation secrets (retry-safe cleanup)
│   │       ├── history.py         # durable event mirror + history/list_sessions readers
│   │       ├── tracing.py         # AgentEvent stream → Cloud Trace spans (TurnTracer)
│   │       ├── resources.py       # worker CPU/RAM self-sampling (OOM forensics)
│   │       └── pool.py            # WarmPool worker side
│   ├── ports/
│   │   ├── blobstore.py           # BlobStore + GcsBlobStore + LocalBlobStore
│   │   ├── eventsink.py           # EventSink + CloudLoggingSink + InMemorySink
│   │   ├── dispatch.py            # DispatchTransport + PubSubDispatch + InMemoryDispatch
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
  tailing; structured outputs. Permissions runbook.
  *First-milestone gate:* **the minimal generic example runs locally *and* on Gemini Agent Runtime with a
  warm start.** *README:* the **deploy quickstart** + engine management.
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

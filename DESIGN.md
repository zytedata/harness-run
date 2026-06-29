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
- Multi-harness / multi-LLM implementations. We design the **seam** (a `Harness` protocol) but ship
  only the **Claude Code (Agent SDK)** binding. Codex / custom-LLM harnesses come later, behind it.
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
> The underlying Google Python SDK is still imported as `vertexai` / `google-cloud-aiplatform`; that's
> an internal detail, not part of our surface.

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
5. **Runtime resolution, never pickle secrets/paths.** Agent Engine cloudpickles the agent; absolute paths
   and secrets must resolve **at runtime** from env / Secret Manager, not be baked in. (A PoC crash taught
   this.) The library enforces it: `AgentSpec` carries *references* (secret names, skill sources), not values.
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
    mcp_servers=[McpServer.github()],
    secrets=["ZYTE_API_KEY", "GH_PAT"],      # names → resolved from Secret Manager at runtime, never pickled
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
```

**Managing deployed engines** (control plane):

```python
gemini.deploy(spec, project=..., location=...)   # create / update; mints a new version
gemini.get_engine("spider-builder")              # latest version (app code default)
gemini.get_engine("spider-builder", version=3)   # pin a version
gemini.list_engines(project=..., location=...)   # discover what's deployed
engine.versions()                                # list versions of one engine
engine.name, engine.version, engine.resource     # identity / underlying resource name
```

Engine identity maps `spec.name` → a stable engine (Agent Engine `display_name`); each `deploy` mints a
new version. App code pins or takes latest; it does not deploy.

**Key types** (sketch — finalized in `spec.py` / `runtime/base.py`):

- `AgentSpec` — the declarative definition. Serializable (`to_dict` / `AgentSpec.from_yaml`).
- `SystemPrompt.inherit(append=..., exclude_dynamic_sections=...)` — inherit the harness's built-in
  system prompt and append. A plain `str` replaces it entirely.
- `SkillSource` — `.git(url, ref=)`, `.local(path)`, `.builtin(name)`; a list allows base + extra sources
  (multi-source skills, §12). Resolved & staged at deploy/run time.
- `McpServer` — `.github()`, `.remote(name, url)`, `.stdio(...)`; credentials come from `secrets`, not here.
- `Engine` — `start_session()`, `get_session(id)`, `versions()`, `name`/`version`/`resource`.
- `Session` — `run()`, `send()`, `interrupt()`, `status`, `stop_reason`, `last_result`, `resume()`, `fork()`.
- `Run` — `__await__` (→ `RunResult`), `__aiter__` (→ `AgentEvent`s), `done`, `status`, `result`.
- `AgentEvent` — `kind`, `summary`, `raw`; cost/usage carried on the terminal event.
- `RunResult` — `text`, `structured_output`, `is_error`, `num_turns`, `cost_usd`, `usage`, `session_id`,
  `artifacts`.

---

## 6. Execution model & platform contracts (the hard-won knowledge)

These are facts measured during the PoC. The library encodes them so consumers inherit them for free.

**Execution paths on Gemini Agent Runtime**
- **Async `run_query_job`** — long-running (validated 64.5 min; supports up to 7 days). But **~2.5 min
  per-job worker provisioning**, and it is *per job*, not a one-time cold start. **No config knob reduces
  it** — image size, `container_concurrency`, `min_instances` were all ruled out. The genai
  `Client().agent_engines` surface exposes only `run_query_job` / `check_query_job` / `cancel_query_job`.
- **Sync `stream_query`** — a warm instance returns in **~3–9 s**, but has a **~600 s per-request ceiling**
  and lives only on the classic `vertexai.agent_engines.get(name)` surface.
- **Warm pool** — the way to get fast *and* long: pre-warmed jobs blocked on an inbound channel.
  Dispatch→result measured **~5.4 s** with pre-warm. See below.

**Warm pool (the `DispatchTransport` + worker loop)**
- A job submitted with a `__POOL_WAIT__` sentinel blocks pulling a shared Pub/Sub subscription
  (competing-consumers = atomic claim). The dispatched message may carry a resume directive and is
  processed as a normal turn.
- The worker emits **heartbeats** while idle (the async executor kills silent jobs) and **pre-warms**
  during the wait (provision skills + establish the GCS channel) so post-assignment setup is ~0.
- On claim, the control plane refills the pool so a warm worker is ready for the next turn.

**Checkpoint / resume**
- Conversation: the Claude SDK `SessionStore` (append/load) + `resume=session_id`. Our `GcsSessionStore`
  keys by **`session_id` alone** (ignores `project_key`) so **any worker, any cwd** can resume — the
  default cwd-keyed local store is fatal for serverless.
- Workspace: tar the job dir to GCS; restore on resume. Pin the Claude session id up front via
  `options.session_id` (the cloud binary doesn't surface it on messages otherwise).
- **Finalize inline at the terminal `result` event.** The async executor **stops draining the generator
  after the result event**, so post-loop checkpoint/artifact code is dead in-cloud — it must run as a
  side-effect at the terminal event, with Cloud Logging as the reliable emit channel.

**Observability**
- Async `run_query_job` surfaces **no** stderr, **no** traces, and only coarse (~12 min) GCS flushes.
  **Cloud Logging (`log_struct`) is the only near-real-time channel** — the harness emits a per-step
  structured log; the client tails it filtered by `session_id`. This is the `EventSink` port.

**Deploy contracts (applied automatically by `gemini.deploy`)**
- **uv ships as a Python requirement** + its bin dir prepended to PATH — **build-script filesystem changes
  do NOT persist** into the runtime container (only `git` survives from the base image).
- **glibc base, Python 3.12** — the `claude-agent-sdk` wheel ships a self-contained glibc ELF `claude`
  binary; no Alpine/musl; SDK supports Python ≤3.13.
- `a2a-sdk>=0.3.4,<0.4` (1.x incompatible with ADK 2.3.0).
- Only `/tmp` is writable → per-job cwd under `/tmp/agent-jobs/<session-id>`.
- `IS_SANDBOX=1` to allow `bypassPermissions` under root.
- `min_instances>=1` for real long sync jobs (min=0 SIGTERM-recycles ~2.5 min); min=0 is fine for the
  cold-start-per-job async path.

**Identity / IAM (two identities — documented in the runbook)**
- Deploy/operator = the **impersonated SA** (publish, read logs, submit jobs).
- **Runtime identity = the RE service agent** `service-<n>@gcp-sa-aiplatform-re.iam.gserviceaccount.com` —
  *all* runtime resource access (Secret Manager, GCS, Pub/Sub, Logging) authorizes against **it**, not the
  operator SA. Granting the operator SA a role does nothing for the running job. The library ships a
  permissions runbook listing every grant each port needs and on which identity.

---

## 7. Ports & adapters

Each is a `typing.Protocol`; concrete adapters ship for prod (GCP) and dev (local/in-memory).

- **`Harness`** — `build_options(spec, ctx)` + an async run loop yielding `AgentEvent`s. Adapter:
  `ClaudeCodeHarness`.
- **`BlobStore`** — `put_bytes / get_bytes / put_tree / get_tree / exists`. Adapters: `GcsBlobStore`,
  `LocalBlobStore`. Backs checkpoint, workspace, artifacts.
- **`EventSink`** — `emit(event)` (write side, runtime) + `tail(session_id, since)` (read side, client).
  Adapters: `CloudLoggingSink`, `InMemorySink`.
- **`DispatchTransport`** — `publish(message)` + `claim(timeout)` (competing-consumers pull). Adapters:
  `PubSubDispatch`, `InMemoryDispatch`.
- **`SecretResolver`** — `resolve(name) -> value`. Adapters: `GcpSecretResolver` (Secret Manager, against
  the RE service agent), `EnvSecretResolver` (local dev).
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
| Deploy | `deploy/deploy_agent_engine.py` | → `runtime/gemini/deploy.py` (contracts encoded) |
| Control-plane dispatch + log tail | `scrape_cli.py` | → `Session` / `EventSink.tail` |
| **Scrapy Cloud, monitoring, scrape prompts** | `selfheal/`, `_INTERACTIVE_SUFFIX` | **stay in gemini-agent-runtime** behind the seam |
| Probes, scratch, `gh_pat.txt` | `scratch_minimal/`, probes | **dropped** |

---

## 9. Decided design choices (2026-06-29)

1. **Harness scope** — seams now, **Claude-only** implementation. Harness-agnostic `AgentSpec` + a
   `Harness` protocol; only the Claude Code binding ships. No speculative second-harness code.
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
├── pyproject.toml                 # core + extras: [gemini] [pubsub] [dev]
├── README.md                      # team onboarding (grows each phase, §11)
├── DESIGN.md                      # this file
├── docs/
│   ├── permissions.md             # the two-identity IAM runbook (per-port grants)
│   └── platform-notes.md          # the §6 contracts, expanded
├── remote_agent_toolkit/
│   ├── __init__.py                # AgentSpec, SystemPrompt, SkillSource, McpServer, gemini, local
│   ├── spec.py                    # AgentSpec + value types (serializable)
│   ├── events.py                  # AgentEvent, RunResult, RunStatus, StopReason, Run handle
│   ├── harness/
│   │   ├── base.py                # Harness protocol
│   │   ├── claude_code.py         # ClaudeCodeHarness (ADK BaseAgent over the Agent SDK)
│   │   └── translate.py           # Claude SDK message → AgentEvent
│   ├── runtime/
│   │   ├── base.py                # Engine + Session protocol + state machine + Run handle
│   │   ├── local.py               # local.deploy / local.run -> LocalEngine
│   │   └── gemini/
│   │       ├── backend.py         # deploy(), get_engine(), list_engines(), GeminiEngine
│   │       ├── deploy.py          # packaging + contracts (uv/glibc/IS_SANDBOX/...)
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

- **P0 — Scaffold.** Repo skeleton (no `src/`), `pyproject` (core + `[gemini]`/`[pubsub]`/`[dev]`), the port
  & harness protocols as real signatures, the conformance harness, CI.
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
  decision.
- **Structured outputs** — prefer the SDK's constrained-decoding `structured_output`; keep "parse last
  JSON block" only as a fallback for harnesses that lack it. Confirm Vertex model support per model.
- **Outcomes / rubrics** — CMA's iterate-until-graded "definition of done" is attractive for autonomous
  background work; candidate post-P4 capability (a `verify=Rubric(...)` on `AgentSpec`).
- **Second harness** — codex (subprocess) or a custom LLMNL harness behind the `Harness` protocol; the
  seam exists from P0, the binding is future.
- **Non-GCP backends** — the ports are protocols; an AWS/Azure adapter set is possible but unscheduled.
- **Scheduled runs** — CMA-style cron `deployments` (each firing = a session) could wrap deploy+session
  for recurring jobs (e.g. self-heal monitoring); future.

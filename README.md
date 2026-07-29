# remote-agent-toolkit

A Python library for **defining and running remote/background AI agents** at Zyte. Define an agent
declaratively as an `AgentSpec`, then run it both **locally** (in-process, for dev) and **remotely** on
**Gemini Agent Runtime** (prod) through *one* API — with custom skills, GitHub access, structured outputs,
checkpoint/resume and warm starts, without re-learning the platform's sharp edges. Two agent harnesses
ship behind the same API: **Claude Code** (the default) and **Codex** (OpenAI models; see
[Choosing the harness](#choosing-the-harness-claude-code-or-codex)).

> **Status: `local` and Gemini Agent Runtime both work — validated live.** Define an `AgentSpec` and run it
> in-process (`local.deploy`), or deploy + run on Agent Runtime (`gemini.deploy` / `gemini.get_engine`), with
> skills, structured output, checkpoint/resume, and a **warm pool** (~10–20 s pickup vs ~2.5 min cold) — all
> exercised end-to-end on real infrastructure. The two paths share one `Engine`/`Session`/`Run` API. Not yet
> built (raise `NotImplementedError` or simply absent): session `fork()`, `get_engine(version=…)` pinning, and
> a sync run path. See [`examples/minimal`](examples/minimal) for a runnable agent and
> [`DESIGN.md`](DESIGN.md) for the architecture, platform contracts (§6), and roadmap (§11).

## Install

Private package — install straight from GitHub (you need repo access). Requires **Python 3.12**
(`claude-agent-sdk` supports ≤3.13):

```bash
pip install "git+ssh://git@github.com/zytedata/remote-agent-toolkit.git"
# or:  uv pip install "git+ssh://git@github.com/zytedata/remote-agent-toolkit.git"
```

Developing on the toolkit itself (early users are expected to contribute)? Clone it and `uv sync` — that
installs the runtime deps plus the `dev` group (pytest, ruff, mypy).

## Define an agent

```python
from remote_agent_toolkit import AgentSpec, SystemPrompt, SkillSource, McpServer, gemini, local

spec = AgentSpec(
    name="spider-builder",
    model="claude-sonnet-4-6",
    system_prompt=SystemPrompt.inherit(append="Prefer the Zyte web-scraping skills."),
    skills=[SkillSource.git("https://github.com/zytedata/claude-skills", ref="0.2.0")],
    mcp_servers=[McpServer.github()],        # token supplied per-invocation, not here
    permission_mode="bypassPermissions",     # the default — safe because each run gets an isolated cwd
    max_turns=120,
    max_budget_usd=10.0,
    checkpoint=True,                         # enable resume / interactive pauses
)
```

The spec carries **no secrets** — credentials are passed per-invocation to `run`/`send` so nothing sensitive
is ever baked into the deployment or shared across runs (see [Secrets & security](#secrets--security)).
Every field except `name` and `model` has a sensible default (see [`spec.py`](remote_agent_toolkit/spec.py));
a two-line spec (`AgentSpec(name=..., model=...)`) is a valid agent. For **structured output**, see the
section below.

## Choosing the harness: Claude Code or Codex

`harness="claude-code"` (the default) runs the Claude Code loop via the Claude Agent SDK.
`harness="codex"` runs OpenAI's Codex instead — same spec, same Engine/Session/Run surface,
both backends:

```python
spec = AgentSpec(
    name="codex-agent",
    model="gpt-5.6-luna",                    # cheapest of the 5.6 family; sol/terra for heavier work
    harness="codex",
    skills=[SkillSource.git("https://github.com/zytedata/codex-skills")],
)
result = await engine.start_session().run(
    "scrape https://books.toscrape.com for title, price",
    secrets={"OPENAI_API_KEY": os.environ["OPENAI_API_KEY"]},
)
```

What to know when running Codex:

- **Model auth**: OpenAI models are called directly (they have no Vertex path), so every run —
  local or remote — needs an `OPENAI_API_KEY` per-invocation secret (local runs fall back to the
  ambient env var). The harness routes it to `codex login` inside an isolated per-job `CODEX_HOME`;
  it never appears in the agent's shell environment.
- **Skills** use the same `<name>/SKILL.md` format and the same `SkillSource` list — they are staged
  into `<cwd>/.agents/skills`, Codex's discovery path (that's the layout of
  [zytedata/codex-skills](https://github.com/zytedata/codex-skills)).
- **Cost & caps**: Codex reports token counts but no dollars, and enforces no turn/budget caps of
  its own — the harness prices tokens via [LiteLLM's pricing dataset](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json)
  (fetched once per process, so new models are priced without a toolkit release; a small baked table
  covers the GPT-5.6 family offline) and interrupts the run at `max_turns` / `max_budget_usd`. For a
  model neither source knows you get `cost_usd=None`, a `cost_unknown` status event, and no budget
  enforcement; the result event records which source priced the run (`price_source`).
- **Checkpoint/resume** works cross-worker: the Codex conversation (a local rollout file) is
  persisted to the blob store alongside the workspace snapshot and restored on `send()`.
- **Reasoning effort**: `spec.reasoning_effort` becomes the SDK's per-turn `effort`, re-applied on
  every turn so resumed conversations keep it. Codex has no `max` level; it maps to `xhigh` with a
  status warning.
- **Gaps**: `allowed_tools`/`disallowed_tools` have no Codex equivalent (ignored with a status
  warning), and Codex has no background-task re-invocation, so `background_task_timeout` is inert.
  `permission_mode` maps onto Codex's sandbox+approval pairs (`bypassPermissions` → full access,
  never ask; `default` → workspace-write with Codex's auto-reviewer).
- Other model providers for Codex (e.g. OpenRouter) are planned as a follow-up.

## Dev: run locally, in-process

Same `Engine` + `Session` API as prod, no GCP beyond model access. Needs **Python 3.12** and Claude Code
auth in your environment (the Agent SDK drives the `claude` CLI). See a complete runnable agent in
[`examples/minimal`](examples/minimal).

```python
import asyncio
from remote_agent_toolkit import local

async def main():
    engine = local.deploy(spec)
    session = engine.start_session()
    result = await session.run("scrape https://books.toscrape.com for title, price")   # wait for completion
    print(result.text, result.structured_output, result.cost_usd)

asyncio.run(main())
```

For a quick sync script, `local.run(spec, "…")` does `deploy → start_session → await run` and returns the
`RunResult`.

**Seeding inputs / collecting artifacts.** `session.workspace` is the agent's working directory on the
host — a `Path` you can drop input files into before `run()` and read the agent's output files from after
(it's created on first access). Use the accessor rather than deriving the path yourself: the layout is
`<workdir>/jobs/<session-id>/workspace/`, where the `workspace/` leaf is deliberate — the agent runs in a
directory whose own name says "this is your workspace", not in the anonymous `jobs/<uuid>` dir (which reads
as disposable temp and tempts weaker models into `cd`-ing away, leaving deliverables outside the collected
dir). The same `workspace/` cwd convention applies on `gemini`, but there the filesystem is remote, so
`session.workspace` raises — seed via the prompt or `spec.repos`, collect via events or a repo push.

## Consuming a run: wait, stream, or poll

`session.run(msg)` (and `session.send(msg)` to resume) returns a `Run` handle, consumable three ways — the
same on `local` and `gemini`:

```python
# (a) wait for completion (simplest — the headline default). Pass any credentials the run
#     needs as per-invocation secrets (name → value); see "Secrets & security" below.
result = await session.run("/scrape ...", secrets={"SH_APIKEY": os.environ["SH_APIKEY"]})

# (b) stream events as they happen (UI / live logs)
async for ev in session.run("/scrape ..."):
    print(ev.kind, ev.summary)               # thinking | tool_use | tool_result | message | status | result
result = session.last_result                 # populated once the run completes

# (c) poll — e.g. a web handler that can't hold the connection, or another process
run = session.run("/scrape ...")             # returns immediately
while not run.done:
    await asyncio.sleep(2)
    print(run.status)                        # pending | running | idle | terminated
result = run.result

# re-attach later / elsewhere by session id, then poll or continue
session = engine.get_session(session_id)
if session.status == "idle" and session.stop_reason == "needs_input":
    await session.send("yes, that schema looks right")     # resumes the conversation (a fresh turn)
```

**Results are kept honest.** `RunResult` keeps its accounting on *error* results too — an
`error_max_turns` run reports its real `cost_usd`/`num_turns`/`usage` (it spent right up to its limit;
treating it as free corrupts budget bookkeeping). And the terminal result is authoritative: if the
underlying `claude` CLI exits non-zero *after* the final result was streamed (it happens transiently even
on finished runs), the run keeps its result and gets `result.warning` set instead of being voided into an
error. For post-mortems, the CLI's stderr is captured to `<workdir>/jobs/<session-id>/stderr.log` (beside,
not inside, the workspace); on a CLI-exit failure its tail is also surfaced as a `claude_stderr` status
event and embedded in the error text.

**Background tasks are honored.** If the agent starts a background job (Bash `run_in_background`) or arms
the Monitor tool and then ends its turn — the trained, efficient behavior for waiting on long processes
like a verification crawl — the run does **not** end there: the harness holds the session open and the
model is re-invoked when the task completes, exactly as in interactive Claude Code. Interim turn ends
surface as `awaiting_tasks` status events; the run's single `result` comes when no work is pending.
`spec.background_task_timeout` (seconds, default 3600) bounds how long a turn waits on still-running
tasks — size it to the longest job the agent legitimately waits on. Two accounting notes: `num_turns` is
cumulative across these re-invocations, and `max_turns` caps each invocation segment rather than the
whole run (`max_budget_usd` remains a global cap). Event-driven waiting instead of poll-loops is exactly
what keeps long crawls nearly free in turns.

Runs are hang-proofed on `gemini`: every log-tail poll is time-bounded (a dead connection costs a ~30 s
retry, not a frozen run), and a **cold** run carries a job watchdog — when the remote job has terminated but
the tail is still silent, the client first recovers the **real result from the durable GCS event mirror**
(covers Cloud Logging ingestion lag, which can run several minutes), and only if no terminal record exists
anywhere (worker OOM, external cancel, engine deleted) ends the run within ~3 minutes with an error result
explaining the job state — instead of waiting out the 1 h tail cap. Warm turns run in a pool worker without
a per-turn job handle, so they get the bounded polls only.

## Structured output

Set `output_schema` to a **pydantic model** (or a JSON-schema `dict`) and `result.structured_output` holds
the parsed, validated object. The schema only describes the *shape* — you must **also tell the agent, in the
prompt, what to emit**. Don't restate the fields by hand; serialize the schema into the prompt so the two
never drift. The toolkit parses JSON out of the agent's final text (a ```` ```json ```` block, a bare object,
or the whole message); it does not constrain decoding.

```python
import json
import pydantic
from remote_agent_toolkit import AgentSpec, local

class PriceCheck(pydantic.BaseModel):
    in_stock: bool
    price_gbp: float

spec = AgentSpec(name="price-check", model="claude-sonnet-4-6", output_schema=PriceCheck)
prompt = (
    "Check the product at <url> and report the result as a JSON object matching this schema:\n"
    f"{json.dumps(PriceCheck.model_json_schema())}"
)
result = await local.deploy(spec).start_session().run(prompt)
print(result.structured_output)    # PriceCheck(in_stock=True, price_gbp=51.77), or None if unparseable
```

Parsing is best-effort: if the final message has no JSON matching the schema, `structured_output` is `None`
and you still have `result.text`. A runnable version is in [`examples/minimal`](examples/minimal).

## Customizing the agent's environment

Beyond skills, several `AgentSpec` fields shape what the agent can do and the environment its tools run in:

- **`model`** — the Claude model id (e.g. `"claude-sonnet-4-6"`, `"claude-haiku-4-5"`).
- **`allowed_tools` / `disallowed_tools`** — tool allow/deny lists. Unset `allowed_tools` defaults to a
  coding-agent set (`Read, Write, Edit, Bash, Glob, Grep, TodoWrite`); the `Skill` tool is enabled
  automatically when skills are present.
- **`max_turns` / `max_budget_usd`** — hard caps on loop length and spend (the run ends with the matching
  stop reason).
- **`reasoning_effort`** — how much reasoning the model spends per response
  (`"none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max"`; unset = harness default). The
  vocabulary is the union of both harnesses'; each maps levels only the other supports to its nearest
  own (Claude: `minimal`/`none` → `low`; Codex: `max` → `xhigh`, with a status warning).
- **`env`** — extra **non-secret** environment variables for the agent's tool subprocess (config flags,
  etc.). Values live in the spec and are baked into the deployed engine, so never put secrets here — pass
  those per-invocation (see [Secrets & security](#secrets--security)).
- **`mcp_servers`** — `McpServer.github()`, `.remote(name, url, headers=...)`, `.stdio(name, command, args=...)`.
  A `github()` server's token is supplied per-invocation under a conventional name (`GH_TOKEN` /
  `GITHUB_TOKEN` / `GH_PAT`) and injected into its headers, not the agent's env.

Credentials (the agent's own API keys, repo push tokens, MCP tokens) are **not** spec fields — they are
passed at call time to `run`/`send`. See [Secrets & security](#secrets--security).

**Python libraries & CLI tools.** The agent's `Bash` tool runs in a real shell with **`uv`** and **`git`** on
`PATH`, so it can fetch and run dependencies on the fly — e.g. `uv run --with httpx script.py`, or
`uv pip install …` inside its working directory. That covers most "use library X" needs with no change to
the deployment. To pin libraries into the deployed engine instead, see
[Pre-baked engine dependencies](#pre-baked-engine-dependencies) below.

**Long-running commands.** The Bash tool's timeouts are Claude Code defaults, not platform limits: a command
is killed after **120 s** unless the agent passes a per-call `timeout`, which is itself capped at **10 min**.
For agents that legitimately run long commands (crawls, builds), raise the caps via `env` — they are ordinary
Claude Code environment settings:

```python
spec = AgentSpec(
    name="crawler",
    model="claude-sonnet-4-6",
    env={"BASH_DEFAULT_TIMEOUT_MS": "600000", "BASH_MAX_TIMEOUT_MS": "3600000"},  # 10 min / 1 h
)
```

(Alternatively the agent can start long work with the Bash tool's `run_in_background` and poll it.) Also
tunable: `BASH_MAX_OUTPUT_LENGTH` for commands with very large output.

## Cloning a git repo

A common setup is to clone a repo into the agent's working directory **before it runs** — so it can read and
modify the code, then commit and push. Declare repos on the spec, naming (via `auth`) the per-invocation
secret that authenticates each one:

```python
from remote_agent_toolkit import AgentSpec, RepoSource

spec = AgentSpec(
    name="repo-fixer",
    model="claude-sonnet-4-6",
    repos=[
        RepoSource.git("https://github.com/zytedata/some-repo", ref="main", auth="GH_TOKEN"),
        # Bitbucket API token authenticates as <account>:<token> — name the account via auth_user:
        RepoSource.git("https://bitbucket.org/acme/spiders", auth="BITBUCKET_API_TOKEN",
                       auth_user="my-bitbucket-account"),
    ],
)

# The token values are supplied at call time, never baked into the spec/engine:
result = await session.run(
    "fix the failing test and push a branch",
    secrets={"GH_TOKEN": gh, "BITBUCKET_API_TOKEN": bb},
)
```

Each repo is cloned into the agent's cwd before the loop starts. When the `auth` secret is supplied, the
clone is **authenticated and push-ready**: the token is embedded into `origin` as
`https://<user>:<token>@host/...` and a commit identity is configured, so the agent can `git push`. The
userinfo `<user>` defaults per host (`x-access-token` for GitHub, `x-token-auth` for Bitbucket access tokens,
`oauth2` for GitLab); set `auth_user` for schemes that pair the token with a real account name — e.g. a
Bitbucket **API token** (`https://<account>:<token>@bitbucket.org/...`). Without `auth` (or for a repo with no
`auth`) the clone is read-only, which is fine for public repos. The push token is **not** placed in the agent's environment, and it is
scrubbed from `.git/config` before any checkpoint snapshot (then re-embedded on resume from the freshly
supplied secret). To let the agent open PRs, also add `McpServer.github()`.

> **Push auth is reachable by the agent — scope it, don't rely on hiding it.** To `git push`, the token
> must be usable by the agent (it can read `.git/config`). The real defense is a **scoped, short-lived**
> credential — a GitHub App installation token limited to the target repo, a Bitbucket repository/workspace
> access token — not a broad personal token. See [Secrets & security](#secrets--security).

## Secrets & security

Secrets are **passed per-invocation**, never declared on the spec and never baked into the deployed engine.
You hand `run`/`send` a `name → value` map; the values live only for that run:

```python
result = await session.run(
    "scrape the catalog and push results",
    secrets={
        "SH_APIKEY": sh_key,             # the agent's own API key (its code/tools read it)
        "GH_TOKEN": gh_installation_tok, # a repo's auth= token (git push) — see "Cloning a git repo"
    },
)
```

**How each secret is routed.** The toolkit puts a secret where its *consumer* needs it, and nowhere else:

| Secret | Consumer | Placed in the agent's env? |
|---|---|---|
| the agent's own keys (`SH_APIKEY`, …) | the agent's code/tools | **yes** — that's the point |
| a repo's `auth=` token | embedded in `git` `origin` for clone/push | **no** |
| a `McpServer.github()` token (`GH_TOKEN`/`GITHUB_TOKEN`/`GH_PAT`) | injected into MCP request headers | **no** |

**The blast-radius model — the honest part.** A background agent can be steered by hostile input (a page it
scrapes, a file in a repo) into revealing whatever it can reach: environment variables, `.git/config`, a
token it was handed. The toolkit does **not** pretend to hide credentials from the agent. Instead it shrinks
the blast radius:

1. **Per-invocation, not baked/shared** — a run only ever has the secrets that *that* call passed. One
   deployed engine serves many tenants without any of them sharing credentials.
2. **Scope the credentials** — prefer short-lived, narrowly-scoped tokens (a GitHub **App installation
   token** for one repo, a Bitbucket **repository access token**) over broad personal tokens. If a run is
   coerced, the exposure is limited to that one scoped token for that one run.
3. **Least secrets per run** — pass only what the task needs.

**LLM API key.** By default `gemini` routes the model through **Vertex** (the engine authenticates as its own
GCP identity — **no LLM key is ever in the agent's environment**). This is the recommended, prompt-injection-
safe default. `deploy(..., use_vertex=False)` switches to API-key mode, where you pass `ANTHROPIC_API_KEY` as
a per-invocation secret — but then the agent's process (hence the `Bash` tool) can read it, so use Vertex for
anything exposed to untrusted input. Locally the agent likewise inherits your shell's environment (including
your own `ANTHROPIC_API_KEY`); local is a trusted-dev context.

**In transit & at rest.** Secret values never travel in the invocation payload itself — the platform
*persists* a job's input verbatim (`jobs/<sid>_input.jsonl` in the output bucket), and a Pub/Sub message is
retained until acked, so values in either would linger. Instead the values are staged at a **single-use GCS
object** in the output bucket (readable only by the bucket's principals: your operator identity and the RE
service agent); the invocation carries just that pointer, and the worker **deletes the object the moment it
reads it** (the client also cleans it up on run completion as a backstop). The toolkit **never** writes secret
values to Cloud Logging, to an `AgentEvent`, or to a checkpoint — push tokens are scrubbed from `.git/config`
before a workspace snapshot and re-embedded on resume. Don't log the `secrets` dict yourself.

## Pre-baked engine dependencies

Runtime `uv` (above) is great for experimentation, but for **pinned versions, a private index, or packages
you don't want re-fetched on every run**, declare them on the spec. `gemini.deploy` bakes them into the
engine image (via the platform's uv-via-requirements contract), so a deployed agent starts with them already
installed:

```python
spec = AgentSpec(
    name="data-agent",
    model="claude-sonnet-4-6",
    packages=["pandas==2.2.*", "httpx>=0.27"],   # pip/uv requirement specifiers, baked at deploy time
)
```

**`local` honors `packages` too**: `local.deploy` resolves them into a per-engine venv (via `uv`, with the
engine contract's Python 3.12 — uv provisions the interpreter if your machine lacks it) and activates it in
the agent's environment. Same spec, same starting packages on both backends — and since `uv` hardlinks from
its global cache, a warm local deploy takes seconds, so iteration stays fast. The venv is per-engine, so an
agent installing extras mid-run never leaks into other runs. Note this covers *Python package* parity;
OS-level parity (glibc, system libs) is what the [`dev/` parity image](dev/) is for. Because building the
gemini engine image is slow (~10 min), pin only what genuinely needs to be pre-installed and lean on runtime
`uv` for the rest.

> **Troubleshooting install issues locally.** A dev image under [`dev/`](dev/) mirrors the engine's install
> contract (Debian/glibc, Python 3.12, `uv`, `git`, your `packages`) so dependency failures surface in seconds
> instead of through ~10-min cloud rebuilds: `make parity-build PACKAGES="pandas==2.2.*"`, then `make
> parity-check` (or `parity-shell` to run your agent inside it). It reproduces the *install* environment, not
> the full managed runtime.

## Prod: deploy once, look up and run

Requires GCP setup — see [GCP setup & required permissions](#gcp-setup--required-permissions) below.

```python
# Ops / CI deploys once (rare):
engine = gemini.deploy(spec, project="my-project", location="us-central1", warm_pool=True, pool_size=2)
engine.wait_until_warm()                          # block until a pool worker reports ready (warm only)

# App code looks the engine up by name and runs — it never deploys:
engine = gemini.get_engine("spider-builder", project="my-project", location="us-central1",
                           spec=spec, warm_pool=True)   # pass spec for structured output; warm_pool to dispatch
session = engine.start_session()
result = await session.run("/scrape https://books.toscrape.com title, price")   # default: wait for the result
```

**Managing deployed engines** (control plane):

```python
gemini.deploy(spec, project=..., location=...)   # create; ops/CI only (warm_pool=True, pool_size=N for a pool)
gemini.deploy(spec, ..., resource_limits={"cpu": "4", "memory": "16Gi"})  # container CPU/RAM (default 4 / 4Gi)
gemini.get_engine("spider-builder", project=..., location=...)   # look up by name (app code)
gemini.get_engine("spider-builder", ..., spec=spec)              # pass spec for structured output
gemini.list_engines(project=..., location=...)   # discover what's deployed
engine.name, engine.resource                     # identity / underlying resource name
engine.wait_until_warm(timeout=300)              # warm pools: wait for a ready worker before dispatching
engine.delete(delete_pool_resources=True)        # tear down: cancels pool workers, removes engine (+ topic/sub)
```

Note: `engine.delete()` is the correct way to tear a warm pool down — its idle workers are long-running jobs
that otherwise block deletion until they expire. Version pinning (`get_engine(..., version=N)`) and session
`fork()` are not built yet; `get_engine` resolves the latest engine of that name.

## Past jobs: listing sessions & reading history

Every `gemini` run leaves a durable record, and the library reads it back — from any process, long after
the run:

```python
engine = gemini.get_engine("spider-builder", project=..., location=..., spec=spec)

for info in engine.list_sessions():            # newest first: {"session_id", "sources", "last_file"}
    session = engine.get_session(info["session_id"])
    events = session.history()                 # the persisted AgentEvents, oldest first
    result = session.last_result               # reconstructed from the terminal event (None if unfinished)
    print(info["session_id"], result and result.text)
```

Where the record lives (what `history()` reads, in order of preference):

1. **Mirrored events** — `gs://<output_bucket>/events/<session_id>/<ts>.jsonl`, one file per turn, written
   by the worker at the end of every turn. The canonical history: it covers **warm-pool turns** (whose
   platform job output goes to a throwaway path) and keeps **all turns** of a multi-turn session.
2. **Platform job output** — `gs://<output_bucket>/jobs/<session_id>.jsonl`, the raw ADK event stream the
   platform writes for a cold job (covers engines deployed before mirroring; a resume overwrites it).
3. **Cloud Logging** — the `remote_agent_toolkit_steps` per-step log (bounded by log retention, ~30 days by
   default; also handy interactively: filter by `jsonPayload.session_id`).

`session.last_result` on a re-attached session does one storage/logging round-trip per access until a
result exists — poll `run.done` for in-flight runs, not this. Caveats: `list_sessions`'s GCS layers are
bucket-wide, so engines sharing an output bucket see each other's sessions; the `local` runtime keeps no
durable event log (`list_sessions` shows its workdir's session dirs; `history()` raises).

## Tracing: see what the agent did, span by span

Every `gemini` turn is exported to **Cloud Trace** as a span tree — one root span per turn
(`invoke_agent <name>`) with a child span per tool call (`execute_tool Bash`, …) whose durations are real
(opened at the tool call, closed at its result), plus the assistant's messages/thinking as point-in-time
annotations on the root. Where to look:

- **Console → Agent Platform → your deployment → Traces tab**: pick *session view* to group turns of one
  conversation (spans carry `gen_ai.conversation.id` = the session id), or *span view* for the flat list.
- **Cloud Trace explorer** works too (filter on span name `invoke_agent` or the label
  `gen_ai.conversation.id:<session_id>`).

The root span carries the model, agent name, token usage and cost (`rat.cost_usd`, `rat.num_turns`); a
failed turn or failed tool call marks its span with error status, so a trace of a broken run shows *where*
it broke at a glance. Span values are truncated one-line summaries — the same text that already flows to
Cloud Logging and the event mirror (never secret values), so tracing adds no new exposure surface.

There's nothing to turn on: the runtime's default service-agent role already includes
`telemetry.traces.write`, and the toolkit force-flushes OpenTelemetry at the end of every turn — which is
the load-bearing part: the platform initializes telemetry in job workers but never flushes it on the
async job path the toolkit uses, so without that flush no span would ever leave the worker (verified with
a standalone repro; raised with Google). The only prerequisites are the `telemetry.googleapis.com` +
`cloudtrace.googleapis.com` APIs on the project, and `roles/cloudtrace.user` for whoever wants to *view*
traces.

**Known gaps** (platform-side, as of 2026-07, raised with Google): the console's *session conversation*
panel stays empty ("No chat conversation data") — it is fed by platform instrumentation that doesn't run
for the async job path — and the trace tree shows a cosmetic "(Missing span ID …)" placeholder above the
turn (the platform tears the job worker down before its own wrapper span is exported). Span values here
are one-line summaries; for full prompts/outputs use [`session.history()`](#past-jobs-listing-sessions--reading-history). Traces are diagnostics, not the
record of a run — for programmatic history use [`session.history()`](#past-jobs-listing-sessions--reading-history).
Cloud Trace has a free monthly span quota; a Claude-agent turn produces tens of spans, not thousands.
Tracing is a `gemini`-runtime feature — the `local` runtime emits no spans (its event stream is already
in-process).

## GCP setup & required permissions

Deploying on Gemini Agent Runtime involves **two identities** — granting roles to the wrong one is the
single most common setup mistake, so they're called out explicitly. Everything below can be created by a team
in their own project; the concrete values are the shared `my-project` setup we use for testing.

**1. The operator service account** — you (a human or CI) *impersonate* it to run the control plane:
`gemini.deploy`, `get_engine`, `list_engines`, submitting runs, and tailing Cloud Logging. Roles on the
project (tighten to your policy):

| Role | Why |
|---|---|
| `roles/aiplatform.user` | create/list engines, run query jobs, create sessions |
| `roles/storage.admin` (or objectAdmin on the buckets) | stage the deploy bundle; read job output |
| `roles/logging.viewer` | tail the per-step event stream from the client |
| `roles/cloudbuild.builds.editor` | the deploy builds the engine image |
| `roles/pubsub.editor` _(warm pool only)_ | create the dispatch topic/subscription + publish turns |

The principal that impersonates it needs `roles/iam.serviceAccountTokenCreator` **on this SA**.

**2. The Agent Runtime service agent** — the engine's *runtime* identity, auto-created by Google as
`service-<PROJECT_NUMBER>@gcp-sa-aiplatform-re.iam.gserviceaccount.com`. **All runtime resource access
authorizes against this agent, not the operator SA** — granting the operator SA a runtime role does nothing
for the running job. Grant it:

| Role | Scope | Why |
|---|---|---|
| `roles/storage.objectAdmin` | the output/checkpoint bucket | workspace snapshots, artifacts, the session store |
| `roles/logging.logWriter` | project | the agent emits structured step logs |
| `roles/pubsub.subscriber` | project _(warm pool)_ | warm-pool workers pull turns; project-level since the toolkit auto-creates a per-engine subscription |

No `secretmanager.secretAccessor` is needed for the runtime agent: **secrets are passed per-invocation, not
resolved from Secret Manager by the engine** (see [Secrets & security](#secrets--security)). If a *caller*
keeps secret values in Secret Manager, that caller (the operator identity) reads them before the call.

**Prerequisites** (create with admin creds; `deploy` ensures the buckets it needs):

- A staging bucket `gs://<project>-agent-staging` and an output bucket `gs://<project>-agent-output`.
- **Claude model access** — see the note below; the deployed engine can't run without it.
- _(warm pool)_ a Pub/Sub topic + subscription for turn dispatch — `gemini.deploy(warm_pool=True)` **creates
  these for you** (given the operator SA's `pubsub.editor`); no manual setup needed.

**Claude model access.** By default the toolkit routes Claude through **Vertex AI** (the engine authenticates
as its own GCP identity — no API key to manage). For that to work:

1. **Enable the Claude models you use in Vertex Model Garden** (accept the Anthropic terms once per project).
   Until a model is enabled, a deployed run fails with *"model … may not exist or you may not have access to
   it."*
2. Set `spec.model` to the id shown in Model Garden — for current Claude models that's the family alias (e.g.
   `claude-opus-4-8`), the same form you use locally. The toolkit defaults `CLOUD_ML_REGION` to `global`,
   where these models are served; override `vertex_region` at deploy if you need a specific location.

Prefer an API key (e.g. for models you haven't enabled on Vertex)? Deploy with `use_vertex=False` and pass
`ANTHROPIC_API_KEY` as a per-invocation secret — the toolkit then uses the key (no Vertex routing), so any
model alias the key supports works. Note the security trade-off: in API-key mode the agent's process can read
the key, so prefer Vertex for anything exposed to untrusted input (see
[Secrets & security](#secrets--security)).

**Concrete shared setup** (`my-project`): location `us-central1` (Claude: `us-central1` + `global`);
operator SA `agent-runtime@my-project.iam.gserviceaccount.com`; runtime agent
`service-123456789012@gcp-sa-aiplatform-re.iam.gserviceaccount.com`. Sketch for a fresh project:

```bash
PROJECT=your-project; NUM=$(gcloud projects describe $PROJECT --format='value(projectNumber)')
RE="service-$NUM@gcp-sa-aiplatform-re.iam.gserviceaccount.com"     # runtime identity
OP="agent-runtime@$PROJECT.iam.gserviceaccount.com"                # operator SA you create

gcloud iam service-accounts create agent-runtime --project $PROJECT
for R in roles/aiplatform.user roles/storage.admin roles/logging.viewer roles/cloudbuild.builds.editor \
         roles/pubsub.editor; do  # pubsub.editor only needed for warm pools
  gcloud projects add-iam-policy-binding $PROJECT --member "serviceAccount:$OP" --role $R; done
gcloud projects add-iam-policy-binding $PROJECT --member "serviceAccount:$RE" --role roles/logging.logWriter
gcloud services enable telemetry.googleapis.com cloudtrace.googleapis.com --project $PROJECT  # tracing
# grant $RE objectAdmin on the output bucket; let yourself impersonate $OP (no secretAccessor needed —
# secrets are passed per-invocation, not read from Secret Manager by the engine):
gcloud iam service-accounts add-iam-policy-binding $OP --member "user:you@org.com" \
  --role roles/iam.serviceAccountTokenCreator
```

Then authenticate impersonating the operator SA (`gcloud auth application-default login
--impersonate-service-account=$OP`) before running `gemini.deploy`.

## Latency & cost (the `gemini` path)

Platform realities the toolkit encodes (measured on Agent Runtime; the `local` path has none of them):

**Deploy** is a rare ops/CI action and takes **minutes** (image build + engine provisioning). App code never
deploys — it looks an engine up by name and runs.

**Running a turn** has these latency profiles:

| Path | Start latency | Ceiling | Use for |
|---|---|---|---|
| Async (default) | **~2.5 min** per-job worker provisioning | long-running | one-shot / long autonomous jobs |
| **Warm pool** (`warm_pool=True`) | **~10–20 s** to first *observed* event | long-running | interactive *and* long — best of both |
| Sync | ~3–9 s | ~600 s (10 min) hard | _not supported yet (see below)_ |

The ~2.5 min async start is **per job, not a one-time cold start** — it's Vertex provisioning a dedicated
worker, and it is *not* reducible via concurrency / min-instances knobs. So for one long task, do it in a
**single** job (pay the start once) rather than many short jobs.

The platform also offers a **sync** path (fast start, but a hard ~10 min ceiling). The toolkit **does not
expose it yet** — the run plane is built around the async + warm-pool paths, which cover long *and*
interactive work. We can add sync later if a genuinely short-turn use case needs the lower start latency.

**How `warm_pool=True` works.** `gemini.deploy(spec, warm_pool=True)` keeps a pool of pre-provisioned workers,
each blocked on a Pub/Sub subscription (a competing-consumers *atomic claim*). A worker warms its Cloud
Logging + storage channels during its idle wait and reports ready — `engine.wait_until_warm()` blocks on that
signal. A run is then dispatched to a free worker, so the turn goes nearly straight to the model. The first
**observed** event lands **~10–20 s** after dispatch (vs ~2.5 min cold) — that window is dominated by **Cloud
Logging's write→queryable ingestion lag**, which varies run to run and is inherent to a log-tail channel; the
worker's *actual* pickup is ~5 s. On claim the pool refills, so the next turn is warm too.

> _Keeping the pool full:_ warm workers are themselves long-running jobs and will eventually exit at the
> platform's max-job-duration limit (whose exact value we haven't pinned down). Topping the pool back up
> across that boundary is a future refinement; in practice the refill-on-claim cadence likely keeps enough
> workers warm, so it shouldn't bite early on.

**Cost.** A deployed engine itself is (almost) free while idle: the toolkit deploys with `min_instances=0`
(no standing container — the async path provisions a worker per job, so a min-instances container would serve
only the unused sync path while billing continuously). Warm workers are the exception: they are long-running
jobs sitting idle waiting for work, so you **pay for that idle compute** continuously — `warm_pool=True`
trades money for latency. Size the pool to your concurrency, and leave it off for batch / non-interactive
agents where a ~2.5 min start is fine. Tear a pool down with `engine.delete(delete_pool_resources=True)` (it
cancels the idle workers, which otherwise keep billing until they expire). Model token cost is the same either
way and is reported per run as `result.cost_usd`.

## Learn more

See [`DESIGN.md`](DESIGN.md) — the agreed architecture, the hard-won platform contracts (§6), the ports &
adapters seam (§7), and the phased roadmap (§11). The README grows with the code, phase by phase.

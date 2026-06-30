# remote-agent-toolkit

A Python library for **defining and running remote/background AI agents** at Zyte. Define an agent
declaratively as an `AgentSpec`, then run it both **locally** (in-process, for dev) and **remotely** on
**Gemini Agent Runtime** (prod) through *one* API — with custom skills, GitHub access, structured outputs,
checkpoint/resume and warm starts, without re-learning the platform's sharp edges.

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
    mcp_servers=[McpServer.github()],
    secrets=["ZYTE_API_KEY", "GH_PAT"],      # names → resolved from Secret Manager at runtime, never pickled
    permission_mode="bypassPermissions",     # the default — safe because each run gets an isolated cwd
    max_turns=120,
    max_budget_usd=10.0,
    checkpoint=True,                         # enable resume / interactive pauses
)
```

Every field except `name` and `model` has a sensible default (see [`spec.py`](remote_agent_toolkit/spec.py));
a two-line spec (`AgentSpec(name=..., model=...)`) is a valid agent. For **structured output**, see the
section below.

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

## Consuming a run: wait, stream, or poll

`session.run(msg)` (and `session.send(msg)` to resume) returns a `Run` handle, consumable three ways — the
same on `local` and `gemini`:

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
    print(run.status)                        # pending | running | idle | terminated
result = run.result

# re-attach later / elsewhere by session id, then poll or continue
session = engine.get_session(session_id)
if session.status == "idle" and session.stop_reason == "needs_input":
    await session.send("yes, that schema looks right")     # resumes the conversation (a fresh turn)
```

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
- **`env`** — extra environment variables exported to the agent's tool subprocess (config flags, etc.).
- **`secrets`** — secret *names*, resolved at runtime (env locally, Secret Manager on `gemini`) and exported
  by name. Conventional GitHub token names (`GH_TOKEN` / `GITHUB_TOKEN` / `GH_PAT`) also wire up `McpServer.github()`.
- **`mcp_servers`** — `McpServer.github()`, `.remote(name, url, headers=...)`, `.stdio(name, command, args=...)`.

**Python libraries & CLI tools.** The agent's `Bash` tool runs in a real shell with **`uv`** and **`git`** on
`PATH`, so it can fetch and run dependencies on the fly — e.g. `uv run --with httpx script.py`, or
`uv pip install …` inside its working directory. That covers most "use library X" needs with no change to
the deployment. To pin libraries into the deployed engine instead, see
[Pre-baked engine dependencies](#pre-baked-engine-dependencies) below.

## Cloning a git repo

A common setup is to clone a repo into the agent's working directory **before it runs** — so it can read and
modify the code, then commit and push. Declare repos on the spec:

```python
from remote_agent_toolkit import AgentSpec, RepoSource

spec = AgentSpec(
    name="repo-fixer",
    model="claude-sonnet-4-6",
    repos=[RepoSource.git("https://github.com/zytedata/some-repo", ref="main")],
    secrets=["GH_TOKEN"],   # a GitHub token makes the clone push-ready (and clones private repos)
)
```

Each repo is cloned into the agent's cwd before the loop starts. If `secrets` carries a GitHub token
(`GH_TOKEN` / `GITHUB_TOKEN` / `GH_PAT`), the clone is **authenticated and push-ready** — the token is injected
into `origin` and a commit identity is configured, so the agent can `git push` without handling credentials;
otherwise it's a read-only clone (fine for public repos). On `gemini` the token comes from Secret Manager, on
`local` from your environment. (To make the agent open PRs, also add `McpServer.github()`.)

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

The **`local` path ignores `packages`** — locally the agent uses your machine's environment plus whatever it
installs at runtime, so you iterate without a rebuild. Because building the engine image is slow (~10 min),
pin only what genuinely needs to be baked in and lean on runtime `uv` for the rest.

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
| `roles/secretmanager.secretAccessor` | each secret in `spec.secrets` | the platform injects secret_refs, read as this agent |
| `roles/storage.objectAdmin` | the output/checkpoint bucket | workspace snapshots, artifacts, the session store |
| `roles/logging.logWriter` | project | the agent emits structured step logs |
| `roles/pubsub.subscriber` | project _(warm pool)_ | warm-pool workers pull turns; project-level since the toolkit auto-creates a per-engine subscription |

**Prerequisites** (create with admin creds; `deploy` ensures the buckets it needs):

- A staging bucket `gs://<project>-agent-staging` and an output bucket `gs://<project>-agent-output`.
- One Secret Manager secret per `spec.secrets` entry, **named exactly the same as the env var** (1:1 — e.g. a
  secret literally named `ZYTE_API_KEY`).
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

Prefer an API key (e.g. for models you haven't enabled on Vertex)? List `"ANTHROPIC_API_KEY"` in
`spec.secrets` (with a matching Secret Manager secret) — the toolkit then uses the key and skips Vertex
routing, so any model alias the key supports works.

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
# grant $RE secretAccessor per-secret and objectAdmin on the output bucket; let yourself impersonate $OP:
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

**Cost.** Warm workers are long-running jobs sitting idle waiting for work, so you **pay for that idle compute**
continuously — `warm_pool=True` trades money for latency. Size the pool to your concurrency, and leave it off
for batch / non-interactive agents where a ~2.5 min start is fine. Tear a pool down with
`engine.delete(delete_pool_resources=True)` (it cancels the idle workers, which otherwise keep billing until
they expire). Model token cost is the same either way and is reported per run as `result.cost_usd`.

## Learn more

See [`DESIGN.md`](DESIGN.md) — the agreed architecture, the hard-won platform contracts (§6), the ports &
adapters seam (§7), and the phased roadmap (§11). The README grows with the code, phase by phase.

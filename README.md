# remote-agent-toolkit

A Python library for **defining and running remote/background AI agents** at Zyte. Define an agent
declaratively as an `AgentSpec`, then run it both **locally** (in-process, for dev) and **remotely** on
**Gemini Agent Runtime** (prod) through *one* API — with custom skills, GitHub access, structured outputs,
checkpoint/resume and warm starts, without re-learning the platform's sharp edges.

> **Status: P1 — the local run plane works.** You can define an `AgentSpec` and run it in-process today
> (`local.deploy` → `Session`/`Run`, with skills, structured output, and checkpoint/resume). The **`gemini`
> backend is P2** (deploy / warm pool / lookup are still stubs). See [`examples/minimal`](examples/minimal)
> for a runnable agent, and [`DESIGN.md`](DESIGN.md) for the architecture, platform contracts (§6), and
> roadmap (§11). Sections below marked _(P2)_ are the north-star target, not yet runnable.

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
    checkpoint=True,                          # enable resume / interactive pauses
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
    print(run.status)                        # running | idle | terminated
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
[Pre-baked engine dependencies](#pre-baked-engine-dependencies-p2) below.

## Pre-baked engine dependencies _(P2)_

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

## Prod: deploy once, look up and run _(P2)_

```python
# Ops / CI deploys once (rare):
gemini.deploy(spec, project="my-project", location="us-central1", warm_pool=True)

# App code looks the engine up by name (optionally a version) and runs — it never deploys:
engine = gemini.get_engine("spider-builder")               # latest deployed version
session = engine.start_session()
result = await session.run("/scrape https://books.toscrape.com title, price")   # default: wait for the result
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

## Latency & cost _(P2 — the `gemini` path)_

Platform realities the toolkit encodes (measured on Agent Runtime; the `local` path has none of them):

**Deploy** is a rare ops/CI action and takes **minutes** (image build + engine provisioning). App code never
deploys — it looks an engine up by name and runs.

**Running a turn** has these latency profiles:

| Path | Start latency | Ceiling | Use for |
|---|---|---|---|
| Async (default) | **~2.5 min** per-job worker provisioning | long-running | one-shot / long autonomous jobs |
| **Warm pool** (`warm_pool=True`) | **~5 s** | long-running | interactive *and* long — best of both |
| Sync | ~3–9 s | ~600 s (10 min) hard | _not supported yet (see below)_ |

The ~2.5 min async start is **per job, not a one-time cold start** — it's Vertex provisioning a dedicated
worker, and it is *not* reducible via concurrency / min-instances knobs. So for one long task, do it in a
**single** job (pay the start once) rather than many short jobs.

The platform also offers a **sync** path (fast start, but a hard ~10 min ceiling). The toolkit **does not
expose it yet** — the run plane is built around the async + warm-pool paths, which cover long *and*
interactive work. We can add sync later if a genuinely short-turn use case needs the lower start latency.

**How `warm_pool=True` works.** `gemini.deploy(spec, warm_pool=True)` keeps a pool of pre-provisioned workers,
each blocked on a Pub/Sub subscription (a competing-consumers *atomic claim*). A run is dispatched to a free
worker that has already provisioned skills and warmed its storage channel during its idle wait, so the turn
goes nearly straight to the model (**~5 s** dispatch→first step vs ~2.5 min cold). On claim the pool refills,
so the next turn is warm too.

> _Keeping the pool full:_ warm workers are themselves long-running jobs and will eventually exit at the
> platform's max-job-duration limit (whose exact value we haven't pinned down). Topping the pool back up
> across that boundary is a future refinement; in practice the refill-on-claim cadence likely keeps enough
> workers warm, so it shouldn't bite early on.

**Cost.** Warm workers are long-running jobs sitting idle waiting for work, so you **pay for that idle compute**
continuously — `warm_pool=True` trades money for latency. Size the pool to your concurrency, and leave it off
for batch / non-interactive agents where a ~2.5 min start is fine. Model token cost is the same either way and
is reported per run as `result.cost_usd`.

## Learn more

See [`DESIGN.md`](DESIGN.md) — the agreed architecture, the hard-won platform contracts (§6), the ports &
adapters seam (§7), and the phased roadmap (§11). The README grows with the code, phase by phase.

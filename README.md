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
pip install "git+https://github.com/zytedata/remote-agent-toolkit.git"
# or:  uv pip install "git+https://github.com/zytedata/remote-agent-toolkit.git"
```

One install pulls **everything** — local *and* Gemini Agent Runtime — with no extras to choose between.
`import remote_agent_toolkit` itself needs no third-party deps (backend imports are lazy), so import stays
fast. Developing on the toolkit itself? Clone it and `uv sync` (installs the runtime deps plus the `dev`
group: pytest, ruff, mypy).

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

## Structured output

Set `output_schema` to a **pydantic model** (or a JSON-schema `dict`) and `result.structured_output` holds
the parsed, validated object. The schema only describes the *shape* — you must **also tell the agent, in the
prompt, to emit that shape as its final message**. The toolkit parses JSON out of the agent's final text
(from a ```` ```json ```` block, a bare object, or the whole message); it does not constrain decoding.

```python
import pydantic
from remote_agent_toolkit import AgentSpec, local

class PriceCheck(pydantic.BaseModel):
    in_stock: bool
    price_gbp: float

spec = AgentSpec(name="price-check", model="claude-sonnet-4-6", output_schema=PriceCheck)
engine = local.deploy(spec)
result = await engine.start_session().run(
    "Check the product at <url> and report the result as a JSON object with keys "
    "in_stock (boolean) and price_gbp (number)."          # ← the schema must be spelled out here
)
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
the deployment.

_(P2)_ For libraries that should be **pre-baked into the deployed `gemini` engine** (pinned versions, a private
index, or system packages), `gemini.deploy` will accept a declared dependency set and encode it into the engine
image, so a deployed agent starts with them already installed instead of fetching per run. Locally the agent
just uses your machine's environment plus whatever it installs at runtime.

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

**Running a turn** has three latency profiles:

| Path | Start latency | Ceiling | Use for |
|---|---|---|---|
| Async (default) | **~2.5 min** per-job worker provisioning | long-running | one-shot / long autonomous jobs |
| Sync | **~3–9 s** | ~600 s (10 min) hard | short, interactive turns |
| **Warm pool** (`warm=True`) | **~5 s** | long-running | interactive *and* long — best of both |

The ~2.5 min async start is **per job, not a one-time cold start** — it's Vertex provisioning a dedicated
worker, and it is *not* reducible via concurrency / min-instances knobs. So for one long task, do it in a
**single** job (pay the start once) rather than many short jobs.

**How `warm=True` works.** `gemini.deploy(spec, warm_pool=True)` keeps a pool of pre-provisioned workers, each
blocked on a Pub/Sub subscription (a competing-consumers *atomic claim*). A run is dispatched to a free worker
that has already provisioned skills and warmed its storage channel during its idle wait, so the turn goes
nearly straight to the model (**~5 s** dispatch→first step vs ~2.5 min cold). On claim the pool refills, so the
next turn is warm too.

**Cost.** Warm workers are long-running jobs sitting idle waiting for work, so you **pay for that idle compute**
continuously — `warm_pool=True` trades money for latency. Size the pool to your concurrency, and leave it off
for batch / non-interactive agents where a ~2.5 min start is fine. Model token cost is the same either way and
is reported per run as `result.cost_usd`.

## Learn more

See [`DESIGN.md`](DESIGN.md) — the agreed architecture, the hard-won platform contracts (§6), the ports &
adapters seam (§7), and the phased roadmap (§11). The README grows with the code, phase by phase.

# remote-agent-toolkit

A Python library for **defining and running remote/background AI agents** at Zyte. Define an agent
declaratively as an `AgentSpec`, then run it both **locally** (in-process, for dev) and **remotely** on
**Gemini Agent Runtime** (prod) through *one* API — with custom skills, GitHub access, structured outputs,
checkpoint/resume and warm starts, without re-learning the platform's sharp edges.

> **Status: early scaffold (P0).** This is a typed skeleton — real signatures (protocols + dataclasses) with
> `NotImplementedError` stub bodies. See [`DESIGN.md`](DESIGN.md) for the full architecture, the platform
> contracts (§6), and the phased roadmap (§11). The example below is the **north-star target**, not yet runnable.

## Install

```bash
pip install remote-agent-toolkit[gemini]
```

Core install pulls only `claude-agent-sdk` + `google-adk`. Extras: `[gemini]` (Agent Runtime + warm pool),
`[pubsub]` (Pub/Sub dispatch), `[dev]` (test/lint tooling). `import remote_agent_toolkit` itself needs no
third-party deps — backend imports are lazy.

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
    permission_mode="bypassPermissions",
    max_turns=120,
    max_budget_usd=10.0,
    checkpoint=True,                          # enable resume / interactive pauses
    output_schema=BookSchema,                 # optional: structured output (pydantic / JSON schema)
)
```

## Dev: run locally, in-process

Same `Engine` + `Session` API as prod, no GCP beyond model access.

```python
engine = local.deploy(spec)
session = engine.start_session()
result = await session.run("scrape https://books.toscrape.com for title, price")   # wait for completion
print(result.text, result.structured_output, result.cost_usd)
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

## Prod: deploy once, look up and run

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

## Learn more

See [`DESIGN.md`](DESIGN.md) — the agreed architecture, the hard-won platform contracts (§6), the ports &
adapters seam (§7), and the phased roadmap (§11). The README grows with the code, phase by phase.

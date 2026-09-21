# Getting started

`agent-run` defines a coding agent once, as an `AgentSpec`, and runs it through one API on two runtimes:
`local` (in-process, for development) and `sandbox` (Google Cloud's Agent Sandbox, for production). This
page gets a first agent running locally and points at what to read next.

## Install

Requires Python 3.12 or newer.

```bash
pip install agent-run
```

The base install covers the **remote path** (deploying and driving sandbox engines — the sandbox image
installs the harness SDKs from its own baked requirements). Running agents **locally** (`local.deploy`)
also runs the harness SDKs on your machine, so add the `local` extra:

```bash
pip install "agent-run[local]"
```

The SDKs are kept out of the base install because they are heavy and pin aggressively — `claude-agent-sdk`
requires `mcp<2`, which conflicts with applications on the mcp 2.x SDK, and `openai-codex` bundles the
Codex CLI binary.

Releases are git tags; pin one to shield yourself from in-development changes on `main`. `CHANGELOG.md`
records what is in each release and how to upgrade across breaking changes.

## Credentials

The agent needs credentials for the model it calls. Which ones depends on the harness and the runtime:

| Harness | Locally | On `sandbox` |
| --- | --- | --- |
| `claude-code` | `ANTHROPIC_API_KEY` in your environment, or a logged-in `claude` CLI | none — Claude is routed through Vertex AI by default |
| `codex` | `OPENAI_API_KEY`, passed as a per-run secret (falls back to the environment) | `OPENAI_API_KEY` as a per-run secret |
| either, with an `openrouter/` model | `OPENROUTER_API_KEY` as a per-run secret | the same |

Per-run secrets are a `name → value` map handed to `run()`/`send()`; they are never baked into a spec or a
deployment. [Secrets and security](secrets-and-security.md) has the full routing table and the reasoning.

## Define an agent

```python
from agent_run import AgentSpec, SystemPrompt, SkillSource, McpServer

spec = AgentSpec(
    name="code-agent",
    harness="codex",
    model="gpt-5.6-luna",
    system_prompt=SystemPrompt.inherit(append="Prefer the skills below when they apply."),
    skills=[SkillSource.git("https://github.com/zytedata/codex-skills")],
    mcp_servers=[McpServer.github()],        # token supplied per-invocation, not here
    permission_mode="bypassPermissions",     # the default — safe because each run gets an isolated cwd
    max_turns=120,
    max_budget_usd=10.0,
    checkpoint=True,                         # enable resume / interactive pauses
)
```

The spec carries **no secrets** — credentials are passed per-invocation to `run`/`send` so nothing sensitive
is ever baked into the deployment or shared across runs (see [Secrets & security](secrets-and-security.md)).

For authenticated remote MCP servers and migration of credential-bearing config, see
[runtime secret references and validation boundaries](secrets-and-security.md#runtime-secret-references-in-persistent-configuration).
Every field except `name`, `model` and `harness` has a sensible default (see
[`spec.py`](../agent_run/spec.py)); a three-line spec (`AgentSpec(name=..., model=..., harness=...)`) is
a valid agent. [Configuration](configuration.md) walks through the rest of the fields, and
[Structured output](structured-output.md) covers `output_schema`.


## Run it locally

Same `Engine` → `Session` → `Run` API as production, no GCP beyond model access. Codex reads
`OPENAI_API_KEY` from your environment here (remotely it is passed as a per-run secret):

```python
import asyncio
from agent_run import local

async def main():
    engine = local.deploy(spec)
    session = engine.start_session()
    result = await session.run("Add type hints to utils.py and run the tests")   # wait for completion
    print(result.text, result.structured_output, result.cost_usd)

asyncio.run(main())
```

`await session.run(...)` waits for the turn to finish; the same `Run` handle can instead be streamed
(`async for ev in run`) or polled, and `session.send(...)` continues the conversation on a later turn.
[Sessions and runs](runs-and-sessions.md) covers all of that, including steering a turn while it runs.

The deployed spec is the *default*; a session can override parts of it without redeploying, locally and on
`sandbox` alike:

```python
from agent_run import SessionConfig, TurnConfig

session = engine.start_session(config=SessionConfig(model="gpt-5.6-sol"))      # this conversation only
result = await session.run("…", config=TurnConfig(max_budget_usd=0.5))          # this turn only
```

[Configuration](configuration.md) explains the three scopes — deploy, session and turn — and which
field belongs to which.

## Run it remotely

The same spec deploys to Agent Sandbox with `sandbox.deploy(spec, project=..., location=...)`, after which
application code looks the engine up by name with `sandbox.get_engine(...)` and runs turns exactly as
above. Deployment needs a prepared GCP project — one command, `agent-run-gcp-setup --project <your-project>`,
audits and applies everything — and the Docker CLI logged into the project's image registry.
[GCP setup](gcp-setup.md) is the prerequisite; [Sandbox runtime](sandbox-runtime.md) covers deploying,
warm pools, engine versions, and what a turn costs.

## Where next

- [Harnesses and models](harnesses-and-models.md) — Claude Code, Codex, model ids, reasoning effort
- [Configuration](configuration.md) — every `AgentSpec` field; session and turn overlays; repos, MCP servers, packages
- [Sessions and runs](runs-and-sessions.md) — wait, stream, poll, resume, steer, interrupt, `exec()`
- [Local runtime](local-runtime.md) — which credential pays, workspaces, tool-call hooks
- [Sandbox runtime](sandbox-runtime.md) — deploy, versions, latency and cost
- [Structured output](structured-output.md), [OpenRouter](openrouter.md),
  [Secrets and security](secrets-and-security.md), [Observability](observability.md), [GCP setup](gcp-setup.md)

A complete runnable agent is in [`examples/minimal`](../examples/minimal); [`examples/sandbox`](../examples/sandbox)
deploys and runs one remotely.

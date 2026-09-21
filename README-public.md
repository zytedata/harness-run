# agent-run

`agent-run` is a Python library for running **coding agents** locally and in remote sandbox environments through a common API.

It provides an abstraction over agent harnesses such as **Claude Code** and **Codex**, with support for multiple model providers. Your application can switch harnesses or models without being tightly coupled to one vendor's agent runtime.

The same `Engine` → `Session` → `Run` API works locally for development and remotely on **GCP Agent Sandbox** for production. With a warm sandbox pool, remote agents can start and complete small tasks in just a few seconds.

## Install

```bash id="9t2v4e"
pip install agent-run
```

To also run agents locally:

```bash id="r1h7mf"
pip install "agent-run[local]"
```

## Example

Define a coding agent explicitly using the Claude Code harness:

```python id="vctval"
from agent_run import AgentSpec

spec = AgentSpec(
    name="code-agent",
    harness="claude-code",
    model="claude-sonnet-4-6",
    checkpoint=True,
)
```

Run it locally:

```python id="s5jgj1"
from agent_run import local

engine = local.deploy(spec)
session = engine.start_session()

result = await session.run(
    "Create a Python function fibonacci(n), add a few tests, and run them."
)

print(result.text)
print(result.cost_usd)
```

A session can continue across multiple turns:

```python id="fthbc2"
result = await session.send(
    "Now make it iterative instead of recursive and run the tests again."
)
```

### Run remotely

The same agent can run remotely on **GCP Agent Sandbox**. `agent-run` handles building and deploying the agent environment and running each turn inside an isolated sandbox.

```python id="gn3vuz"
from agent_run import sandbox

# Deployment is normally done by CI / ops.
sandbox.deploy(
    spec,
    project="my-project",
    location="us-central1",
    warm_pool=True,
)

# Application code looks up the deployed engine by name.
engine = sandbox.get_engine(
    "code-agent",
    project="my-project",
    location="us-central1",
)

session = engine.start_session()

result = await session.run(
    "Create a Python function fibonacci(n), add a few tests, and run them."
)

session_id = session.id
```

With `warm_pool=True`, a ready sandbox is kept available for low-latency execution. Small turns typically start in about a second and can complete in a few seconds.

Another process can later re-attach to the session and continue the conversation:

```python id="vwdsrn"
engine = sandbox.get_engine(
    "code-agent",
    project="my-project",
    location="us-central1",
)

session = engine.get_session(session_id)

result = await session.send(
    "Add type hints and explain the time complexity."
)
```

Want Codex or another model instead? The application-level API stays the same:

```python id="llaos4"
spec = AgentSpec(
    name="code-agent",
    harness="codex",
    model="gpt-5.6-sol",
    checkpoint=True,
)
```

Harness and model can also be configured at session or turn level, so a deployed engine does not need to represent a single fixed model configuration.

## Features

Beyond the basic example, `agent-run` supports:

- **Multiple agent harnesses and models** — use Claude Code or Codex with Claude, OpenAI, and OpenRouter models.
- **Local and remote execution** through the same API, with remote runs on GCP Agent Sandbox.
- **Low-latency remote runs** using warm sandbox pools.
- **Multi-turn sessions** with checkpoint/resume and durable remote history.
- **Per-session and per-turn configuration** for models, prompts, budgets, tools, output schemas, and more.
- **Streaming and polling** as alternatives to awaiting a completed run.
- **Structured output** with Pydantic models or JSON Schema.
- **Repository and workspace setup**, MCP servers, and agent skills.
- **Per-run secrets** without baking credentials into deployed agents.
- **Steering and interruption** while an agent is running.
- **Usage, cost, and resource reporting** for production workloads.

## Documentation

Start with [Getting started](docs/getting-started.md).

- [Harnesses and models](docs/harnesses-and-models.md) — Claude Code, Codex, model selection, and reasoning settings
- [Configuration](docs/configuration.md) — `AgentSpec`, `SessionConfig`, and `TurnConfig`
- [Sessions and runs](docs/runs-and-sessions.md) — multi-turn conversations, streaming, polling, resume, steering, and interruption
- [Local runtime](docs/local-runtime.md) — local development and workspaces
- [Sandbox runtime](docs/sandbox-runtime.md) — GCP Agent Sandbox deployment, remote execution, warm pools, and engine versions
- [Structured output](docs/structured-output.md)
- [OpenRouter](docs/openrouter.md) — models and provider routing
- [Secrets and security](docs/secrets-and-security.md)
- [Observability](docs/observability.md) — events, history, transcripts, cost, usage, and resources
- [GCP setup](docs/gcp-setup.md)

For contributors, see [TESTING.md](TESTING.md). For implementation details and platform architecture, see [DESIGN.md](DESIGN.md). Release notes and upgrade instructions are in [CHANGELOG.md](CHANGELOG.md).

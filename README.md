# agent-run

`agent-run` is a Python library for running **coding agents** locally and in remote sandbox environments through a common API.

It provides an abstraction over agent harnesses such as **Claude Code** and **Codex**, with support for multiple model providers. Your application can switch harnesses or models without being tightly coupled to one vendor's agent runtime.

The same `Engine` → `Session` → `Run` API works locally for development and remotely on **GCP Agent Sandbox** for production. With a warm sandbox pool, remote agents can start and complete small tasks in just a few seconds.

## Install

```bash
pip install agent-run
```

To also run agents locally:

```bash
pip install "agent-run[local]"
```

## Requirements

- Python 3.12 or newer.
- Credentials for the model the agent uses. Codex needs `OPENAI_API_KEY`, passed to each run as a secret (locally it also picks up the environment variable). Claude Code needs `ANTHROPIC_API_KEY` in the environment or a logged-in `claude` CLI locally; on GCP Agent Sandbox, Claude models are routed through Vertex AI by default, so no Anthropic key is needed there.
- For remote runs, a GCP project prepared with `agent-run-gcp-setup --project <your-project>` (installed with the library). See [GCP setup](docs/gcp-setup.md).

## Example

Define a coding agent explicitly using the Codex harness:

```python
from agent_run import AgentSpec

spec = AgentSpec(
    name="code-agent",
    harness="codex",
    model="gpt-5.6-luna",
    checkpoint=True,
)
```

Run it locally:

```python
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

```python
result = await session.send(
    "Now make it iterative instead of recursive and run the tests again."
)
```

### Run remotely

The same agent can run remotely on **GCP Agent Sandbox**. `agent-run` handles building and deploying the agent environment and running each turn inside an isolated sandbox. Credentials are passed to each run rather than baked into the deployed agent:

```python
import os
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
    "Create a Python function fibonacci(n), add a few tests, and run them.",
    secrets={"OPENAI_API_KEY": os.environ["OPENAI_API_KEY"]},
)

session_id = session.session_id
```

With `warm_pool=True`, a ready sandbox is kept available for low-latency execution. Small turns typically start in about a second and can complete in a few seconds.

Another process can later re-attach to the session and continue the conversation:

```python
engine = sandbox.get_engine(
    "code-agent",
    project="my-project",
    location="us-central1",
)

session = engine.get_session(session_id)

result = await session.send(
    "Add type hints and explain the time complexity.",
    secrets={"OPENAI_API_KEY": os.environ["OPENAI_API_KEY"]},
)
```

Want Claude Code instead? The application-level API stays the same. On GCP Agent Sandbox, Claude models are called through Vertex AI by default, so these runs need no API key at all:

```python
spec = AgentSpec(
    name="code-agent",
    harness="claude-code",
    model="claude-sonnet-4-6",
    checkpoint=True,
)
```

The model can also be overridden per session or per turn, and an engine deployed with both harnesses baked in can pick either one per session, so a deployed engine does not need to represent a single fixed model configuration.

## Features

Beyond the basic example, `agent-run` supports:

- **Multiple agent harnesses and models** — Claude Code with Claude models, Codex with OpenAI models, and either harness with models served through OpenRouter.
- **Local and remote execution** through the same API, with remote runs on GCP Agent Sandbox.
- **Low-latency remote runs** using warm sandbox pools.
- **Multi-turn sessions** with checkpoint/resume and durable remote history.
- **Per-session and per-turn configuration** for models, prompts, budgets, tools, output schemas, and more.
- **Streaming and polling** as alternatives to awaiting a completed run.
- **Structured output** with Pydantic models or JSON Schema.
- **Repository and workspace setup**, MCP servers, and agent skills.
- **Per-run secrets** without baking credentials into deployed agents.
- **Steering, interruption, and shell access** (`exec()`) while an agent is running.
- **Usage, cost, and resource reporting** for production workloads.

## Documentation

Start with [Getting started](docs/getting-started.md).

- [Harnesses and models](docs/harnesses-and-models.md) — Claude Code, Codex, model selection, and reasoning settings
- [Configuration](docs/configuration.md) — `AgentSpec`, `SessionConfig`, and `TurnConfig`
- [Sessions and runs](docs/runs-and-sessions.md) — multi-turn conversations, streaming, polling, resume, steering, interruption, and `exec()`
- [Local runtime](docs/local-runtime.md) — local development and workspaces
- [Sandbox runtime](docs/sandbox-runtime.md) — GCP Agent Sandbox deployment, remote execution, warm pools, and engine versions
- [Structured output](docs/structured-output.md)
- [OpenRouter](docs/openrouter.md) — models and provider routing
- [Secrets and security](docs/secrets-and-security.md)
- [Observability](docs/observability.md) — events, history, transcripts, cost, usage, and resources
- [GCP setup](docs/gcp-setup.md)

For contributors, see [TESTING.md](TESTING.md). For implementation details and platform architecture, see [DESIGN.md](DESIGN.md). Release notes and upgrade instructions are in [CHANGELOG.md](CHANGELOG.md).

## License and provenance

`agent-run` is developed by [Zyte](https://www.zyte.com) and released under the [Apache-2.0](LICENSE) license.

Much of the code was written with AI coding agents, chiefly Claude Code, with the maintainers directing the work and reviewing the result.

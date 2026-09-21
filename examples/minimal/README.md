# minimal example agent

A tiny, generic (non-Zyte) agent that proves the core API end-to-end: define an
`AgentSpec`, `local.deploy` it, run one task, stream events, and read a structured result.
The same spec runs remotely, in a Google Agent Sandbox, by swapping `local` → `sandbox.deploy` /
`sandbox.get_engine` (see the top-level README and [`../sandbox`](../sandbox)).

[`agent.py`](agent.py) asks the agent to write `is_prime(n)`, run it on 2–10, and report a
`{checked, primes}` JSON object — exercising the `Write` + `Bash` tools and structured
output, with a verifiable answer.

## Prerequisites

- **Python 3.12** (the `claude-agent-sdk` supports ≤3.13; the repo pins `>=3.12,<3.14`).
- The toolkit installed into that environment, e.g. with [uv](https://docs.astral.sh/uv/):

  ```bash
  uv venv --python 3.12 .venv
  uv pip install --python .venv/bin/python -e .
  ```

- **Claude Code auth available in your environment.** The Agent SDK drives the `claude`
  CLI, so it uses whatever auth that CLI finds (an `ANTHROPIC_API_KEY`, or a logged-in
  Claude Code session). This example makes a **real model call** (a few cents on Haiku).

## Run

```bash
.venv/bin/python examples/minimal/agent.py
```

You'll see streamed progress lines (`tool_use`, `tool_result`, `message`, …) followed by
the final text, the parsed `PrimeReport`, and the turn/cost summary.

## What it shows

- **Declarative spec** → `local.deploy` → `start_session` → `run` (same API as `sandbox`).
- **Streaming** a `Run` with `async for` (also awaitable, and pollable via `run.done`).
- **Structured output** — `output_schema=PrimeReport` parses the final message into a
  validated pydantic model on `result.structured_output`.

Swap `model="claude-haiku-4-5"` for `"claude-sonnet-4-6"` for tougher tasks. To add custom
skills, pass `skills=[SkillSource.local("path/to/skills")]` (or `SkillSource.git(...)`).

# Local runtime

Same `Engine` + `Session` API as prod, no GCP beyond model access. Needs the `local` extra installed and
credentials for the harness in your environment (Claude Code: the Agent SDK drives the `claude` CLI). See a complete runnable agent in
[`examples/minimal`](../examples/minimal).

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

For a quick sync script, `local.run(spec, "…")` does `deploy → start_session → await run` and returns the
`RunResult`.

The deployed spec is the *default*; a session can override parts of it without redeploying — locally and
on sandbox alike ([full story](configuration.md#deploy--session--turn-the-three-configuration-scopes)):

```python
from agent_run import SessionConfig, TurnConfig

session = engine.start_session(config=SessionConfig(model="claude-opus-4-6"))   # this conversation only
result = await session.run("…", config=TurnConfig(max_budget_usd=0.5))          # this turn only
```

**Which credential pays for a local run.** The Agent SDK inherits your whole shell environment (it only
*adds* what the library passes), and `HOME` isn't isolated, so the `claude` CLI sees your own login. It picks
the first of these that is configured — the library does not choose for you:

| # | Source | Notes |
|---|---|---|
| 1 | `CLAUDE_CODE_USE_VERTEX` / `_USE_BEDROCK` / `_USE_FOUNDRY` (truthy) | routes to that cloud; auth is the cloud's own (ADC for Vertex) and steps 2–6 are not consulted |
| 2 | `ANTHROPIC_AUTH_TOKEN` | bearer-token override |
| 3 | `CLAUDE_CODE_OAUTH_TOKEN` | long-lived subscription token from `claude setup-token` (the CI-friendly way to use a subscription) |
| 4 | `ANTHROPIC_API_KEY` | metered API spend |
| 5 | `apiKeyHelper` (settings) | |
| 6 | **your `claude.ai` login** (`~/.claude/.credentials.json`) | **the fallback** — a Pro/Max/Team seat's quota, no API bill |

So on a dev box with `claude` logged in and no key exported, local runs quietly consume **your subscription
seat** — usually the cheaper option on Team, but worth knowing, because it also means two developers running
the same code can be billed differently. Note that step 4 beats step 6: an `ANTHROPIC_API_KEY` sitting in your
shell silently turns subscription runs into metered ones (the CLI warns about this on startup).

**Forcing the API key** (or any other source) for local runs — either of these beats an ambient login,
because both land in the environment the library hands the CLI:

```python
## per run — scoped to this invocation, and the only form that is safe remotely too
await session.run("…", secrets={"ANTHROPIC_API_KEY": os.environ["MY_KEY"]})

## every run of a locally-deployed spec
spec = AgentSpec(name="…", model="…", env={"ANTHROPIC_API_KEY": os.environ["MY_KEY"]})
```

> Keep keys out of `spec.env` for anything you `sandbox.deploy` — the spec is serialized *into* the deployed
> engine, so a value there is baked into the deployment and shared by every run. Per-invocation `secrets` (or
> Vertex routing, which needs no key at all) is the deployable answer.

To force the **subscription** instead, unset `ANTHROPIC_API_KEY` (and the other higher-priority variables) in
the shell you launch from — the library cannot unset an inherited variable for you. To mirror **prod** locally,
set `CLAUDE_CODE_USE_VERTEX=1` with `ANTHROPIC_VERTEX_PROJECT_ID` and `CLOUD_ML_REGION`, which is what
`sandbox.deploy` bakes by default (see [Prod](sandbox-runtime.md#prod-deploy-once-look-up-and-run)); a deployed engine therefore
never touches anyone's subscription.

**Seeding inputs / collecting artifacts.** `session.workspace` is the agent's working directory on the
host — a `Path` you can drop input files into before `run()` and read the agent's output files from after
(it's created on first access). Use the accessor rather than deriving the path yourself: the layout is
`<workdir>/jobs/<session-id>/workspace/`, where the `workspace/` leaf is deliberate — the agent runs in a
directory whose own name says "this is your workspace", not in the anonymous `jobs/<uuid>` dir (which reads
as disposable temp and tempts weaker models into `cd`-ing away, leaving deliverables outside the collected
dir). The same `workspace/` cwd convention applies on `sandbox`, but there the filesystem is remote, so
`session.workspace` raises — seed via the prompt or `spec.repos`, collect via events or a repo push.

**Picking the agent's cwd.** `local.deploy(spec, workspace="/path/of/your/choosing")` runs every session of
that engine in the directory you name, and `session.workspace` returns it. The cwd is part of the agent's
system prompt, so a per-session path means a per-session prompt prefix and no [prompt cache][cache] hit
across sessions; pointing many sessions at one directory keeps the prefix identical and the cache warm. That
is an explicit opt-out of isolation: the directory is yours, concurrent sessions share it, and the library
makes no promise that a session sees only its own files there. Name it something that reads like a workspace,
for the same reason the default leaf is called one — the path is what the agent sees, and a temp-looking cwd
tempts weaker models into `cd`-ing away from it.

Because the directory is yours, the library stops writing to it on your behalf: `repos` is rejected (every
session would clone into the same path), and `checkpoint=True` snapshots the conversation only, leaving the
files alone — a resume continues the conversation in the directory as it stands now, and nothing rolls back
to what the session last saw. This is `local`-only; `sandbox.deploy(workspace=…)` raises, since a worker's cwd
is its own `/tmp`.

[cache]: https://docs.claude.com/en/docs/build-with-claude/prompt-caching

**Observing or gating individual tool calls.** `run(hooks=...)` / `send(hooks=...)` take the Claude Agent
SDK's [hooks](https://docs.claude.com/en/docs/claude-code/hooks) mapping for that turn, e.g. to see every
tool call as it is about to happen and block the ones you do not want:

```python
from claude_agent_sdk import HookMatcher

async def gate(input_data, tool_use_id, context):
    print("about to run", input_data["tool_name"])
    if input_data["tool_name"] == "WebFetch":
        return {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                       "permissionDecision": "deny",
                                       "permissionDecisionReason": "no network from this run"}}
    return {}

run = session.run("…", hooks={"PreToolUse": [HookMatcher(hooks=[gate])]})
```

`PreToolUse` fires under every `permission_mode`, unlike the SDK's `can_use_tool`, which `bypassPermissions`
(the library default) shadows entirely. Hooks are live callables, so they are a `local`-only argument —
`sandbox` runs the turn in a remote worker and rejects them, as does the `codex` harness.

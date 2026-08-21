# remote-agent-toolkit

A Python library for **defining and running remote/background AI agents** at Zyte. Define an agent
declaratively as an `AgentSpec`, then run it both **locally** (in-process, for dev) and **remotely** on
**Gemini Agent Runtime** (prod) through *one* API — with custom skills, GitHub access, structured outputs,
checkpoint/resume and warm starts, without re-learning the platform's sharp edges. Deploy an engine once
per agent role, then customize **per session** (`SessionConfig`: repo/ref, prompt, model, skills) and
**per turn** (`TurnConfig`: budgets, model, output schema) without redeploying — see
[Deploy / session / turn](#deploy--session--turn-the-three-configuration-scopes). Two agent harnesses
ship behind the same API: **Claude Code** (the default) and **Codex**. Between them they run Claude
models, OpenAI GPT models, and — through OpenRouter — Kimi, GLM and DeepSeek. See
[Harnesses and models](#harnesses-and-models).

> **Status: `local` and Gemini Agent Runtime both work — validated live.** Define an `AgentSpec` and run it
> in-process (`local.deploy`), or deploy + run on Agent Runtime (`gemini.deploy` / `gemini.get_engine`), with
> skills, structured output, checkpoint/resume, per-session/per-turn configuration, and a **warm pool**
> (~4 s pickup vs ~2.5 min cold) — all exercised end-to-end on real infrastructure. The two paths share one `Engine`/`Session`/`Run` API. Not yet
> built (raise `NotImplementedError` or simply absent): session `fork()` and a sync run path.
> See [`examples/minimal`](examples/minimal) for a runnable agent and
> [`DESIGN.md`](DESIGN.md) for the architecture, platform contracts (§6), and roadmap (§11).

## Install

Private package — install straight from GitHub (you need repo access). Requires **Python 3.12**
(`claude-agent-sdk` supports ≤3.13):

```bash
pip install "git+ssh://git@github.com/zytedata/remote-agent-toolkit.git"
# or:  uv pip install "git+ssh://git@github.com/zytedata/remote-agent-toolkit.git"
```

Releases are git tags — pin one to shield yourself from in-development changes on `main`
(see [`CHANGELOG.md`](CHANGELOG.md) for what's in each release and how to upgrade across
breaking changes):

```bash
pip install "git+ssh://git@github.com/zytedata/remote-agent-toolkit.git@v0.1.0"
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

## Harnesses and models

A **harness** is the coding-agent loop; the **model** is what it calls. Two harnesses ship, and
both run on both backends (`local` and `gemini`) through the same Engine/Session/Run API.

| Harness | Agent loop | Models it runs | Per-invocation secret |
| --- | --- | --- | --- |
| `claude-code` (default) | Claude Code, via the Claude Agent SDK | Claude models (`claude-sonnet-4-6`, `claude-haiku-4-5`, …) | none on Vertex (the default), else `ANTHROPIC_API_KEY` |
| `codex` | OpenAI Codex, via the `openai-codex` SDK | OpenAI models (`gpt-5.6-sol` / `-terra` / `-luna`, `gpt-5.3-codex`) | `OPENAI_API_KEY` |
| **either one** | the same loop, pointed at OpenRouter | `openrouter/<vendor>/<model>` — Kimi, GLM, DeepSeek (see [below](#openrouter-models-either-harness)) | `OPENROUTER_API_KEY` |

The harness is selected for a session, and its CLI must be included at deploy time.
The model can change on each turn. One deployed engine can therefore serve several models:

```python
# One engine, several models — no redeploy:
session.run(task, config=TurnConfig(model="gpt-5.6-luna"))
session.send(task, config=TurnConfig(model="openrouter/z-ai/glm-5.3"))

# To let sessions pick the harness too, bake both CLIs at deploy:
spec = AgentSpec(name="either", model="claude-sonnet-4-6", harnesses=("claude-code", "codex"))
engine.start_session(config=SessionConfig(harness="codex"))   # selects among what is baked
```

Model ids are free-form strings passed to the harness — any id the underlying CLI accepts works, so a
new model needs no toolkit release. The ids listed above are the ones we price offline (and therefore
can enforce `max_budget_usd` against) and, for OpenRouter, have validated live.

### Claude Code (the default)

Omit `harness` entirely. Auth resolves in the order documented under
[Secrets & security](#secrets--security) — deployed engines route to Vertex by default, so no key is
needed; locally, `ANTHROPIC_API_KEY` or a logged-in `claude` CLI both work.

```python
spec = AgentSpec(name="spider-builder", model="claude-sonnet-4-6")   # harness defaults to claude-code
```

### Codex

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
- **OpenRouter models** reach the same surface — see below.

### OpenRouter models (either harness)

Prefix an OpenRouter model id with `openrouter/`. Set `harness` to `"codex"` or
`"claude-code"`. The same configuration works locally and on Gemini Agent Runtime:

```python
spec = AgentSpec(
    name="kimi-agent",
    model="openrouter/moonshotai/kimi-k3",
    harness="codex",  # "claude-code" also works
)
result = await engine.start_session().run(
    "Fix the failing tests",
    secrets={"OPENROUTER_API_KEY": os.environ["OPENROUTER_API_KEY"]},
)
```

To run the same model with Claude Code, change only the harness:

```python
claude_spec = AgentSpec(
    name="kimi-agent",
    model="openrouter/moonshotai/kimi-k3",
    harness="claude-code",
)
```

The model can change on each turn:

```python
await session.send(task, config=TurnConfig(model="openrouter/z-ai/glm-5.3"))
```

These four models have a known context size and a fallback price. The fallback is used only
when OpenRouter does not return the exact charge. OpenRouter may charge a different rate for
the selected provider.

| Model id | Fallback USD / 1M input → output | Context | Recommended harness | Claude Code test result |
| --- | --- | --- | --- | --- |
| `openrouter/moonshotai/kimi-k3` | 3.00 → 15.00 | ~1M | Either | Local and remote tests passed |
| `openrouter/z-ai/glm-5.3` | 1.40 → 4.40 | ~1M | Either | Local and remote tests passed |
| `openrouter/deepseek/deepseek-v4-flash` | 0.084 → 0.168 | ~1M | Codex | Some turns ended without a final answer |
| `openrouter/deepseek/deepseek-v4-pro` | 1.60 → 3.20 | ~1M | Codex | Final remote test ended without an answer in 2/2 attempts |

Both harnesses can run all four models. Use Codex for DeepSeek v4 Flash and Pro when reliable
completion matters. During testing, both DeepSeek models sometimes finished a Claude Code turn
without returning a final answer. The controlled Codex runs completed normally. Claude Code turns
with this problem return `error_no_final_text`, which allows the caller to retry or switch harness.

Other `openrouter/*` model ids also work. Codex uses generic context metadata for an unknown
model. Exact cost reporting still works when OpenRouter supplies it.

Each OpenRouter response uses [router metadata](https://openrouter.ai/docs/guides/features/router-metadata)
and emits an `openrouter_request` status event. It includes the selected
upstream provider, the provider's model name, region when available, and exact request cost.
The terminal result uses the sum of those exact costs. `max_budget_usd` is checked between model
responses, so one response may take the total above the cap.

#### Providers and repeatable experiments

OpenRouter may offer the same model through several providers. By default, it chooses among
available providers and may use another one when the first choice is unavailable. Providers can
differ in quantization, price, context and output limits, tool support, latency, and throughput.
List the current options without paying for a model call:

```bash
.venv/bin/python dev/openrouter_endpoints.py moonshotai/kimi-k3
```

Provider choice matters when comparing models, harnesses, prompts, or settings. Without a fixed
provider, two runs may use different model hosting and produce different costs, speeds, or results.
That variation can make an experiment misleading.

For a repeatable comparison:

- Create an [OpenRouter preset](https://openrouter.ai/docs/guides/features/presets) that allows one
  provider and disables provider fallbacks.
- Keep the preset unchanged for the whole experiment. Also keep the model, harness, prompt,
  reasoning effort, and other generation settings unchanged unless one of them is the subject of
  the comparison.
- Save each `openrouter_request` event with the results. It records the provider, provider model,
  region, and exact charge for that response.
- Run the provider check below before a larger experiment. It fails if OpenRouter reports a
  different provider.

Put the explicit model before the preset so the toolkit keeps its known context and fallback price:

```python
AgentSpec(
    name="pinned",
    model="openrouter/moonshotai/kimi-k3@preset/kimi-firstparty",
    harness="codex",  # or "claude-code"
)
```

Create the preset in OpenRouter and set its provider rules there. A direct
`openrouter/@preset/<slug>` id also works, although the toolkit cannot know its model or context
before the first response.

The paid tests can verify a real preset on both harnesses. They fail if any request reports a
different provider:

```bash
OPENROUTER_PRESET_MODEL="openrouter/moonshotai/kimi-k3@preset/kimi-firstparty" \
OPENROUTER_EXPECTED_PROVIDER="Moonshot AI" \
make live-openrouter
```

Pinning removes one important source of variation. Model output can still vary between requests,
and a provider may update its serving software or model version. Record the date and the reported
provider details with experimental results.

Important details:

- The `OPENROUTER_API_KEY` is passed per invocation. Both harnesses remove provider credentials
  from the tool environment. Codex also disables its automatic login shell because it could reload
  keys from `~/.bashrc`.
- Claude Code's own dollar estimate is wrong for these custom models. The result uses OpenRouter's
  exact reported charge and keeps the CLI value as `cli_reported_cost_usd` for comparison.
- The four known models use a 1,048,576-token context window on both harnesses.
- Codex disables its built-in web-search tool for OpenRouter turns because OpenRouter rejects the
  tool format Codex sends. Shell, file, MCP, and skill tools remain available.
- Codex uses `low` reasoning when the spec leaves it unset because OpenRouter's Responses endpoint
  requires reasoning.
- `output_schema` is also written into the prompt. OpenRouter accepts the JSON schema but may leave
  enforcement to the model. Structured output was tested separately for every model on both
  harnesses, locally and on Gemini Agent Runtime. All 16 checks returned schema-valid output.
- Reasoning events vary by model. Kimi K3 and both DeepSeek models emitted them during testing;
  GLM-5.3 did not.
- A plain string in `spec.system_prompt` replaces Claude Code's larger preset. This can reduce token
  use when the task does not need the preset's tool guidance.

The paid local and remote tests run a basic turn and a structured-output turn with all four models
on both harnesses. They also check tool use, credential removal, exact cost, budgets, preset
handling, provider reporting, and resume. The remote test checks the Gemini runtime's history,
resource, and trace data. See
`make live-openrouter` and `make live-openrouter-remote`.

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

The deployed spec is the *default*; a session can override parts of it without redeploying — locally and
on gemini alike ([full story](#deploy--session--turn-the-three-configuration-scopes)):

```python
from remote_agent_toolkit import SessionConfig, TurnConfig

session = engine.start_session(config=SessionConfig(model="claude-opus-4-6"))   # this conversation only
result = await session.run("…", config=TurnConfig(max_budget_usd=0.5))          # this turn only
```

**Which credential pays for a local run.** The Agent SDK inherits your whole shell environment (it only
*adds* what the toolkit passes), and `HOME` isn't isolated, so the `claude` CLI sees your own login. It picks
the first of these that is configured — the toolkit does not choose for you:

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
because both land in the environment the toolkit hands the CLI:

```python
# per run — scoped to this invocation, and the only form that is safe remotely too
await session.run("…", secrets={"ANTHROPIC_API_KEY": os.environ["MY_KEY"]})

# every run of a locally-deployed spec
spec = AgentSpec(name="…", model="…", env={"ANTHROPIC_API_KEY": os.environ["MY_KEY"]})
```

> Keep keys out of `spec.env` for anything you `gemini.deploy` — the spec is serialized *into* the deployed
> engine, so a value there is baked into the deployment and shared by every run. Per-invocation `secrets` (or
> Vertex routing, which needs no key at all) is the deployable answer.

To force the **subscription** instead, unset `ANTHROPIC_API_KEY` (and the other higher-priority variables) in
the shell you launch from — the toolkit cannot unset an inherited variable for you. To mirror **prod** locally,
set `CLAUDE_CODE_USE_VERTEX=1` with `ANTHROPIC_VERTEX_PROJECT_ID` and `CLOUD_ML_REGION`, which is what
`gemini.deploy` bakes by default (see [Prod](#prod-deploy-once-look-up-and-run)); a deployed engine therefore
never touches anyone's subscription.

**Seeding inputs / collecting artifacts.** `session.workspace` is the agent's working directory on the
host — a `Path` you can drop input files into before `run()` and read the agent's output files from after
(it's created on first access). Use the accessor rather than deriving the path yourself: the layout is
`<workdir>/jobs/<session-id>/workspace/`, where the `workspace/` leaf is deliberate — the agent runs in a
directory whose own name says "this is your workspace", not in the anonymous `jobs/<uuid>` dir (which reads
as disposable temp and tempts weaker models into `cd`-ing away, leaving deliverables outside the collected
dir). The same `workspace/` cwd convention applies on `gemini`, but there the filesystem is remote, so
`session.workspace` raises — seed via the prompt or `spec.repos`, collect via events or a repo push.

**Picking the agent's cwd.** `local.deploy(spec, workspace="/path/of/your/choosing")` runs every session of
that engine in the directory you name, and `session.workspace` returns it. The cwd is part of the agent's
system prompt, so a per-session path means a per-session prompt prefix and no [prompt cache][cache] hit
across sessions; pointing many sessions at one directory keeps the prefix identical and the cache warm. That
is an explicit opt-out of isolation: the directory is yours, concurrent sessions share it, and the toolkit
makes no promise that a session sees only its own files there. Name it something that reads like a workspace,
for the same reason the default leaf is called one — the path is what the agent sees, and a temp-looking cwd
tempts weaker models into `cd`-ing away from it.

Because the directory is yours, the toolkit stops writing to it on your behalf: `repos` is rejected (every
session would clone into the same path), and `checkpoint=True` snapshots the conversation only, leaving the
files alone — a resume continues the conversation in the directory as it stands now, and nothing rolls back
to what the session last saw. This is `local`-only; `gemini.deploy(workspace=…)` raises, since a worker's cwd
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
(the toolkit default) shadows entirely. Hooks are live callables, so they are a `local`-only argument —
`gemini` runs the turn in a remote worker and rejects them, as does the `codex` harness.

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
the parsed, validated object. The harness sends the schema through the CLI's native structured-output
option. Tell the agent what content to emit in the prompt; the schema describes its shape. OpenRouter may
leave schema enforcement to the selected model, so those turns also receive the schema in their prompt.
The toolkit validates the SDK's structured value when available and can parse JSON from the final text (a
```` ```json ```` block, a bare object, or the whole message) as a fallback.

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
- **`max_buffer_size`** — bytes allowed in a single message read from the Claude Code CLI's stdout
  (default 32 MiB; `claude-code` only). One oversized message — a `Read` of a large image arrives
  base64-encoded, and a big tool result is one line — kills the turn with no result at all, so raise this
  for an agent whose tool outputs are legitimately enormous rather than discovering the ceiling in
  production. (The SDK's own default is 1 MiB; the toolkit's is deliberately far above it.)
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
your own `ANTHROPIC_API_KEY`, or — with no key set — your `claude.ai` subscription login: see
[which credential pays for a local run](#dev-run-locally-in-process)); local is a trusted-dev context.

**In transit & at rest.** Secret values never travel in the invocation payload itself — the platform
*persists* a job's input verbatim (`jobs/<sid>_input.jsonl` in the output bucket), and a Pub/Sub message is
retained until acked, so values in either would linger. Instead the values are staged at a **per-invocation
GCS object** in the output bucket (readable only by the bucket's principals: your operator identity and the RE
service agent); the invocation carries just that pointer, and the worker **deletes the object once the turn's
terminal result is out** — not on read, because the platform automatically re-runs a crashed job attempt with
the same payload, and that retry must still find its credentials. The client also cleans the object up on run
completion, and deploy installs a 1-day GCS lifecycle rule on the `invocation-secrets/` prefix as the
last-resort reaper for objects orphaned by a crash on both sides. The toolkit **never** writes secret
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

**The platform-critical layer is pinned in-tree.** Engines resolve their requirements at *build* time, so
without pins two deploys of the same toolkit commit could produce different runtimes.
[`runtime/gemini/constraints.txt`](remote_agent_toolkit/runtime/gemini/constraints.txt) pins the packages the
toolkit's engine contract depends on (aiplatform, google-adk, claude-agent-sdk, the OTel export stack, …)
with pip-constraints semantics, merged into the requirements at deploy time; refreshing a pin is a normal
reviewed diff (bump → canary deploy → live turn → clean telemetry). Your `packages` merge with these — a pin
that contradicts a platform constraint fails fast at deploy, before the billable build. Deploy also verifies
your venv matches the pickle-coupled pins (aiplatform, cloudpickle, pydantic): the engine build unpickles an
object your venv pickled, so those versions must agree — if deploy raises, sync your venv to the constraint
(or refresh the constraint deliberately).

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
                           warm_pool=True)     # pure addressing; warm_pool to dispatch to the pool
session = engine.start_session()
result = await session.run("/scrape https://books.toscrape.com title, price")   # default: wait for the result
```

### Deploy / session / turn: the three configuration scopes

An engine is deployed per agent *role*, but much of its configuration is naturally
per-**session** (which repo/ref this conversation operates on, the prompt or model variant
of an A/B experiment) or per-**turn** (budgets, structured output for the final turn) —
and requiring a ~4 min engine redeploy per combination would be unworkable. Configuration
therefore has three scopes, each with its own type, named by the lifetime of what it
configures:

| Scope | Type | Bound at | What belongs here |
|---|---|---|---|
| deploy | `AgentSpec` | `gemini.deploy(spec)` | identity (`name`), image contents (`packages`, engine `env`, the harness CLIs — `harnesses=(...)` bakes several), and the *defaults* for everything below |
| session | `SessionConfig` | `engine.start_session(config=...)` | the conversation's world: `repos`, `skills`, `mcp_servers`, `system_prompt`, `harness` (selects among the baked CLIs), `checkpoint`/`interactive`, `extra_env` — plus session-wide defaults for the turn knobs |
| turn | `TurnConfig` | `session.run(config=...)` / `send(config=...)` | the knobs the harness re-reads every invocation: `model`, `reasoning_effort`, `max_turns`, `max_budget_usd`, `max_buffer_size`, `background_task_timeout`, `permission_mode`, tool lists, `output_schema` |

Both config types are **sparse overlays**: a field left at `INHERIT` (the default) keeps
the value from the layer below; a set field replaces it wholesale (`extra_env` is the one
additive exception — it adds onto the deployed `env`). Deploy-only facts have no config
field at all, so "different `packages` per run" is a `TypeError`, not a silent no-op.

```python
from remote_agent_toolkit import RepoSource, SessionConfig, TurnConfig

engine = gemini.get_engine("spider-builder", project=..., location=..., warm_pool=True)

session = engine.start_session(config=SessionConfig(          # bound ONCE, for good
    repos=[RepoSource.git("https://bitbucket.org/o/store", ref="heal/issue-123",
                          auth="bb-token")],                   # named secret, never inline
    model="claude-sonnet-4-6",                                 # overrides the deployed default
))
run  = await session.run("Fix the price selector", secrets={"bb-token": tok})
run2 = await session.send(                                     # same session config; per-turn overlay:
    "Summarize what you changed as JSON",
    secrets={"bb-token": tok},
    config=TurnConfig(output_schema=ChangeReport, reasoning_effort="low"),
)
```

**Why the session config binds once.** A session's first turn clones `repos` and provisions
`skills` into a fresh workspace; every later turn restores a snapshot of the previous
turn's workspace (with the agent's uncommitted work) and never re-reads those fields. A
mid-conversation change could not be honored — so the API refuses to express it: `send()`
takes no session config, and `get_session(sid)` re-attach reads back the config the opener
persisted rather than accepting one.

The contracts behind this:

* **The worker executes the merged spec** (deploy-baked ← session ← turn) and **echoes it
  as an `effective_spec` event** — the durable ground-truth record of what actually ran
  (drive post-mortems and replays from it, not from what you think you passed). The config
  objects themselves are persisted in GCS next to the session's records (30-day lifecycle)
  for the same reason.
* **Fail closed**: a missing config object, a config field this engine revision doesn't
  know, or a `harness` the image doesn't bake all FAIL the turn with a terminal error —
  never a silent fall-back to the baked spec (a wrong-configuration run is exactly what
  this exists to prevent). Client and engine must be deployed from the same toolkit
  revision (already the rule — the invocation payload is a wire contract).
* **No config → nothing staged**: a plain `start_session()`/`run()` is byte-for-byte the
  pre-config behavior; the turn runs the deploy-baked spec with zero extra moving parts.
* **Deploy-time fields stay deploy-time**: `packages` and harness *availability* come from
  the image; a session selects among what is baked (`AgentSpec(harnesses=("claude-code",
  "codex"))` bakes both CLIs so sessions can pick either). Baked skills are reused as a
  staging fast path when the effective `skills` match the deployed ones; a differing
  declaration resolves from its own sources at run time.
* **No credentials in configs**: config objects are referenced from persisted payloads and
  kept for debugging, so `SessionConfig` refuses a `RepoSource.url` embedding
  `user:token@` at construction — name the token via `RepoSource(auth=..., auth_user=...)`
  and pass the value in `run(secrets=...)`.

**Managing deployed engines** (control plane):

```python
gemini.deploy(spec, project=..., location=...)   # create / update; ops/CI only (warm_pool=True, pool_size=N)
gemini.deploy(spec, ..., resource_limits={"cpu": "4", "memory": "16Gi"})  # container CPU/RAM (default 4 / 4Gi)
gemini.get_engine("spider-builder", project=..., location=...)   # look up by name (app code; addressing only)
gemini.list_engines(project=..., location=...)   # discover what's deployed
engine.name, engine.version, engine.resource     # identity / serving revision / underlying resource name
engine.wait_until_warm(timeout=300)              # warm pools: wait for a ready worker before dispatching
engine.delete(delete_pool_resources=True)        # tear down: cancels pool workers, removes engine (+ topic/sub)
```

Note: `engine.delete()` is the correct way to tear a warm pool down — its idle workers are long-running jobs
that otherwise block deletion until they expire. Session `fork()` is not built yet.

### Versions: engines have revisions

Engine identity is `spec.name`. Deploying a name that already exists **updates that engine**, and Agent
Runtime mints a new immutable *runtime revision* of it — so app code's `get_engine("spider-builder")` picks
the new code up without re-pointing at anything, and the previous revision stays around to roll back to.
(Only the first deploy of a name creates an engine; pass `new_engine=True` for a deliberate side-by-side.)

```python
engine.versions()                        # ['7', '6', '5'] — revision ids, newest first
engine.revisions()                       # + create_time / state / which one is `serving`
engine.version                           # the revision this handle's runs execute on
engine.set_traffic("6")                  # roll back: send 100% of traffic to revision 6
engine.set_traffic()                     # back to always-latest (the platform default)
engine.delete_version("5")               # prune an old revision (the serving one can't go)

gemini.get_engine("spider-builder", ..., version="7")   # assert we're running revision 7
```

**`version=` is an assertion, not routing.** Agent Runtime exposes `asyncQuery` — the toolkit's whole run
plane — on the *engine* only; a revision has `query`/`streamQuery` but no async form. So which revision runs
a turn is decided by the engine's traffic config, not by the caller: `get_engine(..., version=N)` verifies
that revision exists *and* is the one serving (a deploy-then-pin CI flow catches a rolled-back engine at
lookup instead of running unknown code), and `set_traffic` is the ops action that actually moves traffic.
Two callers cannot address two revisions of one engine concurrently.

A traffic pin survives later deploys — a fresh revision won't serve until you `set_traffic()` again, and
`deploy` warns when it lands in that state. For warm pools, note that workers already blocked on the
dispatch subscription keep running the revision they cold-started with, so an update has a window where
turns may land on either; `delete()` the pool (or deploy with `new_engine=True`) for a hard cutover.

## Past jobs: listing sessions & reading history

Every `gemini` run leaves a durable record, and the library reads it back — from any process, long after
the run:

```python
engine = gemini.get_engine("spider-builder", project=..., location=...)

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

### Reading the harness transcript

`AgentEvent`s are summaries. When you need the harness's own record — per-turn usage, tool statuses,
subagent trees, permission denials — deploy with `transcript=True` and read it back on both runtimes:

```python
spec = AgentSpec(name="scorer", model="claude-sonnet-4-6", transcript=True)
...
transcripts = await session.transcripts()   # {"main": [...], "subagents/<id>": [...], ...}
```

`checkpoint=True` implies it (resume needs the transcript), but `transcript=True` alone adds **no
workspace snapshot** — a run with a multi-hundred-MB working directory pays for the transcript only. It is
purely observational: continuing a conversation across turns (`send`) still takes `checkpoint=True`. The
`codex` harness keeps its conversation under `checkpoint` alone, so `transcripts()` reads back `{}` there
(the run says so, as a `spec_warning` status event). Without either flag `transcripts()` raises; `{}` means
persistence is on and nothing has been written yet.

Unlike events, a transcript is verbatim: prompts, tool inputs and outputs, anything the agent saw or
echoed — including a secret a coerced agent printed. Treat the checkpoint prefix, and everything
`transcripts()` returns, as sensitive.

## Monitoring job CPU/RAM (OOM forensics)

The platform gives you **no** resource metrics for agent jobs: query-job containers run in a Google tenant
project, so their Cloud Run metrics and OOM-kill events never reach your project. The worker therefore
samples **itself** (its cgroup) and ships the record to your Cloud Logging, where it survives a mid-turn
kill:

- **Per-session samples** — every ~20s to the `remote_agent_toolkit_resources` log (not the event stream, so
  no noise). After a crash, the last sample sits at most one interval before death:
  `logName="projects/<project>/logs/remote_agent_toolkit_resources" AND labels.session_id="<sid>"`.
- **A visible warning** — the first time memory crosses 85% of the limit, a `memory pressure: …` status
  event lands on the normal event stream, so a watcher sees trouble before the platform kills the worker at
  the limit (the fix: deploy with higher `resource_limits`, see above).
- **Peak in every result** — the terminal result's `raw` carries `memory_peak_bytes` /
  `memory_limit_bytes` / `cpu_usec`, so completed turns report their high-water mark for free.

Default on; tune or disable with the `AGENT_RESOURCE_SAMPLE_S` env var on the engine (seconds; `0`
disables). Read a session's samples back with:

```python
session = engine.get_session("<session-id>")      # or any session you already hold
for row in session.resource_samples():            # oldest first; ~30-day log retention
    print(row["time"], row.get("memory_current_bytes"), row.get("memory_limit_bytes"))
```

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

There's nothing to turn on: the runtime service agent's default role covers the export, and the toolkit
force-flushes OpenTelemetry at the end of every turn — which is the load-bearing part: the platform
initializes telemetry in job workers but never flushes it on the async job path the toolkit uses, so
without that flush no span would ever leave the worker (verified with a standalone repro; raised with
Google). The only prerequisites are the `telemetry.googleapis.com` + `cloudtrace.googleapis.com` APIs on
the project, and `roles/cloudtrace.user` for whoever wants to *view* traces.

**If every export fails with `Failed to export span batch code: 403, reason: Forbidden`** in the engine
log (metrics batches too) and no trace reaches the console, the engine's pickle almost certainly carries
the **wrong GCP project**. `AdkApp` snapshots the aiplatform global config's project at *construction*
time on the deploy machine; the worker later force-feeds that pickled project into
`GOOGLE_CLOUD_PROJECT` and routes every span/metric batch to it — so a stray local default (e.g.
gcloud's org-wide `other-project`) bakes a cross-project telemetry write into the engine, which the
runtime service agent is (rightly) forbidden to perform. The breakage is **persistent per engine**
(it's in the pickle) and survives redeploys from the same misconfigured environment — which is what made
it masquerade as a server-side incident for three days in 2026-08. The toolkit now pins the deploy
target into the app (`build_adk_app`) and refuses to ship a pickle that captured anything else; on older
toolkit versions, fix the deploy environment (`gcloud config set project <target>` or
`GOOGLE_CLOUD_PROJECT=<target>`) and redeploy. Diagnosis shortcut: `python -m pickletools
agent_engine.pkl | grep -A1 project` on the staged pickle — the project string is visible in cleartext.
The turns themselves are unaffected throughout (traces are a diagnostic channel).

**Known gaps** (platform-side, as of 2026-07/08, raised with Google): the console's *session conversation*
panel stays empty ("No chat conversation data") — it is fed by platform instrumentation that doesn't run
for the async job path — and the trace tree shows a cosmetic "(Missing span ID …)" placeholder above the
turn (the platform tears the job worker down before its own wrapper span is exported). As of 2026-08-05,
new spans land in the telemetry-backed store: the console (Traces tab / Trace explorer) shows them, but
the legacy Cloud Trace **v1 list API** no longer returns them — don't use it to check whether tracing
works. Span values here
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
| **Warm pool** (`warm_pool=True`) | **~4 s** to first *observed* event | long-running | interactive *and* long — best of both |
| Sync | ~3–9 s | ~600 s (10 min) hard | _not supported yet (see below)_ |

The ~2.5 min async start is **per job, not a one-time cold start** — it's Vertex provisioning a dedicated
worker, and it is *not* reducible via concurrency / min-instances knobs. So for one long task, do it in a
**single** job (pay the start once) rather than many short jobs.

The platform also offers a **sync** path (fast start, but a hard ~10 min ceiling). The toolkit **does not
expose it yet** — the run plane is built around the async + warm-pool paths, which cover long *and*
interactive work. We can add sync later if a genuinely short-turn use case needs the lower start latency.

**How `warm_pool=True` works.** `gemini.deploy(spec, warm_pool=True)` keeps a pool of pre-provisioned workers,
each blocked on a Pub/Sub subscription (a competing-consumers *atomic claim*). A worker warms its logging +
storage channels during its idle wait and reports ready — `engine.wait_until_warm()` blocks on that signal.
A run is then dispatched to a free worker, so the turn goes nearly straight to the model. The first
**observed** event lands **~4 s** after dispatch (measured: 3.8 s to first event, 10.9 s to the result of a
one-tool Haiku turn, vs ~2.5 min cold) — the worker's claim pickup plus one ~0.5 s mirror flush and one
tail poll; events then stream **~1–2 s** behind the agent for the rest of the turn. (Before the GCS event
stream this number was ~10–20 s, dominated by Cloud Logging's ingestion lag.) On claim the pool refills, so
the next turn is warm too.

> _Keeping the pool full:_ an idle worker waits `pool_max_wait_s` (a `deploy()` parameter; default a day)
> for an assignment, then exits — **without replacement**. And a pool that has drained to empty does not
> self-recover: the automatic one-worker refill after each dispatch just claims that pending dispatch
> itself on an empty pool, so net pool size stays 0 and every turn goes cold (~2.5 min) until
> `engine.fill_pool(n)` re-warms it by hand. Size `pool_max_wait_s` to your dispatch gaps — any quiet
> stretch longer than it drains the pool. The platform's max **job** duration (7 days at the time of
> writing — a platform limit that can change) caps the wait regardless: a worker that outlives it is
> killed, also without replacement.

**Event streaming scales with your fleet.** The stream you consume with `async for ev in run` is the
session's **GCS event mirror**, tailed live: the worker writes small batches as events happen and the
client polls the object listing — strongly consistent (no ingestion lag) and free of any restrictive
read quota, so tens of concurrently-streamed runs in one project are a non-event. The same objects are
the durable history `session.history()` reads. Cloud Logging still receives every step (it's the
indexed store the [debugging recipes](TESTING.md#debugging-a-live-run) query), but no client run
depends on reading it.

> _Engines deployed before event streaming_ wrote the mirror only at end-of-turn, so against them the
> stream delivers all of a turn's events in one batch with the terminal result (still a correct run —
> just not live), and `wait_until_warm` times out soft (its readiness marker only reached Cloud
> Logging, which clients no longer read: its read path is capped at
> [60 requests/min per project](https://cloud.google.com/logging/quotas), fixed and shared by
> everything in the project — the reason it was dropped as a data plane). **Redeploy an engine to
> move it to live streaming.**

**Cost.** A deployed engine itself is (almost) free while idle: the toolkit deploys with `min_instances=0`
(no standing container — the async path provisions a worker per job, so a min-instances container would serve
only the unused sync path while billing continuously). Warm workers are the exception: they are long-running
jobs sitting idle waiting for work, so you **pay for that idle compute** continuously — `warm_pool=True`
trades money for latency. Size the pool to your concurrency, and leave it off for batch / non-interactive
agents where a ~2.5 min start is fine. `deploy(..., pool_max_wait_s=...)` bounds an idle worker's life
(default a day — see the pool-draining note above before lowering it). Tear a pool down with
`engine.delete(delete_pool_resources=True)` (it cancels the idle workers, which otherwise keep billing until
they expire). Model token cost is the same either way and is reported per run as `result.cost_usd`.

## Learn more

See [`DESIGN.md`](DESIGN.md) — the agreed architecture, the hard-won platform contracts (§6), the ports &
adapters seam (§7), and the phased roadmap (§11). The README grows with the code, phase by phase.

Contributing? [`TESTING.md`](TESTING.md) covers the testing ladder — the offline suite (`make test`), the
install-parity image ([`dev/`](dev/)), and **live validation** on real Agent Runtime (`make live-smoke`):
when each is required and how to debug a live run.

Changes ship as tagged releases documented in [`CHANGELOG.md`](CHANGELOG.md) — its header spells out
the versioning policy (0.x minor releases may break; every breaking change carries update notes) and
the release checklist. If your PR changes behavior, add an entry under `Unreleased`.

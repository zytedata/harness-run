# agent-run

A Python library for **defining and running remote/background AI agents** at Zyte. Define an agent
declaratively as an `AgentSpec`, then run it both **locally** (in-process, for dev) and **remotely** in
**Gemini Enterprise Agent Platform sandboxes** (prod) through *one* API — with custom skills, GitHub access,
structured outputs, checkpoint/resume and a ready pool, without re-learning the platform's sharp edges. Deploy an engine once
per agent role, then customize **per session** (`SessionConfig`: repo/ref, prompt, model, skills) and
**per turn** (`TurnConfig`: budgets, model, output schema) without redeploying — see
[Deploy / session / turn](#deploy--session--turn-the-three-configuration-scopes). Two agent harnesses
ship behind the same API: **Claude Code** (the default) and **Codex**. Between them they run Claude
models, OpenAI GPT models, and — through OpenRouter — Kimi, GLM and DeepSeek. See
[Harnesses and models](#harnesses-and-models).

> **Status: `local` and the sandbox runtime both work — validated live.** Define an `AgentSpec` and run it
> in-process (`local.deploy`), or deploy + run in Agent Sandbox (`sandbox.deploy` / `sandbox.get_engine`), with
> skills, structured output, checkpoint/resume, per-session/per-turn configuration, and a **ready pool**
> (~1 s to the first event; ~20 s when no sandbox is ready) — all exercised end-to-end on real infrastructure.
> The two paths share one `Engine`/`Session`/`Run` API. Not yet built (raise `NotImplementedError` or simply
> absent): session `fork()`.
> See [`examples/minimal`](examples/minimal) for a runnable agent and
> [`DESIGN.md`](DESIGN.md) for the architecture, platform contracts (§6), and roadmap (§11).

## Install

Private package — install straight from GitHub (you need repo access). Requires **Python 3.12**
(`claude-agent-sdk` supports ≤3.13):

```bash
pip install "git+ssh://git@github.com/zytedata/remote-agent-toolkit.git"
# or:  uv pip install "git+ssh://git@github.com/zytedata/remote-agent-toolkit.git"
```

The base install covers the **remote path** (deploying and driving sandbox engines — the
sandbox image installs the harness SDKs from its own baked requirements). Running agents
**locally** (`local.deploy`) also runs the harness SDKs on your machine, so add the `local`
extra:

```bash
pip install "agent-run[local] @ git+ssh://git@github.com/zytedata/remote-agent-toolkit.git"
```

(Kept out of the base install because the SDKs are heavy and pin aggressively — e.g.
`claude-agent-sdk` requires `mcp<2`, which conflicts with apps on the mcp 2.x SDK, and
`openai-codex` bundles the codex CLI binary.)

Releases are git tags — pin one to shield yourself from in-development changes on `main`
(see [`CHANGELOG.md`](CHANGELOG.md) for what's in each release and how to upgrade across
breaking changes). Every release is also published to Zyte's internal PyPI, which is what
Scrapy Cloud builds and other environments without access to this git repo should use:

```bash
# From the internal PyPI (read credentials: the usual pkgrepo user, or ask IT support):
pip install "agent-run==0.3.0" --extra-index-url "https://<user>:<password>@pypi.internal.example/simple/"
# Straight from the git tag:
pip install "git+ssh://git@github.com/zytedata/remote-agent-toolkit.git@v0.3.1"
```

Every push to `main` also publishes a dev build there, `X.Y.Z.dev<n>`; it sorts before the
release of the same version, so you only get one by pinning it exactly.

Developing on the toolkit itself (early users are expected to contribute)? Clone it and `uv sync` — that
installs the runtime deps plus the `dev` group (pytest, ruff, mypy, and the harness SDKs —
the suite exercises both harnesses).

## Define an agent

```python
from agent_run import AgentSpec, SystemPrompt, SkillSource, McpServer, sandbox, local

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

For authenticated remote MCP servers and migration of credential-bearing config, see
[runtime secret references and validation boundaries](docs/secret-configuration.md).
Every field except `name` and `model` has a sensible default (see [`spec.py`](agent_run/spec.py));
a two-line spec (`AgentSpec(name=..., model=...)`) is a valid agent. For **structured output**, see the
section below.

## Harnesses and models

A **harness** is the coding-agent loop; the **model** is what it calls. Two harnesses ship, and
both run on both backends (`local` and `sandbox`) through the same Engine/Session/Run API.

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
new model needs no toolkit release. `max_budget_usd` is enforced against a different number per
harness: Claude Code reports its own spend, the OpenAI ids listed above are priced offline by the
toolkit, and OpenRouter ids use the charge OpenRouter reports. The four OpenRouter models listed
above have been validated live on both harnesses.

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
  enforcement; the result event records which source priced the run (`price_source`). Collab-subagent
  spend is invisible on Codex's wire, so the harness recovers it post-hoc from the subagent rollout
  files — the terminal `usage`/`cost_usd` include it (see "What the numbers count"), each thread priced
  at the model its own rollout names (a subagent role can run a different model than the parent; one on a model
  `pricing` doesn't know makes `cost_usd` `None` — never a guess — with a `subagent_price_unknown`
  status event; its tokens stay in `usage`). A thread
  seen on the wire with no rollout found raises a `subagent_usage_missing` status event — the tripwire
  for the recovery's reliance on non-public rollout details (`make live-usage` re-checks them live). The mid-turn
  `max_turns`/`max_budget_usd` checks can't see it while the turn is running; the budget is re-checked
  at turn end once it is known, so a turn that subagent spend pushes over the cap still ends
  `error_budget_exceeded` (it just can't be interrupted early).
- **Checkpoint/resume** works cross-worker: the Codex conversation (a local rollout file) is
  persisted to the blob store alongside the workspace snapshot and restored on `send()`.
- **Reasoning effort**: `spec.reasoning_effort` becomes the SDK's per-turn `effort`, re-applied on
  every turn so resumed conversations keep it. Codex has no `max` level; it maps to `xhigh` with a
  status warning.
- **Which model actually ran**: every Codex turn emits one `model_routing` status event carrying
  what the Codex app-server recorded for the thread, and whether that matches what was asked for.
- **Gaps**: `allowed_tools`/`disallowed_tools` have no Codex equivalent and are rejected
  before launch (including an explicitly empty allow list). Unknown `permission_mode`
  values are also rejected; they never fall back to full access. Remove unsupported
  settings only if the selected Codex sandbox provides the restrictions you need.
  Codex has no background-task re-invocation, so `background_task_timeout` is inert.
  `permission_mode` maps onto Codex's sandbox+approval pairs (`bypassPermissions` → full access,
  never ask; `default` → workspace-write with Codex's auto-reviewer).
- **Any other Codex setting**: `spec.codex_config` takes `config.toml` keys the spec has no field for,
  e.g. `{"sandbox_workspace_write.network_access": True, "web_search": "disabled"}`, passed as
  `--config` overrides after the harness's own so they win. Settable per session and per turn.
- **OpenRouter models** reach the same surface — see below.

### OpenRouter models (either harness)

Prefix an OpenRouter model id with `openrouter/`. Set `harness` to `"codex"` or
`"claude-code"`. The same configuration works locally and in the sandbox runtime:

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

The table summarizes the paid local and remote (sandbox) tests. Each model was tested
separately on both harnesses:

- ✅ passed locally and remotely
- ⚠️ works, with the reliability note in the last column

All four models have a 1,048,576-token context window, on both harnesses.

| Model | Harness | Completes a turn | Structured output | Shell tools | Exact cost | Provider selection | Notes |
| --- | --- | :---: | :---: | :---: | :---: | :---: | --- |
| **Kimi K3**<br>`openrouter/moonshotai/kimi-k3` | Codex | ✅ | ✅ | ✅ | ✅ | ✅ |  |
|  | Claude Code | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| **GLM-5.3**<br>`openrouter/z-ai/glm-5.3` | Codex | ✅ | ✅ | ✅ | ✅ | ✅ |  |
|  | Claude Code | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| **DeepSeek v4 Flash**<br>`openrouter/deepseek/deepseek-v4-flash` | Codex | ✅ | ✅\* | ✅ | ✅ | ✅ | Recommended harness |
|  | Claude Code | ⚠️ | ✅ | ✅ | ✅ | ✅ | See below |
| **DeepSeek v4 Pro**<br>`openrouter/deepseek/deepseek-v4-pro` | Codex | ✅ | ✅\* | ✅ | ✅ | ✅ | Recommended harness |
|  | Claude Code | ⚠️ | ✅ | ✅ | ✅ | ✅ | See below |

\* The DeepSeek structured-output checks on Codex run without a pinned provider, because Novita —
the provider the other DeepSeek checks pin — rejects the `json_schema` format Codex sends. So those
two boxes are the only ones where structured output and provider selection were not proven in the
same request.

**Shell tools** means the agent can run commands in its workspace, such as Python, `git`, or a
test command, and read the output. The tests also confirm that model-provider credentials are
absent from the command's environment.

**Provider selection** is set on the spec, session or turn — see
[Providers and repeatable experiments](#providers-and-repeatable-experiments) below.

**Exact cost** is the charge OpenRouter reports for each response. There is no estimated
price: OpenRouter routes one model id to providers whose prices differ by up to 2.5x, so a
per-model estimate would only match whoever served the call (the measured gap is in the
[`pricing`](agent_run/harness/pricing.py) module docstring). A turn OpenRouter reports no
charge for gets `cost_usd=None` on its result event, plus a `cost_unknown` status event, and its
`max_budget_usd` cannot be enforced. `RunResult.cost_usd` is `float | None` and carries that same
`None`, so an unreported charge stays apart from a turn that really was free.

The ⚠️ is about intermittency, not a permanent failure: under Claude Code, both DeepSeek models have
finished a turn without returning a final answer in earlier runs, while the newest local and remote
runs (2026-08-24) had both of them answer on the first attempt. The same models under Codex have not
shown it. So prefer Codex for DeepSeek v4 Flash and Pro when a single run has to produce an answer.
A Claude Code turn that hits this is reported as `error_no_final_text` rather than a success, so the
caller can retry or switch harness.

For an `openrouter/*` id the toolkit does not list, only the context window is missing: Codex falls
back to generic model metadata and warns, and Claude Code assumes 200k tokens, so it compacts the
conversation earlier than the model needs. Add the model to `OPENROUTER_CONTEXT_WINDOWS` in
`harness/pricing.py` to fix that.

What can stop a new model is the OpenRouter account, not the toolkit. An account restricted to
zero-data-retention endpoints refuses a model that has none, with HTTP 404 and a data-policy
message. The run fails honestly — the `openrouter_request` event records the 404 — but note that
the Claude Code CLI rewrites the message into a vague "model may not exist or you may not have
access", so read the recorded status rather than the result text.

Every model call goes through a local proxy the toolkit runs for the turn, on either harness. The
provider choice reaches OpenRouter through it, and the exact charge comes back through it. The
`OPENROUTER_API_KEY` is passed per invocation and stays in the calling process: the CLI gets a
random per-run token for the proxy instead. Both harnesses also remove provider credentials from the
tool environment, and Codex has its automatic login shell disabled because it could reload keys from
`~/.bashrc`. [DESIGN.md](DESIGN.md) explains why the proxy is there.

Each response uses [router metadata](https://openrouter.ai/docs/guides/features/router-metadata)
and emits an `openrouter_request` status event. It reports the selected upstream provider,
the provider's model name, region when available, the HTTP status, the exact request cost, and the
provider routing the request asked for (`requested_routing`).
Failed responses are reported too, so a retried request is visible. The result uses the sum of
the exact costs. `max_budget_usd` is checked between model responses, so one response may take
the total above the cap. The proxy then refuses the next request (a zero budget refuses the first).
This is admission between responses, not a hard spending ceiling: concurrent requests can pass it
together, and a response OpenRouter reports no cost for adds nothing to the total. For a hard ceiling,
use the provider's own prepaid or spending limits.

```python
run = session.run(task, secrets={"OPENROUTER_API_KEY": key})
async for event in run:
    data = event.raw or {}
    if data.get("event") == "openrouter_request":
        print(data["provider"], data["provider_model"], data["region"], data["cost_usd"])
```

A Codex turn also emits one `model_routing` event (see the Codex section above). It answers "did the
model override take effect", where `openrouter_request` answers "who served each call".

#### Providers and repeatable experiments

OpenRouter may offer the same model through several providers. Its
[provider routing](https://openrouter.ai/docs/guides/routing/provider-selection) normally chooses
among available providers and may use another one when the first choice is unavailable. Providers
can differ in quantization, price, context and output limits, tool support, latency, and throughput.
List the current options without paying for a model call:

```bash
.venv/bin/python dev/openrouter_endpoints.py moonshotai/kimi-k3
```

Provider choice matters when comparing models, harnesses, prompts, or settings. Without a fixed
provider, two runs may use different model hosting and produce different costs, speeds, or results.
That variation can make an experiment misleading.

Choose one provider directly in the agent definition:

```python
spec = AgentSpec(
    name="kimi-first-party",
    model="openrouter/moonshotai/kimi-k3",
    harness="codex",  # or "claude-code"
    openrouter_provider="moonshotai",
)
```

The toolkit sends this rule with every model request:

```json
{"provider": {"only": ["moonshotai"], "allow_fallbacks": false}}
```

That object is also what the `openrouter_request` event reports as `requested_routing`, because it
is what the request carried.

OpenRouter must use that provider, and cannot switch to another one. A request that provider
rejects or cannot serve therefore fails instead of moving elsewhere. You see it as an
`openrouter_request` event carrying the HTTP status and OpenRouter's message, and the CLI retries.
Observed with Novita on DeepSeek v4 Flash: one rejected request, then a normal completion.

The selected provider must support every feature used by the request. For example, a provider may
support tools but reject Codex's `json_schema` structured-output format. With strict selection,
OpenRouter returns that error to the agent. Check the exact model, harness, provider, and feature
combination before a larger experiment. `dev/openrouter_endpoints.py` shows whether each endpoint
advertises tools and `response_format`, although providers can support different response formats.

Provider selection can also be set for one session or one turn:

```python
engine.start_session(config=SessionConfig(openrouter_provider="moonshotai"))
await session.send(task, config=TurnConfig(openrouter_provider="moonshotai"))
```

Use `openrouter_provider=None` in a turn config to return to OpenRouter's normal routing. Either
provider setting requires an `openrouter/` model, so clear whichever one is set in the same config
when a turn switches to a non-OpenRouter model.

For a repeatable comparison:

- Set `openrouter_provider` and keep it unchanged for the whole experiment. Also keep the model,
  harness, prompt, reasoning effort, and other generation settings unchanged unless one of them is
  the subject of the comparison.
- Save each `openrouter_request` event with the results.
- Run the paid provider check below before a larger experiment.

The provider-list command prints a base provider slug and an exact endpoint slug. A base slug such
as `moonshotai` allows every endpoint for that provider. A full slug such as
`moonshotai/mxfp4` selects that endpoint only. Either form works as the `openrouter_provider`
value. The paid check selects one known provider for every included model and fails if OpenRouter
reports a different provider:

```bash
make live-openrouter
```

Choosing a fixed provider removes one important source of variation. Model output can still vary
between requests, and a provider may update its serving software or model version. Record the date
and the reported provider details with experimental results.

#### Full provider routing control

`openrouter_provider` is a shorthand for one provider with fallbacks off. `openrouter_routing`
takes OpenRouter's whole
[`provider` object](https://openrouter.ai/docs/guides/routing/provider-selection) and sends it
verbatim, so nothing in that API is out of reach:

```python
spec = AgentSpec(
    name="kimi-first-party",
    model="openrouter/moonshotai/kimi-k3",
    harness="codex",  # or "claude-code"
    openrouter_routing={
        "order": ["moonshotai", "fireworks"],
        "allow_fallbacks": True,
    },
)
```

That asks for Moonshot first, then Fireworks, then anyone else who serves the model. The fields
OpenRouter accepts there:

| Key | Type | What it does |
| --- | --- | --- |
| `order` | list of slugs | Try these first, in this order. |
| `only` | list of slugs | Allow only these, in no particular order. |
| `ignore` | list of slugs | Never use these. Everyone else stays eligible. |
| `allow_fallbacks` | bool | Whether OpenRouter may go on to a provider outside an `order` list, rather than failing once the list is exhausted. Default true. An `only` list confines fallbacks to its own members, so this changes nothing there. |
| `sort` | `"price"`, `"throughput"`, `"latency"` | Order every eligible provider by that measure instead of OpenRouter's default. |
| `require_parameters` | bool | Skip providers that do not support every parameter the request sends. |
| `data_collection` | `"allow"`, `"deny"` | Exclude providers that may store the request. |
| `quantizations` | list | Only endpoints at these quantization levels. |
| `max_price` | object | A ceiling on what a provider may charge, per price component. |
| `zdr` | bool | Zero-data-retention endpoints only. |

A one-entry `only` is the strict pin again, since the list leaves nothing to fall back to.

The lists are honored, and they are used as lists. In the 2026-08-22 paid checks on Kimi K3,
`{"order": ["fireworks", "moonshotai"], "allow_fallbacks": true}` went to Fireworks first every
time, and one remote turn under `{"only": ["moonshotai", "fireworks"]}` was served by Fireworks for
one request and Moonshot AI for the next. So a set of several providers bounds who may serve a turn
without keeping the turn on one of them.

Whether the reported provider is checked depends on what the routing object allows. A closed
set — `only`, or `order` with `allow_fallbacks: false` — is checked against the provider
OpenRouter reports, and `provider_matches_request` is true or false. Anything that leaves the set
open (an `order` list with fallbacks on, an `ignore` list, a `sort` preference) reports
`provider_matches_request: null`, because the response could legitimately come from a provider the
request never named.

`openrouter_routing` and `openrouter_provider` cannot be combined: both write the same request
field, so one would silently win. The routing object goes in a `SessionConfig` or `TurnConfig` the
same way as the single slug.

Both slug forms above work in any of these lists. Two model-id suffixes do the same job as `sort`
without a routing object at all: `openrouter/z-ai/glm-5.3:nitro` sorts by throughput and `:floor`
sorts by price. They are part of the model id, so they need no other setting.

Important details:

- On the Claude Code harness, the CLI's own dollar figure is kept as `cli_reported_cost_usd`, next
  to the exact charge, for comparison. The Codex CLI reports no dollar figure at all, so there is
  none to keep there.
- Codex disables its built-in web-search tool for OpenRouter turns because OpenRouter rejects the
  tool format Codex sends. Shell, file, MCP, and skill tools remain available.
- Codex uses `low` reasoning when the spec leaves it unset because OpenRouter's Responses endpoint
  requires reasoning.
- `output_schema` also rides in the prompt on these turns (see [Structured output](#structured-output)).
- Reasoning events vary by model. Kimi K3 and both DeepSeek models emitted them during testing;
  GLM-5.3 did not.
- A plain string in `spec.system_prompt` replaces Claude Code's larger built-in system prompt. This
  can reduce token use when the task does not need that prompt's tool guidance.

Two paid tests cover these models, `make live-openrouter` locally and
`make live-openrouter-remote` on the sandbox runtime. [TESTING.md](TESTING.md) lists what each one
checks and what it costs.

## Dev: run locally, in-process

Same `Engine` + `Session` API as prod, no GCP beyond model access. Needs **Python 3.12** and Claude Code
auth in your environment (the Agent SDK drives the `claude` CLI). See a complete runnable agent in
[`examples/minimal`](examples/minimal).

```python
import asyncio
from agent_run import local

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
on sandbox alike ([full story](#deploy--session--turn-the-three-configuration-scopes)):

```python
from agent_run import SessionConfig, TurnConfig

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

> Keep keys out of `spec.env` for anything you `sandbox.deploy` — the spec is serialized *into* the deployed
> engine, so a value there is baked into the deployment and shared by every run. Per-invocation `secrets` (or
> Vertex routing, which needs no key at all) is the deployable answer.

To force the **subscription** instead, unset `ANTHROPIC_API_KEY` (and the other higher-priority variables) in
the shell you launch from — the toolkit cannot unset an inherited variable for you. To mirror **prod** locally,
set `CLAUDE_CODE_USE_VERTEX=1` with `ANTHROPIC_VERTEX_PROJECT_ID` and `CLOUD_ML_REGION`, which is what
`sandbox.deploy` bakes by default (see [Prod](#prod-deploy-once-look-up-and-run)); a deployed engine therefore
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
is an explicit opt-out of isolation: the directory is yours, concurrent sessions share it, and the toolkit
makes no promise that a session sees only its own files there. Name it something that reads like a workspace,
for the same reason the default leaf is called one — the path is what the agent sees, and a temp-looking cwd
tempts weaker models into `cd`-ing away from it.

Because the directory is yours, the toolkit stops writing to it on your behalf: `repos` is rejected (every
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
(the toolkit default) shadows entirely. Hooks are live callables, so they are a `local`-only argument —
`sandbox` runs the turn in a remote worker and rejects them, as does the `codex` harness.

## Consuming a run: wait, stream, or poll

`session.run(msg)` (and `session.send(msg)` to resume) returns a `Run` handle, consumable three ways — the
same on `local` and `sandbox`:

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
if (run := session.current_run) is not None:             # a turn still running (even under another process)
    async for ev in run:
        print(ev.kind, ev.summary)
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

**What the numbers count.** `result.cost_usd` is the turn's total spend in dollars (or `None` when the
backend can't price it — never a guess). `result.usage` is the toolkit-normalized token record, identical
in shape and meaning on every harness and both runtimes: a flat dict whose five keys are always present —
`input_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`, `output_tokens`,
`reasoning_output_tokens`. The three input buckets are disjoint (fresh / cache-read / cache-written), and
`reasoning_output_tokens` is the reasoning *share* of `output_tokens`, not a further bucket — so total
tokens = the three input buckets + `output_tokens`. A key is `None` when the backend doesn't report that number
(Anthropic has no reasoning split; OpenAI doesn't bill cache writes separately, the count is
informational). Both metrics cover the **whole turn**: subagent sessions and background-task
re-invocation segments included — on Claude Code aggregated from the CLI's complete per-model record, on
Codex recovered from the subagent rollout files (Codex reports no subagent usage on its wire). The
backends' own verbatim records stay on the result event's `raw`: `model_usage` and `cli_usage` on Claude
Code, `subagent_usage` (per-thread) on Codex. `result.num_turns` counts the **main agent's** model calls
(same unit on both harnesses): cumulative across re-invocation segments, subagent calls **not** included —
the same scope `max_turns` caps.

**Background tasks are honored.** If the agent starts a background job (Bash `run_in_background`) or arms
the Monitor tool and then ends its turn — the trained, efficient behavior for waiting on long processes
like a verification crawl — the run does **not** end there: the harness holds the session open and the
model is re-invoked when the task completes, exactly as in interactive Claude Code. Interim turn ends
surface as `awaiting_tasks` status events; the run's single `result` comes when no work is pending.
`spec.background_task_timeout` (seconds, default 3600) bounds how long a turn waits on still-running
tasks — size it to the longest job the agent legitimately waits on. Two accounting notes: `num_turns`,
`cost_usd` and `usage` are cumulative across these re-invocations (see "What the numbers count"), and
`max_turns` caps each invocation segment rather than the
whole run (`max_budget_usd` remains a global cap). Event-driven waiting instead of poll-loops is exactly
what keeps long crawls nearly free in turns.

On `sandbox`, live events and recovery records carry the submitted turn's ID. Reattaching a
session and immediately sending again cannot replay an earlier result. Per-worker warm
dispatch also matches the addressed worker: only its `turn_started` acknowledges pickup,
and late events from abandoned workers are ignored. Every storage poll is time-bounded.
Cold jobs and per-worker warm jobs have a watchdog that recovers a matching durable result
after job termination, or reports an error after its grace period. Legacy shared pools
have no per-worker job handle and rely on bounded polling.

When upgrading to turn IDs, redeploy engines before updating clients. Submission fails
before dispatch if the serving revision does not advertise the protocol. With pinned or
split traffic, every serving revision must support it. Older clients can drive upgraded
workers during the rollout; `Session.history()` continues to return all turns.

## Talking to a running turn: steer, interrupt, stop, exec

The session API is turn-based, and an interactive loop like Claude Code's needs two more things while
a turn is **running**: send a message into it, and cut it short without losing the session (a third,
looking into its workspace, is [below](#looking-into-a-running-turn-exec)). Both work the same on
`local` and `sandbox`, and from any process that holds the session id (`engine.get_session(session_id)`):

```python
run = session.run("Build the spider")

# The operator changes their mind while the agent works. Same Run, same worker, same workspace:
session.send("Actually skip the images, only product fields")            # steer: the model sees it at its next step
session.send("Stop that approach, use the sitemap instead", interrupt=True)  # interrupt, then continue from this message

# Or just stop. The turn ends through its normal end-of-turn path (checkpoint, transcript, accounting):
await session.interrupt()
assert session.stop_reason == "interrupted"     # idle and resumable
await session.send("OK, now do X")              # a normal turn, from the interrupted turn's checkpoint
```

- **`send()` on a running session** delivers the message into the running turn and returns the **same
  `Run`**: its events keep flowing and its single `result` covers everything. With `interrupt=False`
  the message is queued for the model's next step (both harnesses: Claude Code's mid-turn `query()`,
  Codex's `turn/steer`). With `interrupt=True` the model is interrupted first and continues from the
  message in the same harness session — no checkpoint round-trip, no second job. On an idle session
  `send()` is the usual resume (`interrupt` is ignored).
- **The `user` event is the acknowledgement.** Each delivered message appears on the stream, in the
  mirror and in `history()` as an event of kind `user` (the text as `summary`; `raw["message_id"]`,
  `raw["interrupt"]`), emitted when the harness hands it to the model. Pass your own
  `message_id=` to match it up; the worker dedupes on it, so a retried send after re-attaching never
  reaches the model twice. Until that event arrives the message is "waiting" — on `sandbox` well under a
  second to reach the worker, plus whatever tool call the model is in the middle of.
- **A message sent while the turn is still starting is queued, not refused.** Between `run()` and
  the worker's `control_ready` event (~1 s from the ready pool, 15–25 s when a sandbox has to be
  created) the session holds the message and posts it the moment the worker is ready — in order,
  acknowledged by the same `user` event, on the same `Run`. An `interrupt=True` message queued that
  early interrupts the harness right after it starts, so the model's first step is the message. If
  the turn ends before the queue drained (dispatch failed, the sandbox died) the messages are dropped
  and `result.warning` says how many. No retry loop needed on the caller's side.
- **`interrupt()` is "interrupt and stop"**: the harness stops what the model is doing and the turn
  ends with `StopReason.INTERRUPTED` — not an error; `cost_usd` / `usage` / `num_turns` are real, the
  checkpoint includes the interrupted turn's workspace changes, and the next `send()` resumes from it.
  Only when the worker cannot be reached (an engine deployed before this existed, or a cold job that
  has not started yet) or does not stop within the timeout does it fall back to cancelling the run:
  an error result whose `warning` says so, and no checkpoint.
- **Guards.** `run()` on a running session raises: a second concurrent turn under one session id would
  corrupt its checkpoint and transcript. `secrets` / `config` / `hooks` cannot change mid-turn and are
  rejected on a running `send()`. `send()` raises `ControlUnavailable` (a typed exception; nothing was
  delivered) only when a *ready* worker did not take the message: the platform proxy failed, or the
  turn ended between your `busy` check and the post. Mind that race on your side: `send()` on a session
  that has just gone idle is a resume, i.e. a new turn with no secrets — check `session.busy` right
  before, and catch `ControlUnavailable`.
- **From another process (`sandbox`).** `engine.get_session(id)` in the process that started the turn
  returns the live session. In a *different* process (a poller, or a worker adopting a job whose
  owner died mid-turn) the session holds no run, so its first `send()`, `interrupt()`, `exec()`,
  `run()`, `current_run`, `busy` or `last_result` reads the session's event record once — every
  event names its turn and sandbox — and, if a turn is still running there, **adopts** it: its
  events replay from the worker and then flow live on the session's `Run` — `session.current_run`
  hands you that `Run`, so an adopter consumes the events exactly as the owner would (`async for ev
  in run`, `await run`) instead of polling `last_result` — `send()` steers it (no second turn is
  started; `run()` raises as it would locally), `interrupt()` stops it, `exec()` probes it, and
  `last_result` lands at its end. The adopter also keeps the turn's tokens fresh and deletes the
  sandbox at the result (a few seconds late, in case the owner is still reading), so a job whose
  owner died is cleaned up — a poller that only asks `busy` or `last_result` gets the same, so a
  long turn does not lose model access for want of the right accessor. Adoption drives the turn on
  the running event loop; a plain sync poller with no loop keeps re-reading the record on each
  `last_result` until the result appears. An adopter that merely exits leaves the turn to its owner.
  With nothing running there `send()` is the usual resume. Same-process re-attach needs none of
  this: the engine hands back the live session.
- **Transport (`sandbox`).** The client POSTs the message to the worker's `/control` endpoint through
  the platform's proxy (sub-second); the worker feeds it to the harness and dedupes on `message_id`.
  The worker announces the channel with a `control_ready` status event at the start of the turn;
  until then the message waits in the client's queue (above). A message can only reach a running
  turn — nothing carries over to the session's next turn. On `local` the same channel is an
  in-process queue.

The interactive system-prompt suffix (`checkpoint=True`) still tells the model to end its turn for a
genuine decision; steering is additive — the way to talk to an agent that is *already* working.

### Looking into a running turn: `exec()`

`await session.exec(command)` runs a shell command in the running turn's workspace and returns its
output — a read-only probe for showing the agent's work *while* it works. The events are not enough
for that: a Claude Code edit arrives as old/new strings without context, a Codex change as paths only,
and a `git checkout` or a shell one-liner escapes both. One `git diff` in the workspace is exact and
harness-independent:

```python
run = session.run("Fix the spider")
while not run.done:
    r = await session.exec(
        "git add -A --intent-to-add . && git diff origin/main",   # what the agent has changed so far
        timeout=30,
    )
    if r.ok:
        render(r.stdout)                                          # e.g. refresh a change panel
    await asyncio.sleep(10)
```

- **What it returns.** `ExecResult(stdout, stderr, returncode, truncated, duration_ms)` (`r.ok` is
  `returncode == 0`). The command runs under `/bin/bash -c` with the agent's working directory as cwd
  (`cwd=` relative to it, or absolute). `timeout` in seconds (default 60; on `sandbox` at most 240,
  the platform proxy cuts a call at ~300 s) kills the command and reports `returncode` 124. Output is
  kept up to 500 000 characters per stream and `truncated` says when it was cut — a probe never
  fails for printing too much. The command's own failure is data, not an exception: read
  `returncode` and `stderr`.
- **Only a running turn has a workspace.** On `sandbox` the sandbox exists for the turn's duration
  and is deleted at the terminal event; `local` keeps the same rule so code written against it
  behaves the same remotely (between turns, read `session.workspace` directly there). Called right
  after `run()`, before the sandbox is known, `exec()` waits for the worker (~1 s from the ready
  pool, 15–25 s on a fresh sandbox) — so a poller can start at once. Called with no turn running,
  or after the turn ended, it raises `ControlUnavailable`; a poller treats that as "the turn is
  over" and reads `run.result`.
- **Read-only is your side of the contract.** Nothing polices the command; one that writes into
  the workspace races the agent.
- **From another process it works too.** A re-attached session's `exec()` first adopts the turn
  running there (see [From another process](#talking-to-a-running-turn-steer-interrupt-stop-exec))
  and probes its sandbox; with nothing running it raises `ControlUnavailable`, as usual.
- **Transport (`sandbox`).** The worker's `/exec` endpoint through the platform proxy (~0.3 s plus
  the command itself); the same endpoint `dev/live_smoke.py` uses to test the sandbox's isolation.

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
from agent_run import AgentSpec, local

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

- **`model`** — the model id for the chosen harness: a Claude id (`"claude-sonnet-4-6"`), an OpenAI
  id (`"gpt-5.6-luna"`), or an OpenRouter id (`"openrouter/z-ai/glm-5.3"`). See
  [Harnesses and models](#harnesses-and-models).
- **`openrouter_provider`** — pin one OpenRouter provider for an `openrouter/` model, so every
  request goes to the same one (see [OpenRouter models](#openrouter-models-either-harness)). Also
  settable per session and per turn.
- **`openrouter_routing`** — OpenRouter's whole `provider` object instead of one pinned slug:
  several providers, a deny list, fallbacks on, a price or throughput sort (see
  [Full provider routing control](#full-provider-routing-control)). Also settable per session and
  per turn.
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
- **`codex_config`** — extra Codex `config.toml` settings as a `{dotted.key: value}` mapping (strings,
  booleans, numbers, lists), for the `codex` harness only; `claude-code` ignores it with a status
  warning. See [Codex](#codex).

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
from agent_run import AgentSpec, RepoSource

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
| `OPENROUTER_API_KEY` | the local proxy, which is what calls OpenRouter | **no** — the CLI gets a per-run proxy token |

**The blast-radius model — the honest part.** A background agent can be steered by hostile input (a page it
scrapes, a file in a repo) into revealing whatever it can reach: environment variables, `.git/config`, a
token it was handed. The toolkit does **not** pretend to hide credentials from the agent. Instead it shrinks
the blast radius:

1. **Per-invocation, not baked/shared** — a run only ever has the secrets that *that* call passed, and
   (since the run-scoped GCS tokens below) can reach only its own staged objects, so one deployed engine
   serves many tenants without any of them sharing credentials.
2. **Scope the credentials** — prefer short-lived, narrowly-scoped tokens (a GitHub **App installation
   token** for one repo, a Bitbucket **repository access token**) over broad personal tokens. If a run is
   coerced, the exposure is limited to that one scoped token for that one run.
3. **Least secrets per run** — pass only what the task needs.

**LLM API key.** By default `sandbox` routes the model through **Vertex** with **hourly tokens** the client
mints from a predict-only service account (`agent-run-model@<project>`, created by `agent-run-gcp-setup`) and the
sandbox worker serves to Claude Code from a loopback metadata server (the client pushes a fresh one every
25 minutes while the turn runs, so long turns just work — see [Long turns](#gcp-setup--required-permissions)).
The token is not in the agent's environment, but the agent's shell can fetch it the way the CLI does — the
sandbox has no other identity — and it is worth `aiplatform.endpoints.predict` for at most an hour and
nothing else; see
[What the agent's shell can reach](#what-the-agents-shell-can-reach) below. `deploy(..., use_vertex=False)`
switches to API-key mode, where you pass `ANTHROPIC_API_KEY` as a per-invocation secret — then the agent can
read a long-lived key, so prefer Vertex for anything exposed to untrusted input. Locally the agent likewise inherits your shell's environment (including
your own `ANTHROPIC_API_KEY`, or — with no key set — your `claude.ai` subscription login: see
[which credential pays for a local run](#dev-run-locally-in-process)); local is a trusted-dev context.
An [OpenRouter model](#openrouter-models-either-harness) is the one case where a model key is never
in the agent's environment even in API-key mode.

**In transit & at rest.** Secret values travel in exactly one place: the HTTPS body of the turn the client
posts to the sandbox's worker through the platform's authenticated proxy. Nothing is persisted by the
platform — no job input, no message queue, no staged object. The toolkit **never** writes secret values to
an `AgentEvent`, to the event mirror or to a checkpoint — push tokens are scrubbed from `.git/config` before
a workspace snapshot and re-embedded on resume. Don't log the `secrets` dict yourself.

### What the agent's shell can reach

The agent runs inside a **gVisor sandbox with no Google identity**: the metadata server answers with a
tenant-project workload identity that is 403 on everything in your project (verified live 2026-09-10 and
re-checked by `make live-smoke`'s isolation check on every run). Whatever a prompt-injected agent finds in
its shell is therefore what the turn itself brought along, and the toolkit keeps that small:

- **Run-scoped GCS token.** The worker's GCS work (the event mirror, checkpoints, transcripts, artifacts)
  runs on a short-lived, downscoped token the client mints per turn — this bucket, only this run's object
  prefixes (`runtime/sandbox/scoped_gcs.py`). Another run's records are unreachable with it. The client
  refreshes it for turns longer than an hour through an object under the run's own prefix.
- **Model token.** An hourly Vertex token for the predict-only model service account (above), served by
  the worker's loopback metadata server (`GCE_METADATA_HOST` in the agent's env points at it): model calls,
  nothing else. Spend past the run's `max_budget_usd` is the only thing a coerced agent could add.
- **The turn's own secrets** — those the caller passed, routed as the table above says.

Sandboxes are created per turn and deleted at its end, so nothing a run leaves behind survives into the
next one; multi-turn continuity rides the checkpoint (a credential-scrubbed tar in your bucket). There is
no runtime service account, no per-worker message channel and no platform-persisted job input
(DESIGN.md §6 and §13).

## Pre-baked engine dependencies

Runtime `uv` (above) is great for experimentation, but for **pinned versions, a private index, or packages
you don't want re-fetched on every run**, declare them on the spec. `sandbox.deploy` bakes them into the
sandbox image (a `pip install` layer of the generated Dockerfile), so a deployed agent starts with them
already installed:

```python
spec = AgentSpec(
    name="data-agent",
    model="claude-sonnet-4-6",
    packages=["pandas==2.2.*", "httpx>=0.27"],   # pip/uv requirement specifiers, baked at deploy time
)
```

**The image's base layer is pinned in-tree.** `runtime/sandbox/_image.py` lists what every sandbox image
installs besides the toolkit (the `claude-agent-sdk` pin that ships the `claude` binary, `openai-codex` when
the Codex harness is baked, `google-cloud-storage`, `uv`, …). Your `packages` merge with these — a pin that
contradicts the base fails fast at deploy, before the build. The image tag is a content digest of the whole
build context (the toolkit's source, the spec, the requirements, the Dockerfile), so a deploy of an unchanged
spec from an unchanged toolkit finds its image in the registry and skips the build.

**`local` honors `packages` too**: `local.deploy` resolves them into a per-engine venv (via `uv`, with the
engine contract's Python 3.12 — uv provisions the interpreter if your machine lacks it) and activates it in
the agent's environment. Same spec, same starting packages on both backends — and since `uv` hardlinks from
its global cache, a warm local deploy takes seconds, so iteration stays fast. The venv is per-engine, so an
agent installing extras mid-run never leaks into other runs. Note this covers *Python package* parity;
OS-level parity (glibc, system libs) is what the [`dev/` parity image](dev/) is for. The image build runs on
your machine with Docker (~30 s warm, a few minutes cold) and every package adds pull time when a sandbox
starts, so pin only what genuinely needs to be pre-installed and lean on runtime `uv` for the rest.

> **Troubleshooting install issues locally.** A dev image under [`dev/`](dev/) mirrors the engine's install
> contract (Debian/glibc, Python 3.12, `uv`, `git`, your `packages`) so dependency failures surface with a
> shell at hand: `make parity-build PACKAGES="pandas==2.2.*"`, then `make parity-check` (or `parity-shell` to
> run your agent inside it). It reproduces the *install* environment, not the sandbox.

## Prod: deploy once, look up and run

Requires GCP setup — see [GCP setup & required permissions](#gcp-setup--required-permissions) below.

```python
# Ops / CI deploys once (rare): builds + pushes the agent's image (Docker), creates a template,
# fills a ready pool of two sandboxes.
engine = sandbox.deploy(spec, project="my-project", location="us-central1", warm_pool=True, pool_size=2)
engine.wait_until_warm()                          # block until a ready sandbox is in the pool (pool only)

# App code looks the engine up by name and runs — it never deploys:
engine = sandbox.get_engine("spider-builder", project="my-project", location="us-central1")
session = engine.start_session()
result = await session.run("/scrape https://books.toscrape.com title, price")   # default: wait for the result
```

A turn claims a ready sandbox off the pool (or creates one, ~20 s), hands it the turn over HTTPS and
streams the events straight back from it; the sandbox is deleted at the terminal event. `get_engine`
follows what the deploy recorded (`warm_pool`, the model identity, the baked spec) — pass
`warm_pool=False` to bypass a pool deliberately.

**What `deploy` waits for.** Pass `log=print` to follow it: image build and push, then `creating template …`,
a `template <name>: PROVISIONING for 30s` line every 30 s while the platform provisions, `template <id> is
ACTIVE`, pool fill. The template's provisioning time varies a lot — 12–20 s on most days, ten minutes on
others (observed 2026-09-15 for 4 CPU / 8 GiB templates while 1 CPU / 1 GiB ones took 12 s); `deploy` returns
the moment the template lists as ACTIVE rather than waiting for the platform's long-running operation, which
has been seen to complete nine minutes after that. A template that ends FAILED makes `deploy` raise with the
platform's reason (from the operation); FAILED versions stay visible in `engine.revisions()` with their
`state`, `get_engine` skips them (with a warning) and resolves to the newest ACTIVE version, and the next
successful deploy retires them. The platform itself gives up on a template after about 30 minutes of
PROVISIONING and fails it with a bare `INTERNAL` (seen five times in two days for 4 CPU templates, never for
1 CPU ones, with images and configs identical to templates that came up in 90 s); every such stall followed an
earlier stuck or FAILED template, and one completed 34 s after that FAILED template was deleted, so a failed
template appears to hold its warm-pool capacity until deleted. `deploy` therefore deletes every FAILED template
under the host instance before creating a new one (logging each), deletes its own template when it ends FAILED,
and after 32 minutes of PROVISIONING gives up with the operation name, deleting the stuck template so it does
not hold up the re-run (idempotent). If a create runs past ten minutes, `engine.revisions()` and
`delete_version()` on anything FAILED is the manual version of the same fix.

**The deploy record.** `deploy` writes `deploys/<name>/<template id>.json` under the output bucket
right after the template exists, before the pool is filled: the baked spec, the image, the model
service account, the pool settings. `get_engine` reads it; without it the handle can still address
the template but a Vertex-routed turn fails for lack of a model service account, so `get_engine`
**warns** at lookup when the record is missing. The usual cause is a `deploy` interrupted while it
waited for the platform to create the template (the template came up, the record was never written).
**The fix is to re-run the same `deploy`**: it is idempotent — the image is found in the registry and
not rebuilt, the newest template already serves it so no new version is created, and the record is
written.

### Deploy / session / turn: the three configuration scopes

An engine is deployed per agent *role*, but much of its configuration is naturally
per-**session** (which repo/ref this conversation operates on, the prompt or model variant
of an A/B experiment) or per-**turn** (budgets, structured output for the final turn) —
and requiring an image rebuild per combination would be unworkable. Configuration
therefore has three scopes, each with its own type, named by the lifetime of what it
configures:

| Scope | Type | Bound at | What belongs here |
|---|---|---|---|
| deploy | `AgentSpec` | `sandbox.deploy(spec)` | identity (`name`), image contents (`packages`, the agent `env`, the harness CLIs — `harnesses=(...)` bakes several), and the *defaults* for everything below |
| session | `SessionConfig` | `engine.start_session(config=...)` | the conversation's world: `repos`, `skills`, `mcp_servers`, `system_prompt`, `harness` (selects among the baked CLIs), `checkpoint`/`interactive`, `extra_env` — plus session-wide defaults for the turn knobs |
| turn | `TurnConfig` | `session.run(config=...)` / `send(config=...)` | the knobs the harness re-reads every invocation: `model`, `openrouter_provider`, `openrouter_routing`, `reasoning_effort`, `max_turns`, `max_budget_usd`, `max_buffer_size`, `background_task_timeout`, `permission_mode`, tool lists, `output_schema` |

Both config types are **sparse overlays**: a field left at `INHERIT` (the default) keeps
the value from the layer below; a set field replaces it wholesale (`extra_env` is the one
additive exception — it adds onto the deployed `env`). Deploy-only facts have no config
field at all, so "different `packages` per run" is a `TypeError`, not a silent no-op.

```python
from agent_run import RepoSource, SessionConfig, TurnConfig

engine = sandbox.get_engine("spider-builder", project=..., location=...)

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
persisted rather than accepting one. Only a definite "no such object" from the bucket means the
session has no config: a permission error, a timeout or a malformed record fails the re-attach
instead of silently running the baked spec, and the next attempt re-reads.

The contracts behind this:

* **The worker executes the merged spec** (deploy-baked ← session ← turn) and **echoes it
  as an `effective_spec` event** — the durable ground-truth record of what actually ran
  (drive post-mortems and replays from it, not from what you think you passed). The configs
  ride the turn's HTTP body; copies are persisted in GCS next to the session's records
  (30-day lifecycle) for the same reason, and `get_session` re-attach reads the session's back.
* **Fail closed**: a config field this image's toolkit doesn't know, or a `harness` the
  image doesn't bake, FAILS the turn with a terminal error — never a silent fall-back to the
  baked spec (a wrong-configuration run is exactly what this exists to prevent). Client and
  image must come from the same toolkit revision (the turn body is a wire contract).
* **No config → nothing recorded**: a plain `start_session()`/`run()` is byte-for-byte the
  pre-config behavior; the turn runs the deploy-baked spec with zero extra moving parts.
* **Deploy-time fields stay deploy-time**: `packages` and harness *availability* come from
  the image; a session selects among what is baked (`AgentSpec(harnesses=("claude-code",
  "codex"))` bakes both CLIs so sessions can pick either). Baked skills are reused as a
  staging fast path when the effective `skills` match the deployed ones; a differing
  declaration resolves from its own sources at run time.
* **No credentials in configs**: config objects are persisted and kept for debugging, so
  `SessionConfig` refuses a `RepoSource.url` embedding
  `user:token@` at construction — name the token via `RepoSource(auth=..., auth_user=...)`
  and pass the value in `run(secrets=...)`.

**Managing deployed engines** (control plane):

```python
sandbox.deploy(spec, project=..., location=...)   # build + push the image, create a template; ops/CI only
sandbox.deploy(spec, ..., warm_pool=True, pool_size=2, pool_max_wait_s=3600)   # + a ready pool, idle life 1 h
sandbox.deploy(spec, ..., resource_limits={"cpu": "8", "memory": "16Gi"})      # sandbox CPU/RAM (default 4 / 4Gi; max 8 vCPU)
sandbox.deploy(spec, ..., image="…-docker.pkg.dev/proj/agent-run/my-agent:tag")     # use an image you pushed; no build
sandbox.get_engine("spider-builder", project=..., location=...)   # look up by name (app code; addressing only)
sandbox.list_engines(project=..., location=...)   # discover what's deployed: {name, resource, versions}
engine.name, engine.version, engine.resource     # identity / template id / the template's resource name
engine.wait_until_warm(timeout=300)              # pools: wait for a ready sandbox before dispatching
engine.fill_pool(2)                              # top a pool up (after its sandboxes idled out)
engine.delete()                                  # tear down: every ready sandbox and every version's template
```

Deploying needs the Docker CLI logged into the registry (`docker login -u oauth2accesstoken
--password-stdin <region>-docker.pkg.dev` with an access token, or `gcloud auth configure-docker`) and
the Artifact Registry repo from [GCP setup](#gcp-setup--required-permissions). Session `fork()` is not
built yet.

### Versions: a template per deploy

Engine identity is `spec.name`. Every deploy whose image or resources differ from the newest version
creates a new immutable **sandbox template** displayed under that name — a template is a version — so app
code's `get_engine("spider-builder")` resolves the newest one without re-pointing at anything. A deploy
of an unchanged spec from an unchanged toolkit is a no-op (same image digest, same template). Once the new
version's pool is filled, the previous versions' idle sandboxes and templates are retired.

```python
engine.versions()                        # ['85216549199151104', …] — template ids, newest first
engine.revisions()                       # + image / create_time / state / which one is `current`
engine.version                           # the template this handle's turns run on
engine.delete_version("…")               # delete one version (its template + ready sandboxes)

sandbox.get_engine("spider-builder", ..., version="…")   # pin: turns run on that template
```

**`version=` routes.** Any template can be dispatched to, so a pinned handle keeps running its version
after a newer deploy — useful for a canary or a rollback (redeploy the old spec: same digest, same image,
a new template). "Serving" simply means "newest"; there is no traffic configuration.

**A display name is one lineage.** Because `get_engine(name)` resolves the newest template and a deploy
retires the previous versions once its pool is filled, an engine name is a single-revision resource by
design: two deployers on different toolkit revisions (or different specs) addressing the same name
replace each other's version. Consumers that run several toolkit revisions side by side — a `dev`
branch next to `main`, a pinned release next to a candidate — should put the revision in the name
(`spider-builder-b040d27`) and tear down the names they stop using with `engine.delete()`. Deleting an
engine touches only sandboxes created from *its* templates, so `x` can be torn down next to `x-<rev>`.

## Past jobs: listing sessions & reading history

Every `sandbox` run leaves a durable record, and the library reads it back — from any process, long after
the run:

```python
engine = sandbox.get_engine("spider-builder", project=..., location=...)

for info in engine.list_sessions():            # newest first: {"session_id", "sources", "last_file"}
    session = engine.get_session(info["session_id"])
    events = session.history()                 # the persisted AgentEvents, oldest first
    result = session.last_result               # reconstructed from the terminal event (None if unfinished)
    print(info["session_id"], result and result.text)
```

Where the record lives: the **event mirror** — `gs://<output_bucket>/events/<session_id>/*.jsonl`, small
batch files the worker writes with the run-scoped token as the turn runs, keeping **all turns** of a
multi-turn session. It is also the client's fallback stream when a sandbox stops answering (an OOM shows up
as the sandbox going away mid-turn: the run ends with an explained `sandbox_unreachable` error result unless
the mirror already holds the terminal result).

`session.last_result` on a re-attached session reads the record on first access and adopts a turn still
running there (["From another process"](#from-another-process-sandbox) above); without an event loop it
re-reads the record on each access until a result exists — prefer `current_run` / `run.done` for
in-flight runs. Caveats: the mirror is bucket-wide, so engines
sharing an output bucket see each other's sessions in `list_sessions`; the `local` runtime keeps no durable
event log (`list_sessions` shows its workdir's session dirs; `history()` raises).

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

## Observability: what you get, what you don't

The sandbox runtime keeps **one record**: the event mirror (`session.history()`, and the live stream).
There is no Cloud Logging step log and no Cloud Trace span tree any more — the sandbox has no Google
identity to ship telemetry with. What replaces them:

- **Events are the record.** Every turn streams `turn_started` (with the sandbox id), `effective_spec`,
  `workspace_ready`, the harness events and the terminal `result`; `session.history()` reads the same
  objects back from any process. Large payloads survive (the mirror has no per-entry size cap; the live
  channel pages).
- **A sandbox that dies** (OOM, TTL, deleted elsewhere) ends the run with an explained
  `sandbox_unreachable` error result within ~30 s — never a hang, never a silent retry. Raise
  `resource_limits` for agents whose turns build dependencies or import big projects (up to 8 vCPU;
  16 GiB verified).
- **Cost** is reported per run as `result.cost_usd` as before.

**CPU / memory.** The platform exposes **no** resource metrics for sandboxes (no Cloud Monitoring metric
type, no usage fields on the sandbox resource — checked 2026-09-16), so the worker samples **itself**: gVisor
mounts cgroup v1 accounting and reports the template's memory limit as `MemTotal`. Three outputs:

- **Per-session samples** — every 20 s (`AGENT_RUN_RESOURCE_SAMPLE_S` in the image env; `0` disables) to the
  session's **event mirror only**, not the live stream, so a watcher is not drowned and the record survives a
  mid-turn kill: after an OOM the last sample sits at most one interval before death. `session.history()`
  skips them (`history(include_samples=True)` keeps them); read them as rows with:

  ```python
  session = engine.get_session("<session-id>")      # or any session you already hold
  for row in session.resource_samples():            # oldest first
      print(row["time"], row.get("memory_current_bytes"), row.get("memory_limit_bytes"), row.get("cpu_usec"))
  ```
- **A visible warning** — the first time memory crosses 85 % of the limit, a `memory pressure: …` status
  event lands on the normal event stream, so a watcher sees trouble before the platform kills the sandbox
  at the limit (the fix: deploy with higher `resource_limits`).
- **Peak in every result** — `result.resources` (and the terminal result event's `raw`) carries
  `memory_peak_bytes` / `memory_limit_bytes` / `cpu_usec`, so completed *and failed* turns report their
  high-water mark for free.

The idle baseline with Claude Code is ~550 MiB (the CLI ~500 MiB, the worker ~110 MiB); what the samples
show above that is your agent's own work.

## GCP setup & required permissions

Everything below can be created by a team in their own project; the concrete values are the shared
`my-project` setup we use for testing.

> **One command sets all of this up:** `agent-run-gcp-setup --project <your-project>` (installed with the
> toolkit; plain ADC, no gcloud needed) audits a project against everything in this section, shows
> what's missing, asks for confirmation, applies it, and re-audits. It is **additive only** and
> idempotent — safe to run, and re-run, against existing non-empty projects. `--check` audits without
> changing anything (exit 0 iff ready); `--yes` skips the prompt (CI/agents); `--verify` proves the
> end state with a real throwaway deploy (image build + push with your Docker, a pool of one) + one Haiku
> turn (a few cents, ~5 min), and refuses to spend on the deploy while any check it depends on is still
> failing (the model check runs as a 1-token live probe in the first report, so a missing Model Garden
> enablement surfaces before any money is spent). Two things stay manual: enabling Claude in Vertex Model
> Garden (the tool live-probes each model — by default Haiku 4.5, Sonnet 5 and Opus 5 as required plus
> Fable 5 as optional/non-blocking; tune with `--model`/`--optional-model` — and links the exact console
> page for a missing one) and, on a project with APIs fully disabled, the Service Usage API bootstrap.
> The tables below remain the reference for what it grants and why.

Two identities take part; the sandbox itself has none:

**1. The operator identity** — you (a human or CI) *impersonate* the operator service account, or run as
your own identity, to drive the whole control plane: `sandbox.deploy`, `get_engine`, `list_engines`,
running turns. Grants (tighten to your policy):

| Role | Scope | Why |
|---|---|---|
| `roles/aiplatform.user` | project | create/list/delete sandbox templates and sandboxes, execute into them (`aiplatform.sandboxEnvironments.*`, `aiplatform.sandboxEnvironmentTemplates.*`), and the host `reasoningEngine` they hang off |
| `roles/storage.admin` | the output bucket | the records (event mirror, configs, checkpoints), the ready-pool roster, the lifecycle-rule update `deploy` performs, and minting the per-turn run-scoped GCS tokens (a downscoped token can only carry rights its source already has) |
| `roles/artifactregistry.writer` | the image repo | `deploy` pushes the agent image |
| `roles/iam.serviceAccountTokenCreator` | **on the model service account** | every turn mints a one-hour Vertex token for the sandbox by impersonating it |

Why `roles/aiplatform.user` and not a narrower custom role: the platform authorizes template calls with
`aiplatform.sandboxEnvironmentTemplates.{list,get,create,delete}`, permissions that (as of 2026-09-15) are
absent from IAM's public catalog — not testable on the project or on the host reasoning engine, listed by no
predefined role, so a custom role cannot carry them. A custom role with the published
`aiplatform.sandboxEnvironments.*` + `aiplatform.reasoningEngines.{create,get,list}` was tried live and
failed on the first template listing. `roles/aiplatform.user` evidently includes them through an unpublished
grant. Revisit when Agent Sandbox reaches GA.

The principal that impersonates the operator SA needs `roles/iam.serviceAccountTokenCreator` **on that SA**
(and on the model SA, if it drives turns as itself). Impersonation is the pattern for callers that already
have a Google identity (humans, GCP-hosted services, CI with workload identity). A production app running
*outside* GCP instead authenticates **as** the operator SA directly with a service-account **key** stored
in the app's secret store (`gcloud iam service-accounts keys create key.json --iam-account=<operator SA>`,
then point `GOOGLE_APPLICATION_CREDENTIALS` at it). A key is a long-lived credential, so prefer
impersonation or workload identity federation where they're available, and rotate keys you do hand out.

**2. The model service account** — `agent-run-model@<project>.iam.gserviceaccount.com` (created by
`agent-run-gcp-setup`; `sandbox.deploy(model_service_account=)` names another one). The sandbox runs the model
on a token minted for this account, and the agent's shell can read that token, so it holds **only** a custom
role with `aiplatform.endpoints.predict` (`agentRunPredict`) — model calls and nothing else. Never
`roles/aiplatform.user` here: it would hand the shell every sandbox and template in the project.

**Platform side**: the Google-managed **Agent Sandbox service agent**,
`service-<PROJECT_NUMBER>@gcp-sa-vertex-sandbox.iam.gserviceaccount.com`, pulls the agent image when a
sandbox starts and needs `roles/artifactregistry.reader` on the image repo. It gets nothing else; the
sandbox it starts runs as a zero-permission tenant identity.

**Prerequisites** (`agent-run-gcp-setup` creates them; `deploy` ensures the lifecycle rules):

- An output bucket `gs://<project>-agent-output` with uniform bucket-level access.
- An Artifact Registry Docker repo `agent-run` in the location (`sandbox.deploy(image_repo=)` names another).
- **Claude model access** — see the note below.
- The Docker CLI on the deploying machine, logged into the registry.

**Claude model access.** By default the toolkit routes Claude through **Vertex AI** with the model service
account's token. For that to work:

1. **Enable the Claude models you use in Vertex Model Garden** (accept the Anthropic terms once per project).
   Until a model is enabled, a deployed run fails with *"model … may not exist or you may not have access to
   it."*
2. Set `spec.model` to the id shown in Model Garden — for current Claude models that's the family alias (e.g.
   `claude-opus-4-8`), the same form you use locally. The toolkit defaults `CLOUD_ML_REGION` to `global`,
   where these models are served; override `vertex_region` at deploy if you need a specific location.
3. **Long turns need no setup.** IAM mints the model token for one hour, but the sandbox worker serves it to
   Claude Code from a loopback **metadata server** (the CLI's Google auth fetches it as on a VM and refreshes
   it itself when it nears expiry), and the client re-mints a token every 25 minutes and pushes it to the
   worker for as long as the turn runs. The one condition: a client process must hold the session while the
   turn runs — the one that started it, or one that re-attached with `get_session` (see
   [From another process](#talking-to-a-running-turn-steer-interrupt-stop)) — because the sandbox itself
   cannot mint tokens; a turn whose last client exits keeps running on its current tokens and loses model
   access within the hour. `max_turn_s` (a `deploy` knob, default 8 h) bounds the sandbox's lifetime, not
   the token.

Prefer an API key (e.g. for models you haven't enabled on Vertex)? Deploy with `use_vertex=False` and pass
`ANTHROPIC_API_KEY` as a per-invocation secret — the toolkit then uses the key (no Vertex routing), so any
model alias the key supports works. Note the security trade-off: in API-key mode the agent's process can read
a long-lived key, so prefer Vertex for anything exposed to untrusted input (see
[Secrets & security](#secrets--security)).

**Concrete shared setup** (`my-project`): location `us-central1` (Claude: `us-central1` + `global`);
operator SA `agent-runtime@my-project.iam.gserviceaccount.com`; image repo
`us-central1-docker.pkg.dev/my-project/agent-run`. The setup tool's default operator SA is
`agent-runtime@`, so audit this project with `agent-run-gcp-setup --project my-project
--operator-sa agent-runtime --check`; the tool discovers the ADC principal itself (a plain
`gcloud auth application-default login` included) and `--impersonator user:...` only names *other*
people to grant. Sketch for a fresh project:

```bash
PROJECT=your-project; REGION=us-central1; NUMBER=$(gcloud projects describe $PROJECT --format='value(projectNumber)')
OP="agent-runtime@$PROJECT.iam.gserviceaccount.com"        # operator SA you create
MODEL="agent-run-model@$PROJECT.iam.gserviceaccount.com"        # model identity you create
OUT="gs://$PROJECT-agent-output"

gcloud services enable aiplatform.googleapis.com artifactregistry.googleapis.com storage.googleapis.com \
  iamcredentials.googleapis.com --project $PROJECT
gcloud iam service-accounts create agent-runtime --project $PROJECT
gcloud iam service-accounts create agent-run-model --project $PROJECT
gcloud projects add-iam-policy-binding $PROJECT --member "serviceAccount:$OP" --role roles/aiplatform.user
# the model identity reaches Vertex only for model calls: a custom role, never roles/aiplatform.user
gcloud iam roles create agentRunPredict --project $PROJECT --stage GA \
  --title "agent-run model identity: model calls only" --permissions aiplatform.endpoints.predict
gcloud projects add-iam-policy-binding $PROJECT --member "serviceAccount:$MODEL" \
  --role projects/$PROJECT/roles/agentRunPredict
gcloud storage buckets create $OUT --project $PROJECT --uniform-bucket-level-access
gcloud storage buckets add-iam-policy-binding $OUT --member "serviceAccount:$OP" --role roles/storage.admin
gcloud artifacts repositories create agent-run --repository-format=docker --location=$REGION --project $PROJECT
gcloud artifacts repositories add-iam-policy-binding agent-run --location=$REGION --project $PROJECT \
  --member "serviceAccount:$OP" --role roles/artifactregistry.writer
gcloud artifacts repositories add-iam-policy-binding agent-run --location=$REGION --project $PROJECT \
  --member "serviceAccount:service-$NUMBER@gcp-sa-vertex-sandbox.iam.gserviceaccount.com" \
  --role roles/artifactregistry.reader
# who mints tokens: the operator SA mints model tokens; you impersonate the operator SA
gcloud iam service-accounts add-iam-policy-binding $MODEL --member "serviceAccount:$OP" \
  --role roles/iam.serviceAccountTokenCreator
gcloud iam service-accounts add-iam-policy-binding $OP --member "user:you@org.com" \
  --role roles/iam.serviceAccountTokenCreator
gcloud auth configure-docker $REGION-docker.pkg.dev
```

Then authenticate impersonating the operator SA (`gcloud auth application-default login
--impersonate-service-account=$OP`); `sandbox.deploy` pushes to the `agent-run` repo and mints model tokens
from `$MODEL` by default (pass `image_repo=` / `model_service_account=` for other names).

## Latency & cost (the `sandbox` path)

Platform realities the toolkit encodes (measured on Agent Sandbox, 2026-09-10/11; the `local` path has none
of them):

**Deploy** is a rare ops/CI action: an image build on your machine (~30 s with a warm Docker cache, a few
minutes cold), a push, a template (~20–80 s), and the pool fill (~15–35 s per sandbox, in parallel). App
code never deploys — it looks an engine up by name and runs.

**Running a turn** has two latency profiles:

| Path | Start latency | Use for |
|---|---|---|
| **Ready pool** (`warm_pool=True`) | **~1 s** to the first event, ~4–7 s to the result of a one-tool Haiku turn | interactive *and* long |
| No ready sandbox | **~20 s** (2 s to create, 12–30 s until the platform's proxy routes to it) | batch / one-shot |

A ready sandbox's worker is already running and reachable; a turn is one HTTPS round trip to hand it the
prompt, tokens and configs, and the events stream straight back from it (one long-poll per event batch,
~0.2 s proxy round trip). Sandboxes are created from Google's pre-warmed pool per template, so the ~20 s
of the no-pool path is route propagation, not boot. DESIGN.md §13 has the architecture and the measured
numbers.

**How `warm_pool=True` works.** `sandbox.deploy(spec, warm_pool=True, pool_size=N)` creates N sandboxes from
the new template, waits until each answers, and records them in a **roster** (client-owned GCS objects under
`pool/<template>/` in the output bucket). A turn takes the oldest ready sandbox off the roster (claimed
atomically, so several client processes share one pool), hands it the turn, and refills the pool in the
background; the used sandbox is deleted at the terminal event. `engine.wait_until_warm()` blocks until the
roster has a live entry. A turn that finds the roster empty creates a sandbox for itself (the ~20 s path),
never a stranded turn.

> _Keeping the pool full:_ a ready sandbox idles for `pool_max_wait_s` (a `deploy()` parameter; default a
> day), then the platform deletes it — **without replacement** (the only refill is the one after each
> dispatch; `engine.fill_pool(n)` re-warms ahead of time). Its TTL is set at creation and cannot be extended,
> so the toolkit creates pool sandboxes with `pool_max_wait_s + max_turn_s` of life: one claimed at the end
> of its idle life still has the whole `max_turn_s` (default 8 h) for its turn. Size `pool_max_wait_s` to
> your dispatch gaps.

**Event streaming.** The stream you consume with `async for ev in run` comes **directly from the sandbox**:
the client long-polls the worker's `/events` (held until something new exists, so an event is seen within
one proxy round trip of its emission) and the worker keeps writing the **GCS event mirror** as the durable
record — the fallback stream when a sandbox stops answering, and what `session.history()` reads. Nothing
reads Cloud Logging; there is no shared read quota to run into.

**Cost.** A template costs nothing while nothing runs on it; a sandbox bills while it exists
($0.085/vCPU-hour + $0.009/GiB-hour at the time of writing — $0.38/h for the default 4 vCPU / 4 GiB). A
ready pool is therefore idle compute you pay for continuously: `warm_pool=True` trades money for latency —
size the pool to your concurrency and `pool_max_wait_s` to your quiet gaps, and leave it off for batch
agents where a ~20 s start is fine. Google keeps a pool of pre-started containers per template (two
sandboxes created from a 34-hour-old template both had a PID 1 that was 34 hours old), and that pool is
Google's cost, not yours: ten idle 4 CPU / 4 GiB templates with no sandbox ever created from them were
kept for a full billing day (2026-09-17) and the project's Agent Platform Compute / Memory lines went
*down* that day (55 vCPU-h / 145 GiB-h, below the ready-pool sandboxes alone), where billed pools would
have added ~1000–1900 GiB-h. So a cold engine costs only its image storage; still delete engine versions
nobody runs, because each ACTIVE template competes for provisioning capacity. Turn sandboxes are deleted
at the terminal event, so a turn costs its own
duration. Tear a pool down with `engine.delete()`. Model token cost is the same either way and is reported
per run as `result.cost_usd`.

**Limits** (measured 2026-09-11, undocumented by the platform): at most 8 vCPU per sandbox (16 GiB works);
one proxied HTTP call may take up to ~5 min (the toolkit's long-poll holds 20 s); a request body over ~1 MB
fails (a prompt that large would); a response over ~2 MB fails (the worker pages events under 1 MB); sandbox
TTLs up to 30 days are accepted; `/tmp` has ~60 GB.

## Learn more

See [`DESIGN.md`](DESIGN.md) — the agreed architecture, the hard-won platform contracts (§6), the ports &
adapters seam (§7), and the phased roadmap (§11). The README grows with the code, phase by phase.

Contributing? [`TESTING.md`](TESTING.md) covers the testing ladder — the offline suite (`make test`), the
install-parity image ([`dev/`](dev/)), and **live validation** on the real sandbox platform (`make live-smoke`,
plus `make live-openrouter` and `make live-openrouter-remote` for the OpenRouter models): when each is
required and how to debug a live run.

Changes ship as tagged releases documented in [`CHANGELOG.md`](CHANGELOG.md) — its header spells out
the versioning policy (0.x minor releases may break; every breaking change carries update notes) and
the release checklist. If your PR changes behavior, add an entry under `Unreleased`.

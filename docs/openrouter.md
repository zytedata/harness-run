# OpenRouter models

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
[`pricing`](../harness_run/harness/pricing.py) module docstring). A turn OpenRouter reports no
charge for gets `cost_usd=None` on its result event, plus a `cost_unknown` status event, and its
`max_budget_usd` cannot be enforced. `RunResult.cost_usd` is `float | None` and carries that same
`None`, so an unreported charge stays apart from a turn that really was free.

The ⚠️ is about intermittency, not a permanent failure: under Claude Code, both DeepSeek models have
finished a turn without returning a final answer in earlier runs, while the newest local and remote
runs (2026-08-24) had both of them answer on the first attempt. The same models under Codex have not
shown it. So prefer Codex for DeepSeek v4 Flash and Pro when a single run has to produce an answer.
A Claude Code turn that hits this is reported as `error_no_final_text` rather than a success, so the
caller can retry or switch harness.

For an `openrouter/*` id the library does not list, only the context window is missing: Codex falls
back to generic model metadata and warns, and Claude Code assumes 200k tokens, so it compacts the
conversation earlier than the model needs. Add the model to `OPENROUTER_CONTEXT_WINDOWS` in
`harness/pricing.py` to fix that.

What can stop a new model is the OpenRouter account, not the library. An account restricted to
zero-data-retention endpoints refuses a model that has none, with HTTP 404 and a data-policy
message. The run fails honestly — the `openrouter_request` event records the 404 — but note that
the Claude Code CLI rewrites the message into a vague "model may not exist or you may not have
access", so read the recorded status rather than the result text.

Every model call goes through a local proxy the library runs for the turn, on either harness. The
provider choice reaches OpenRouter through it, and the exact charge comes back through it. The
`OPENROUTER_API_KEY` is passed per invocation and stays in the calling process: the CLI gets a
random per-run token for the proxy instead. Both harnesses also remove provider credentials from the
tool environment, and Codex has its automatic login shell disabled because it could reload keys from
`~/.bashrc`. [DESIGN.md](../DESIGN.md) explains why the proxy is there.

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

## Providers and repeatable experiments

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

The library sends this rule with every model request:

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

## Full provider routing control

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
- `output_schema` also rides in the prompt on these turns (see [Structured output](structured-output.md)).
- Reasoning events vary by model. Kimi K3 and both DeepSeek models emitted them during testing;
  GLM-5.3 did not.
- A plain string in `spec.system_prompt` replaces Claude Code's larger built-in system prompt. This
  can reduce token use when the task does not need that prompt's tool guidance.

Two paid tests cover these models, `make live-openrouter` locally and
`make live-openrouter-remote` on the sandbox runtime. [TESTING.md](../TESTING.md) lists what each one
checks and what it costs.

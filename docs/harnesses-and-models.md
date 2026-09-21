# Harnesses and models

A **harness** is the coding-agent loop; the **model** is what it calls. Two harnesses ship, and
both run on both backends (`local` and `sandbox`) through the same Engine/Session/Run API.

| Harness | Agent loop | Models it runs | Per-invocation secret |
| --- | --- | --- | --- |
| `claude-code` | Claude Code, via the Claude Agent SDK | Claude models (`claude-sonnet-4-6`, `claude-haiku-4-5`, …) | none on Vertex (the default), else `ANTHROPIC_API_KEY` |
| `codex` | OpenAI Codex, via the `openai-codex` SDK | OpenAI models (`gpt-5.6-sol` / `-terra` / `-luna`, `gpt-5.3-codex`) | `OPENAI_API_KEY` |
| **either one** | the same loop, pointed at OpenRouter | `openrouter/<vendor>/<model>` — Kimi, GLM, DeepSeek (see [below](openrouter.md)) | `OPENROUTER_API_KEY` |

The harness is selected for a session, and its CLI must be included at deploy time.
The model can change on each turn. One deployed engine can therefore serve several models:

```python
## One engine, several models — no redeploy:
session.run(task, config=TurnConfig(model="gpt-5.6-luna"))
session.send(task, config=TurnConfig(model="openrouter/z-ai/glm-5.3"))

## To let sessions pick the harness too, bake both CLIs at deploy:
spec = AgentSpec(name="either", model="claude-sonnet-4-6", harnesses=("claude-code", "codex"))
engine.start_session(config=SessionConfig(harness="codex"))   # selects among what is baked
```

Model ids are free-form strings passed to the harness — any id the underlying CLI accepts works, so a
new model needs no library release. `max_budget_usd` is enforced against a different number per
harness: Claude Code reports its own spend, the OpenAI ids listed above are priced offline by the
library, and OpenRouter ids use the charge OpenRouter reports. The four OpenRouter models listed
above have been validated live on both harnesses.

## Claude Code

`harness="claude-code"`. Auth resolves in the order documented under
[Secrets & security](secrets-and-security.md) — deployed engines route to Vertex by default, so no key is
needed; locally, `ANTHROPIC_API_KEY` or a logged-in `claude` CLI both work.

```python
spec = AgentSpec(name="code-agent", model="claude-sonnet-4-6", harness="claude-code")
```

## Codex

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
    "Add type hints to utils.py and run the tests",
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
  (fetched once per process, so new models are priced without a library release; a small baked table
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
- **OpenRouter models** reach the same surface — see below.


## Reasoning effort

`reasoning_effort` sets how much reasoning the model spends per response:
`"none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max"`; unset means the harness default. The
vocabulary is the union of both harnesses'; each maps levels only the other supports to its nearest own
(Claude: `minimal`/`none` → `low`; Codex: `max` → `xhigh`, with a status warning). It is a turn-scoped
knob, so it can be set on the spec, a `SessionConfig` or a `TurnConfig`.

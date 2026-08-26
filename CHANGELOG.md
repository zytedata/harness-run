# Changelog

All notable changes to `remote-agent-toolkit` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning is [SemVer](https://semver.org/)-style with the usual 0.x caveat:
**minor releases (0.X) may contain backwards-incompatible changes**; patch
releases (0.X.Y) do not. Every incompatible change is listed under a
**Backwards-incompatible** heading together with update notes describing how to
migrate.

Releases are git tags (`vX.Y.Z`) on `main`; install a specific release with

```
uv pip install "remote-agent-toolkit @ git+https://github.com/zytedata/remote-agent-toolkit@vX.Y.Z"
```

Release checklist: update this file (move the `Unreleased` section into a new
version heading with the date), bump `version` in `pyproject.toml`, commit,
tag `vX.Y.Z`, push the commit and the tag.

## Unreleased

### Backwards-incompatible

- `RunResult.usage` is now a toolkit-normalized token record: the same flat dict on
  every harness, whose five keys are always present — `input_tokens`,
  `cache_read_input_tokens`, `cache_creation_input_tokens`, `output_tokens`,
  `reasoning_output_tokens` — with `None` for a number the harness does not report
  (never a fake `0`). The three input buckets are disjoint; `reasoning_output_tokens`
  is the reasoning share of `output_tokens`. It also became complete: it covers the
  whole turn, subagent sessions and background-task re-invocation segments included.
  Previously it was each CLI's own dict, verbatim — on Claude Code that meant main
  conversation loop only and, on re-invoked turns, the final segment only (up to a
  ~2x undercount on subagent-heavy runs); on Codex it meant a different shape
  (inclusive `input_tokens`, `cached_input_tokens`, `total_tokens`) that missed
  collab-subagent threads entirely, since Codex reports no subagent usage on its
  wire ([openai/codex#14642](https://github.com/openai/codex/issues/14642)) — the
  harness now recovers those post-hoc from their rollout files, and a natively
  priced Codex `cost_usd` includes that spend too — each subagent thread priced at
  the model its rollout names (`raw["subagent_usage"][thread]["model"]`), the
  parent's rate standing in, with a `subagent_price_unknown` status event, for a
  model `pricing` does not know. The mid-turn `max_turns`/
  `max_budget_usd` checks still cannot see it while the turn runs, but the budget
  is re-checked at turn end, so subagent spend that crosses the cap yields
  `error_budget_exceeded` rather than `success`; a subagent still running at turn
  end is flagged by a `subagent_usage_partial` status event.
  **Update notes:** readers of Claude Code's verbatim dicts find them on the result
  event's `raw` — `raw["model_usage"]` (the CLI's complete per-model record) and
  `raw["cli_usage"]` (the old `usage` value). Codex readers: `cached_input_tokens`
  is now `cache_read_input_tokens`, `input_tokens` no longer includes the
  cached/cache-written share, `total_tokens` is gone (it is the three input buckets
  + `output_tokens`; `reasoning_output_tokens` is already inside `output_tokens`), and
  per-thread subagent detail is at `raw["subagent_usage"]`. The harness conformance
  suite now asserts the normalized shape. The harness code runs inside the engine,
  so a deployed gemini engine keeps emitting the legacy `usage` shape until it is
  redeployed with this version (the client passes results through unchanged), and
  events already recorded stay in the legacy shape. `num_turns` is unchanged and now documented: the
  main agent's model calls on both harnesses, cumulative across re-invocation
  segments, subagent calls not counted — the scope `max_turns` caps. ([#36])

### Added

- Both harnesses can run `openrouter/*` models with an `OPENROUTER_API_KEY`
  per-invocation secret. Kimi K3, GLM-5.3, and DeepSeek v4 Flash/Pro have known
  context sizes. Each OpenRouter response reports its selected upstream and exact
  charge as an `openrouter_request` event. Results and budget checks use that charge,
  and an OpenRouter model is never priced from the toolkit's own table: a turn
  OpenRouter reported no charge for gets `cost_usd=None` and a `cost_unknown` event
  (why an estimate cannot replace it: the `harness/pricing.py` docstring).
  `openrouter_provider` selects one OpenRouter provider per agent, session, or turn and
  disables fallbacks. Paid local and Gemini Agent Runtime checks cover both
  harnesses. ([#34])
- `openrouter_routing` carries OpenRouter's whole `provider` object instead of that one
  pinned slug — several providers, a deny list, fallbacks on, a price or throughput sort —
  and sends it verbatim. Same three scopes, and the two fields cannot be combined.
  `provider_matches_request` is only reported when the object names a closed set of
  providers (`only`, or `order` with `allow_fallbacks` false); an open set reports `null`
  instead of a verdict the request cannot support. Every `openrouter_request` event carries
  the object as `requested_routing`, including the one a pinned slug resolves to — the slug is
  turned into that object before the proxy starts. Because a config field is new, a client staging it
  against an older engine fails closed, as the same-revision rule requires. ([#34])
- Codex emits a `model_routing` status event from the app-server's thread record. ([#34])
- Every OpenRouter model call, on both harnesses, passes through a per-run localhost
  proxy, because the CLIs can neither send OpenRouter's provider field nor report what
  OpenRouter charged. It also enforces the selected model and the budget, and keeps the
  provider key in the parent process. See DESIGN.md.
  The proxy's lifecycle, the key lookup, the schema steer, the metadata drain and the
  `cost_unknown` event live once in `harness/_shared.py`; each binding keeps only how its
  own CLI is pointed at the proxy (environment variables for Claude Code, `--config`
  overrides for Codex). ([#34])

### Backwards-incompatible

- `RunResult.cost_usd` is now `float | None`, and defaults to `None`. It used to be
  `float` defaulting to `0.0`, so a run whose spend the toolkit could not determine
  reached the caller as `0.0` and read as a free run. The terminal event already
  carried `None` for that case, alongside a `cost_unknown` status event; the result
  object now carries it too. A run that died before reporting anything also reports
  `None` rather than `0.0`. **Update note:** code that does arithmetic or formatting
  on `result.cost_usd` needs a `None` check, e.g. `result.cost_usd or 0.0`. ([#34])

### Changed

- The harness SDKs are now pinned exactly: `claude-agent-sdk==0.2.130` and
  `openai-codex==0.147.0`, matching what a deployed engine installs. Both were
  validated together on the local and Agent Runtime paths, and `openai-codex`
  ships the Codex CLI itself, whose behavior moves between versions (0.147
  dropped the `chat` wire API the OpenRouter route used to have a choice
  about). Installing this library therefore fixes those two versions ([#34]).

### Fixed

- Claude Code turns that finish without a final assistant message now return
  `error_no_final_text` for OpenRouter models. This has occurred intermittently with
  DeepSeek v4 and previously looked like a successful run with placeholder text. ([#34])
- Claude Code now passes `output_schema` to the Agent SDK as its native JSON-schema
  output format and preserves `ResultMessage.structured_output` in the terminal event.
  Earlier versions only parsed the final text on the client, so the model received no
  schema constraint and SDK-provided structured data was discarded. OpenRouter turns
  also receive the schema in their prompt because its selected model may be responsible
  for following it. ([#34])

[#34]: https://github.com/zytedata/remote-agent-toolkit/pull/34
[#36]: https://github.com/zytedata/remote-agent-toolkit/pull/36

## 0.2.0 — 2026-08-24

### Added

- Configuration now has three scopes, one type each: the `AgentSpec` baked at
  deploy, a `SessionConfig` bound once at `engine.start_session(config=…)`
  (the conversation's world: `repos`, `skills`, `mcp_servers`,
  `system_prompt`, `harness`, `checkpoint`, `extra_env`, plus session-wide
  knob defaults), and a `TurnConfig` passed per turn to `run()`/`send()`
  (`model`, `reasoning_effort`, budgets, `permission_mode`, tool lists,
  `output_schema`). Both configs are sparse overlays: a field left at
  `INHERIT` keeps the value from the layer below. Full parity between the
  `local` and `gemini` backends; no more ~4 min engine redeploy to vary
  per-conversation or per-turn settings.
- The worker echoes the merged configuration it actually executed as an
  `effective_spec` event in the session's stream — the durable ground-truth
  record for debugging and replay.
- `AgentSpec(harnesses=("claude-code", "codex"))` bakes several harness CLIs
  into one image, so each session can pick its harness; selecting one the
  image doesn't carry fails the turn loudly.

(all [#16])

- `gemini.deploy(..., pool_max_wait_s=...)` sets how long an idle warm-pool
  worker waits for an assignment before exiting. The worker side always read
  `AGENT_POOL_MAX_WAIT_S`, but nothing plumbed it into the engine env, so
  the knob was unreachable; passing it without `warm_pool=True` now fails
  loudly instead of being swallowed by `deploy()`'s kwargs catch-all, and
  invalid values (zero, negative, NaN, infinity) are rejected at the
  `deploy()` boundary — before the pub/sub ensure, so bad input never leaves
  an orphaned topic/subscription behind ([#28]).
- `AgentSpec(transcript=True)` persists the harness's own transcript — the full
  per-turn record: usage, tool statuses, subagent trees, permission denials —
  and `Session.transcripts()` reads it back on both runtimes, keyed by `"main"`
  plus each subagent subpath. Reading a transcript used to require
  `checkpoint=True`, which also archives the entire working directory to blobs
  at every turn: hundreds of MB per run for an agent that writes a lot, paid
  purely to get at a JSONL. `checkpoint=True` still implies `transcript`
  (resume needs the transcript), so existing specs are unaffected. On `gemini`
  the flag is what opens the checkpoint bucket, so a transcript-only spec needs
  a redeploy with an `output_bucket`. `transcript` on its own is purely
  observational: `send()` still needs `checkpoint=True` to continue a
  conversation. `transcripts()` raises when the spec persists nothing and reads
  `{}` when persistence is on but nothing is written yet, so an empty result is
  never a misconfiguration in disguise; the `codex` harness persists its
  conversation under `checkpoint` alone, so it reads `{}` there and the run says
  so with a `spec_warning` event. Treat what `transcripts()` returns as
  sensitive: unlike events, it is the verbatim record of everything the agent
  saw ([#26]).
- `run(hooks=…)` / `send(hooks=…)` pass Claude Agent SDK hook callbacks for one
  turn, so a caller can observe or gate every individual tool call — a
  `PreToolUse` hook fires under every `permission_mode`, unlike the SDK's
  `can_use_tool`, which the default `bypassPermissions` shadows entirely. Hooks
  are live callables, so they ride the run plane next to `secrets` rather than a
  config overlay (configs are serialized data) and are `local`-only: `gemini`
  runs the turn in a remote worker and rejects them, and the `codex` harness
  fails the turn rather than run it with the hooks never called ([#25]).
- `local.deploy(spec, workspace=…)` (and `local.run(…, workspace=…)`) runs every
  session of that engine in a directory you name instead of a per-session
  `<workdir>/jobs/<session-id>/workspace`. The cwd is part of the agent's system
  prompt, so a suite of short sessions pays prompt-cache creation on every one of
  them; one shared cwd keeps the prefix identical and the cache warm (measured 2x
  on a 111-attempt suite). The directory is yours: sessions are no longer isolated
  from each other there, `repos` is rejected (they would all clone to the same
  path), and `checkpoint=True` checkpoints the conversation only, so a resume
  continues in the directory as it stands instead of restoring a snapshot over it.
  `gemini.deploy(workspace=…)` raises — a worker's cwd is its own `/tmp` ([#29]).
- `run_harness_conformance()` is the `Harness` port's conformance suite, the
  sibling of `run_session_store_conformance()`: hand it a callable that runs one
  turn and it asserts what callers read off `AgentEvent.raw` beyond the event's
  own fields — an `init` status event and the terminal `result` event's spend
  (a float, or `None` when the backend cannot price the run), usage and turn
  count. `raw` is documented as a pass-through, so a harness or SDK reshaping
  it broke downstream readers silently; run this against your own `Harness`
  implementation, or against a shipped one after a toolkit or SDK upgrade, to
  catch that at test time. `run_claude_code_harness_conformance()` layers
  Claude Code's stricter promise on top — the `init` event's backend payload
  under `raw["data"]`, with the session id in it — which Codex's `init` event
  does not carry ([#27]).

### Changed

- Claude Code runs now pass `--strict-mcp-config`: MCP servers come from
  `mcp_servers` in the spec/configs only. MCP config is loaded outside the
  setting sources, so the existing project-only isolation did not stop the CLI
  from picking up a `.mcp.json` sitting in the workspace — and the workspace
  is a cloned repo, which could ship a `stdio` server (an arbitrary command)
  for the agent to run. A repo whose `.mcp.json` servers *are* wanted must
  mirror them into `mcp_servers` ([#21]).
- The pinned engine `google-cloud-aiplatform` is now 1.165.1 (Google's release
  made fresh venvs drift past the old `==1.164.0` pin, failing
  `verify_deploy_env()` — and CI — by design). Deploy venvs must carry 1.165.1
  from now on (`uv sync` suffices); already-deployed engines are unaffected,
  the pin is baked into their image. Verified with a clean `make live-smoke`
  ([#24], [#33]).
- The default idle life of a warm-pool worker is now a day, up from 30
  minutes. An idle-expired worker exits **without replacement**, and a pool
  that drains to empty never self-recovers (the post-dispatch refill worker
  claims the pending dispatch itself) — so the short default silently turned
  any pool quiet for half an hour permanently cold. Note the cost implication
  on redeploy: idle workers now bill for up to a day; pass `pool_max_wait_s`
  to dial it back for pools with steady traffic ([#28]).

### Fixed

- A single oversized message on the Claude Code CLI's stdout no longer destroys the
  whole turn. The Agent SDK caps one NDJSON message at 1 MiB and raises from inside
  its read loop when a message exceeds it, so the harness generator died mid-run and
  the turn ended as `agent run failed: Failed to decode JSON: JSON message exceeded
  maximum buffer size of 1048576 bytes` with no result — everything the agent had
  done was discarded. Nothing plumbed the SDK's `max_buffer_size` option, so the
  1 MiB default was in force; an agent that `Read`s a screenshot (base64-encoded into
  one line) or gets a large tool result hit it routinely. The cap is now
  `AgentSpec.max_buffer_size`, default 32 MiB, and settable per session or per turn
  like the other invocation knobs. `claude-code` only: the Codex app-server SDK
  frames its own stream and has no equivalent ([#32]).
- Structured output is no longer lost when a background-task notification arrives
  after the agent has already delivered its answer: the model's reply to the stale
  notification became the turn's final message, and structured parsing — which reads
  the final message only — returned nothing, so a successful run read as having no
  deliverable. The final result event now carries the displaced segment-boundary
  results' texts (`raw["segment_summaries"]`, newest first) and result building
  falls back over them; `RunResult.structured_output_recovered` marks a recovered
  value ([#20]).
- An unset `system_prompt` now gives the Claude Code harness its own preset
  prompt, instead of an *empty* system prompt (the Agent SDK's `None` means
  `--system-prompt ""`). Previously a default spec produced a bare agent on
  non-interactive runs — no Claude Code identity, no `CLAUDE.md` — while
  interactive runs got the full preset; now `None` behaves like
  `SystemPrompt.inherit()` on both harnesses. Note the behavior change on
  redeploy: default-spec agents get the preset prompt and start honoring
  `CLAUDE.md` from cloned repos ([#22]).
- A second `local.deploy()` into the same `workdir` no longer crashes on
  provisioning the engine venv (uv ≥ 0.12 errors on `uv venv` over an existing
  venv), which broke cross-process re-attach — `get_session`, checkpoint
  resume — for specs with `packages`. The existing venv is now reused (declared
  packages are still re-applied onto it), so an agent's mid-run installs
  survive re-attach; the `UV_VENV_CLEAR=1` workaround discarded them. Reuse
  requires the venv's Python to match the engine contract's at major.minor
  (checked via `pyvenv.cfg`) — an out-of-contract or interpreter-less venv is
  re-provisioned from scratch. Note convergence is one-way: packages *removed*
  from the spec stay installed in a reused venv; delete the workdir's `venv/`
  to rebuild from the spec alone ([#23]).
- The synchronous `local.run(spec, message, …)` convenience now forwards
  `config` and `hooks` to the turn it runs; both were silently swallowed by the
  backend-symmetry `**_` catch-all, so a `TurnConfig` passed there had no
  effect ([#25]).

### Backwards-incompatible

- `get_engine(spec=…)` is removed, with no deprecation shim — passing `spec`
  raises a `TypeError` pointing at the replacement ([#16]). It was
  only ever a client-side parsing hint; runs always executed the
  deploy-baked spec. **To update:** `get_engine(…)` now only addresses an
  engine; bind a `SessionConfig` at `start_session` for anything the old
  spec was supposed to change — including `output_schema`/`checkpoint`,
  which also restore the client-side structured parsing and idle
  stop-reason the old hint provided. Per-turn knobs go to
  `run(config=TurnConfig(…))`.
- Engines deployed from an older toolkit revision keep working with new
  client code **as long as you pass no configs** (nothing config-related
  rides the invocation then). To *use* `SessionConfig`/`TurnConfig` against
  one, redeploy it first: an old worker predates the fail-closed contract,
  so it would silently run its baked spec instead (and on the cold path the
  unrecognized directive line would additionally leak into the prompt).

[#16]: https://github.com/zytedata/remote-agent-toolkit/pull/16
[#20]: https://github.com/zytedata/remote-agent-toolkit/pull/20
[#21]: https://github.com/zytedata/remote-agent-toolkit/pull/21
[#22]: https://github.com/zytedata/remote-agent-toolkit/pull/22
[#23]: https://github.com/zytedata/remote-agent-toolkit/pull/23
[#24]: https://github.com/zytedata/remote-agent-toolkit/pull/24
[#25]: https://github.com/zytedata/remote-agent-toolkit/pull/25
[#26]: https://github.com/zytedata/remote-agent-toolkit/pull/26
[#27]: https://github.com/zytedata/remote-agent-toolkit/pull/27
[#28]: https://github.com/zytedata/remote-agent-toolkit/pull/28
[#29]: https://github.com/zytedata/remote-agent-toolkit/pull/29
[#32]: https://github.com/zytedata/remote-agent-toolkit/pull/32
[#33]: https://github.com/zytedata/remote-agent-toolkit/pull/33

## 0.1.0 — 2026-08-07

Initial release. What's in the box:

### Agent definition

- `AgentSpec`: one declarative spec for an agent — system prompt, skills
  (local or packaged), MCP servers (local or remote), extra Python/apt
  packages, git repositories to provision (with host-aware auth), per-run
  secrets, structured `output_schema`, model override, `reasoning_effort`
  ([#3]), and container `resource_limits` (CPU/memory, [#6]).
- Harnesses: Claude Code (via the Claude Agent SDK) and Codex (OpenAI models,
  [#2]) behind the same `AgentSpec`/engine/session API.

### Run planes

- Local: run the same spec on your machine for fast iteration; a dev parity
  image mirrors the engine's install contract.
- Gemini Agent Runtime (Vertex AI Agent Engine): `deploy()` /
  `get_engine()` / sessions with multi-turn `run()`, real interrupt,
  checkpoint/resume (canonical-UUID session ids, [#1]), and agent versions —
  `deploy` creates revisions and `get_engine(version=…)` pins one ([#13]).
- Warm pool: pre-provisioned workers claimed per turn for lower latency,
  with readiness signaling, pre-warm tuning, and robust cross-process
  teardown.
- Live event streaming over a GCS mirror (quota-free); Cloud Logging is
  emit-only. Clock-free turn boundaries prevent stale-result replay on fast
  resume. Out-of-order live logs are reconciled on history recovery
  ([#10], [#15]).
- Worker CPU/RAM self-sampling with `Session.resource_samples()` for OOM
  forensics ([#7]).

### Deploy reliability

- The platform-critical engine layer (aiplatform, cloudpickle, pydantic,
  google-adk, telemetry stack, …) is pinned via an in-tree
  `constraints.txt`, merged into engine requirements client-side;
  `verify_deploy_env()` fails fast when the deploying venv drifts from the
  pickle-coupled pins ([#18]).
- The deploy target project/location is pinned into the pickled `AdkApp`
  (`build_adk_app`), so engine telemetry can no longer follow the deployer's
  stray local gcloud default into the wrong project (persistent span/metric
  export 403s) ([#17]).
- Retry-safe secrets handoff: per-invocation secrets are deleted at turn
  end, not on first read ([#5]).

[#1]: https://github.com/zytedata/remote-agent-toolkit/pull/1
[#2]: https://github.com/zytedata/remote-agent-toolkit/pull/2
[#3]: https://github.com/zytedata/remote-agent-toolkit/pull/3
[#5]: https://github.com/zytedata/remote-agent-toolkit/pull/5
[#6]: https://github.com/zytedata/remote-agent-toolkit/pull/6
[#7]: https://github.com/zytedata/remote-agent-toolkit/pull/7
[#10]: https://github.com/zytedata/remote-agent-toolkit/pull/10
[#13]: https://github.com/zytedata/remote-agent-toolkit/pull/13
[#15]: https://github.com/zytedata/remote-agent-toolkit/pull/15
[#17]: https://github.com/zytedata/remote-agent-toolkit/pull/17
[#18]: https://github.com/zytedata/remote-agent-toolkit/pull/18

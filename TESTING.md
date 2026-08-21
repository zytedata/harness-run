# Testing — from unit tests to live validation

Three rungs, cheapest first. Every change runs rung 1; run the later rungs when your change
can only break in ways the earlier rungs can't see.

| Rung | Command | Catches | Cost |
|---|---|---|---|
| Offline tests | `make test` | logic, event plumbing, contracts we encode | ~40 s (parallel), free |
| Install parity | `make parity-build` / `-check` | dependency/install/glibc breakage | ~1 min, free |
| **Live validation** | `make live-smoke` | **platform-contract breakage** | ~10 min, ~$0.10 + build |
| Model-provider check | `make live-openrouter` | provider-contract breakage (OpenRouter) | ~1 min, a few cents |
| Model-provider check, remote | `make live-openrouter-remote` | the same models + remote visibility on Agent Runtime | ~8-15 min, ~$0.30 |
| Model attribution | `make live-attribution` | did the turn run the model we asked for — both harnesses | ~10 s, a few cents |

## 1. Offline tests (`make test`)

`make test` runs with `-n auto` (pytest-xdist). Each worker is its own process, so the
autouse fixtures, the pricing cache and the signal-handler test stay isolated; `make
test-serial` gives readable output when debugging. The wall clock is bounded by a few
deliberate poll-cadence tests; the slowest takes about 15 seconds. Expect about 40 seconds
overall.

The pytest suite makes **no network calls**: no GCP, no model, no subprocesses talking to
real services. Backends are exercised against the fakes in [`tests/fakes.py`](tests/fakes.py) /
[`tests/codex_fakes.py`](tests/codex_fakes.py), storage against `LocalBlobStore`, logging
against `InMemorySink`. Useful conventions when adding tests:

- **Patch the seam, not the internals** — e.g. monkeypatch `handoff.GcsBlobStore` to
  `LocalBlobStore`, or `eventsink.CloudLoggingSink` to `InMemorySink`, and drive the real
  code path above it.
- **Re-attach sessions by id** — `GeminiSession(engine, "sid")` constructs a handle without
  any platform call; `engine.start_session()` would call the live sessions API.
- Ruff must pass too: `make lint`.

## 2. Install parity (`dev/` image)

Catches install-time breakage (wheels, glibc, uv resolution) locally in seconds instead of
through ~10-minute cloud rebuilds. See [`dev/README.md`](dev/README.md).

## 3. Live validation on Gemini Agent Runtime

### Why offline green isn't enough

The platform's behavior changes **server-side, with zero client changes** — the offline
suite stays green through it. Measured example (2026-07-28): new engines' job workers
stopped resolving a default `class_method` and stopped exporting the managed-session env
var, and the job runner started kill-and-retrying workers — all discovered live, none
visible to any local test. The §6 contracts in [`DESIGN.md`](DESIGN.md) exist because of
such findings; live validation is how we keep them true.

**Run a live check when your change touches:** deploy packaging or `_deploy.py` contracts,
the pickled ADK app template or a Google SDK migration, `class_method` dispatch, the
secrets handoff lifecycle, warm-pool dispatch, or the event-sink/Cloud Logging plumbing.
Prose-only / pure-logic changes with solid offline coverage don't need it.

### The standard smoke test

```bash
make live-smoke                    # both paths; or MODE=cold / MODE=warm
PROJECT=my-proj LOCATION=us-central1 make live-smoke   # non-default project
```

[`dev/live_smoke.py`](dev/live_smoke.py) deploys **throwaway engines from your checkout**
(named `ratk-smoke-{cold,warm}-<you>`), runs one tool-using Haiku turn through each dispatch
path, and tears everything down in `finally`:

- **cold** — `run_query_job`, the production default: every job provisions its own worker
  (~2.5 min startup before the turn runs).
- **warm** — pub/sub dispatch to a pre-warmed pool worker (~4 s to first observed event).

Pass criteria per engine: terminal result with `error=False`, `turns > 0`, and the expected
answer in the text. The `*-session-config` / `*-turn-config` checks additionally exercise the
config transport on a second session per mode: the session-config turn must carry the
run-time system prompt's marker (the worker ran the session's config, not the deploy-baked
spec), the turn-config turn must return `structured_output` parsed via a per-turn
`output_schema` **and** still carry the marker (the session config persists across turns),
and both turns must stream the worker's `effective_spec` echo with the config pointers.
Typical numbers: deploy ~3.5–4 min (the two run in parallel), cold turn ~3 min end-to-end,
warm ~1 min, a few cents of model spend. Exit code is non-zero on any FAIL, so you can gate
on it.

Events stream via the **GCS mirror** (no read quota, no ingestion lag — see DESIGN §6), so
concurrent tests don't contend. Against an engine deployed *before* event streaming, all of
a turn's events arrive in one batch with the result (its mirror was written at end-of-turn)
— redeploy it for live streaming.

Prerequisites: the GCP setup from the
[README "GCP setup & required permissions"](README.md#gcp-setup--required-permissions)
(ADC login, IAM grants, Haiku enabled in Vertex Model Garden). Defaults target the shared
`my-project` test project.

### The revision probe (`make live-revisions`)

[`dev/live_revisions.py`](dev/live_revisions.py) covers what the smoke test can't: the
**control plane's versioning**. It deploys one throwaway engine (`ratk-rev-<you>`) twice and
asserts that the second deploy *updates* it into a new runtime revision rather than creating
a second engine, that traffic follows the newest revision, that `set_traffic` rolls back and
a turn still runs, that `get_engine(version=…)` accepts the serving revision and rejects a
non-serving one, and that `delete_version` prunes. Run it when you touch deploy, versioning,
or traffic config. ~10 min: the two builds are **sequential** (the second is the update under
test), so it costs about the same wall-clock as the smoke test's parallel pair.

### The OpenRouter model check

```bash
OPENROUTER_API_KEY=... make live-openrouter
MODELS=openrouter/z-ai/glm-5.3 make live-openrouter  # one model, both harnesses
```

This paid local test runs every model on both harnesses. Each model must call a shell tool,
return its output, produce schema-valid structured output, hide provider credentials from the
tool, report its selected upstream, and use OpenRouter's exact cost. It also checks resume, a
missing preset error, and a tiny budget cap on both harnesses.

To verify a real provider pin, supply a preset model and its expected provider:

```bash
OPENROUTER_PRESET_MODEL="openrouter/moonshotai/kimi-k3@preset/kimi-firstparty" \
OPENROUTER_EXPECTED_PROVIDER="Moonshot AI" \
make live-openrouter
```

The test reads every `openrouter_request` event and fails if the provider differs. Set
`SERIAL=1` for ordered output while debugging.

**It costs real money** (a few cents a pass) and needs a key, so run it **by hand,
sparingly, locally**. It must never run in CI: `pytest -q` stays free and credential-less
(see §1 and `.github/workflows/ci.yml`) — the offline tests pin the config the harness
emits, and that is what CI checks.

### The OpenRouter model check on Agent Runtime

```bash
OPENROUTER_API_KEY=... make live-openrouter-remote
```

This paid remote test repeats the local checks on Gemini Agent Runtime. One engine contains both
CLIs and serves every model through per-turn overrides. Every model runs a structured-output turn
on both harnesses. Resume, preset errors, and budget caps also run on both harnesses.

It also checks the remote-only surface: the worker's `effective_spec` echo names the model,
`session.resource_samples()` returns worker CPU/RAM, `memory_peak_bytes` is stamped on the
terminal result, `session.history()` replays the events, and **that session's** Cloud Trace
root span carries the model and its cost. It needs
`uv pip install google-cloud-trace` (dev-only, not a toolkit dependency). The trace check
reports SKIP when the package is missing.

The engine is deleted in `finally`; a failed teardown prints loudly, because an engine
bills while it exists. `KEEP=1` leaves it up for debugging and hands you the cleanup.

**Costs real money** (mostly the build) and usually takes **~8-15 min**. Checks run
concurrently; build time varies widely. Give it a generous timeout. Teardown
also runs on SIGTERM/SIGINT, because a `timeout` that fires mid-run would otherwise leave an
engine billing.

DeepSeek v4 sometimes returns no final message under Claude Code. Flash produced this most
often in the earlier measurements; Pro produced it on both attempts in the final remote
validation. The test retries one soft failure and reports when the retry was used. It stays
failed when the retry also has no answer.

### Model attribution: did we run what we asked for?

```bash
OPENROUTER_API_KEY=... OPENAI_API_KEY=... make live-attribution
```

`dev/live_model_attribution.py` checks evidence returned by the CLI, app-server, and
OpenRouter:

- **claude-code** — the CLI's `system/init` message names the model it resolved.
- **codex** — the app-server's `thread.read()` names the thread's bound `model_provider`,
  surfaced as a `model_routing` event. This is what proves an OpenRouter override took
  effect; a mismatch is reported as `matches_request: false`.
- **OpenRouter on both harnesses** — `openrouter_request` names the selected upstream and
  reports exact cost for each model response.

Checks run concurrently and cost a few cents. `SERIAL=1` gives ordered output.

### Writing a bespoke live probe

When the smoke test doesn't cover your change (e.g. validating crash/retry behavior, or a
new event field), follow the same pattern — it's what keeps live testing safe and cheap.
Worked examples in [`dev/`](dev): `live_two_turn_probe.py` (two turns on one session over
the event stream — the probe that caught the stale-result replay bug) and
`live_warm_latency_probe.py` (wait_until_warm + measured dispatch→first-event latency):

- **Throwaway, named engines**: suffix with something identifying (`-itest`, your name) so
  leftovers are attributable; never point a probe at someone's standing engine.
- **Cheap model, tight caps**: Haiku, low `max_turns` / `max_budget_usd` — the platform
  round-trip is the subject, not the model's work.
- **Teardown in `finally`**, warm pools with `delete_pool_resources=True` (idle pool
  workers are long-running billed jobs; `engine.delete()` cancels them first).
- **Check for leftovers** after any failed run — engines bill while they exist:

  ```python
  from remote_agent_toolkit import gemini
  for e in gemini.list_engines("my-project", "us-central1"):
      print(e)
  ```

### Debugging a live run

- **Server-side truth is the steps log**, not the client tail. Every worker event lands in
  Cloud Logging as `remote_agent_toolkit_steps`, filter
  `labels.session_id="<adk session id>"`. The client's streamed events are the same records
  *after* ingestion.
- **Never infer server timing from client event arrival** — Cloud Logging ingestion lags by
  minutes under load. For timing claims, compare wall-clock at the client against the
  timestamps *inside* the step-log entries.
- **Job containers run in a Google tenant project**: their Cloud Run logs/metrics and kill
  events are invisible from our project. The worker's cgroup self-sampling
  (`remote_agent_toolkit_resources` log, `session.resource_samples()`, and the
  `memory_peak_bytes` stamped into the terminal result) is the only usage record — see the
  [README monitoring section](README.md#monitoring-job-cpuram-oom-forensics).
- A session is also inspectable in the console playground:
  `https://console.cloud.google.com/vertex-ai/agents/agent-engines` → engine → session.

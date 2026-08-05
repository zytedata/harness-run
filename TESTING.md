# Testing — from unit tests to live validation

Three rungs, cheapest first. Every change runs rung 1; run the later rungs when your change
can only break in ways the earlier rungs can't see.

| Rung | Command | Catches | Cost |
|---|---|---|---|
| Offline tests | `make test` | logic, event plumbing, contracts we encode | seconds, free |
| Install parity | `make parity-build` / `-check` | dependency/install/glibc breakage | ~1 min, free |
| **Live validation** | `make live-smoke` | **platform-contract breakage** | ~10 min, ~$0.10 + build |

## 1. Offline tests (`make test`)

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
- **warm** — pub/sub dispatch to a pre-warmed pool worker (~10–20 s pickup).

Pass criteria per engine: terminal result with `error=False`, `turns > 0`, and the expected
answer in the text (the `*-spec` checks additionally require the run-scoped system prompt's
marker — proving the worker ran the handle's spec, not the deploy-baked one). Typical
numbers: deploy ~3.5–4 min (the two run in parallel), cold turn ~3 min end-to-end, warm
~1 min, a few cents of model spend. Exit code is non-zero on any FAIL, so you can gate on it.

**Running on end-user ADC (no SA impersonation)?** Two gotchas, both observed live:
set `GOOGLE_CLOUD_QUOTA_PROJECT=<project>` or every Cloud Logging read 429s (end-user
credentials without a quota project bill reads against a default consumer with no
quota) — and even then, the event tail polls at 1 Hz (= 60 reads/min) while the default
per-user logging read quota is 60/min, so the smoke's two parallel mode tails can still
429. `IMPERSONATE_SA` avoids both; otherwise run modes separately (`MODE=cold`, then
`MODE=warm`) via `dev/_smoke_slowpoll.py`, which lowers the tail cadence for the smoke run.

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

### Writing a bespoke live probe

When the smoke test doesn't cover your change (e.g. validating crash/retry behavior, or a
new event field), follow the same pattern — it's what keeps live testing safe and cheap:

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

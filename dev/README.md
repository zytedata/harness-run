# dev/ — local parity with the engine image

[`Dockerfile`](Dockerfile) mirrors the **install contract** of the Gemini Agent Runtime engine
(Debian/glibc, Python 3.12, `uv`, `git`, the toolkit, and your agent's declared `packages`). The point is to
catch dependency/install failures **locally, in seconds** — instead of discovering them through ~10-minute
cloud rebuilds.

It is **not** the full managed runtime: there's no Secret Manager, Cloud Logging, tracing, or warm pool. It
reproduces the environment the agent's *tools* run in, and you exercise it with the exact same
`local.deploy(...)` code you'd run anywhere.

## Use it

```bash
# build (optionally with the same packages your AgentSpec declares)
make parity-build PACKAGES="pandas==2.2.* httpx>=0.27"

# sanity check — toolkit imports, uv present (no model call)
make parity-check

# drop into a shell with the repo mounted and your Claude auth forwarded
ANTHROPIC_API_KEY=... make parity-shell
#   then, inside the container:
#   python examples/minimal/agent.py
```

`make parity-shell` mounts the repo at `/work` and forwards `ANTHROPIC_API_KEY`, so you can edit a script on
your host and run it inside the parity environment. Claude auth is resolved from the environment exactly as
the deployed engine resolves it.

## Relation to `gemini.deploy`

`gemini.deploy` installs `AgentSpec.packages` into the engine image via the same uv-requirements path this
image uses, so a clean `make parity-check` (and your agent running under `parity-shell`) is a strong signal
that the deploy's install step will succeed too. See `DESIGN.md` §12 (local parity) for the open design
questions.

## Live probes

This directory also holds the paid probes. They deploy or run against the real thing and spend real
money, so they run by hand, never in CI. See [`TESTING.md`](../TESTING.md) for when each is required.

| Script | Make target | What it covers |
|---|---|---|
| [`live_smoke.py`](live_smoke.py) | `make live-smoke` | the standard live validation: throwaway engines from your checkout, one turn through the cold and warm dispatch paths, teardown in `finally` |
| [`live_revisions.py`](live_revisions.py) | `make live-revisions` | the revision control plane: deploy-as-update, traffic rollback, pinning |
| [`live_pool_cutover_probe.py`](live_pool_cutover_probe.py) | `make live-pool-cutover` | the warm-pool redeploy cutover (#38): fresh dispatch pair per deploy, old workers exit, `get_engine` discovery, post-redeploy turn on the new revision |
| [`live_openrouter_probe.py`](live_openrouter_probe.py) | `make live-openrouter` | the OpenRouter models on both harnesses, locally |
| [`live_openrouter_remote_probe.py`](live_openrouter_remote_probe.py) | `make live-openrouter-remote` | the same models on Agent Runtime, plus the remote-only visibility surface |
| [`live_model_attribution.py`](live_model_attribution.py) | `make live-attribution` | did the turn run the model we asked for, from evidence the CLI and provider return |
| [`live_usage_probe.py`](live_usage_probe.py) | `make live-usage` | usage/cost accounting on both harnesses — incl. the Codex subagent rollout recovery, which rides non-public rollout details; `SCENARIO=<name>` for one scenario |
| [`openrouter_endpoints.py`](openrouter_endpoints.py) | — | lists a model's OpenRouter providers and their advertised features; free unless you pass `--probe` |

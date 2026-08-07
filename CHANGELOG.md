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

## 0.1.0 — 2026-08-07

Initial release. What's in the box:

### Agent definition

- `AgentSpec`: one declarative spec for an agent — system prompt, skills
  (local or packaged), MCP servers (local or remote), extra Python/apt
  packages, git repositories to provision (with host-aware auth), per-run
  secrets, structured `output_schema`, model override, `reasoning_effort`,
  and container `resource_limits` (CPU/memory).
- Harnesses: Claude Code (via the Claude Agent SDK) and Codex (OpenAI models)
  behind the same `AgentSpec`/engine/session API.

### Run planes

- Local: run the same spec on your machine for fast iteration; a dev parity
  image mirrors the engine's install contract.
- Gemini Agent Runtime (Vertex AI Agent Engine): `deploy()` /
  `get_engine()` / sessions with multi-turn `run()`, real interrupt,
  checkpoint/resume (canonical-UUID session ids), and agent versions —
  `deploy` creates revisions and `get_engine(version=…)` pins one.
- Warm pool: pre-provisioned workers claimed per turn for lower latency,
  with readiness signaling, pre-warm tuning, and robust cross-process
  teardown.
- Live event streaming over a GCS mirror (quota-free); Cloud Logging is
  emit-only. Clock-free turn boundaries prevent stale-result replay on fast
  resume.
- Worker CPU/RAM self-sampling with `Session.resource_samples()` for OOM
  forensics.

### Deploy reliability

- The platform-critical engine layer (aiplatform, cloudpickle, pydantic,
  google-adk, telemetry stack, …) is pinned via an in-tree
  `constraints.txt`, merged into engine requirements client-side;
  `verify_deploy_env()` fails fast when the deploying venv drifts from the
  pickle-coupled pins.
- The deploy target project/location is pinned into the pickled `AdkApp`
  (`build_adk_app`), so engine telemetry can no longer follow the deployer's
  stray local gcloud default into the wrong project (persistent span/metric
  export 403s).
- Retry-safe secrets handoff: per-invocation secrets are deleted at turn
  end, not on first read.

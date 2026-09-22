# Changelog

All notable changes to `harness-run` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning is [SemVer](https://semver.org/)-style with the usual 0.x caveat:
**minor releases (0.X) may contain backwards-incompatible changes**; patch
releases (0.X.Y) do not. Every incompatible change is listed under a
**Backwards-incompatible** heading together with update notes describing how to
migrate.

Releases are git tags (`vX.Y.Z`) on `main`, published to [PyPI](https://pypi.org/project/harness-run/)
by `.github/workflows/publish.yml` through PyPI's Trusted Publishing; install a specific release with

```
pip install "harness-run==X.Y.Z"
```

Release checklist: update this file (move the `Unreleased` section into a new
version heading with the date), bump `version` in `pyproject.toml`, merge that
commit to `main` (PR), tag the merge commit `vX.Y.Z` and push the tag. The
publish workflow refuses a tag whose version differs from `pyproject.toml`; once
it has uploaded the release, create the GitHub Release for the tag with this
file's section as the notes.

## 0.4.0 — 2026-09-23

First public release.

`harness-run` was developed at [Zyte](https://www.zyte.com) as an internal library
(`remote-agent-toolkit`, versions 0.1.0–0.3.1) and is released here under the Apache-2.0
license with a new name. It runs coding agents — Claude Code and Codex, with Anthropic, OpenAI,
Vertex AI and OpenRouter models — locally and in GCP Agent Sandbox through one API: declarative
`AgentSpec`s, multi-turn sessions with checkpoint/resume, streaming events, structured output,
skills, MCP servers, repository setup, per-run secrets and usage/cost reporting. The
[README](README.md) and [docs/](docs/) describe the release; the development history before it
is in the git log.

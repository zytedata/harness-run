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

## Live smoke test (`live_smoke.py`)

This directory also holds [`live_smoke.py`](live_smoke.py) — the standard **live validation** run
(`make live-smoke`): throwaway engines deployed from your checkout, one turn through the cold and warm
dispatch paths, teardown in `finally`. It's the opposite end of the ladder from the parity image — real
platform, real money. See [`TESTING.md`](../TESTING.md) for when it's required.

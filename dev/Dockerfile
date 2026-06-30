# Dev-only image mirroring the Gemini Agent Runtime *install contract* — Debian/glibc,
# Python 3.12, `uv`, `git`, the toolkit, and the agent's declared `packages` — so
# dependency / install issues surface locally in seconds instead of through ~10-min cloud
# rebuilds. This is NOT the full managed runtime (no Secret Manager, tracing, or warm pool):
# it reproduces the environment the agent's tools run in, and you exercise it with the very
# same `local.deploy(...)` code.
#
#   Build:  docker build -f dev/Dockerfile -t ratk-dev [--build-arg EXTRA_PACKAGES="pandas httpx"] .
#   Shell:  docker run --rm -it -e ANTHROPIC_API_KEY -v "$PWD":/work -w /work ratk-dev
#
# (the Makefile `parity-*` targets wrap these). Claude auth is passed in via the
# environment (e.g. ANTHROPIC_API_KEY), matching how the deployed engine resolves it.

# Debian (glibc) + Python 3.12 — the claude-agent-sdk wheel ships a self-contained glibc
# `claude` binary (no Alpine/musl), and the SDK supports Python <=3.13.
FROM python:3.12-slim-bookworm

# `git` survives in Google's managed base image; skills/agents shell out to it. Build tools
# cover the rare sdist in the dependency tree.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install the toolkit. This also installs `uv` (a core dependency), mirroring the platform's
# "uv ships as a Python requirement" contract. Copy only what the install needs.
WORKDIR /opt/toolkit
COPY pyproject.toml README.md ./
COPY remote_agent_toolkit ./remote_agent_toolkit
RUN pip install --no-cache-dir .

# The agent's declared spec.packages, installed with uv exactly as the deployed engine will
# (pass them at build time so version/import conflicts show up here, not in the cloud).
ARG EXTRA_PACKAGES=""
RUN if [ -n "$EXTRA_PACKAGES" ]; then uv pip install --system $EXTRA_PACKAGES; fi

# Match the runtime contract: bypassPermissions requires IS_SANDBOX under root; /tmp is the
# writable per-job root.
ENV IS_SANDBOX=1 \
    AGENT_JOBS_ROOT=/tmp/agent-jobs \
    PYTHONUNBUFFERED=1

WORKDIR /work
CMD ["bash"]

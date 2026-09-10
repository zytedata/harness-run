# Sandbox-spike image: the toolkit harness (Claude Code, bundled in claude-agent-sdk) behind
# the tiny HTTP runner in server.py, packaged the way an Agent Sandbox custom container
# needs it: Linux, non-root, one declared port. See dev/sandbox_spike/README.md.
#
#   Build (from the repo root; the context must be the repo):
#     docker build -f dev/sandbox_spike/Dockerfile -t ratk-sandbox-spike .
#   Run locally:
#     docker run --rm -p 8080:8080 ratk-sandbox-spike

FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates curl procps \
    && rm -rf /var/lib/apt/lists/*

# The toolkit with the [local] extra (claude-agent-sdk ships the glibc `claude` binary).
WORKDIR /opt/toolkit
COPY pyproject.toml README.md ./
COPY remote_agent_toolkit ./remote_agent_toolkit
RUN pip install --no-cache-dir ".[local]"

COPY dev/sandbox_spike/server.py /opt/spike/server.py

# Non-root, as the sandbox requires. /workspace is the per-sandbox scratch root; HOME must be
# writable (Claude Code keeps ~/.claude).
RUN useradd --uid 1000 --create-home --shell /bin/bash appuser \
    && mkdir -p /workspace && chown -R appuser:appuser /workspace /opt/spike
USER appuser
ENV HOME=/home/appuser \
    PORT=8080 \
    SPIKE_WORKSPACE=/workspace \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /workspace
EXPOSE 8080
CMD ["python", "/opt/spike/server.py"]

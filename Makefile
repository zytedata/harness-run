# Developer convenience targets. The `parity-*` targets build/run a dev image that mirrors
# the sandbox image's install contract (see dev/Dockerfile and dev/README.md), so
# dependency/install issues surface locally before a deploy.

VENV ?= .venv
IMAGE ?= ratk-dev
PACKAGES ?=

.PHONY: test test-serial lint live-smoke live-openrouter live-openrouter-remote live-attribution live-usage live-interactive chat parity-build parity-shell parity-check

# Parallel by default: the suite is dominated by a few deliberate poll-cadence tests, so
# -n auto takes it from ~55s to ~40s and keeps scaling as tests are added. Use test-serial
# when you need readable output or are debugging an ordering question.
test:
	$(VENV)/bin/python -m pytest -q -n auto

test-serial:
	$(VENV)/bin/python -m pytest -q

lint:
	$(VENV)/bin/ruff check .

# Live validation on the real sandbox platform (a throwaway engine, torn down after).
# Costs a few cents + ~5 min (image build included); needs Docker logged into the registry.
# See TESTING.md for when it's required and how to configure. LONG_MINUTES=n adds a long turn.
live-smoke:
	$(VENV)/bin/python dev/live_smoke.py

# Live check of the OpenRouter models on both harnesses (local runtime, no cloud).
# COSTS REAL MONEY and needs OPENROUTER_API_KEY — run it by hand, sparingly, never in CI.
# Run it when you touch the harness's provider wiring or the model list. TESTING.md has the
# current cost and runtime.
live-openrouter:
	$(VENV)/bin/python dev/live_openrouter_probe.py

# Same models on the sandbox runtime, plus the remote-only visibility surface
# (effective_spec echo, history). One throwaway engine serves all four models (model is a
# per-turn knob), deleted in `finally`. COSTS REAL MONEY and takes ~5-10 min. Checks run
# concurrently. Give it a generous timeout — by hand, never in CI.
live-openrouter-remote:
	$(VENV)/bin/python dev/live_openrouter_remote_probe.py

# Did the turn actually run the model we asked for? Checks both harnesses against evidence
# that comes back from the CLI/app-server/provider, not from our own request. Concurrent,
# a few cents, needs keys. COSTS REAL MONEY — by hand, never in CI.
live-attribution:
	$(VENV)/bin/python dev/live_model_attribution.py

# Re-check usage/cost accounting live, incl. the Codex subagent rollout recovery
# (non-public details — run after a Codex CLI bump). SCENARIO=codex-subagent etc.
SCENARIO ?= all
live-usage:
	$(VENV)/bin/python dev/live_usage_probe.py $(SCENARIO)

# Turn control on the local runtime, both harnesses, against real models: steer into a
# running turn, interrupt + continue, interrupt() to a resumable INTERRUPTED stop, resume.
# A few cents (Haiku + a small Codex turn), ~2 min, needs ANTHROPIC_API_KEY + OPENAI_API_KEY.
# COSTS REAL MONEY — by hand, never in CI. HARNESSES=codex (or claude-code) runs one.
live-interactive:
	$(VENV)/bin/python dev/live_interactive_probe.py

# Local chat UI for trying turn control by hand (dev/chat.py): type while the agent works to
# steer, "Interrupt & send", "Stop" and resume. Same status as the live probes: a dev tool,
# COSTS REAL MONEY, needs the model keys plus fastapi + uvicorn in the venv, run it from an
# unsandboxed shell. HARNESS=codex for Codex. Then open http://127.0.0.1:8765 .
HARNESS ?= claude-code
chat:
	$(VENV)/bin/python dev/chat.py --harness $(HARNESS)

# Build the parity image. Pass the agent's spec.packages so they install exactly as on the
# engine, e.g.:  make parity-build PACKAGES="pandas==2.2.* httpx>=0.27"
parity-build:
	docker build -f dev/Dockerfile -t $(IMAGE) $(if $(PACKAGES),--build-arg EXTRA_PACKAGES="$(PACKAGES)",) .

# Interactive shell in the parity image, with the repo mounted at /work and your Claude auth
# forwarded — run `python examples/minimal/agent.py` (or your own script) inside it.
parity-shell:
	docker run --rm -it -e ANTHROPIC_API_KEY -v "$(CURDIR)":/work -w /work $(IMAGE)

# Sanity check (no model call): confirm the toolkit imports and uv is present in the image.
parity-check:
	docker run --rm $(IMAGE) python -c "import remote_agent_toolkit, shutil, sys; \
print('python', sys.version.split()[0]); \
print('toolkit import OK'); \
print('uv on PATH:', shutil.which('uv') is not None)"

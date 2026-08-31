# Developer convenience targets. The `parity-*` targets build/run a dev image that mirrors
# the Gemini Agent Runtime install contract (see dev/Dockerfile and dev/README.md), so
# dependency/install issues surface locally instead of via ~10-min cloud rebuilds.

VENV ?= .venv
IMAGE ?= ratk-dev
PACKAGES ?=

.PHONY: test test-serial lint live-smoke live-revisions live-openrouter live-openrouter-remote live-attribution live-usage parity-build parity-shell parity-check

# Parallel by default: the suite is dominated by a few deliberate poll-cadence tests, so
# -n auto takes it from ~55s to ~40s and keeps scaling as tests are added. Use test-serial
# when you need readable output or are debugging an ordering question.
test:
	$(VENV)/bin/python -m pytest -q -n auto

test-serial:
	$(VENV)/bin/python -m pytest -q

lint:
	$(VENV)/bin/ruff check .

# Live validation on real Gemini Agent Runtime (throwaway engines, torn down after).
# Costs real money + ~10 min; see TESTING.md for when it's required and how to configure.
live-smoke:
	$(VENV)/bin/python dev/live_smoke.py

# Live check of the revision control plane (deploy-as-update, traffic rollback, pinning).
# Two SEQUENTIAL builds — ~10 min; run it when you touch deploy/versioning.
live-revisions:
	$(VENV)/bin/python dev/live_revisions.py

# Live check of the OpenRouter models on both harnesses (local runtime, no cloud).
# COSTS REAL MONEY and needs OPENROUTER_API_KEY — run it by hand, sparingly, never in CI.
# Run it when you touch the harness's provider wiring or the model list. TESTING.md has the
# current cost and runtime.
live-openrouter:
	$(VENV)/bin/python dev/live_openrouter_probe.py

# Same models on Gemini Agent Runtime, plus the remote-only visibility surface
# (effective_spec echo, resource samples, memory peak, history, traces). One throwaway
# engine serves all four models (model is a per-turn knob), deleted in `finally`.
# COSTS REAL MONEY and takes ~8-15 min, mostly for the engine build. Checks run
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

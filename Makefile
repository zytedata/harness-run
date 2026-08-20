# Developer convenience targets. The `parity-*` targets build/run a dev image that mirrors
# the Gemini Agent Runtime install contract (see dev/Dockerfile and dev/README.md), so
# dependency/install issues surface locally instead of via ~10-min cloud rebuilds.

VENV ?= .venv
IMAGE ?= ratk-dev
PACKAGES ?=

.PHONY: test lint live-smoke live-revisions live-openrouter live-openrouter-remote parity-build parity-shell parity-check

test:
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

# Live check of the OpenRouter models on the codex harness (local runtime, no cloud).
# COSTS REAL MONEY (a few cents) and needs OPENROUTER_API_KEY — run it by hand, sparingly,
# never in CI. Run it when you touch the harness's provider wiring or the model list.
live-openrouter:
	$(VENV)/bin/python dev/live_openrouter_probe.py

# Same models on Gemini Agent Runtime, plus the remote-only visibility surface
# (effective_spec echo, resource samples, memory peak, history, traces). One throwaway
# engine serves all four models (model is a per-turn knob), deleted in `finally`.
# COSTS REAL MONEY (~$0.30) and takes ~35-40 min: a ~5 min build plus seven turns that each
# pay cold-start latency. Give it a generous timeout — by hand, never in CI.
live-openrouter-remote:
	$(VENV)/bin/python dev/live_openrouter_remote_probe.py

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

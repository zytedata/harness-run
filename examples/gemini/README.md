# gemini example — deploy, run, tear down

[`deploy_and_run.py`](deploy_and_run.py) runs the **prod path** end-to-end on Gemini Agent Runtime:
`gemini.deploy` → `start_session` → stream a run → `engine.delete`. It's the same `AgentSpec` and
`Engine`/`Session`/`Run` API as [`../minimal`](../minimal) — only the backend changes.

This mirrors the scripts used to validate the backend live (deploy ~4 min, warm pickup ~12 s, clean teardown).

## Prerequisites

GCP setup from the top-level [README "GCP setup & required permissions"](../../README.md#gcp-setup--required-permissions):
the operator SA + RE-agent IAM grants, and the Claude model (`spec.model`) **enabled in your Vertex Model
Garden**. Plus a Python 3.12 env with the toolkit installed (`uv pip install -e .`).

## Run

```bash
# reuse-or-deploy, then run a turn (cold). Deploys only if no engine of this name exists.
.venv/bin/python examples/gemini/deploy_and_run.py

# warm pool: pre-warmed workers, ~12 s pickup instead of ~2.5 min
WARM=1 .venv/bin/python examples/gemini/deploy_and_run.py

# tear the engine down — cancels warm-pool workers + removes the engine and dispatch topic/sub
TEARDOWN=1 .venv/bin/python examples/gemini/deploy_and_run.py

# point at your own project / least-priv SA
PROJECT=my-proj LOCATION=us-central1 IMPERSONATE_SA=agent-runtime@my-proj.iam.gserviceaccount.com \
  .venv/bin/python examples/gemini/deploy_and_run.py
```

The engine is **reused** if one of this name already exists (so a second run skips the ~4 min deploy), and is
**left running** afterwards for that reuse. A deployed engine bills while it exists — warm pools especially,
with idle workers — so run with `TEARDOWN=1` when you're done; that calls
`engine.delete(delete_pool_resources=True)`, which cancels the pool workers and removes the engine + topic/sub.
Defaults target the shared `my-project` test project.

> Deploy builds an engine image (~4 min) and creates a billed engine. The model run itself is a few cents.

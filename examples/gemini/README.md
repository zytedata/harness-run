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
# cold (default): one deploy, one run, teardown
.venv/bin/python examples/gemini/deploy_and_run.py

# warm pool: pre-warmed workers, ~12 s pickup instead of ~2.5 min
WARM=1 .venv/bin/python examples/gemini/deploy_and_run.py

# point at your own project / least-priv SA
PROJECT=my-proj LOCATION=us-central1 IMPERSONATE_SA=agent-runtime@my-proj.iam.gserviceaccount.com \
  .venv/bin/python examples/gemini/deploy_and_run.py
```

Defaults target the shared `my-project` test project. The script **always tears the engine down**
(`engine.delete(delete_pool_resources=True)`) in a `finally` block, so a deployed engine never lingers and
bills — including cancelling warm-pool workers.

> Deploy builds an engine image (~4 min) and creates a billed engine. The model run itself is a few cents.

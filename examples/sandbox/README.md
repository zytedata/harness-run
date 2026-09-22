# sandbox example — deploy, run, tear down

[`deploy_and_run.py`](deploy_and_run.py) runs the **prod path** end-to-end in a Google Agent Sandbox:
`sandbox.deploy` (image build + push + template) → `start_session` → stream a run → `engine.delete`. It's the
same `AgentSpec` and `Engine`/`Session`/`Run` API as [`../minimal`](../minimal) — only the backend changes.

## Prerequisites

GCP setup from the top-level [GCP setup](../../docs/gcp-setup.md)
(`agent-run-gcp-setup`: image repo, model service account, output bucket), the Claude model (`spec.model`)
**enabled in your Vertex Model Garden**, the Docker CLI logged into the registry, and a Python 3.12 env with
the toolkit installed (`uv pip install -e .`).

## Run

```bash
# reuse-or-deploy, then run a turn on a fresh sandbox (~20 s to the result). Deploys only if no engine of this name exists.
.venv/bin/python examples/sandbox/deploy_and_run.py

# ready pool: one pre-warmed sandbox, ~1 s to the first event instead of ~15-25 s
WARM=1 .venv/bin/python examples/sandbox/deploy_and_run.py

# tear the engine down — deletes its ready sandboxes and every version's template
TEARDOWN=1 .venv/bin/python examples/sandbox/deploy_and_run.py

# point at your own project / least-priv SA
PROJECT=my-proj LOCATION=us-central1 IMPERSONATE_SA=agent-runtime@my-proj.iam.gserviceaccount.com \
  .venv/bin/python examples/sandbox/deploy_and_run.py
```

The engine is **reused** if one of this name already exists (a second run skips the image build), and is
**left in place** afterwards for that reuse. A template itself costs nothing; a ready pool's idle sandboxes
bill while they exist, so run with `TEARDOWN=1` when you're done. Set `PROJECT` to your GCP project.

> The first deploy builds and pushes the agent's image (~1 min with a warm Docker cache) and creates the
> template (seconds for a 1 CPU template; the platform has taken up to 30 min for 4 CPU ones). The model run
> itself is a few cents.

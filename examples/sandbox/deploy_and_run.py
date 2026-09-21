"""Deploy the example agent as a sandbox engine, run a turn, tear it down.

The same `AgentSpec` and `Engine`/`Session`/`Run` API as the local example — only the
backend (`local` → `sandbox`) changes. This is the prod path the top-level README's "Prod"
section describes, as a runnable template.

Prerequisites (see the top-level README "GCP setup & required permissions"):
  * `agent-run-gcp-setup` done for the project (image repo, model service account, bucket), the Claude
    model enabled in Vertex Model Garden, and Docker logged into the registry (the deploy builds
    and pushes the agent's image).
Configure via env (defaults are the shared my-project test setup):
  PROJECT, LOCATION, IMPERSONATE_SA (optional least-priv impersonation), WARM=1 (ready pool).

Run:
  .venv/bin/python examples/sandbox/deploy_and_run.py             # reuse-or-deploy, then run a turn
  TEARDOWN=1 .venv/bin/python examples/sandbox/deploy_and_run.py  # tear the engine down (templates + ready sandboxes)

The engine is REUSED if one of this name already exists (a subsequent run skips the image
build); it's left in place afterwards for that reuse. A template itself costs nothing; a ready
pool's idle sandboxes bill while they exist, so tear it down with TEARDOWN=1 when you're done.
"""

from __future__ import annotations

import asyncio
import os

import pydantic

from agent_run import AgentSpec, SystemPrompt, sandbox

PROJECT = os.environ.get("PROJECT", "my-project")
LOCATION = os.environ.get("LOCATION", "us-central1")
WARM = os.environ.get("WARM") == "1"


def _credentials():
    """Optional least-privilege SA impersonation; omit (return None) to use your own ADC."""
    sa = os.environ.get("IMPERSONATE_SA")
    if not sa:
        return None
    import google.auth
    from google.auth import impersonated_credentials

    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    source, _ = google.auth.default(scopes=scopes)
    return impersonated_credentials.Credentials(
        source_credentials=source, target_principal=sa, target_scopes=scopes
    )


class Answer(pydantic.BaseModel):
    product: int


SPEC = AgentSpec(
    harness="claude-code",
    name="agent-run-example",
    # A family alias enabled in your Vertex Model Garden (see the README model-access note).
    model="claude-haiku-4-5",
    system_prompt=SystemPrompt.inherit(append="Be concise. Verify your work by running code."),
    max_turns=8,
    max_budget_usd=2.0,
    output_schema=Answer,
)
TASK = 'Compute 6*7 by writing and running a one-line Python script, then report JSON {"product": <int>}.'


async def _run_turn(engine) -> None:
    session = engine.start_session()
    print(f"session {session.session_id} — running", flush=True)
    run = session.run(TASK)
    async for event in run:  # stream events straight from the sandbox as they happen
        print(f"  [{event.kind:11}] {' '.join((event.summary or '').split())[:120]}", flush=True)
    result = run.result
    print(f"\nstructured: {result.structured_output}  cost=${result.cost_usd:.4f}  error={result.is_error}")


def get_or_deploy(creds):
    """Reuse the named engine if it exists; deploy it only if not. Returns ``(engine, reused)``."""
    try:
        engine = sandbox.get_engine(SPEC.name, project=PROJECT, location=LOCATION,
                                   warm_pool=WARM, credentials=creds)
        print(f"reusing existing engine {engine.resource}", flush=True)
        return engine, True
    except LookupError:
        print(f"no engine named {SPEC.name!r} — deploying (image build + push + template)...", flush=True)
        engine = sandbox.deploy(SPEC, project=PROJECT, location=LOCATION, credentials=creds,
                               warm_pool=WARM, pool_size=1)
        print(f"deployed {engine.resource}", flush=True)
        return engine, False


def teardown(creds) -> None:
    """Tear the named engine down — deletes its ready sandboxes and every version's template."""
    try:
        engine = sandbox.get_engine(SPEC.name, project=PROJECT, location=LOCATION,
                                   warm_pool=WARM, credentials=creds)
    except LookupError:
        print(f"no engine named {SPEC.name!r} — nothing to tear down", flush=True)
        return
    engine.delete()
    print(f"torn down {engine.resource} (templates + sandboxes)", flush=True)


def main() -> None:
    creds = _credentials()
    # TEARDOWN=1 deletes the engine (its templates and ready sandboxes) and exits.
    if os.environ.get("TEARDOWN") == "1":
        teardown(creds)
        return

    engine, reused = get_or_deploy(creds)
    if WARM:
        if reused:
            engine.fill_pool(1)  # top up — a reused pool's sandboxes may have idled out
        print("waiting for a ready sandbox...", flush=True)
        engine.wait_until_warm(timeout=300)
    asyncio.run(_run_turn(engine))

    # Left in place so the next run reuses it (skipping the image build). Ready sandboxes
    # bill while they exist — tear it down when done:
    #     TEARDOWN=1 .venv/bin/python examples/sandbox/deploy_and_run.py
    print(f"\nengine left running for reuse: {engine.resource}"
          "\n  tear down with:  TEARDOWN=1 .venv/bin/python examples/sandbox/deploy_and_run.py", flush=True)


if __name__ == "__main__":
    main()

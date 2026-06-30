"""Deploy the example agent to Gemini Agent Runtime, run a turn, tear it down.

The same `AgentSpec` and `Engine`/`Session`/`Run` API as the local example — only the
backend (`local` → `gemini`) changes. This is the prod path the top-level README's "Prod"
section describes, as a runnable template.

Prerequisites (see the top-level README "GCP setup & required permissions"):
  * the operator SA + RE-agent IAM grants, and the Claude model enabled in Vertex Model Garden.
Configure via env (defaults are the shared my-project test setup):
  PROJECT, LOCATION, IMPERSONATE_SA (optional least-priv impersonation), WARM=1 (warm pool).

Run:  .venv/bin/python examples/gemini/deploy_and_run.py
Note: deploy builds the engine image (~4 min) and creates a billed engine; this script
always tears it down (engine.delete) in a finally block.
"""

from __future__ import annotations

import asyncio
import os

import pydantic

from remote_agent_toolkit import AgentSpec, SystemPrompt, gemini

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
    name="ratk-example",
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
    async for event in run:  # stream events (Cloud Logging tail) as they happen
        print(f"  [{event.kind:11}] {' '.join((event.summary or '').split())[:120]}", flush=True)
    result = run.result
    print(f"\nstructured: {result.structured_output}  cost=${result.cost_usd:.4f}  error={result.is_error}")


def main() -> None:
    creds = _credentials()
    print(f"Deploying {SPEC.name} to {PROJECT}/{LOCATION} (warm={WARM}) ... ~4 min build", flush=True)
    engine = gemini.deploy(SPEC, project=PROJECT, location=LOCATION, credentials=creds,
                           warm_pool=WARM, pool_size=1)
    try:
        print(f"deployed {engine.resource}", flush=True)
        if WARM:
            print("waiting for a pool worker to warm up...", flush=True)
            engine.wait_until_warm(timeout=300)
        asyncio.run(_run_turn(engine))
    finally:
        engine.delete(delete_pool_resources=True)  # cancels pool workers + removes engine (+ topic/sub)
        print("torn down", flush=True)


if __name__ == "__main__":
    main()

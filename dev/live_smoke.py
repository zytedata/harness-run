"""Live smoke test — deploy throwaway engines from THIS checkout, run one turn, tear down.

The offline suite (`make test`) can't catch platform-contract breakage: the Agent Runtime
side changes underneath us (see TESTING.md). This script is the standard live check for a
branch that touches any deploy/runtime contract — it validates, on real infrastructure:

  * ``gemini.deploy``: packaging, ``AgentEngineConfig``, the pickled ADK app template
  * worker startup: the engine container unpickles the staged app and resolves
    ``async_stream_query`` (the explicit ``class_method`` both dispatch paths use)
  * a full turn round-trip per mode — **cold** (``run_query_job``, the prod default) and
    **warm** (pub/sub dispatch to a pre-warmed pool worker)

Pass criteria per engine: terminal result with ``error=False``, ``turns > 0``, and the
expected answer in the text. Engines are deleted in ``finally`` (the warm one including its
dispatch topic/sub); exit code is non-zero if any mode fails.

Configure via env (defaults are the shared my-project test setup):
  PROJECT, LOCATION, IMPERSONATE_SA (optional), MODE=cold|warm|both (default both),
  SUFFIX (engine-name suffix; defaults to your username, so contributors don't collide).

Run:
  make live-smoke                       # or:
  .venv/bin/python dev/live_smoke.py

Costs real money and ~10 min: two ~4 min builds (parallel) + a few cents of Haiku turns.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import getpass
import os
import re
import sys
import time
import traceback

from remote_agent_toolkit import AgentSpec, gemini

PROJECT = os.environ.get("PROJECT", "my-project")
LOCATION = os.environ.get("LOCATION", "us-central1")
MODE = os.environ.get("MODE", "both")
SUFFIX = re.sub(r"[^a-z0-9-]", "-", (os.environ.get("SUFFIX") or getpass.getuser()).lower())

TASK = (
    'Run `python3 -c "print(6 * 7)"` in the shell and reply with just the number it prints.'
)


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


def _spec(name: str) -> AgentSpec:
    # Haiku + tight caps: the point is the round-trip, not the model's work.
    return AgentSpec(name=name, model="claude-haiku-4-5", max_turns=8, max_budget_usd=1.0)


def _deploy(name: str, warm: bool, credentials):
    print(f"[{name}] {time.strftime('%H:%M:%S')} deploying (warm_pool={warm}) ...", flush=True)
    t0 = time.time()
    engine = gemini.deploy(
        _spec(name), PROJECT, LOCATION, warm_pool=warm, pool_size=1, credentials=credentials
    )
    print(f"[{name}] deployed in {time.time() - t0:.0f}s", flush=True)
    return engine


async def _exercise(label: str, engine) -> bool:
    session = engine.start_session()
    print(f"[{label}] {time.strftime('%H:%M:%S')} session started, sending turn", flush=True)
    t0 = time.time()
    run = session.run(TASK)
    async for ev in run:
        summary = " ".join((ev.summary or "").split())[:90]
        print(f"[{label}] {time.strftime('%H:%M:%S')} {ev.kind:11} {summary}", flush=True)
    r = run.result
    text = " ".join((r.text or "").split())
    print(
        f"[{label}] RESULT after {time.time() - t0:.0f}s: text={text[:120]!r} "
        f"turns={r.num_turns} cost=${r.cost_usd or 0:.4f} error={r.is_error}",
        flush=True,
    )
    return (not r.is_error) and bool(r.num_turns) and "42" in text


async def main() -> int:
    modes = ["cold", "warm"] if MODE == "both" else [MODE]
    credentials = _credentials()
    engines: dict[str, object] = {}
    verdicts: dict[str, bool] = {}
    loop = asyncio.get_running_loop()
    try:
        with concurrent.futures.ThreadPoolExecutor(len(modes)) as pool:
            futs = {
                mode: loop.run_in_executor(
                    pool, _deploy, f"ratk-smoke-{mode}-{SUFFIX}", mode == "warm", credentials
                )
                for mode in modes
            }
            for mode, fut in futs.items():
                try:
                    engines[mode] = await fut
                except Exception:
                    print(f"[{mode}] DEPLOY FAILED:\n{traceback.format_exc()}", flush=True)
                    verdicts[mode] = False

        async def guarded(mode):
            try:
                verdicts[mode] = await _exercise(mode, engines[mode])
            except Exception:
                print(f"[{mode}] RUN FAILED:\n{traceback.format_exc()}", flush=True)
                verdicts[mode] = False

        await asyncio.gather(*(guarded(mode) for mode in engines))
    finally:
        for mode, engine in engines.items():
            try:
                engine.delete(delete_pool_resources=(mode == "warm"))
                print(f"[{mode}] engine deleted", flush=True)
            except Exception:
                print(f"[{mode}] TEARDOWN FAILED — delete the engine manually! "
                      f"gemini.get_engine(...).delete()\n{traceback.format_exc()}", flush=True)

    print("\n=== VERDICTS ===", flush=True)
    for mode in modes:
        print(f"  {mode}: {'PASS' if verdicts.get(mode) else 'FAIL'}", flush=True)
    return 0 if all(verdicts.get(mode) for mode in modes) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

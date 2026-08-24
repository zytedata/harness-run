"""Bespoke live probe: warm pickup latency over the GCS event stream (see TESTING.md).

Deploys a warm-pool engine, blocks on ``wait_until_warm`` (exercising its GCS pool-ready
branch live), then dispatches one turn and measures dispatch → first observed event and
dispatch → result. On the old Cloud Logging tail the first observed event lagged ~10-20 s
(ingestion); the stream should cut that to roughly worker-pickup + one flush + one poll.

Throwaway engine, Haiku, teardown (with pool resources) in finally.
"""

from __future__ import annotations

import asyncio
import getpass
import os
import re
import sys
import time
import traceback

from remote_agent_toolkit import AgentSpec, gemini

PROJECT = os.environ.get("PROJECT", "my-project")
LOCATION = os.environ.get("LOCATION", "us-central1")
SUFFIX = re.sub(r"[^a-z0-9-]", "-", (os.environ.get("SUFFIX") or getpass.getuser()).lower())
NAME = f"ratk-warmlat-{SUFFIX}"

TASK = 'Run `python3 -c "print(6 * 7)"` in the shell and reply with just the number it prints.'


async def main() -> int:
    spec = AgentSpec(name=NAME, model="claude-haiku-4-5", max_turns=8, max_budget_usd=1.0)
    print(f"{time.strftime('%H:%M:%S')} deploying {NAME} (warm_pool=True) ...", flush=True)
    engine = gemini.deploy(spec, PROJECT, LOCATION, warm_pool=True, pool_size=1)
    ok = False
    try:
        t0 = time.time()
        warm = await asyncio.to_thread(engine.wait_until_warm)
        print(f"wait_until_warm={warm} after {time.time() - t0:.0f}s", flush=True)

        t0 = time.time()
        session = engine.start_session()
        run = session.run(TASK)
        first = None
        async for ev in run:
            first = first or time.time() - t0
            print(f"  +{time.time() - t0:5.1f}s {ev.kind:11} "
                  f"{' '.join((ev.summary or '').split())[:70]}", flush=True)
        r = run.result
        total = time.time() - t0
        text = " ".join((r.text or "").split())
        ok = warm and (not r.is_error) and "42" in text
        print(f"\nMEASURED: dispatch->first event {first:.1f}s, dispatch->result {total:.1f}s, "
              f"cost={'unknown' if r.cost_usd is None else f'${r.cost_usd:.4f}'}", flush=True)
    except Exception:
        print(f"PROBE FAILED:\n{traceback.format_exc()}", flush=True)
    finally:
        try:
            engine.delete(delete_pool_resources=True)
            print("engine deleted", flush=True)
        except Exception:
            print(f"TEARDOWN FAILED — delete {NAME} manually!\n{traceback.format_exc()}",
                  flush=True)
    print(f"\n=== VERDICT: {'PASS' if ok else 'FAIL'} ===", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

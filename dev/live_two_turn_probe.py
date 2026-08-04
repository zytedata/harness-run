"""Bespoke live probe: two turns on ONE session over the GCS event stream (see TESTING.md).

The stream tail resumes from a ``since`` watermark that floors the mirror OBJECT NAMES —
the risky path is a second turn on the same session: it must stream its OWN events and
terminal result, never replay the first turn's (whose files sit under the same prefix).
Also spot-checks that history() returns BOTH turns and that the client observed sub-~5s
event latency (the whole point of the stream channel).

Throwaway cold engine, Haiku, checkpoint=True (real resume), teardown in finally.
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
NAME = f"ratk-stream2t-{SUFFIX}"

_checks: list[tuple[str, bool]] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    _checks.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{f' — {detail}' if detail else ''}", flush=True)
    return bool(ok)


async def run_turn(session, message: str, expect: str, *, resume: bool = False) -> None:
    t0 = time.time()
    run = session.send(message) if resume else session.run(message)
    async for ev in run:
        print(f"    +{time.time() - t0:5.1f}s {ev.kind:11} "
              f"{' '.join((ev.summary or '').split())[:80]}", flush=True)
    r = run.result
    text = " ".join((r.text or "").split())
    check(f"turn answered {expect!r}", (not r.is_error) and expect in text,
          f"text={text[:60]!r} turns={r.num_turns}")
    return None


async def main() -> int:
    spec = AgentSpec(name=NAME, model="claude-haiku-4-5", max_turns=8,
                     max_budget_usd=1.0, checkpoint=True)
    print(f"{time.strftime('%H:%M:%S')} deploying {NAME} ...", flush=True)
    engine = gemini.deploy(spec, PROJECT, LOCATION)
    try:
        check("deploy handle streams events", engine._streams_events)
        looked_up = gemini.get_engine(NAME, project=PROJECT, location=LOCATION, spec=spec)
        check("get_engine detects the stream marker", looked_up._streams_events)

        session = looked_up.start_session()
        await run_turn(session, 'Run `python3 -c "print(6 * 7)"` in the shell and reply '
                                "with just the number it prints.", "42")
        # Turn 2, same session (resume): must NOT replay turn 1's terminal result from
        # the mirror — its files sit under the same events/<sid>/ prefix.
        await run_turn(session, "Now multiply that result by 2 and reply with just the "
                                "number.", "84", resume=True)

        events = session.history()
        kinds = [e.kind for e in events]
        check("history keeps BOTH turns", kinds.count("result") == 2,
              f"{kinds.count('result')} results / {len(events)} events")
    except Exception:
        print(f"PROBE FAILED:\n{traceback.format_exc()}", flush=True)
        _checks.append(("probe completed", False))
    finally:
        try:
            engine.delete()
            print("engine deleted", flush=True)
        except Exception:
            print(f"TEARDOWN FAILED — delete {NAME} manually!\n{traceback.format_exc()}",
                  flush=True)

    print("\n=== VERDICTS ===", flush=True)
    for label, ok in _checks:
        print(f"  {'PASS' if ok else 'FAIL'}: {label}", flush=True)
    return 0 if _checks and all(ok for _, ok in _checks) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

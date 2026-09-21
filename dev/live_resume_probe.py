"""Paid, manual Claude resume regression: stop with a pending background task, then reply.

Run with the real pinned CLI and Anthropic credentials:
    python dev/live_resume_probe.py

For a throwaway sandbox engine deployed from this checkout, set ENGINE=<resource>,
PROJECT and LOCATION. The caller owns that engine's teardown. Logs go to stdout;
this probe never runs in the offline suite. A missing pending-task state is a failure,
not evidence that resume worked.
"""

from __future__ import annotations

import asyncio
import os
import tempfile

from agent_run import AgentSpec, sandbox, local
from agent_run.events import StopReason

TASK = (
    "Use Bash with run_in_background=true to run exactly this command: "
    "`printf partial > note.txt; sleep 120; echo finished`. "
    "Do not wait for it, poll it, or use TaskOutput. As soon as it is backgrounded, "
    "end your response with WAITING."
)
RESUME = "Read note.txt in the current directory and reply with exactly its contents."


async def probe(engine, secrets: dict[str, str] | None = None) -> None:
    session = engine.start_session()
    if secrets is None:
        secrets = {k: os.environ[k] for k in ("ANTHROPIC_API_KEY",) if os.environ.get(k)}
    run = session.run(TASK, secrets=secrets)
    stopped = False
    async with asyncio.timeout(600):
        async for event in run:
            raw = event.raw or {}
            print("stop", event.kind, raw.get("event"), event.summary[:180], flush=True)
            if raw.get("event") == "awaiting_tasks" and raw.get("pending") and not stopped:
                # Let the background command create the file before killing its process.
                await asyncio.sleep(1)
                await session.interrupt()
                stopped = True
        assert stopped, "probe did not interrupt while a background task was pending"
        assert session.stop_reason == StopReason.INTERRUPTED, session.stop_reason
        assert run.result is not None and not run.result.is_error

        resumed = session.send(RESUME, secrets=secrets)
        events = []
        async for event in resumed:
            events.append(event)
            print("resume", event.kind, (event.raw or {}).get("event"),
                  event.summary[:180], flush=True)
        result = resumed.result
        assert result is not None and not result.is_error, result
        assert result.num_turns > 0 and result.text.strip() == "partial", result
        assert sum(e.kind == "result" for e in events) == 1
        print("PASS: interrupted a pending task; resume answered from the saved workspace",
              flush=True)


async def main() -> None:
    resource = os.environ.get("ENGINE")
    if resource:
        engine = sandbox.get_engine(
            resource, project=os.environ.get("PROJECT", "my-project"),
            location=os.environ.get("LOCATION", "us-central1"), warm_pool=True,
        )
    else:
        engine = local.deploy(
            AgentSpec(name="resume-probe", model="claude-haiku-4-5-20251001",
                      checkpoint=True, max_turns=12, max_budget_usd=1.0),
            workdir=tempfile.mkdtemp(prefix="agent-run-resume-"),
        )
    await probe(engine, secrets={} if resource else None)


if __name__ == "__main__":
    asyncio.run(main())

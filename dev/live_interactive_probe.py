"""Live check of turn control (issue #44) on the local runtime, both harnesses.

Runs the four transitions against real models, in one session per harness:

1. **steer** — ``send()`` while the model runs a long shell command; the model must answer
   the steer inside the same run (one result), with a ``user`` event as the ack.
2. **interrupt + continue** — ``send(..., interrupt=True)`` while a command runs; the reply
   to the new message must arrive in the same run.
3. **stop** — ``interrupt()`` while a command runs; the session must end idle with
   ``StopReason.INTERRUPTED``, its checkpoint including a file the interrupted turn wrote.
4. **resume** — a plain ``send()`` afterwards must continue from that checkpoint (the
   model sees the file).

It also prints a latency table: from the ``send()`` / ``interrupt()`` call to the harness
reacting (``interrupt_requested``), to the ``user`` ack (the model has the message) and to
the ``result`` (end to end, model time included). The 2026-09-04 numbers are in TESTING.md.

COSTS REAL MONEY (a few cents: Haiku + a small Codex turn) and needs ``ANTHROPIC_API_KEY``
and ``OPENAI_API_KEY``. Run it by hand when touching the harness run loops or
``control.py``; the offline suite covers the same transitions against fakes. Run from an
unsandboxed shell: the Claude CLI's Bash tool needs to write under ``~/.claude``.

    make live-interactive            # both harnesses, concurrently
    HARNESSES=codex make live-interactive
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time

from agent_run import AgentSpec, local
from agent_run.events import RunStatus, StopReason

MODELS = {"claude-code": "claude-haiku-4-5-20251001", "codex": "gpt-5.6-luna"}

STEER_TASK = (
    "Run the shell command `sleep 8 && echo one`, then run `sleep 8 && echo two`, then "
    "reply with the word DONE."
)
STEER_MSG = "STOP whatever you were doing. Do not run more commands. Reply with just the single word BANANA."
INTERRUPT_TASK = "Run the shell command `sleep 40 && echo slow`, then reply with the word DONE."
INTERRUPT_MSG = "Forget that. Do not run more commands. Reply with just the single word OK."
STOP_TASK = (
    "First create a file named note.txt in the current directory containing the word "
    "partial. Then run the shell command `sleep 40 && echo slow`. Then reply with DONE."
)
RESUME_MSG = "Which files are in the current directory? Reply with just their names."


class Check:
    def __init__(self, harness: str) -> None:
        self.harness = harness
        self.failures: list[str] = []
        self.latency: dict[str, float] = {}  # "<step> <from> -> <to>" -> seconds

    def lat(self, step: str, clock: dict, to: str, start: str = "acted") -> None:
        """Record ``clock[to] - clock[start]`` under a readable name, when both marks exist."""
        if start in clock and to in clock:
            self.latency[f"{step}: send -> {to}" if start == "acted" else f"{step}: {start} -> {to}"] = (
                clock[to] - clock[start]
            )

    def expect(self, cond: bool, what: str) -> None:
        tag = "PASS" if cond else "FAIL"
        print(f"[{self.harness}] {tag}: {what}", flush=True)
        if not cond:
            self.failures.append(what)


async def _drive(run, *, on_tool_use=None, label: str, harness: str, clock: dict | None = None):
    """Consume the run; call ``on_tool_use`` once, ~1.5 s after the first tool starts.

    ``clock`` (optional) collects wall-clock marks: ``clock["acted"]`` right before
    ``on_tool_use`` runs, and ``clock[<kind or event tag>]`` the first time that event kind
    (``user``, ``result``) or status tag (``interrupt_requested``) is seen. The latency table
    at the end is these marks against ``acted``.
    """
    events = []
    acted = False
    t0 = time.monotonic()
    async for ev in run:
        now = time.monotonic()
        events.append(ev)
        tag = (ev.raw or {}).get("event") or (ev.raw or {}).get("subtype") or ""
        if clock is not None:
            clock.setdefault(ev.kind, now)
            if tag:
                clock.setdefault(tag, now)
        print(f"[{harness} {label} {now - t0:6.1f}s] {ev.kind:11} {tag:20} "
              f"{' '.join(ev.summary.split())[:90]}", flush=True)
        if on_tool_use is not None and not acted and ev.kind == "tool_use":
            acted = True
            await asyncio.sleep(1.5)
            if clock is not None:
                clock["acted"] = time.monotonic()
            await on_tool_use()
    return events


async def probe(harness: str) -> Check:
    check = Check(harness)
    spec = AgentSpec(name=f"ctl-{harness}", model=MODELS[harness], harness=harness,
                     checkpoint=True, max_turns=20, max_budget_usd=1.0)
    workdir = tempfile.mkdtemp(prefix=f"agent-run-ctl-{harness}-")
    engine = local.deploy(spec, workdir=workdir)
    session = engine.start_session()
    secrets = {k: os.environ[k] for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY") if k in os.environ}

    # 1. steer
    run = session.run(STEER_TASK, secrets=secrets)

    async def steer():
        same = session.send(STEER_MSG, message_id="steer-1")
        check.expect(same is run, "send() on the running session returned the same Run")

    clock: dict = {}
    events = await _drive(run, on_tool_use=steer, label="steer", harness=harness, clock=clock)
    check.lat("steer", clock, "user")
    check.lat("steer", clock, "result")
    users = [e for e in events if e.kind == "user"]
    results = [e for e in events if e.kind == "result"]
    check.expect(len(users) == 1 and users[0].raw["message_id"] == "steer-1",
                 "one user event acknowledged the steer with its message_id")
    check.expect(len(results) == 1, "the steered turn produced exactly one result")
    check.expect("BANANA" in (run.result.text or ""), f"the model answered the steer: {run.result.text!r}")
    check.expect(session.status == RunStatus.IDLE and session.stop_reason == StopReason.NEEDS_INPUT,
                 f"session idle after the steered turn (stop_reason={session.stop_reason})")

    # 2. interrupt + continue
    run = session.send(INTERRUPT_TASK, secrets=secrets)

    async def interrupt_continue():
        same = session.send(INTERRUPT_MSG, interrupt=True, message_id="int-1")
        check.expect(same is run, "send(interrupt=True) returned the same Run")

    clock = {}
    events = await _drive(run, on_tool_use=interrupt_continue, label="interrupt", harness=harness,
                          clock=clock)
    check.lat("interrupt+continue", clock, "interrupt_requested")
    check.lat("interrupt+continue", clock, "user")
    check.lat("interrupt+continue", clock, "result")
    tags = [(e.raw or {}).get("event") for e in events]
    check.expect("interrupt_requested" in tags, "the harness reported the interrupt request")
    check.expect(any(e.kind == "user" and e.raw.get("interrupt") for e in events),
                 "the user event carries interrupt=True")
    check.expect(sum(e.kind == "result" for e in events) == 1,
                 "interrupt + continue produced exactly one result")
    check.expect("OK" in (run.result.text or "").upper() and not run.result.is_error,
                 f"the reply to the interrupting message arrived in the same run: {run.result.text!r}")
    check.expect(run.result.cost_usd is None or run.result.cost_usd > 0,
                 f"accounting covers the whole run (cost_usd={run.result.cost_usd})")

    # 3. stop
    run = session.send(STOP_TASK, secrets=secrets)
    stopped_at = {}

    async def stop():
        # Give the model time to write the file first; the sleep is the tool we interrupt.
        for _ in range(60):
            if (session.workspace / "note.txt").exists():
                break
            await asyncio.sleep(0.5)
        stopped_at["t"] = time.monotonic()
        await session.interrupt()
        stopped_at["returned"] = time.monotonic()

    events = await _drive(run, on_tool_use=stop, label="stop", harness=harness)
    if "returned" in stopped_at:
        check.latency["stop: interrupt() -> returned"] = stopped_at["returned"] - stopped_at["t"]
    check.expect(run.done and session.stop_reason == StopReason.INTERRUPTED,
                 f"interrupt() ended the turn as INTERRUPTED (got {session.stop_reason})")
    check.expect(run.result is not None and not run.result.is_error, "the interrupted result is not an error")
    check.expect(any((e.raw or {}).get("event") == "checkpoint_saved" for e in events),
                 "a checkpoint was saved for the interrupted turn")
    check.expect((session.workspace / "note.txt").exists(),
                 "the interrupted turn's file exists in the workspace")
    check.expect(run.result.num_turns >= 1, f"accounting kept (num_turns={run.result.num_turns})")

    # 4. resume from the interrupted turn's checkpoint
    run = session.send(RESUME_MSG, secrets=secrets)
    events = await _drive(run, label="resume", harness=harness)
    check.expect("note.txt" in (run.result.text or ""),
                 f"a later send() resumed with the interrupted turn's workspace: {run.result.text!r}")
    check.expect(session.stop_reason == StopReason.NEEDS_INPUT, "session idle again after the resume")
    return check


async def main() -> int:
    harnesses = [h.strip() for h in os.environ.get("HARNESSES", "claude-code,codex").split(",") if h.strip()]
    checks = await asyncio.gather(*(probe(h) for h in harnesses))
    failed = [(c.harness, f) for c in checks for f in c.failures]
    print()
    for c in checks:
        print(f"{c.harness}: {'PASS' if not c.failures else 'FAIL'} ({len(c.failures)} failed checks)")
    print()
    print("Latency (seconds, local runtime; 'send' = the moment send()/interrupt() was called):")
    for c in checks:
        for name, secs in c.latency.items():
            print(f"  [{c.harness}] {name:45} {secs:6.1f}")
    for harness, what in failed:
        print(f"  FAIL [{harness}] {what}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

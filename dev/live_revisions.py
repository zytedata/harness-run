"""Live probe for runtime revisions — deploy twice, verify versioning, roll back, tear down.

The revision surface (``deploy`` updating an engine instead of creating one,
``engine.versions()``, ``engine.set_traffic``, ``get_engine(version=…)``) is control-plane
behavior the offline suite can only fake. This probe checks it on real infrastructure, in
the shape TESTING.md prescribes for a bespoke live check: one throwaway engine, cheap model,
tight caps, deleted in ``finally``. It validates:

  * **deploy #1** creates the engine and it has exactly one runtime revision, serving.
  * **deploy #2** of the same ``spec.name`` UPDATES it — same engine resource, a second
    revision, and the new one is what serves (the platform's always-latest default).
  * ``engine.set_traffic(old)`` rolls traffic back; ``engine.version`` follows it, and
    ``get_engine(version=new)`` now fails the assertion while ``version=old`` passes.
  * a turn actually runs after the rollback (asyncQuery is engine-level, so this is what
    proves traffic config and the run plane agree).
  * ``set_traffic()`` restores always-latest and the pruning path (``delete_version``) works
    on the now non-serving revision.

Configure via env (defaults are the shared my-project test setup):
  PROJECT, LOCATION, IMPERSONATE_SA (optional),
  SUFFIX (engine-name suffix; defaults to your username, so contributors don't collide).

Run:
  make live-revisions                   # or:
  .venv/bin/python dev/live_revisions.py

Costs real money and ~10 min: TWO sequential ~4 min builds (the second must follow the
first — it's the update under test) + a few cents of Haiku.
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
NAME = f"ratk-rev-{SUFFIX}"

TASK = 'Run `python3 -c "print(6 * 7)"` in the shell and reply with just the number it prints.'

_checks: list[tuple[str, bool]] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    _checks.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{f' — {detail}' if detail else ''}", flush=True)
    return bool(ok)


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


def _spec(note: str) -> AgentSpec:
    # Haiku + tight caps: the point is the control-plane round-trip, not the model's work.
    # The appended note differs per deploy so the two revisions are genuinely different code.
    return AgentSpec(
        name=NAME,
        model="claude-haiku-4-5",
        max_turns=8,
        max_budget_usd=1.0,
        system_prompt=f"Be concise. (build: {note})",
    )


def _deploy(note: str, credentials):
    print(f"{time.strftime('%H:%M:%S')} deploying ({note}) ...", flush=True)
    t0 = time.time()
    engine = gemini.deploy(_spec(note), PROJECT, LOCATION, credentials=credentials)
    print(f"  deployed in {time.time() - t0:.0f}s -> {engine.resource}", flush=True)
    return engine


async def _run_a_turn(engine) -> bool:
    session = engine.start_session()
    run = session.run(TASK)
    async for ev in run:
        summary = " ".join((ev.summary or "").split())[:90]
        print(f"    {ev.kind:11} {summary}", flush=True)
    result = run.result
    text = " ".join((result.text or "").split())
    return (not result.is_error) and bool(result.num_turns) and "42" in text


async def main() -> int:
    credentials = _credentials()
    engine = None
    try:
        engine = _deploy("one", credentials)
        first_resource = engine.resource
        v1 = engine.versions()
        check("deploy #1 mints one revision", len(v1) == 1, f"versions={v1}")
        check("that revision serves", engine.version == v1[0], f"version={engine.version}")

        engine = _deploy("two", credentials)
        check("deploy #2 reuses the engine", engine.resource == first_resource, engine.resource)
        v2 = engine.versions()
        check("deploy #2 mints a second revision", len(v2) == 2, f"versions={v2}")
        newest, previous = v2[0], v2[1]
        check("the new revision serves", engine.version == newest, f"version={engine.version}")
        check("the first revision is still there", previous in v2 and previous == v1[0])

        print(f"{time.strftime('%H:%M:%S')} rolling traffic back to {previous} ...", flush=True)
        engine.set_traffic(previous)
        rolled = gemini.get_engine(
            NAME, project=PROJECT, location=LOCATION, version=previous, credentials=credentials
        )
        check("get_engine(version=previous) resolves", rolled.version == previous)
        try:
            gemini.get_engine(
                NAME, project=PROJECT, location=LOCATION, version=newest, credentials=credentials
            )
            check("get_engine(version=newest) rejects the non-serving pin", False)
        except ValueError as exc:
            check("get_engine(version=newest) rejects the non-serving pin", True, str(exc)[:60])

        print(f"{time.strftime('%H:%M:%S')} running a turn on the rolled-back engine ...", flush=True)
        check("a turn runs after the rollback", await _run_a_turn(rolled))

        engine.set_traffic()  # back to always-latest
        check("set_traffic() restores always-latest", engine.versions()[0] == newest
              and gemini.get_engine(NAME, project=PROJECT, location=LOCATION,
                                    version=newest, credentials=credentials).version == newest)

        engine.delete_version(previous)
        check("delete_version prunes a non-serving revision", previous not in engine.versions())
    except Exception:
        print(f"PROBE FAILED:\n{traceback.format_exc()}", flush=True)
        _checks.append(("probe completed", False))
    finally:
        if engine is not None:
            try:
                engine.delete()
                print("engine deleted", flush=True)
            except Exception:
                print("TEARDOWN FAILED — delete the engine manually! "
                      f"gemini.get_engine({NAME!r}, ...).delete()\n{traceback.format_exc()}",
                      flush=True)

    print("\n=== VERDICTS ===", flush=True)
    for label, ok in _checks:
        print(f"  {'PASS' if ok else 'FAIL'}: {label}", flush=True)
    return 0 if _checks and all(ok for _, ok in _checks) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

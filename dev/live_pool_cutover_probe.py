"""Live probe for the warm-pool redeploy cutover (issue #38 / PR #39).

The #38 fix makes each ``deploy(warm_pool=True)`` mint a generation-scoped dispatch
pair, retire the previous pair once the new pool is filled, and makes ``get_engine``
discover the live pair from the deployed env. All of that is control-plane + worker
behavior the offline suite can only fake; this probe checks it on real infrastructure:

  * **deploy #1** creates a warm engine whose subscription carries a generation token,
    and ``wait_until_warm`` sees the generation-scoped readiness marker.
  * **deploy #2** (same name) mints a DIFFERENT pair, deletes the old one, and the old
    idle worker exits within a couple of minutes (claim failure on the deleted
    subscription) instead of lingering for ``pool_max_wait_s``.
  * ``get_engine(warm_pool=True)`` discovers deploy #2's subscription from the env.
  * a turn dispatched through that handle runs on the NEW revision — its system prompt
    carries a per-deploy marker, and the reply must be the new deploy's marker (the
    exact regression from #38: post-redeploy turns served with the old baked spec).

Configure via env (defaults are the shared my-project test setup):
  PROJECT, LOCATION, IMPERSONATE_SA (optional),
  SUFFIX (engine-name suffix; defaults to your username, so contributors don't collide).

Run:
  make live-pool-cutover                # or:
  .venv/bin/python dev/live_pool_cutover_probe.py

Costs real money and ~20 min: TWO sequential ~4 min builds (the second must follow the
first — it's the cutover under test), two warm-pool fills, + a few cents of Haiku.
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
NAME = f"ratk-cutover-{SUFFIX}"

TASK = "What is your revision marker? Reply with exactly the marker and nothing else."

# The generation token new_generation() mints into the subscription name.
_GENERATION_SUB = re.compile(r"-[0-9a-f]{8}-dispatch-sub$")

# How long the old idle worker gets to notice its subscription is gone and exit. The
# claim long-poll is 5 s, so failure surfaces within seconds; the job needs a little
# longer to be REPORTED terminal by the platform.
_OLD_WORKER_EXIT_S = 300.0

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


def _spec(marker: str) -> AgentSpec:
    # Haiku + tight caps: the point is the cutover, not the model's work. The marker
    # differs per deploy, so which revision served a turn is visible in its reply.
    return AgentSpec(
        name=NAME,
        model="claude-haiku-4-5",
        max_turns=8,
        max_budget_usd=1.0,
        system_prompt=(
            f"Be concise. Your revision marker is {marker}. When asked for your "
            "revision marker, reply with exactly it and nothing else."
        ),
    )


def _deploy(marker: str, credentials):
    print(f"{time.strftime('%H:%M:%S')} deploying ({marker}, warm_pool=True) ...", flush=True)
    t0 = time.time()
    engine = gemini.deploy(
        _spec(marker), PROJECT, LOCATION, warm_pool=True, pool_size=1, credentials=credentials
    )
    print(f"  deployed in {time.time() - t0:.0f}s -> {engine.resource}", flush=True)
    print(f"  dispatch subscription: {engine._subscription}", flush=True)
    return engine


def _subscription_exists(subscription: str, credentials) -> bool:
    from google.api_core import exceptions as gax
    from google.cloud import pubsub_v1

    kwargs = {"credentials": credentials} if credentials else {}
    try:
        pubsub_v1.SubscriberClient(**kwargs).get_subscription(subscription=subscription)
        return True
    except gax.NotFound:
        return False


def _await_worker_exit(engine, jobs: list[str], timeout: float) -> str | None:
    """Poll the tracked pool-worker jobs until all are terminal; return the last status."""
    ae = engine._agent_engines()
    deadline = time.monotonic() + timeout
    status = None
    while time.monotonic() < deadline:
        statuses = []
        for job in jobs:
            try:
                statuses.append(getattr(ae.check_query_job(name=job), "status", None))
            except Exception:  # noqa: BLE001 — a finished op can stop resolving; call it done
                statuses.append("SUCCESS")
        status = ",".join(str(s) for s in statuses)
        if all(s in ("SUCCESS", "FAILED") for s in statuses):
            return status
        time.sleep(10)
    return status


async def _run_a_turn(engine) -> str:
    session = engine.start_session()
    run = session.run(TASK)
    async for ev in run:
        summary = " ".join((ev.summary or "").split())[:90]
        print(f"    {ev.kind:11} {summary}", flush=True)
    result = run.result
    if result.is_error:
        return f"<error: {' '.join((result.text or '').split())[:120]}>"
    return " ".join((result.text or "").split())


async def main() -> int:
    credentials = _credentials()
    engine = None
    try:
        engine = _deploy("REV1", credentials)
        sub1, jobs1 = engine._subscription, list(engine._pool_jobs)
        check("deploy #1 subscription is generation-scoped", bool(_GENERATION_SUB.search(sub1)),
              sub1.rsplit("/", 1)[-1])

        t0 = time.time()
        warm = await asyncio.to_thread(engine.wait_until_warm)
        check("wait_until_warm sees the generation-scoped marker", warm,
              f"after {time.time() - t0:.0f}s")

        engine = _deploy("REV2", credentials)
        sub2 = engine._subscription
        check("deploy #2 mints a different pair", bool(_GENERATION_SUB.search(sub2))
              and sub2 != sub1, sub2.rsplit("/", 1)[-1])
        check("the old subscription is deleted", not _subscription_exists(sub1, credentials))

        print(f"{time.strftime('%H:%M:%S')} waiting for the old idle worker to exit ...",
              flush=True)
        status = _await_worker_exit(engine, jobs1, _OLD_WORKER_EXIT_S)
        check("the old idle worker exits promptly (not pool_max_wait_s)",
              status is not None and all(s in ("SUCCESS", "FAILED")
                                         for s in status.split(",")),
              f"job status={status}")

        t0 = time.time()
        warm = await asyncio.to_thread(engine.wait_until_warm)
        check("the new pool reports warm", warm, f"after {time.time() - t0:.0f}s")

        looked_up = gemini.get_engine(
            NAME, project=PROJECT, location=LOCATION, warm_pool=True, credentials=credentials
        )
        check("get_engine discovers deploy #2's subscription",
              looked_up._subscription == sub2,
              looked_up._subscription.rsplit("/", 1)[-1] if looked_up._subscription else "None")

        print(f"{time.strftime('%H:%M:%S')} running a turn through the looked-up handle ...",
              flush=True)
        text = await _run_a_turn(looked_up)
        check("the post-redeploy turn ran on the NEW revision",
              "REV2" in text and "REV1" not in text, f"reply={text[:80]!r}")
    except Exception:
        print(f"PROBE FAILED:\n{traceback.format_exc()}", flush=True)
        _checks.append(("probe completed", False))
    finally:
        if engine is not None:
            try:
                engine.delete(delete_pool_resources=True)
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

"""Live smoke test — deploy throwaway engines from THIS checkout, run turns, tear down.

The offline suite (`make test`) can't catch platform-contract breakage: the Agent Runtime
side changes underneath us (see TESTING.md). This script is the standard live check for a
branch that touches any deploy/runtime contract — it validates, on real infrastructure:

  * ``gemini.deploy``: packaging, ``AgentEngineConfig``, the pickled ADK app template
  * worker startup: the engine container unpickles the staged app and resolves
    ``async_stream_query`` (the explicit ``class_method`` both dispatch paths use)
  * a full turn round-trip per mode — **cold** (``run_query_job``, the prod default) and
    **warm** (pub/sub dispatch to a pre-warmed pool worker)
  * **session/turn config transport** (three checks on one extra session per mode):
      - ``get_engine(name)`` is pure addressing; ``start_session(config=SessionConfig)``
        binds a run-time system prompt that plants a marker — the reply must carry it,
        proving the worker ran the session's config, not the deploy-baked spec;
      - a second turn on the SAME session passes ``TurnConfig(output_schema=...)`` — the
        worker steers the model to a JSON final message and the client parses it
        (``result.structured_output``), proving the per-turn overlay reaches both sides;
        the session config must STILL be in force: the schema leaves no room for the
        marker in the text, so its persistence shows in the turn's ``effective_spec``
        echo, whose system prompt must still carry the marker;
      - both turns must stream the worker's ``effective_spec`` echo event carrying the
        config pointers — the durable ground-truth record of what actually ran.

Pass criteria per engine: terminal result with ``error=False``, ``turns > 0``, and the
expected answer in the text (plus the config checks above). Engines are deleted in
``finally`` (the warm one including its dispatch topic/sub); exit code is non-zero if any
check fails.

Configure via env (defaults are the shared my-project test setup):
  PROJECT, LOCATION, IMPERSONATE_SA (optional), MODE=cold|warm|both (default both),
  SUFFIX (engine-name suffix; defaults to your username, so contributors don't collide).

Run:
  make live-smoke                       # or:
  .venv/bin/python dev/live_smoke.py

Costs real money and ~15 min: two ~4 min builds (parallel) + a few cents of Haiku turns.
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

from remote_agent_toolkit import AgentSpec, SessionConfig, SystemPrompt, TurnConfig, gemini

PROJECT = os.environ.get("PROJECT", "my-project")
LOCATION = os.environ.get("LOCATION", "us-central1")
MODE = os.environ.get("MODE", "both")
SUFFIX = re.sub(r"[^a-z0-9-]", "-", (os.environ.get("SUFFIX") or getpass.getuser()).lower())

TASK = (
    'Run `python3 -c "print(6 * 7)"` in the shell and reply with just the number it prints.'
)
JSON_TASK = (
    'Run `python3 -c "print(6 * 7)"` in the shell, then reply with a JSON object of the '
    'form {"answer": <the number it printed>}.'
)
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "integer"}},
    "required": ["answer"],
}

# The session-config check plants this marker via a run-time system prompt; seeing it in
# the reply proves the worker executed the session's config, not the deploy-baked spec.
MARKER = "SESSION-CONFIG-OK"


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


async def _drive(label: str, run):
    """Stream a run to completion; return (result, echo_events)."""
    t0 = time.time()
    echoes = []
    async for ev in run:
        if (ev.raw or {}).get("event") == "effective_spec":
            echoes.append(ev.raw)
        summary = " ".join((ev.summary or "").split())[:90]
        print(f"[{label}] {time.strftime('%H:%M:%S')} {ev.kind:11} {summary}", flush=True)
    r = run.result
    text = " ".join((r.text or "").split())
    print(
        f"[{label}] RESULT after {time.time() - t0:.0f}s: text={text[:120]!r} "
        f"turns={r.num_turns} cost={'unknown' if r.cost_usd is None else f'${r.cost_usd:.4f}'} "
        f"error={r.is_error} "
        f"structured={r.structured_output!r}",
        flush=True,
    )
    return r, echoes


async def _exercise_baked(label: str, engine) -> bool:
    """One plain turn on the deploy handle: the baked spec runs, nothing is staged."""
    session = engine.start_session()
    print(f"[{label}] {time.strftime('%H:%M:%S')} session started, sending turn", flush=True)
    r, _ = await _drive(label, session.run(TASK))
    text = " ".join((r.text or "").split())
    return (not r.is_error) and bool(r.num_turns) and "42" in text


async def _exercise_configs(mode: str, name: str, credentials, verdicts) -> None:
    """The config checks: session config binds; turn config overlays; echoes stream."""
    # Pure addressing: no spec anywhere — the session config is the only run-time input.
    engine = await asyncio.to_thread(
        gemini.get_engine, name, PROJECT, LOCATION,
        warm_pool=(mode == "warm"), credentials=credentials,
    )
    session = engine.start_session(config=SessionConfig(
        system_prompt=SystemPrompt.inherit(
            append=f"CRITICAL: end your final reply with the exact token {MARKER}."
        ),
    ))

    label = f"{mode}-session-config"
    try:
        r, echoes = await _drive(label, session.run(TASK))
        text = " ".join((r.text or "").split())
        ok = (not r.is_error) and bool(r.num_turns) and "42" in text and MARKER in text
        # The worker's ground-truth echo must confirm what ran and where it came from.
        ok = ok and any(
            e.get("session_config_gcs") and MARKER in str(e["spec"].get("system_prompt"))
            for e in echoes
        )
        verdicts[label] = ok
    except Exception:
        print(f"[{label}] RUN FAILED:\n{traceback.format_exc()}", flush=True)
        verdicts[label] = False
        return

    label = f"{mode}-turn-config"
    try:
        r, echoes = await _drive(label, session.send(
            JSON_TASK,
            config=TurnConfig(output_schema=ANSWER_SCHEMA, reasoning_effort="low"),
        ))
        ok = (not r.is_error) and bool(r.num_turns)
        # The per-turn overlay reached the worker (JSON steering) and the client (parse).
        ok = ok and r.structured_output == {"answer": 42}
        # The SESSION config persists across turns. The schema steers the final message to
        # pure JSON, so the marker cannot appear in the text — the proof lives in the
        # worker's echo: the effective spec of THIS turn must still carry the marker prompt.
        ok = ok and any(
            e.get("turn_config_gcs") and e.get("session_config_gcs")
            and e["spec"].get("reasoning_effort") == "low"
            and MARKER in str(e["spec"].get("system_prompt"))
            for e in echoes
        )
        verdicts[label] = ok
    except Exception:
        print(f"[{label}] RUN FAILED:\n{traceback.format_exc()}", flush=True)
        verdicts[label] = False


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
                verdicts[mode] = await _exercise_baked(mode, engines[mode])
            except Exception:
                print(f"[{mode}] RUN FAILED:\n{traceback.format_exc()}", flush=True)
                verdicts[mode] = False
            await _exercise_configs(
                mode, f"ratk-smoke-{mode}-{SUFFIX}", credentials, verdicts
            )

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
    checks = [
        k for mode in modes for k in (mode, f"{mode}-session-config", f"{mode}-turn-config")
    ]
    for check in checks:
        print(f"  {check}: {'PASS' if verdicts.get(check) else 'FAIL'}", flush=True)
    return 0 if all(verdicts.get(check) for check in checks) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

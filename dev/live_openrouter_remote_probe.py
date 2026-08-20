"""Live probe — the OpenRouter models on the **Gemini Agent Runtime**, with the remote-only
visibility surface checked alongside them.

COSTS REAL MONEY (~$0.30: one ~5 min engine build plus ~$0.08 of turns) and takes
**~35-40 min** — measured, not estimated: the build is ~5 min and each of the seven turns
pays cold-start latency (100-420 s apiece, and the platform's variance is wide). Give it a
generous timeout; see the teardown note below for what happens if you don't. Run it by
hand, locally, sparingly — never in CI. The companion
`dev/live_openrouter_probe.py` covers the same models on the local runtime and is the
cheaper first check; this one exists for what only the remote path has.

Why a separate remote probe: on the local runtime the harness runs in-process, so a turn
proves the provider wiring and nothing else. Remotely the same turn also has to survive
being packaged into an engine, unpickled by a worker, handed its key through the GCS
secrets handoff, and streamed back over GCS — and the platform's own visibility
(`effective_spec`, resource samples, traces) has to carry the OpenRouter model the same
way it carries a GPT or Claude one. None of that is exercised locally.

**One engine, four models.** `model` is a per-turn knob, so a single deployment runs all
four via `TurnConfig(model=...)` — one build instead of four, and it demonstrates the claim
that a session can move between models without a redeploy.

What it checks, per model (a turn each on one engine):
  * terminal result with `is_error=False`, the right answer, a `tool_use` event
  * `cost_usd` is a real number (so `max_budget_usd` is enforceable remotely too)
  * the worker's `effective_spec` echo names the model we asked for — the ground-truth
    record that the per-turn override reached the worker, not just the client

Then, once per run:
  * **structured output** on GLM-5.3, the model that ignores OpenRouter's unenforced
    json_schema format (see `harness/codex.py`)
  * **resume**: `run()` then `send()` on one session must remember a token — the workspace
    snapshot and the Codex rollout both round-trip through the blob store
  * **remote-only visibility**, compared against what a GPT/Claude turn gives:
      - `session.resource_samples()` returns worker CPU/RAM rows
      - `memory_peak_bytes` is stamped into the terminal result
      - `session.history()` replays the turn's events after the fact
      - Cloud Trace's root span for THAT session names the OpenRouter model and
        carries its cost (the same span contract a GPT/Claude turn gets)

Pass criteria: every check passes. The engine is deleted in `finally`; a failed teardown
prints loudly, because an engine bills while it exists.

Configure via env:
  OPENROUTER_API_KEY   required
  PROJECT, LOCATION    default my-project / us-central1
  SUFFIX               engine-name suffix (defaults to your username)
  IMPERSONATE_SA       optional service account to impersonate
  KEEP=1               skip teardown (debugging — you then own the cleanup)

Run:
  OPENROUTER_API_KEY=... make live-openrouter-remote
"""

from __future__ import annotations

import asyncio
import getpass
import os
import re
import signal
import sys
import time
import traceback

from remote_agent_toolkit import AgentSpec, TurnConfig, gemini

PROJECT = os.environ.get("PROJECT", "my-project")
LOCATION = os.environ.get("LOCATION", "us-central1")
SUFFIX = re.sub(r"[^a-z0-9-]", "-", (os.environ.get("SUFFIX") or getpass.getuser()).lower())
NAME = f"ratk-openrouter-{SUFFIX}"

MODELS = [
    m.strip()
    for m in os.environ.get(
        "MODELS",
        "openrouter/moonshotai/kimi-k3,"
        "openrouter/z-ai/glm-5.3,"
        "openrouter/deepseek/deepseek-v4-flash,"
        "openrouter/deepseek/deepseek-v4-pro",
    ).split(",")
    if m.strip()
]
# The engine is baked with the cheapest of them; every turn overrides the model anyway.
BAKED_MODEL = "openrouter/deepseek/deepseek-v4-flash"
SCHEMA_MODEL = "openrouter/z-ai/glm-5.3"  # ignores the unenforced json_schema format
TOKEN = "BANANA-77"

TASK = 'Run `python3 -c "print(6 * 7)"` in the shell and reply with just the number it prints.'
SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "integer"}, "note": {"type": "string"}},
    "required": ["answer", "note"],
    "additionalProperties": False,
}

# (label, state) where state is "PASS" / "FAIL" / "SKIP"; SKIP never fails the run.
VERDICTS: list[tuple[str, str, str]] = []


def check(label: str, ok: bool, note: str = "") -> bool:
    state = "PASS" if ok else "FAIL"
    VERDICTS.append((label, state, note))
    print(f"  {state}  {label}" + (f"  — {note}" if note else ""), flush=True)
    return bool(ok)


def skip(label: str, note: str) -> None:
    """Record a check we could not run (a missing optional reader, not a failure)."""
    VERDICTS.append((label, "SKIP", note))
    print(f"  SKIP  {label}  — {note}", flush=True)


async def _drive(label: str, run):
    """Stream one turn; return (result, kinds, effective_spec echoes, terminal event raw).

    The terminal event's ``raw`` is returned because the resource enrichment
    (``memory_peak_bytes`` and friends) is stamped there, not onto ``RunResult``.
    """
    kinds, echoes, final_raw = [], [], {}
    t0 = time.time()
    async for ev in run:
        kinds.append(ev.kind)
        raw = ev.raw or {}
        if raw.get("event") == "effective_spec":
            echoes.append(raw)
        if ev.kind == "result":
            final_raw = raw
        summary = " ".join((ev.summary or "").split())[:80]
        print(f"[{label}] {time.strftime('%H:%M:%S')} {ev.kind:11} {summary}", flush=True)
    r = run.result
    print(
        f"[{label}] done in {time.time() - t0:.0f}s: turns={r.num_turns} "
        f"cost=${r.cost_usd or 0:.4f} error={r.is_error}",
        flush=True,
    )
    return r, kinds, echoes, final_raw


async def _check_model(engine, model: str, key: str) -> None:
    """One turn on `model` via a per-turn override."""
    label = model.removeprefix("openrouter/")
    session = engine.start_session()
    r, kinds, echoes, _ = await _drive(
        label,
        session.run(TASK, secrets={"OPENROUTER_API_KEY": key}, config=TurnConfig(model=model)),
    )
    text = " ".join((r.text or "").split())
    check(f"{label}: turn completed", not r.is_error and bool(r.num_turns), f"text={text[:40]!r}")
    check(f"{label}: answered 42", "42" in text)
    check(f"{label}: called a tool", "tool_use" in kinds)
    check(f"{label}: cost priced", isinstance(r.cost_usd, float), f"${r.cost_usd or 0:.4f}")
    # The echo is the worker's own record of the merged spec it executed.
    echoed = [e.get("spec", {}).get("model") for e in echoes]
    check(f"{label}: effective_spec echoes the model", model in echoed, f"echoed={echoed}")


async def _check_structured_output(engine, key: str) -> None:
    label = f"{SCHEMA_MODEL.removeprefix('openrouter/')} schema"
    session = engine.start_session()
    r, _, _, _ = await _drive(
        label,
        session.run(
            'Run `python3 -c "print(6 * 7)"`. Return the number as `answer` and a one-word `note`.',
            secrets={"OPENROUTER_API_KEY": key},
            config=TurnConfig(model=SCHEMA_MODEL, output_schema=SCHEMA),
        ),
    )
    out = r.structured_output
    check(
        "structured output parses remotely",
        isinstance(out, dict) and out.get("answer") == 42,
        f"structured_output={out!r}",
    )


async def _check_resume(engine, key: str) -> None:
    session = engine.start_session()
    secrets = {"OPENROUTER_API_KEY": key}
    await _drive("resume-1", session.run(f"Remember the word {TOKEN}. Reply with just: stored",
                                        secrets=secrets))
    r2, _, _, final_raw = await _drive(
        "resume-2",
        session.send("What word did I ask you to remember? Reply with just the word.",
                     secrets=secrets),
    )
    text = " ".join((r2.text or "").split())
    check("resume remembers across turns", TOKEN in text.upper(), f"turn2={text[:40]!r}")
    return session, final_raw


async def _check_visibility(session, final_raw: dict, expect_model: str) -> None:
    """The remote-only surface: resource samples, memory peak, history, traces."""
    samples = await asyncio.to_thread(session.resource_samples)
    check(
        "resource_samples returns worker CPU/RAM",
        bool(samples),
        f"{len(samples)} sample(s)" + (f", keys={sorted(samples[-1])[:4]}" if samples else ""),
    )
    # Stamped onto the terminal event's raw by the worker's cgroup sampler.
    peak = final_raw.get("memory_peak_bytes")
    check("memory_peak_bytes stamped on the result", peak is not None, f"peak={peak}")

    history = await asyncio.to_thread(session.history)
    check("history replays the turn's events", bool(history), f"{len(history)} event(s)")

    ok, note = await asyncio.to_thread(_trace_check, session.session_id, expect_model)
    if ok is None:
        skip("Cloud Trace root span carries the model and cost", note)
    else:
        check("Cloud Trace root span carries the model and cost", ok, note)


def _trace_check(session_id: str, expect_model: str) -> tuple[bool | None, str]:
    """Find THIS session's root span and confirm what it reports.

    Scoped to the session on purpose: "some invoke_agent traces exist" proves nothing —
    the project is shared and other engines emit them too. The root span must name the
    OpenRouter model we ran and carry a cost, which is the same contract a GPT or Claude
    turn gets (``runtime/gemini/tracing.py``).

    Returns ``(None, reason)`` when the optional reader library isn't installed — a
    missing dev-only dependency is not a parity failure. Install it with
    ``uv pip install google-cloud-trace`` to turn this into a real check.
    """
    try:
        from google.cloud import trace_v1
    except ImportError:
        return None, "google-cloud-trace not installed (console: engine → Traces tab)"
    try:
        client = trace_v1.TraceServiceClient()
        deadline = time.time() + 120  # spans land a little after the turn
        while True:
            for tr in client.list_traces(request={
                "project_id": PROJECT,
                "filter": "span:invoke_agent",
                "view": trace_v1.ListTracesRequest.ViewType.COMPLETE,
                "page_size": 50,
            }):
                for span in tr.spans:
                    labels = dict(span.labels)
                    if labels.get("gen_ai.conversation.id") != str(session_id):
                        continue
                    model = labels.get("gen_ai.request.model")
                    cost = labels.get("rat.cost_usd")
                    note = f"model={model} cost={cost} turns={labels.get('rat.num_turns')}"
                    return (model == expect_model and cost is not None), note
            if time.time() > deadline:
                return False, f"no root span for session {session_id} within 120s"
            time.sleep(15)
    except Exception as exc:  # noqa: BLE001 — a read failure is worth reporting, not raising
        return False, f"Cloud Trace read failed: {type(exc).__name__}: {str(exc)[:120]}"


_TORN_DOWN = False


def _teardown(engine) -> None:
    """Delete the engine, once, from either the normal path or a signal.

    `finally` alone is not enough: this probe runs for ~35 min, so it is routinely wrapped
    in a `timeout` or interrupted — and SIGTERM/SIGINT kill the process without running
    `finally`, leaving an engine billing. (Observed: a 2400 s `timeout` fired mid-run and
    leaked `ratk-openrouter-<user>`.) So teardown is idempotent and also reachable from a
    signal handler.
    """
    global _TORN_DOWN
    if _TORN_DOWN or engine is None:
        return
    _TORN_DOWN = True
    if os.environ.get("KEEP") == "1":
        print(f"\nKEEP=1 — engine {NAME} left running; delete it yourself.", flush=True)
        return
    try:
        print(f"\n{time.strftime('%H:%M:%S')} deleting {NAME} ...", flush=True)
        engine.delete()
        print("teardown OK", flush=True)
    except Exception:  # noqa: BLE001 — always say so loudly; an engine bills while it exists
        traceback.print_exc()
        print(f"\n!!! TEARDOWN FAILED — delete the engine {NAME} manually !!!", flush=True)


def _install_signal_teardown(engine) -> None:
    """On SIGTERM/SIGINT: delete the engine, then exit non-zero."""

    def handler(signum, _frame):
        print(f"\nreceived signal {signum} — tearing down before exit", flush=True)
        _teardown(engine)
        raise SystemExit(130)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):  # not the main thread / unsupported
            pass


async def main() -> int:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("OPENROUTER_API_KEY is not set — nothing to probe.", file=sys.stderr)
        return 2

    credentials = None
    if sa := os.environ.get("IMPERSONATE_SA"):
        from google.auth import impersonated_credentials, default as adc

        source, _ = adc(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        credentials = impersonated_credentials.Credentials(
            source_credentials=source,
            target_principal=sa,
            target_scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )

    spec = AgentSpec(
        name=NAME,
        model=BAKED_MODEL,
        harness="codex",
        checkpoint=True,  # the resume check needs conversation + workspace continuity
        max_turns=8,
        max_budget_usd=0.50,
    )
    print(f"{time.strftime('%H:%M:%S')} deploying {NAME} (~4 min build) ...", flush=True)
    t0 = time.time()
    engine = await asyncio.to_thread(
        gemini.deploy, spec, PROJECT, LOCATION, credentials=credentials
    )
    print(f"{time.strftime('%H:%M:%S')} deployed in {time.time() - t0:.0f}s", flush=True)
    _install_signal_teardown(engine)

    try:
        for model in MODELS:
            await _check_model(engine, model, key)
        await _check_structured_output(engine, key)
        last_session, final_raw = await _check_resume(engine, key)
        await _check_visibility(last_session, final_raw, BAKED_MODEL)
    except Exception:  # noqa: BLE001 — always reach teardown
        traceback.print_exc()
        check("probe ran without exceptions", False, "see traceback above")
    finally:
        _teardown(engine)

    print("\n=== VERDICTS ===")
    for label, state, note in VERDICTS:
        print(f"{state}  {label}" + (f"  — {note}" if note else ""))
    failed = [label for label, state, _ in VERDICTS if state == "FAIL"]
    skipped = [label for label, state, _ in VERDICTS if state == "SKIP"]
    passed = len(VERDICTS) - len(failed) - len(skipped)
    print(f"\n{passed} passed, {len(failed)} failed, {len(skipped)} skipped")
    if failed:
        print("failed:", ", ".join(failed))
    return 0 if VERDICTS and not failed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

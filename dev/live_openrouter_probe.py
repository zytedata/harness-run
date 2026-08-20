"""Live probe — run each blessed OpenRouter model through the codex harness for real.

COSTS REAL MONEY. Run it sparingly, locally, by hand. It must never run in CI: the
offline suite (`make test`) is free and credential-less by design, and that stays true
(see TESTING.md and .github/workflows/ci.yml). One full pass is a few cents — four short
tool-using turns — but it is real spend on the shared OpenRouter account.

Why it exists: the offline tests pin the config the harness EMITS, and every one of those
settings was chosen because OpenRouter rejected the alternative. Only a live turn proves
the settings still add up to a working agent — that the Responses wire is accepted, that
reasoning is enabled the way the endpoint demands, that Codex's web-search tool is really
gone, and that the model actually calls a shell tool rather than just talking. A provider
can change any of that server-side with no diff on our end.

Every model is checked on **both harnesses** — codex reaches OpenRouter through a provider
config, claude-code through OpenRouter's Anthropic-compatible endpoint.

What it validates per model and harness, on the LOCAL runtime (no GCP, no engine build):

  * the turn completes: terminal result with ``is_error=False``
  * the agent reported the right answer ("42"); tool use is recorded but not required,
    since a model may answer from memory
  * cost accounting works: ``cost_usd`` is a real number, meaning the model resolved a
    price, which is also what makes ``max_budget_usd`` enforceable
  * ``num_turns`` is counted (one per token-usage notification)

Then two checks that pin gaps found the hard way:

  * **resume** (cheapest model, both harnesses): the first turn runs a command, then a
    second turn via ``send()`` must recall its result. Codex re-applies the thread args on
    resume, so a provider config that only works on a fresh thread shows up here.
  * **structured output** (GLM-5.3): OpenRouter accepts Codex's json_schema format but
    does not enforce it, and GLM-5.3 ignored it 3/3 times before the harness started
    stating the schema in the developer instructions too. This is the model that catches
    a regression there.

Pass criteria: every model passes every check, and the resume check passes. Exit code is
non-zero otherwise.

All checks run **concurrently** (they are independent turns against different upstreams),
so a pass is roughly one turn's wall clock rather than six. Set ``SERIAL=1`` to run them in
lockstep when debugging one model.

Configure via env:
  OPENROUTER_API_KEY   required — the key the turns bill to
  MODELS               optional comma-separated override of the model list
  SERIAL=1             run checks one at a time instead of concurrently

Run:
  make live-openrouter                              # or:
  OPENROUTER_API_KEY=... .venv/bin/python dev/live_openrouter_probe.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import traceback

from remote_agent_toolkit import AgentSpec, local

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

# Cheap and unambiguous: the answer is only reachable by running the command, so a model
# that just chats fails the tool check.
TASK = 'Run `python3 -c "print(6 * 7)"` in the shell and reply with just the number it prints.'

# The resume check runs on the cheapest model — it exercises the harness, not the model.
RESUME_MODEL = "openrouter/deepseek/deepseek-v4-flash"
# Both harnesses can reach OpenRouter: codex through a provider config, claude-code through
# the Anthropic-compatible endpoint. Every model is checked on both.
HARNESSES = ["codex", "claude-code"]

# The strictest structured-output case we know: this model ignores the unenforced schema.
SCHEMA_MODEL = "openrouter/z-ai/glm-5.3"
SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "integer"}, "note": {"type": "string"}},
    "required": ["answer", "note"],
    "additionalProperties": False,
}


async def _probe(model: str, key: str, harness: str = "codex", attempts: int = 2) -> dict:
    """Run one turn on ``model`` under ``harness``; return a verdict row.

    Retries once on a soft miss (the turn completed but the answer is absent). Under Claude
    Code's large system prompt these models occasionally return no final message, or echo a
    fragment of the scaffolding — measured at roughly one turn in eight, across models. That
    is model fit, not the wiring this probe is here to check, so one retry keeps the signal
    about the wiring. A retry is always reported in the note.
    """
    for attempt in range(1, attempts + 1):
        row = await _probe_once(model, key, harness)
        if row["ok"]:
            if attempt > 1:
                row["note"] = f"passed on attempt {attempt} (first attempt: soft miss)"
            return row
        if attempt < attempts:
            print(f"[{row['model']}] soft miss, retrying once: {row['note'][:70]}", flush=True)
    row["note"] = f"failed {attempts} attempts — {row['note']}"
    return row


async def _probe_once(model: str, key: str, harness: str = "codex") -> dict:
    """One turn; the verdict row for it."""
    label = f"{model.removeprefix('openrouter/')} [{harness}]"
    # Tight caps: the point is the round-trip, not the model's work.
    spec = AgentSpec(
        name="ratk-openrouter-probe",
        model=model,
        harness=harness,
        max_turns=8,
        max_budget_usd=0.50,
    )
    row = {"model": label, "ok": False, "tools": False, "cost": None, "turns": 0, "note": ""}
    t0 = time.time()
    try:
        engine = local.deploy(spec)
        run = engine.start_session().run(TASK, secrets={"OPENROUTER_API_KEY": key})
        async for ev in run:
            if ev.kind == "tool_use":
                row["tools"] = True
            summary = " ".join((ev.summary or "").split())[:90]
            print(f"[{label}] {time.strftime('%H:%M:%S')} {ev.kind:11} {summary}", flush=True)
        r = run.result
        text = " ".join((r.text or "").split())
        row.update(cost=r.cost_usd, turns=r.num_turns or 0)
        row["ok"] = bool(
            not r.is_error and row["turns"] and "42" in text and isinstance(r.cost_usd, float)
        )
        # Tool use is recorded but not required: a model may answer 6*7 from memory
        # (observed with Kimi K3 on claude-code), which is a model choice, not a fault.
        if not row["ok"]:
            row["note"] = (
                f"error={r.is_error} turns={row['turns']} tools={row['tools']} "
                f"cost={r.cost_usd!r} text={text[:60]!r}"
            )
        print(
            f"[{label}] RESULT after {time.time() - t0:.0f}s: text={text[:80]!r} "
            f"turns={row['turns']} cost=${r.cost_usd or 0:.4f} error={r.is_error}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001 — one model's failure must not hide the rest
        row["note"] = f"{type(exc).__name__}: {exc}"
        print(f"[{label}] EXCEPTION after {time.time() - t0:.0f}s", flush=True)
        traceback.print_exc()
    return row


async def _probe_resume(model: str, key: str, harness: str = "codex", attempts: int = 2) -> dict:
    """Two turns on one session: the routing must survive a resume (retries once)."""
    for attempt in range(1, attempts + 1):
        row = await _probe_resume_once(model, key, harness)
        if row["ok"]:
            if attempt > 1:
                row["note"] = f"passed on attempt {attempt} (first attempt: soft miss)"
            return row
        if attempt < attempts:
            print(f"[{row['model']}] soft miss, retrying once", flush=True)
    row["note"] = f"failed {attempts} attempts — {row['note']}"
    return row


async def _probe_resume_once(model: str, key: str, harness: str = "codex") -> dict:
    """One resume pair; the verdict row for it."""
    label = f"{model.removeprefix('openrouter/')} (resume) [{harness}]"
    spec = AgentSpec(
        name="ratk-openrouter-resume",
        model=model,
        harness=harness,
        checkpoint=True,  # conversation continuity is what `send` resumes
        max_turns=6,
        max_budget_usd=0.30,
    )
    row = {"model": label, "ok": False, "tools": True, "cost": None, "turns": 0, "note": ""}
    try:
        session = local.deploy(spec).start_session()
        secrets = {"OPENROUTER_API_KEY": key}
        first = await session.run(TASK, secrets=secrets)
        # `run` starts a fresh turn by design; `send` is the one that continues it.
        # The first turn does real work on purpose: given only "remember this word", a
        # small model under Claude Code's large system prompt often produces no final
        # message at all (measured with DeepSeek v4 Flash), which is a model-fit problem
        # rather than a continuity one and would make this check meaningless.
        second = await session.send(
            "What number did the command print earlier? Reply with just the number.",
            secrets=secrets,
        )
        text = " ".join((second.text or "").split())
        row.update(cost=second.cost_usd, turns=second.num_turns or 0)
        row["ok"] = bool(not first.is_error and not second.is_error and "42" in text)
        if not row["ok"]:
            row["note"] = f"turn1={(first.text or '')[:20]!r} turn2={text[:60]!r} (expected 42)"
        print(f"[{label}] turn2 text={text[:60]!r}", flush=True)
    except Exception as exc:  # noqa: BLE001 — report, don't hide
        row["note"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    return row


async def _probe_schema(model: str, key: str, harness: str = "codex") -> dict:
    """``output_schema`` must yield a parsed object rather than prose."""
    label = f"{model.removeprefix('openrouter/')} (schema) [{harness}]"
    spec = AgentSpec(
        name="ratk-openrouter-schema",
        model=model,
        harness=harness,
        max_turns=6,
        max_budget_usd=0.40,
        output_schema=SCHEMA,
    )
    row = {"model": label, "ok": False, "tools": True, "cost": None, "turns": 0, "note": ""}
    try:
        run = local.deploy(spec).start_session().run(
            'Run `python3 -c "print(6 * 7)"`. Return the number as `answer` and a one-word `note`.',
            secrets={"OPENROUTER_API_KEY": key},
        )
        async for _ in run:
            pass
        r = run.result
        row.update(cost=r.cost_usd, turns=r.num_turns or 0)
        out = r.structured_output
        row["ok"] = bool(not r.is_error and isinstance(out, dict) and out.get("answer") == 42)
        if not row["ok"]:
            row["note"] = f"structured_output={out!r} text={(r.text or '')[:60]!r}"
        print(f"[{label}] structured_output={out!r}", flush=True)
    except Exception as exc:  # noqa: BLE001 — report, don't hide
        row["note"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    return row


async def _probe_preset(harness: str, key: str) -> dict:
    """Pinning plumbing: an `@preset/<slug>` id must reach OpenRouter as a preset.

    Uses a slug that does not exist, so OpenRouter answers `preset_not_found` — which is
    the proof the id arrived as a preset rather than being mangled or rejected locally.
    The turn is expected to fail; what is checked is HOW. Free: no tokens are billed.
    Also asserts the run reports no cost, since a preset cannot be priced.
    """
    label = f"@preset plumbing [{harness}]"
    spec = AgentSpec(
        name="ratk-openrouter-preset",
        model="openrouter/@preset/ratk-does-not-exist",
        harness=harness,
        max_turns=4,
        max_budget_usd=0.20,
    )
    row = {"model": label, "ok": False, "tools": True, "cost": None, "turns": 0, "note": ""}
    try:
        run = local.deploy(spec).start_session().run(
            "say ok", secrets={"OPENROUTER_API_KEY": key}
        )
        saw_cost_unknown = False
        blob = []
        final = None
        async for ev in run:
            blob.append(f"{ev.kind}:{ev.summary}")
            if (ev.raw or {}).get("event") == "cost_unknown":
                saw_cost_unknown = True
            if ev.kind == "result":
                final = ev
        text = " ".join(blob).lower()
        reached = "preset" in text  # OpenRouter's own preset_not_found wording
        # The terminal EVENT carries cost_usd=None for an unpriced model. Note RunResult
        # flattens that to 0.0 (`cost_usd: float = 0.0`), so the event is what to read.
        event_cost = final.cost_usd if final is not None else "no result event"
        row["ok"] = bool(reached and saw_cost_unknown and event_cost is None)
        row["note"] = (
            f"preset error surfaced={reached} cost_unknown={saw_cost_unknown} "
            f"event cost={event_cost!r}"
        )
        print(f"[{label}] {row['note']}", flush=True)
    except Exception as exc:  # noqa: BLE001 — report, don't hide
        row["note"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    return row


async def main() -> int:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("OPENROUTER_API_KEY is not set — nothing to probe.", file=sys.stderr)
        return 2
    print(f"probing {len(MODELS)} model(s) — this spends real money", flush=True)

    # Parallel: these are independent turns against different upstreams, and the wall
    # clock is otherwise the sum of four cold starts. Each line is prefixed with its model
    # so interleaved output stays readable, and one model's failure cannot mask another's
    # (every check returns a row rather than raising). SERIAL=1 restores lockstep output
    # for debugging a single model.
    if os.environ.get("SERIAL") == "1":
        rows = [await _probe(m, key, h) for m in MODELS for h in HARNESSES]
        for h in HARNESSES:
            rows.append(await _probe_resume(RESUME_MODEL, key, h))
            rows.append(await _probe_schema(SCHEMA_MODEL, key, h))
            rows.append(await _probe_preset(h, key))
    else:
        rows = list(
            await asyncio.gather(
                *(_probe(m, key, h) for m in MODELS for h in HARNESSES),
                *(_probe_resume(RESUME_MODEL, key, h) for h in HARNESSES),
                *(_probe_schema(SCHEMA_MODEL, key, h) for h in HARNESSES),
                *(_probe_preset(h, key) for h in HARNESSES),
            )
        )

    print("\n=== VERDICTS ===")
    for row in rows:
        cost = f"${row['cost']:.4f}" if isinstance(row["cost"], float) else "unknown"
        print(
            f"{'PASS' if row['ok'] else 'FAIL'}  {row['model']:34} "
            f"turns={row['turns']:<3} cost={cost:<10} {row['note']}"
        )
    spent = sum(r["cost"] for r in rows if isinstance(r["cost"], float))
    print(f"total reported spend: ${spent:.4f}")
    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

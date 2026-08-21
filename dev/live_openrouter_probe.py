"""Paid local check for the four OpenRouter models on both harnesses.

Each model must call a shell tool, return its output, hide provider credentials, report the
selected upstream, and use OpenRouter's exact cost. Resume, structured output, preset errors,
and budget caps are checked on both harnesses. Checks run concurrently unless ``SERIAL=1``.

Run by hand with ``OPENROUTER_API_KEY=... make live-openrouter``. Never run this in CI.
Use ``MODELS=...`` to limit the model list. To verify a real preset, also set
``OPENROUTER_PRESET_MODEL`` and ``OPENROUTER_EXPECTED_PROVIDER``.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import sys
import time
import traceback

from remote_agent_toolkit import AgentSpec, StopReason, local

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

_TOOL_INPUT = b"ratk-openrouter-tool-check-2026-08-21"
EXPECTED_DIGEST = hashlib.sha256(_TOOL_INPUT).hexdigest()[:16]
EXPECTED = EXPECTED_DIGEST + ":or0:rt0:aa0:ak0"
TASK = (
    "Run this exact command in the shell and reply with only what it prints: "
    '`python3 -c "import hashlib, os; '
    "h=hashlib.sha256(b'ratk-openrouter-tool-check-2026-08-21').hexdigest()[:16]; "
    "s=':or%d:rt%d:aa%d:ak%d' % tuple(int(bool(os.environ.get(k))) for k in "
    "('OPENROUTER_API_KEY','RATK_OPENROUTER_PROXY_TOKEN',"
    "'ANTHROPIC_AUTH_TOKEN','ANTHROPIC_API_KEY')); "
    'print(h+s)"`'
)

# GLM was stable in the controlled runs, which keeps the resume check focused on continuity.
RESUME_MODEL = "openrouter/z-ai/glm-5.3"
BUDGET_MODEL = "openrouter/deepseek/deepseek-v4-flash"
# Both harnesses can reach OpenRouter: codex through a provider config, claude-code through
# the Anthropic-compatible endpoint. Every model is checked on both.
HARNESSES = ["codex", "claude-code"]
PINNED_MODEL = os.environ.get("OPENROUTER_PRESET_MODEL")
EXPECTED_PROVIDER = os.environ.get("OPENROUTER_EXPECTED_PROVIDER")
PROBE_REASONING_EFFORT = os.environ.get("PROBE_REASONING_EFFORT")

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

    Retries once on a soft miss. DeepSeek v4 Flash sometimes returns no final message under
    Claude Code. A retry is always reported in the verdict.
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
        reasoning_effort=PROBE_REASONING_EFFORT,
    )
    row = {"model": label, "ok": False, "tools": False, "cost": None, "turns": 0, "note": ""}
    providers: list[str | None] = []
    requested_models: list[str | None] = []
    request_costs: list[float | None] = []
    price_source = None
    t0 = time.time()
    try:
        engine = local.deploy(spec)
        run = engine.start_session().run(TASK, secrets={"OPENROUTER_API_KEY": key})
        async for ev in run:
            if ev.kind == "tool_use":
                row["tools"] = True
            if (ev.raw or {}).get("event") == "openrouter_request":
                providers.append(ev.raw.get("provider"))
                requested_models.append(ev.raw.get("requested_model"))
                request_costs.append(ev.raw.get("cost_usd"))
            if ev.kind == "result":
                price_source = (ev.raw or {}).get("price_source")
            summary = " ".join((ev.summary or "").split())[:90]
            print(f"[{label}] {time.strftime('%H:%M:%S')} {ev.kind:11} {summary}", flush=True)
        r = run.result
        text = " ".join((r.text or "").split())
        exact_costs = [float(cost) for cost in request_costs if isinstance(cost, (int, float))]
        bare_model = model.removeprefix("openrouter/")
        row.update(cost=r.cost_usd, turns=r.num_turns or 0)
        row["ok"] = bool(
            not r.is_error
            and row["turns"]
            and EXPECTED in text
            and row["tools"]
            and isinstance(r.cost_usd, float)
            and bool(providers)
            and all(providers)
            and bool(requested_models)
            and all(requested == bare_model for requested in requested_models)
            and bool(exact_costs)
            and math.isclose(r.cost_usd, sum(exact_costs), rel_tol=1e-9, abs_tol=1e-12)
            and price_source == "openrouter"
        )
        if not row["ok"]:
            row["note"] = (
                f"error={r.is_error} turns={row['turns']} tools={row['tools']} "
                f"cost={r.cost_usd!r} request_costs={exact_costs} providers={providers} "
                f"requested={requested_models} source={price_source} "
                f"text={text[:60]!r}"
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
        # The first turn does real work so the second turn has an exact value to recall.
        second = await session.send(
            "What text did the command print earlier? Reply with just that text.",
            secrets=secrets,
        )
        text = " ".join((second.text or "").split())
        row.update(cost=second.cost_usd, turns=second.num_turns or 0)
        row["ok"] = bool(not first.is_error and not second.is_error and EXPECTED in text)
        if not row["ok"]:
            row["note"] = (
                f"turn1={(first.text or '')[:20]!r} turn2={text[:60]!r} (expected {EXPECTED})"
            )
        print(f"[{label}] turn2 text={text[:60]!r}", flush=True)
    except Exception as exc:  # noqa: BLE001 — report, don't hide
        row["note"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    return row


async def _probe_schema(model: str, key: str, harness: str = "codex") -> dict:
    """``output_schema`` must yield a parsed object."""
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
        session = local.deploy(spec).start_session()
        run = session.run(
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

    Uses a slug that does not exist, so OpenRouter answers `preset_not_found`. This confirms
    the id reached OpenRouter as a preset.
    The turn is expected to fail; what is checked is HOW. Free: no tokens are billed.
    Also asserts the failed request reports no cost.
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
        run = local.deploy(spec).start_session().run("say ok", secrets={"OPENROUTER_API_KEY": key})
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


async def _probe_budget(harness: str, key: str) -> dict:
    """The exact OpenRouter charge trips the cap."""
    label = f"exact-cost budget [{harness}]"
    spec = AgentSpec(
        name="ratk-openrouter-budget",
        model=BUDGET_MODEL,
        harness=harness,
        system_prompt="Reply briefly.",
        max_turns=2,
        max_budget_usd=0.000001,
    )
    row = {"model": label, "ok": False, "tools": True, "cost": None, "turns": 0, "note": ""}
    try:
        session = local.deploy(spec).start_session()
        run = session.run("Reply with only: ok", secrets={"OPENROUTER_API_KEY": key})
        final = None
        async for event in run:
            if event.kind == "result":
                final = event
        result = run.result
        row.update(cost=result.cost_usd, turns=result.num_turns or 0)
        source = (final.raw or {}).get("price_source") if final else None
        row["ok"] = bool(
            result.is_error
            and session.stop_reason is StopReason.BUDGET_EXCEEDED
            and isinstance(result.cost_usd, float)
            and result.cost_usd > spec.max_budget_usd
            and source == "openrouter"
        )
        row["note"] = f"stop={session.stop_reason} cost={result.cost_usd:.6f} source={source}"
    except Exception as exc:  # noqa: BLE001 — report, don't hide
        row["note"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    return row


async def _probe_pin(harness: str, key: str, model: str, expected_provider: str) -> dict:
    """Run a caller-owned preset and verify every reported upstream matches it."""
    label = f"preset provider [{harness}]"
    spec = AgentSpec(
        name="ratk-openrouter-pinned",
        model=model,
        harness=harness,
        max_turns=4,
        max_budget_usd=0.30,
    )
    row = {"model": label, "ok": False, "tools": False, "cost": None, "turns": 0, "note": ""}
    try:
        run = local.deploy(spec).start_session().run(TASK, secrets={"OPENROUTER_API_KEY": key})
        providers = []
        async for event in run:
            if event.kind == "tool_use":
                row["tools"] = True
            if (event.raw or {}).get("event") == "openrouter_request":
                providers.append(event.raw.get("provider"))
        result = run.result
        row.update(cost=result.cost_usd, turns=result.num_turns or 0)
        wanted = expected_provider.casefold()
        row["ok"] = bool(
            not result.is_error
            and row["tools"]
            and providers
            and all((provider or "").casefold() == wanted for provider in providers)
        )
        row["note"] = f"expected={expected_provider!r} providers={providers!r} tools={row['tools']}"
    except Exception as exc:  # noqa: BLE001 — report every harness in one run
        row["note"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    return row


async def main() -> int:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("OPENROUTER_API_KEY is not set — nothing to probe.", file=sys.stderr)
        return 2
    if bool(PINNED_MODEL) != bool(EXPECTED_PROVIDER):
        print(
            "Set OPENROUTER_PRESET_MODEL and OPENROUTER_EXPECTED_PROVIDER together.",
            file=sys.stderr,
        )
        return 2
    print(f"probing {len(MODELS)} model(s) — this spends real money", flush=True)

    # Parallel: these are independent turns against different upstreams, and the wall
    # clock is otherwise the sum of four cold starts. Each line is prefixed with its model
    # so interleaved output stays readable, and one model's failure cannot mask another's
    # (every check returns a row). SERIAL=1 restores ordered output
    # for debugging a single model.
    if os.environ.get("SERIAL") == "1":
        rows = [await _probe(m, key, h) for m in MODELS for h in HARNESSES]
        for h in HARNESSES:
            rows.append(await _probe_resume(RESUME_MODEL, key, h))
            rows.append(await _probe_schema(SCHEMA_MODEL, key, h))
            rows.append(await _probe_preset(h, key))
            rows.append(await _probe_budget(h, key))
            if PINNED_MODEL and EXPECTED_PROVIDER:
                rows.append(await _probe_pin(h, key, PINNED_MODEL, EXPECTED_PROVIDER))
    else:
        rows = list(
            await asyncio.gather(
                *(_probe(m, key, h) for m in MODELS for h in HARNESSES),
                *(_probe_resume(RESUME_MODEL, key, h) for h in HARNESSES),
                *(_probe_schema(SCHEMA_MODEL, key, h) for h in HARNESSES),
                *(_probe_preset(h, key) for h in HARNESSES),
                *(_probe_budget(h, key) for h in HARNESSES),
                *(
                    _probe_pin(h, key, PINNED_MODEL, EXPECTED_PROVIDER)
                    for h in HARNESSES
                    if PINNED_MODEL and EXPECTED_PROVIDER
                ),
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

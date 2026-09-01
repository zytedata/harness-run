"""Live re-check of RunResult usage/cost accounting — both harnesses, subagents included.

Why this exists: the Codex subagent recovery (PR #36) relies on NON-PUBLIC details of
Codex's rollout files under ``CODEX_HOME/sessions`` — the ``session_meta``
``parent_thread_id`` marker, the per-turn ``turn_context.model`` line, thread-cumulative
``token_count`` totals (openai/codex#14642 is closed as not-planned, so there is no wire
signal to switch to). A Codex CLI bump can change that layout silently; this probe
re-establishes it live. The Claude scenarios pin what the normalization reads there:
``model_usage`` is subagent-inclusive and cumulative across re-invocation segments.

Scenarios (arg, or ``all``):

  claude-subagent    usage must equal the model_usage aggregate (subagent-inclusive)
  claude-segment     background task forces a re-invocation; usage must be cumulative
  codex-subagent     forced collab spawn; the subagent thread must be recovered from its
                     rollout (billed into usage/cost, no subagent_usage_missing event)
  codex-two-turns    a second turn must NOT re-bill turn 1's subagent (delta state file)
  or-codex-subagent  the same recovery on the OpenRouter path; the proxy's exact charge
                     is already subagent-inclusive (price_source stays "openrouter")
  or-claude          OpenRouter Claude: normalization from model_usage + exact cost

Spends real money (a few turns on Sonnet, GPT luna and DeepSeek v4 Flash — cents).
Writes ``verify-usage-<scenario>.json`` into the working directory and exits non-zero
if any check fails. See TESTING.md for when this rung is required.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from remote_agent_toolkit import AgentSpec, local
from remote_agent_toolkit.harness import pricing
from remote_agent_toolkit.harness._usage import USAGE_KEYS, from_claude_model_usage

OUT_DIR = Path.cwd()

SUBAGENT_TASK_CLAUDE = (
    "You MUST delegate this entire job to a subagent via the Task tool "
    "(subagent_type: general-purpose): the subagent should write and run a Python "
    "script that computes the sum of all primes below 100, and report the number "
    "back to you. Do not compute anything yourself in the main conversation - "
    "spawn the subagent, wait for its answer, then reply with just the number."
)
SEGMENT_TASK = (
    "Using the Bash tool with run_in_background=true, start the command "
    "`sleep 20 && echo finished`. Then immediately end your turn, replying only "
    "with the word 'started'. Later, when you are notified that the background "
    "task completed, reply only with the word 'done'."
)
SUBAGENT_TASK_CODEX = (
    "You have multi-agent collaboration tools (spawnAgent / wait / closeAgent). "
    "You MUST use them: spawn one subagent whose task is to write and run a Python "
    "script computing the sum of all primes below 100, wait for it to finish, then "
    "reply with just the number it reports. Do not compute anything yourself."
)
TRIVIAL_TASK = (
    "Compute 6*7 by writing and running a one-line Python script, then reply "
    "with just the number."
)

SCENARIOS = {
    "claude-subagent": dict(model="claude-sonnet-5", task=SUBAGENT_TASK_CLAUDE),
    "claude-segment": dict(model="claude-sonnet-5", task=SEGMENT_TASK),
    "codex-subagent": dict(model="gpt-5.6-luna", task=SUBAGENT_TASK_CODEX),
    "codex-two-turns": dict(model="gpt-5.6-luna", task=SUBAGENT_TASK_CODEX),
    "or-codex-subagent": dict(
        model="openrouter/deepseek/deepseek-v4-flash", task=SUBAGENT_TASK_CODEX
    ),
    "or-claude": dict(
        model="openrouter/deepseek/deepseek-v4-flash", task=TRIVIAL_TASK,
        harness="claude-code",
    ),
}


def result_record(result, result_event) -> dict:
    raw = (result_event.raw or {}) if result_event is not None else {}
    return {
        "text": (result.text or "")[:200],
        "is_error": result.is_error,
        "num_turns": result.num_turns,
        "cost_usd": result.cost_usd,
        "usage": result.usage,
        "warning": result.warning,
        "raw_keys": sorted(raw),
        "model_usage": raw.get("model_usage"),
        "cli_usage": raw.get("cli_usage"),
        "subagent_usage": raw.get("subagent_usage"),
        "price_source": raw.get("price_source"),
    }


async def drive(run):
    events, result_event = [], None
    async for event in run:
        events.append(event)
        if event.kind == "result":
            result_event = event
        summary = " ".join((event.summary or "").split())[:110]
        print(f"  [{event.kind:11}] {summary}", flush=True)
    return events, result_event


def status_events(events, name: str) -> list:
    return [e.raw for e in events if (e.raw or {}).get("event") == name]


async def run_scenario(name: str) -> list[str]:
    """Run one scenario; returns the names of the checks that FAILED."""
    print(f"\n=== {name} ===")
    sc = SCENARIOS[name]
    harness = sc.get("harness") or (
        "claude-code" if sc["model"].startswith("claude") else "codex"
    )
    spec = AgentSpec(
        name=f"verify-{name}",
        harness=harness,
        model=sc["model"],
        max_turns=40,
        max_budget_usd=3.0,
        background_task_timeout=120,
    )
    engine = local.deploy(spec)
    session = engine.start_session()
    run = session.run(sc["task"])
    events, result_event = await drive(run)
    result = run.result
    report = {
        "scenario": name,
        "model": sc["model"],
        "turn1": result_record(result, result_event),
    }
    raw = (result_event.raw or {}) if result_event is not None else {}
    usage = result.usage or {}

    # ok_* keys are pass/fail; everything else is recorded context.
    checks: dict[str, object] = {}
    checks["ok_no_error"] = not result.is_error
    checks["ok_normalized_keys"] = set(usage) == set(USAGE_KEYS)

    if harness == "claude-code":
        # The contract the normalization reads: model_usage is the CLI's COMPLETE
        # per-model record (subagent-inclusive; final segment's copy is cumulative).
        checks["ok_usage_is_model_usage_aggregate"] = (
            usage == from_claude_model_usage(raw.get("model_usage"))
        )
        mu_cost = sum(
            (entry.get("costUSD") or 0)
            for entry in (raw.get("model_usage") or {}).values()
        )
        checks["model_usage_cost_sum"] = round(mu_cost, 6)
        if sc["model"].startswith("claude") and result.cost_usd is not None:
            checks["ok_cost_is_model_usage_sum"] = (
                abs(result.cost_usd - mu_cost) < 1e-6
            )
    else:
        sub = raw.get("subagent_usage") or {}
        checks["n_subagent_threads"] = len(sub)
        checks["subagent_models"] = sorted(
            {str(d.get("model")) for d in sub.values()}
        )
        # The rollout recovery worked end to end: the wire-announced thread(s) were
        # found on disk (no missing tripwire) and billed into the result.
        checks["ok_subagent_recovered"] = len(sub) >= 1
        checks["ok_no_missing_rollouts"] = not status_events(
            events, "subagent_usage_missing"
        )
        # ...and actually MERGED: for every bucket a thread reports, the aggregate
        # covers at least the threads' sum (the parent's own calls come on top).
        checks["ok_subagent_usage_merged"] = all(
            (usage.get(key) or 0) >= sum(d.get(key) or 0 for d in sub.values())
            for key in USAGE_KEYS
            if any(d.get(key) is not None for d in sub.values())
        )
        # ...and priced into cost_usd: the total is at least the subagent spend alone,
        # re-priced here from the per-thread record (inclusive input = the disjoint
        # buckets recombined). Only checkable when every thread's model has a price.
        thread_prices = {
            tid: pricing.model_price(str(d.get("model"))) for tid, d in sub.items()
        }
        if sub and all(thread_prices.values()) and result.cost_usd is not None:
            sub_spend = sum(
                thread_prices[tid].cost_usd(
                    (d.get("input_tokens") or 0)
                    + (d.get("cache_read_input_tokens") or 0)
                    + (d.get("cache_creation_input_tokens") or 0),
                    d.get("cache_read_input_tokens") or 0,
                    d.get("output_tokens") or 0,
                )
                for tid, d in sub.items()
            )
            checks["subagent_spend_usd"] = round(sub_spend, 6)
            checks["ok_cost_includes_subagents"] = result.cost_usd >= sub_spend
        checks["partial_warnings"] = status_events(events, "subagent_usage_partial")
        if name == "or-codex-subagent":
            # OpenRouter's exact charge is already subagent-inclusive; the harness
            # must not price subagent tokens on top of it.
            checks["ok_price_source_openrouter"] = raw.get("price_source") == "openrouter"

    if name == "codex-two-turns":
        run2 = session.send(
            "Now, WITHOUT spawning any subagent, run `echo second-turn-done` in the "
            "shell yourself and reply with its output."
        )
        events2, result_event2 = await drive(run2)
        report["turn2"] = result_record(run2.result, result_event2)
        sub2 = ((result_event2.raw or {}).get("subagent_usage") or {}) if result_event2 else {}
        # Totals are thread-cumulative; the state file must keep turn 2 from
        # re-billing what turn 1 already did.
        checks["ok_turn2_not_rebilled"] = not sub2

    report["checks"] = checks
    out = OUT_DIR / f"verify-usage-{name}.json"
    out.write_text(json.dumps(report, indent=1, default=str))
    failed = [k for k, v in checks.items() if k.startswith("ok_") and v is not True]
    print(json.dumps({"usage": usage, "cost_usd": result.cost_usd, **checks},
                     indent=1, default=str))
    print(f"written: {out}")
    return [f"{name}: {k}" for k in failed]


async def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = list(SCENARIOS) if arg == "all" else [arg]
    failures: list[str] = []
    for name in names:
        failures.extend(await run_scenario(name))
    print("\n--- verdict ---")
    if failures:
        print("FAILED checks:\n  " + "\n  ".join(failures))
        raise SystemExit(1)
    print(f"all checks passed ({', '.join(names)})")


if __name__ == "__main__":
    asyncio.run(main())

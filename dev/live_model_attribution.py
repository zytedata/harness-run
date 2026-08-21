"""Paid local attribution check for both harnesses.

Claude Code must report its resolved model. Codex must report its bound provider. Every
OpenRouter response must report the selected upstream and exact cost. Checks run concurrently
unless ``SERIAL=1``. Run by hand with ``make live-attribution``; never run this in CI.
"""

from __future__ import annotations

import asyncio
import math
import os
import sys

from remote_agent_toolkit import AgentSpec, local

TASK = "Reply with just the word: ok"
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.6-luna")
OR_MODELS = [
    "openrouter/moonshotai/kimi-k3",
    "openrouter/z-ai/glm-5.3",
    "openrouter/deepseek/deepseek-v4-flash",
    "openrouter/deepseek/deepseek-v4-pro",
]

VERDICTS: list[tuple[str, str, str]] = []


def check(label: str, ok: bool, note: str = "") -> None:
    VERDICTS.append((label, "PASS" if ok else "FAIL", note))
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  — {note}" if note else ""), flush=True)


def skip(label: str, note: str) -> None:
    VERDICTS.append((label, "SKIP", note))
    print(f"  SKIP  {label}  — {note}", flush=True)


async def _events(spec: AgentSpec, secrets: dict[str, str]) -> tuple[list, object]:
    run = local.deploy(spec).start_session().run(TASK, secrets=secrets)
    events = [ev async for ev in run]
    return events, run.result


def _init_model(events) -> str | None:
    init = next((e.raw for e in events if (e.raw or {}).get("subtype") == "init"), None)
    return ((init or {}).get("data") or {}).get("model")


def _openrouter_requests(events) -> list[dict]:
    return [e.raw for e in events if (e.raw or {}).get("event") == "openrouter_request"]


async def check_claude(key: str | None) -> None:
    """The claude CLI reports the model it resolved in its init message."""
    spec = AgentSpec(name="attr-claude", model=CLAUDE_MODEL, max_turns=4, max_budget_usd=0.20)
    events, result = await _events(spec, {"ANTHROPIC_API_KEY": key} if key else {})
    reported = _init_model(events)
    check(
        f"claude-code: CLI ran {CLAUDE_MODEL}",
        reported is not None and CLAUDE_MODEL in str(reported),
        f"init reported model={reported!r}, error={result.is_error}",
    )


async def check_claude_openrouter(model: str, key: str) -> None:
    """claude-code reaching OpenRouter: the CLI must name the model we asked for.

    Also checks that OpenRouter's exact cost replaces the CLI estimate. The original value
    remains available as ``cli_reported_cost_usd``.
    """
    bare = model.removeprefix("openrouter/")
    spec = AgentSpec(name="attr-cc-or", model=model, max_turns=4, max_budget_usd=0.50)
    events, result = await _events(spec, {"OPENROUTER_API_KEY": key})
    reported = _init_model(events)
    check(
        f"claude-code+{bare}: CLI ran the model we asked for",
        reported is not None and bare in str(reported),
        f"init reported model={reported!r}, error={result.is_error}",
    )
    final = next((e for e in reversed(events) if e.kind == "result"), None)
    raw = (final.raw or {}) if final else {}
    cli = raw.get("cli_reported_cost_usd")
    requests = _openrouter_requests(events)
    providers = [request.get("provider") for request in requests]
    request_costs = [
        float(request["cost_usd"])
        for request in requests
        if isinstance(request.get("cost_usd"), (int, float))
    ]
    check(
        f"claude-code+{bare}: OpenRouter reported the upstream for each request",
        bool(requests) and all(providers),
        f"providers={providers}",
    )
    check(
        f"claude-code+{bare}: result equals OpenRouter's exact charges",
        isinstance(result.cost_usd, float)
        and cli is not None
        and bool(request_costs)
        and math.isclose(result.cost_usd, sum(request_costs), rel_tol=1e-9, abs_tol=1e-12)
        and raw.get("price_source") == "openrouter",
        f"result=${result.cost_usd or 0:.6f} requests={request_costs} "
        f"cli=${cli or 0:.6f} source={raw.get('price_source')}",
    )


async def check_codex_openai(key: str) -> None:
    """The app-server reports the thread's provider; for OpenAI it must not say openrouter."""
    spec = AgentSpec(
        name="attr-codex", model=OPENAI_MODEL, harness="codex", max_turns=4, max_budget_usd=0.20
    )
    events, result = await _events(spec, {"OPENAI_API_KEY": key})
    routing = next((e.raw for e in events if (e.raw or {}).get("event") == "model_routing"), None)
    if routing is None:
        skip("codex+OpenAI: thread routing reported", "no model_routing event (read() unsupported)")
        return
    check(
        f"codex+OpenAI: thread bound to a non-OpenRouter provider, model {OPENAI_MODEL}",
        routing.get("resolved_model_provider") != "openrouter" and not result.is_error,
        f"provider={routing.get('resolved_model_provider')!r} "
        f"model={routing.get('resolved_model')!r}",
    )


async def check_codex_openrouter(model: str, key: str) -> None:
    """The override must actually bind the thread to the openrouter provider."""
    label = model.removeprefix("openrouter/")
    spec = AgentSpec(name="attr-or", model=model, harness="codex", max_turns=4, max_budget_usd=0.30)
    events, result = await _events(spec, {"OPENROUTER_API_KEY": key})
    routing = next((e.raw for e in events if (e.raw or {}).get("event") == "model_routing"), None)
    if routing is None:
        skip(f"codex+{label}: thread routing reported", "no model_routing event")
    else:
        check(
            f"codex+{label}: thread bound to the openrouter provider",
            routing.get("matches_request") is True,
            f"provider={routing.get('resolved_model_provider')!r} "
            f"model={routing.get('resolved_model')!r}",
        )
    requests = _openrouter_requests(events)
    providers = [request.get("provider") for request in requests]
    request_costs = [
        float(request["cost_usd"])
        for request in requests
        if isinstance(request.get("cost_usd"), (int, float))
    ]
    check(
        f"codex+{label}: OpenRouter reported the upstream for each request",
        bool(requests) and all(providers),
        f"providers={providers}",
    )
    final = next((e for e in reversed(events) if e.kind == "result"), None)
    check(
        f"codex+{label}: turn completed with exact OpenRouter cost",
        not result.is_error
        and isinstance(result.cost_usd, float)
        and bool(request_costs)
        and math.isclose(result.cost_usd, sum(request_costs), rel_tol=1e-9, abs_tol=1e-12)
        and (final.raw or {}).get("price_source") == "openrouter",
        f"cost={result.cost_usd!r} requests={request_costs} "
        f"source={(final.raw or {}).get('price_source') if final else None}",
    )


async def main() -> int:
    ork = os.environ.get("OPENROUTER_API_KEY")
    oak = os.environ.get("OPENAI_API_KEY")
    ank = os.environ.get("ANTHROPIC_API_KEY")
    print("model-attribution probe — spends real money", flush=True)

    tasks = []
    # No key gate for claude-code: the CLI also accepts a logged-in session or Vertex, so
    # let it try and let the check report what actually happened.
    tasks.append(("claude-code", check_claude(ank)))
    if oak:
        tasks.append(("codex+openai", check_codex_openai(oak)))
    else:
        skip("codex+OpenAI: thread routing reported", "OPENAI_API_KEY unset")
    if ork:
        for m in OR_MODELS:
            tasks.append((f"codex+{m}", check_codex_openrouter(m, ork)))
            tasks.append((f"claude-code+{m}", check_claude_openrouter(m, ork)))
    else:
        skip("openrouter checks", "OPENROUTER_API_KEY unset")

    if os.environ.get("SERIAL") == "1":
        for _, coro in tasks:
            try:
                await coro
            except Exception as exc:  # noqa: BLE001
                check("check raised", False, f"{type(exc).__name__}: {exc}")
    else:
        results = await asyncio.gather(*(c for _, c in tasks), return_exceptions=True)
        for (name, _), r in zip(tasks, results):
            if isinstance(r, BaseException):
                check(f"{name}: check raised", False, f"{type(r).__name__}: {r}")

    print("\n=== VERDICTS ===")
    for label, state, note in VERDICTS:
        print(f"{state}  {label}" + (f"  — {note}" if note else ""))
    failed = [lbl for lbl, st, _ in VERDICTS if st == "FAIL"]
    skipped = [lbl for lbl, st, _ in VERDICTS if st == "SKIP"]
    print(
        f"\n{len(VERDICTS) - len(failed) - len(skipped)} passed, "
        f"{len(failed)} failed, {len(skipped)} skipped"
    )
    if failed:
        print("failed:", ", ".join(failed))
    return 0 if VERDICTS and not failed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

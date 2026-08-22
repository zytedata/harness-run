"""Paid local check for the four OpenRouter models on both harnesses.

Each model must call a shell tool, return its output, produce schema-valid structured output,
hide provider credentials, report the selected upstream, and use OpenRouter's exact cost.
Provider selection, whole routing objects, resume, and budget caps are checked on both
harnesses. Checks run concurrently unless ``SERIAL=1``.

Run by hand with ``OPENROUTER_API_KEY=... make live-openrouter``. Never run this in CI.
Use ``MODELS=...`` to limit the model list. ``OPENROUTER_PROVIDER`` overrides the provider
used by the checks; this is mainly useful with a single model.
``OPENROUTER_ALTERNATE_PROVIDER`` overrides the second provider in the routing checks.
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
HARNESSES = [
    value.strip()
    for value in os.environ.get("HARNESSES", "codex,claude-code").split(",")
    if value.strip()
]
PROBE_REASONING_EFFORT = os.environ.get("PROBE_REASONING_EFFORT")
DEFAULT_PROVIDERS = {
    "openrouter/moonshotai/kimi-k3": "moonshotai",
    "openrouter/z-ai/glm-5.3": "z-ai",
    # The shared account's ZDR policy excludes DeepSeek's own endpoint. Novita serves
    # both models, supports tools, and is allowed by the account. It rejects Codex's
    # json_schema format, so those two schema checks use normal routing below.
    "openrouter/deepseek/deepseek-v4-flash": "novita",
    "openrouter/deepseek/deepseek-v4-pro": "novita",
}


def _provider_for(model: str) -> str:
    """Use a provider known to serve the model unless the operator overrides it."""
    return (
        os.environ.get("OPENROUTER_PROVIDER")
        or DEFAULT_PROVIDERS.get(model)
        or model.removeprefix("openrouter/").split("/", 1)[0]
    )


# Routing objects are checked on Kimi K3: it is served by many providers, so a set of
# two is a real choice. GLM 5.3 has a single endpoint, which would prove nothing.
ROUTING_MODEL = "openrouter/moonshotai/kimi-k3"
# Any second provider that serves ROUTING_MODEL. It never has to be reachable: the
# primary is in both routing objects, so a provider the account cannot use is simply
# not selected.
ROUTING_ALTERNATE = os.environ.get("OPENROUTER_ALTERNATE_PROVIDER", "fireworks")


def _routing_cases(model: str) -> list[tuple[str, dict, bool | None]]:
    """The routing objects to check, with the verdict each one supports.

    A closed set can be checked against the provider OpenRouter reports. An open one
    cannot, and then the toolkit must claim nothing instead of guessing.
    """
    primary = _provider_for(model)
    return [
        ("closed", {"only": [primary, ROUTING_ALTERNATE]}, True),
        ("open", {"order": [ROUTING_ALTERNATE, primary], "allow_fallbacks": True}, None),
    ]


def _looks_like(provider: str | None, slug: str) -> bool:
    """Whether a reported display name is the provider that ``slug`` asked for."""

    def norm(value: str) -> str:
        return "".join(char for char in value.casefold() if char.isalnum())

    return provider is not None and norm(provider) == norm(slug.split("/", 1)[0])


def _schema_provider(model: str, harness: str) -> str | None:
    """Use normal routing where the selected provider lacks Codex JSON Schema support."""
    if os.environ.get("OPENROUTER_PROVIDER"):
        return _provider_for(model)
    if harness == "codex" and model.startswith("openrouter/deepseek/"):
        return None
    return _provider_for(model)


SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "integer"}, "note": {"type": "string"}},
    "required": ["answer", "note"],
    "additionalProperties": False,
}


async def _check_model(model: str, key: str, harness: str = "codex", attempts: int = 2) -> dict:
    """Run one turn on ``model`` under ``harness``; return a verdict row.

    Retries once on a soft miss. DeepSeek v4 Flash sometimes returns no final message under
    Claude Code. A retry is always reported in the verdict.
    """
    for attempt in range(1, attempts + 1):
        row = await _check_model_once(model, key, harness)
        if row["ok"]:
            if attempt > 1:
                row["note"] = f"passed on attempt {attempt} (first attempt: soft miss)"
            return row
        if attempt < attempts:
            print(f"[{row['model']}] soft miss, retrying once: {row['note'][:70]}", flush=True)
    row["note"] = f"failed {attempts} attempts — {row['note']}"
    return row


async def _check_model_once(model: str, key: str, harness: str = "codex") -> dict:
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
        openrouter_provider=_provider_for(model),
    )
    row = {"model": label, "ok": False, "tools": False, "cost": None, "turns": 0, "note": ""}
    requests: list[dict] = []
    price_source = None
    t0 = time.time()
    try:
        engine = local.deploy(spec)
        run = engine.start_session().run(TASK, secrets={"OPENROUTER_API_KEY": key})
        async for ev in run:
            if ev.kind == "tool_use":
                row["tools"] = True
            if (ev.raw or {}).get("event") == "openrouter_request":
                requests.append(dict(ev.raw))
            if ev.kind == "result":
                price_source = (ev.raw or {}).get("price_source")
            summary = " ".join((ev.summary or "").split())[:90]
            print(f"[{label}] {time.strftime('%H:%M:%S')} {ev.kind:11} {summary}", flush=True)
        r = run.result
        text = " ".join((r.text or "").split())
        # Only the proxy reports an HTTP status, so its presence proves the call went
        # through it. Failed responses are recorded too. They have no provider and no
        # cost, so the routing and accounting checks use the successful ones.
        through_proxy = all("http_status" in req for req in requests)
        succeeded = [req for req in requests if req.get("http_status") == 200]
        providers = [req.get("provider") for req in succeeded]
        requested_providers = [req.get("requested_provider") for req in succeeded]
        provider_matches = [req.get("provider_matches_request") for req in succeeded]
        requested_models = [req.get("requested_model") for req in succeeded]
        exact_costs = [
            float(req["cost_usd"])
            for req in succeeded
            if isinstance(req.get("cost_usd"), (int, float))
        ]
        bare_model = model.removeprefix("openrouter/")
        row.update(cost=r.cost_usd, turns=r.num_turns or 0)
        row["ok"] = bool(
            not r.is_error
            and row["turns"]
            and EXPECTED in text
            and row["tools"]
            and isinstance(r.cost_usd, float)
            and through_proxy
            and bool(succeeded)
            and all(providers)
            and all(provider == _provider_for(model) for provider in requested_providers)
            and all(match is True for match in provider_matches)
            and all(requested == bare_model for requested in requested_models)
            and bool(exact_costs)
            and math.isclose(r.cost_usd, sum(exact_costs), rel_tol=1e-9, abs_tol=1e-12)
            and price_source == "openrouter"
        )
        if not row["ok"]:
            statuses = [req.get("http_status") for req in requests]
            row["note"] = (
                f"error={r.is_error} turns={row['turns']} tools={row['tools']} "
                f"cost={r.cost_usd!r} request_costs={exact_costs} providers={providers} "
                f"requested_providers={requested_providers} matches={provider_matches} "
                f"requested={requested_models} statuses={statuses} source={price_source} "
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


async def _check_resume(model: str, key: str, harness: str = "codex", attempts: int = 2) -> dict:
    """Two turns on one session: the routing must survive a resume (retries once)."""
    for attempt in range(1, attempts + 1):
        row = await _check_resume_once(model, key, harness)
        if row["ok"]:
            if attempt > 1:
                row["note"] = f"passed on attempt {attempt} (first attempt: soft miss)"
            return row
        if attempt < attempts:
            print(f"[{row['model']}] soft miss, retrying once", flush=True)
    row["note"] = f"failed {attempts} attempts — {row['note']}"
    return row


async def _check_resume_once(model: str, key: str, harness: str = "codex") -> dict:
    """One resume pair; the verdict row for it."""
    label = f"{model.removeprefix('openrouter/')} (resume) [{harness}]"
    spec = AgentSpec(
        name="ratk-openrouter-resume",
        model=model,
        harness=harness,
        checkpoint=True,  # conversation continuity is what `send` resumes
        max_turns=6,
        max_budget_usd=0.30,
        openrouter_provider=_provider_for(model),
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


async def _check_structured_output(
    model: str, key: str, harness: str = "codex", attempts: int = 2
) -> dict:
    """Structured output on ``model`` under ``harness``; retries once, like ``_check_model``.

    The task asks the model to run a command and report what it printed. A model that
    answers from its head instead gets the arithmetic wrong. Measured on DeepSeek v4 Pro
    under codex: one run answered 6 for ``print(6 * 7)``, two immediate re-runs answered 42.
    """
    for attempt in range(1, attempts + 1):
        row = await _check_structured_output_once(model, key, harness)
        if row["ok"]:
            if attempt > 1:
                row["note"] = f"passed on attempt {attempt} (first attempt: soft miss)"
            return row
        if attempt < attempts:
            print(f"[{row['model']}] soft miss, retrying once: {row['note'][:70]}", flush=True)
    row["note"] = f"failed {attempts} attempts — {row['note']}"
    return row


async def _check_structured_output_once(model: str, key: str, harness: str = "codex") -> dict:
    """``output_schema`` must yield a parsed object."""
    label = f"{model.removeprefix('openrouter/')} (schema) [{harness}]"
    spec = AgentSpec(
        name="ratk-openrouter-schema",
        model=model,
        harness=harness,
        max_turns=6,
        max_budget_usd=0.40,
        output_schema=SCHEMA,
        openrouter_provider=_schema_provider(model, harness),
    )
    row = {"model": label, "ok": False, "tools": True, "cost": None, "turns": 0, "note": ""}
    try:
        session = local.deploy(spec).start_session()
        run = session.run(
            'Run `python3 -c "print(6 * 7)"`. Return the number as `answer` and a one-word `note`.',
            secrets={"OPENROUTER_API_KEY": key},
        )
        events = []
        async for event in run:
            events.append(event)
        r = run.result
        row.update(cost=r.cost_usd, turns=r.num_turns or 0)
        out = r.structured_output
        row["ok"] = bool(not r.is_error and isinstance(out, dict) and out.get("answer") == 42)
        if not row["ok"]:
            summaries = [event.summary for event in events if event.summary]
            row["note"] = f"structured_output={out!r} summaries={summaries!r}"
        print(f"[{label}] structured_output={out!r}", flush=True)
    except Exception as exc:  # noqa: BLE001 — report, don't hide
        row["note"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    return row


async def _check_unpinned(model: str, key: str, harness: str) -> dict:
    """No provider pin: the turn still goes through the proxy, which is where cost is."""
    label = f"{model.removeprefix('openrouter/')} unpinned [{harness}]"
    spec = AgentSpec(
        name="ratk-openrouter-unpinned",
        model=model,
        harness=harness,
        system_prompt="Reply briefly.",
        max_turns=2,
        max_budget_usd=0.30,
    )
    row = {"model": label, "ok": False, "tools": True, "cost": None, "turns": 0, "note": ""}
    requests: list[dict] = []
    price_source = None
    try:
        session = local.deploy(spec).start_session()
        run = session.run("Reply with only: ok", secrets={"OPENROUTER_API_KEY": key})
        async for event in run:
            if (event.raw or {}).get("event") == "openrouter_request":
                requests.append(dict(event.raw))
            if event.kind == "result":
                price_source = (event.raw or {}).get("price_source")
        result = run.result
        row.update(cost=result.cost_usd, turns=result.num_turns or 0)
        succeeded = [req for req in requests if req.get("http_status") == 200]
        costs = [
            float(req["cost_usd"])
            for req in succeeded
            if isinstance(req.get("cost_usd"), (int, float))
        ]
        row["ok"] = bool(
            not result.is_error
            and all("http_status" in req for req in requests)
            and bool(succeeded)
            and all(req.get("provider") for req in succeeded)
            # Nothing was asked for, so nothing is claimed about what was selected.
            and all(req.get("requested_provider") is None for req in succeeded)
            and all(req.get("provider_matches_request") is None for req in succeeded)
            and isinstance(result.cost_usd, float)
            and bool(costs)
            and math.isclose(result.cost_usd, sum(costs), rel_tol=1e-9, abs_tol=1e-12)
            and price_source == "openrouter"
        )
        row["note"] = (
            f"providers={[req.get('provider') for req in succeeded]} "
            f"statuses={[req.get('http_status') for req in requests]} source={price_source}"
        )
    except Exception as exc:  # noqa: BLE001 — report, don't hide
        row["note"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    return row


async def _check_routing(
    model: str,
    key: str,
    harness: str,
    case: str,
    routing: dict,
    expect_match: bool | None,
) -> dict:
    """A whole routing object reaches OpenRouter, and is only judged when it can be."""
    label = f"{model.removeprefix('openrouter/')} routing {case} [{harness}]"
    spec = AgentSpec(
        name="ratk-openrouter-routing",
        model=model,
        harness=harness,
        system_prompt="Reply briefly.",
        max_turns=2,
        max_budget_usd=0.30,
        reasoning_effort=PROBE_REASONING_EFFORT,
        openrouter_routing=routing,
    )
    row = {"model": label, "ok": False, "tools": True, "cost": None, "turns": 0, "note": ""}
    requests: list[dict] = []
    try:
        session = local.deploy(spec).start_session()
        run = session.run("Reply with only: ok", secrets={"OPENROUTER_API_KEY": key})
        async for event in run:
            if (event.raw or {}).get("event") == "openrouter_request":
                requests.append(dict(event.raw))
        result = run.result
        row.update(cost=result.cost_usd, turns=result.num_turns or 0)
        succeeded = [req for req in requests if req.get("http_status") == 200]
        # Judge the reported provider here rather than trusting the toolkit's own verdict.
        allowed = routing.get("only") or []
        served_by_allowed = all(
            any(_looks_like(req.get("provider"), slug) for slug in allowed) for req in succeeded
        )
        row["ok"] = bool(
            not result.is_error
            and bool(succeeded)
            and all(req.get("requested_routing") == routing for req in succeeded)
            # The routing object replaces the single-slug pin; nothing sets both.
            and all(req.get("requested_provider") is None for req in succeeded)
            and all(req.get("provider") for req in succeeded)
            and all(req.get("provider_matches_request") is expect_match for req in succeeded)
            and (not allowed or served_by_allowed)
        )
        row["note"] = (
            f"routing={routing} providers={[req.get('provider') for req in succeeded]} "
            f"matches={[req.get('provider_matches_request') for req in succeeded]}"
        )
    except Exception as exc:  # noqa: BLE001 — report, don't hide
        row["note"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    return row


async def _check_budget(harness: str, key: str) -> dict:
    """The exact OpenRouter charge trips the cap."""
    label = f"exact-cost budget [{harness}]"
    spec = AgentSpec(
        name="ratk-openrouter-budget",
        model=BUDGET_MODEL,
        harness=harness,
        system_prompt="Reply briefly.",
        max_turns=2,
        max_budget_usd=0.000001,
        openrouter_provider=_provider_for(BUDGET_MODEL),
    )
    row = {"model": label, "ok": False, "tools": True, "cost": None, "turns": 0, "note": ""}
    try:
        session = local.deploy(spec).start_session()
        run = session.run("Reply with only: ok", secrets={"OPENROUTER_API_KEY": key})
        final = None
        through_proxy = []
        async for event in run:
            if (event.raw or {}).get("event") == "openrouter_request":
                through_proxy.append("http_status" in event.raw)
            if event.kind == "result":
                final = event
        result = run.result
        row.update(cost=result.cost_usd, turns=result.num_turns or 0)
        raw = final.raw or {} if final else {}
        source = raw.get("price_source")
        subtype = raw.get("subtype")
        row["ok"] = bool(
            result.is_error
            and session.stop_reason is StopReason.BUDGET_EXCEEDED
            and subtype == "error_budget_exceeded"
            and isinstance(result.cost_usd, float)
            and result.cost_usd > spec.max_budget_usd
            and source == "openrouter"
            and bool(through_proxy)
            and all(through_proxy)
        )
        row["note"] = (
            f"stop={session.stop_reason} cost={result.cost_usd:.6f} source={source} "
            f"subtype={subtype} enforcement={raw.get('budget_enforcement')}"
        )
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
    # (every check returns a row). SERIAL=1 restores ordered output
    # for debugging a single model.
    if os.environ.get("SERIAL") == "1":
        rows = [await _check_model(m, key, h) for m in MODELS for h in HARNESSES]
        for h in HARNESSES:
            rows.append(await _check_resume(RESUME_MODEL, key, h))
            for model in MODELS:
                rows.append(await _check_structured_output(model, key, h))
            rows.append(await _check_unpinned(RESUME_MODEL, key, h))
            for case, routing, expect in _routing_cases(ROUTING_MODEL):
                rows.append(
                    await _check_routing(ROUTING_MODEL, key, h, case, routing, expect)
                )
            rows.append(await _check_budget(h, key))
    else:
        rows = list(
            await asyncio.gather(
                *(_check_model(m, key, h) for m in MODELS for h in HARNESSES),
                *(_check_resume(RESUME_MODEL, key, h) for h in HARNESSES),
                *(_check_structured_output(m, key, h) for m in MODELS for h in HARNESSES),
                *(_check_unpinned(RESUME_MODEL, key, h) for h in HARNESSES),
                *(
                    _check_routing(ROUTING_MODEL, key, h, case, routing, expect)
                    for h in HARNESSES
                    for case, routing, expect in _routing_cases(ROUTING_MODEL)
                ),
                *(_check_budget(h, key) for h in HARNESSES),
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

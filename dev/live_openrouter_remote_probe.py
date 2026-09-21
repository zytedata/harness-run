"""Paid remote (Agent Sandbox) check for OpenRouter on both harnesses.

One throwaway engine contains both CLIs. It checks every model's basic and structured-output
turns on both harnesses. It also checks resume, budgets, provider selection, whole routing
objects, cost reporting, tools, credentials, effective specs, resource samples, memory peak,
history, and Cloud Trace.

Run by hand with ``OPENROUTER_API_KEY=... make live-openrouter-remote``. Never run this in
CI. A run usually takes 8-15 minutes and spends real money. The engine is deleted after the
checks and on SIGTERM/SIGINT, including an interrupt during the deploy itself (there is no
engine object yet in that window, so it is deleted by name). ``KEEP=1`` leaves it running
for debugging.

Set ``PROJECT``, ``LOCATION``, ``SUFFIX``, ``IMPERSONATE_SA``, or
``SERIAL=1`` as needed. ``OPENROUTER_PROVIDER`` overrides the provider used by the checks;
this is mainly useful when ``MODELS`` contains one model.
``OPENROUTER_ALTERNATE_PROVIDER`` overrides the second provider in the routing checks.
"""

from __future__ import annotations

import asyncio
import getpass
import hashlib
import math
import os
import re
import signal
import sys
import time
import traceback

from remote_agent_toolkit import AgentSpec, SessionConfig, StopReason, TurnConfig, gemini

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
RESUME_MODEL = "openrouter/z-ai/glm-5.3"
TOKEN = "BANANA-77"
# Room for the concurrent checks; the default of 1 would serialize them.
# Both harnesses are checked remotely. claude-code matters most here: a deployed engine
# bakes CLAUDE_CODE_USE_VERTEX, which outranks the OpenRouter token, so this is where the
# harness blanking that switch is proven on real infrastructure.
HARNESSES = ["codex", "claude-code"]
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


def _pinned_routing(model: str) -> dict:
    """What ``openrouter_provider=_provider_for(model)`` becomes in the request."""
    return {"only": [_provider_for(model)], "allow_fallbacks": False}


def _schema_provider(model: str, harness: str) -> str | None:
    """Use normal routing where the selected provider lacks Codex JSON Schema support."""
    if os.environ.get("OPENROUTER_PROVIDER"):
        return _provider_for(model)
    if harness == "codex" and model.startswith("openrouter/deepseek/"):
        return None
    return _provider_for(model)


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
    """Stream one turn and retain the reporting fields checked by this probe.

    The terminal event's ``raw`` is returned because the resource enrichment
    (``memory_peak_bytes`` and friends) is stamped there, not onto ``RunResult``.
    """
    kinds, echoes, final_raw = [], [], {}
    observations = {
        "openrouter_requests": [],
        "cost_unknown": False,
        "final_cost": None,
        "model_routing": [],
        "init_models": [],
        "summaries": [],
    }
    t0 = time.time()
    async for ev in run:
        kinds.append(ev.kind)
        raw = ev.raw or {}
        if raw.get("event") == "effective_spec":
            echoes.append(raw)
        if raw.get("event") == "openrouter_request":
            observations["openrouter_requests"].append(raw)
        if raw.get("event") == "model_routing":
            observations["model_routing"].append(raw)
        if raw.get("subtype") == "init":
            observations["init_models"].append((raw.get("data") or {}).get("model"))
        if raw.get("event") == "cost_unknown":
            observations["cost_unknown"] = True
        if ev.kind == "result":
            final_raw = raw
            observations["final_cost"] = ev.cost_usd
        summary = " ".join((ev.summary or "").split())[:80]
        observations["summaries"].append(ev.summary or "")
        print(f"[{label}] {time.strftime('%H:%M:%S')} {ev.kind:11} {summary}", flush=True)
    r = run.result
    print(
        f"[{label}] done in {time.time() - t0:.0f}s: turns={r.num_turns} "
        f"cost=${r.cost_usd or 0:.4f} error={r.is_error}",
        flush=True,
    )
    return r, kinds, echoes, final_raw, observations


async def _check_model(engine, model: str, key: str, harness: str = "codex") -> None:
    """One turn on `model` under `harness`, via a per-turn model override."""
    label = f"{model.removeprefix('openrouter/')} [{harness}]"
    retry_note = ""
    for attempt in (1, 2):
        session = engine.start_session(config=SessionConfig(harness=harness))
        r, kinds, echoes, final_raw, observations = await _drive(
            label,
            session.run(
                TASK,
                secrets={"OPENROUTER_API_KEY": key},
                config=TurnConfig(
                    model=model,
                    openrouter_provider=_provider_for(model),
                ),
            ),
        )
        if final_raw.get("subtype") != "error_no_final_text" or attempt == 2:
            break
        retry_note = "first attempt returned no final text; retried once"
        print(f"[{label}] {retry_note}", flush=True)
    text = " ".join((r.text or "").split())
    check(
        f"{label}: turn completed",
        not r.is_error and bool(r.num_turns),
        f"text={text[:40]!r}" + (f"; {retry_note}" if retry_note else ""),
    )
    check(f"{label}: returned the command output", EXPECTED_DIGEST in text)
    check(f"{label}: auth key absent from the tool", EXPECTED in text)
    check(f"{label}: called a tool", "tool_use" in kinds)
    check(f"{label}: cost priced", isinstance(r.cost_usd, float), f"${r.cost_usd or 0:.4f}")
    all_requests = observations["openrouter_requests"]
    # Only the proxy reports an HTTP status, so its presence proves the call went through
    # it. Failed responses have no provider and no cost, so the checks below use the
    # successful ones.
    requests = [request for request in all_requests if request.get("http_status") == 200]
    providers = [request.get("provider") for request in requests]
    requested_models = [request.get("requested_model") for request in requests]
    request_costs = [
        float(request["cost_usd"])
        for request in requests
        if isinstance(request.get("cost_usd"), (int, float))
    ]
    requested_routings = [request.get("requested_routing") for request in requests]
    provider_matches = [request.get("provider_matches_request") for request in requests]
    bare_model = model.removeprefix("openrouter/")
    check(
        f"{label}: every model call went through the proxy",
        bool(all_requests) and all("http_status" in request for request in all_requests),
        f"statuses={[request.get('http_status') for request in all_requests]}",
    )
    check(
        f"{label}: upstream provider reported",
        bool(requests) and all(providers),
        f"providers={providers}",
    )
    check(
        f"{label}: requested provider honored",
        bool(requested_routings)
        and all(routing == _pinned_routing(model) for routing in requested_routings)
        and all(match is True for match in provider_matches),
        f"requested={requested_routings} providers={providers} matches={provider_matches}",
    )
    check(
        f"{label}: OpenRouter reports the requested model",
        bool(requested_models) and all(requested == bare_model for requested in requested_models),
        f"requested={requested_models}",
    )
    check(
        f"{label}: result equals OpenRouter's exact charges",
        final_raw.get("price_source") == "openrouter"
        and isinstance(r.cost_usd, float)
        and bool(request_costs)
        and math.isclose(r.cost_usd, sum(request_costs), rel_tol=1e-9, abs_tol=1e-12),
        f"source={final_raw.get('price_source')} result={r.cost_usd} request_costs={request_costs}",
    )
    # The echo is the worker's own record of the merged spec it executed.
    echoed = [e.get("spec", {}).get("model") for e in echoes]
    check(f"{label}: effective_spec echoes the model", model in echoed, f"echoed={echoed}")
    if harness == "codex":
        routing = observations["model_routing"]
        check(
            f"{label}: Codex thread bound to OpenRouter",
            bool(routing) and routing[-1].get("matches_request") is True,
            f"routing={routing}",
        )
    else:
        init_models = observations["init_models"]
        check(
            f"{label}: Claude Code names the model",
            bool(init_models) and any(bare_model in str(value) for value in init_models),
            f"init_models={init_models}",
        )


async def _check_structured_output(engine, model: str, key: str, harness: str) -> None:
    label = f"{model.removeprefix('openrouter/')} schema [{harness}]"
    session = engine.start_session(config=SessionConfig(harness=harness))
    r, _, _, _, _ = await _drive(
        label,
        session.run(
            'Run `python3 -c "print(6 * 7)"`. Return the number as `answer` and a one-word `note`.',
            secrets={"OPENROUTER_API_KEY": key},
            config=TurnConfig(
                model=model,
                output_schema=SCHEMA,
                openrouter_provider=_schema_provider(model, harness),
            ),
        ),
    )
    out = r.structured_output
    check(
        f"{label}: structured output parses remotely",
        isinstance(out, dict) and out.get("answer") == 42,
        f"structured_output={out!r}",
    )


async def _check_resume(engine, key: str, harness: str):
    session = engine.start_session(
        config=SessionConfig(
            harness=harness,
            model=RESUME_MODEL,
            openrouter_provider=_provider_for(RESUME_MODEL),
        )
    )
    secrets = {"OPENROUTER_API_KEY": key}
    await _drive(
        f"resume-1 [{harness}]",
        session.run(f"Remember the word {TOKEN}. Reply with just: stored", secrets=secrets),
    )
    r2, _, _, final_raw, _ = await _drive(
        f"resume-2 [{harness}]",
        session.send(
            "What word did I ask you to remember? Reply with just the word.", secrets=secrets
        ),
    )
    text = " ".join((r2.text or "").split())
    check(
        f"resume remembers across turns [{harness}]",
        TOKEN in text.upper(),
        f"turn2={text[:40]!r}",
    )
    return session, final_raw


async def _check_routing(
    engine,
    key: str,
    harness: str,
    case: str,
    routing: dict,
    expect_match: bool | None,
) -> None:
    """A whole routing object survives the trip to a deployed worker, and is judged there."""
    label = f"routing {case} [{harness}]"
    session = engine.start_session(config=SessionConfig(harness=harness))
    _, _, _, _, observations = await _drive(
        label,
        session.run(
            "Reply with only: ok",
            secrets={"OPENROUTER_API_KEY": key},
            config=TurnConfig(model=ROUTING_MODEL, openrouter_routing=routing),
        ),
    )
    requests = [
        request
        for request in observations["openrouter_requests"]
        if request.get("http_status") == 200
    ]
    providers = [request.get("provider") for request in requests]
    matches = [request.get("provider_matches_request") for request in requests]
    check(
        f"{label}: routing object reached OpenRouter",
        bool(requests)
        and all(request.get("requested_routing") == routing for request in requests),
        f"routing={[request.get('requested_routing') for request in requests]}",
    )
    # Judge the reported provider here rather than trusting the toolkit's own verdict.
    allowed = routing.get("only") or []
    check(
        f"{label}: provider verdict matches what the routing can prove",
        bool(requests)
        and all(providers)
        and all(match is expect_match for match in matches)
        and (
            not allowed
            or all(any(_looks_like(name, slug) for slug in allowed) for name in providers)
        ),
        f"providers={providers} matches={matches} expected={expect_match}",
    )


async def _check_budget(engine, key: str, harness: str) -> None:
    """A tiny cap must use OpenRouter's charge and stop with the budget reason."""
    label = f"exact-cost budget [{harness}]"
    session = engine.start_session(config=SessionConfig(harness=harness))
    config = TurnConfig(
        model=BAKED_MODEL,
        max_turns=2,
        max_budget_usd=0.000001,
        openrouter_provider=_provider_for(BAKED_MODEL),
    )
    run = session.run(
        "Reply with only: ok",
        secrets={"OPENROUTER_API_KEY": key},
        config=config,
    )
    result, _, _, final_raw, observations = await _drive(label, run)
    check(
        f"{label}: budget stop reason",
        result.is_error and session.stop_reason is StopReason.BUDGET_EXCEEDED,
        f"stop={session.stop_reason}",
    )
    check(
        f"{label}: exact charge exceeds tiny cap",
        isinstance(result.cost_usd, float) and result.cost_usd > config.max_budget_usd,
        f"cost={result.cost_usd}",
    )
    check(
        f"{label}: OpenRouter is the price source",
        final_raw.get("price_source") == "openrouter" and bool(observations["openrouter_requests"]),
        f"source={final_raw.get('price_source')}",
    )


async def _check_visibility(session, final_raw: dict, expect_model: str, harness: str) -> None:
    """The remote-only surface: the durable history the mirror keeps."""
    history = await asyncio.to_thread(session.history)
    check(
        f"history replays the turn's events [{harness}]",
        bool(history) and any(e.kind == "result" for e in history),
        f"{len(history)} event(s)",
    )


_TORN_DOWN = False
_ENGINE = None  # set as soon as gemini.deploy returns; read by the signal handler
_CREDENTIALS = None  # set before the handler goes on, so a delete by name can authenticate


def _delete_by_name() -> str | None:
    """Delete the engine when the run never got a handle for it. ``None`` if it went.

    An interrupt during ``gemini.deploy`` leaves no engine object, but the platform may
    already have created the engine, and it bills while it exists. ``NAME`` is fixed, so
    look the engine up by it and delete it. On failure return the reason as one line: the
    common case is that the deploy had not created anything yet, and a full traceback for
    that would bury the message that matters.
    """
    try:
        gemini.get_engine(NAME, PROJECT, LOCATION, credentials=_CREDENTIALS).delete()
        return None
    except Exception as exc:  # noqa: BLE001 — best effort; the caller says what to do
        return f"{type(exc).__name__}: {str(exc)[:200]}"


def _teardown() -> None:
    """Delete the engine, once, from either the normal path or a signal.

    SIGTERM and SIGINT can end the process before ``finally`` runs. The signal handler and
    this idempotent function ensure the engine is deleted once.

    Before ``gemini.deploy`` returns there is no engine object, so that window deletes by
    name instead.
    """
    global _TORN_DOWN
    if _TORN_DOWN:
        return
    _TORN_DOWN = True
    if os.environ.get("KEEP") == "1":
        print(f"\nKEEP=1 — engine {NAME} left running; delete it yourself.", flush=True)
        return
    if _ENGINE is None:
        # Interrupted before the deploy returned. The engine may exist anyway.
        print(
            f"\n{time.strftime('%H:%M:%S')} interrupted before the deploy returned; "
            f"trying to delete {NAME} by name ...",
            flush=True,
        )
        reason = _delete_by_name()
        if reason is None:
            print("teardown OK", flush=True)
        else:
            print(
                f"\n!!! COULD NOT DELETE {NAME} ({reason}). It may exist in "
                f"{PROJECT}/{LOCATION} and bills while it does. Check and delete it "
                "manually !!!",
                flush=True,
            )
        return
    try:
        print(f"\n{time.strftime('%H:%M:%S')} deleting {NAME} ...", flush=True)
        _ENGINE.delete()
        print("teardown OK", flush=True)
    except Exception:  # noqa: BLE001 — always say so loudly; an engine bills while it exists
        traceback.print_exc()
        print(f"\n!!! TEARDOWN FAILED — delete the engine {NAME} manually !!!", flush=True)


def _install_signal_teardown() -> None:
    """On SIGTERM/SIGINT: delete the engine, then exit non-zero.

    Installed BEFORE ``gemini.deploy``, because that call is the longest part of the run
    (the image build, then a template create the platform has taken up to 30 min over) and
    the process is routinely wrapped in a ``timeout`` or interrupted. An
    interrupt during the deploy would otherwise leave a billing engine behind with nothing
    printed. There is no engine object yet in that window, so ``_teardown`` falls back to
    deleting by name.
    """

    def handler(signum, _frame):
        print(f"\nreceived signal {signum} — tearing down before exit", flush=True)
        _teardown()
        raise SystemExit(130)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):  # not the main thread / unsupported
            pass


async def main() -> int:
    global _ENGINE, _CREDENTIALS
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
    _CREDENTIALS = credentials

    spec = AgentSpec(
        name=NAME,
        model=BAKED_MODEL,
        harness="codex",
        # Both CLIs baked, so sessions can pick either harness against the same engine.
        harnesses=("codex", "claude-code"),
        checkpoint=True,  # the resume check needs conversation + workspace continuity
        max_turns=8,
        max_budget_usd=0.50,
    )
    # Installed before the deploy: the deploy takes 5-10 min, and an interrupt during it
    # must still reach teardown. The deploy sits inside the try below for the same reason.
    _install_signal_teardown()
    print(f"{time.strftime('%H:%M:%S')} deploying {NAME} ...", flush=True)
    t0 = time.time()

    try:
        _ENGINE = engine = await asyncio.to_thread(
            gemini.deploy,
            spec,
            PROJECT,
            LOCATION,
            credentials=credentials,
        )
        print(f"{time.strftime('%H:%M:%S')} deployed in {time.time() - t0:.0f}s", flush=True)
        # Every check uses an independent session. Resume remains sequential inside its
        # session. SERIAL=1 gives ordered output for debugging.
        if os.environ.get("SERIAL") == "1":
            for model in MODELS:
                for h in HARNESSES:
                    await _check_model(engine, model, key, h)
            resumes = {}
            for h in HARNESSES:
                for model in MODELS:
                    await _check_structured_output(engine, model, key, h)
                resumes[h] = await _check_resume(engine, key, h)
                await _check_budget(engine, key, h)
                for case, routing, expect in _routing_cases(ROUTING_MODEL):
                    await _check_routing(engine, key, h, case, routing, expect)
            visibility_targets = [(*resumes[h], h) for h in HARNESSES]
        else:
            model_checks = [
                _check_model(engine, model, key, harness)
                for model in MODELS
                for harness in HARNESSES
            ]
            schema_checks = [
                _check_structured_output(engine, model, key, harness)
                for model in MODELS
                for harness in HARNESSES
            ]
            resume_checks = [_check_resume(engine, key, harness) for harness in HARNESSES]
            budget_checks = [_check_budget(engine, key, harness) for harness in HARNESSES]
            # After the resume checks: their results are read back by position below.
            routing_checks = [
                _check_routing(engine, key, harness, case, routing, expect)
                for harness in HARNESSES
                for case, routing, expect in _routing_cases(ROUTING_MODEL)
            ]
            results = await asyncio.gather(
                *model_checks,
                *schema_checks,
                *resume_checks,
                *budget_checks,
                *routing_checks,
                return_exceptions=True,
            )
            for r in results:
                if isinstance(r, BaseException):
                    traceback.print_exception(type(r), r, r.__traceback__)
                    check("all concurrent checks completed", False, f"{type(r).__name__}: {r}")
            resume_start = len(model_checks) + len(schema_checks)
            visibility_targets = []
            for harness in HARNESSES:
                resume = results[resume_start + HARNESSES.index(harness)]
                if isinstance(resume, BaseException):
                    raise RuntimeError(
                        f"resume check failed for {harness}; skipping its visibility checks"
                    ) from resume
                visibility_targets.append((*resume, harness))
        await asyncio.gather(
            *(
                _check_visibility(session, final_raw, RESUME_MODEL, harness)
                for session, final_raw, harness in visibility_targets
            )
        )
    except Exception:  # noqa: BLE001 — always reach teardown
        traceback.print_exc()
        check("probe ran without exceptions", False, "see traceback above")
    finally:
        _teardown()

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

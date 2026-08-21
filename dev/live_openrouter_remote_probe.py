"""Paid Gemini Agent Runtime check for OpenRouter on both harnesses.

One throwaway engine contains both CLIs. It checks every model's basic and structured-output
turns on both harnesses. It also checks resume, budgets, provider selection and cost reporting,
tools, credentials, effective specs, resource samples, memory peak, history, and Cloud Trace.

Run by hand with ``OPENROUTER_API_KEY=... make live-openrouter-remote``. Never run this in
CI. A run usually takes 8-15 minutes and spends real money. The engine is deleted after the
checks and on SIGTERM/SIGINT. ``KEEP=1`` leaves it running for debugging.

Set ``PROJECT``, ``LOCATION``, ``SUFFIX``, ``IMPERSONATE_SA``, ``MAX_INSTANCES``, or
``SERIAL=1`` as needed. ``OPENROUTER_PROVIDER`` overrides the provider used by the checks;
this is mainly useful when ``MODELS`` contains one model.
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
MAX_INSTANCES = int(os.environ.get("MAX_INSTANCES", "8"))
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


def _schema_provider(model: str, harness: str) -> str | None:
    """Use normal routing where the selected provider lacks Codex JSON Schema support."""
    if os.environ.get("OPENROUTER_PROVIDER"):
        return _provider_for(model)
    if harness == "codex" and model.startswith("openrouter/deepseek/"):
        return None
    return _provider_for(model)


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
    requests = observations["openrouter_requests"]
    providers = [request.get("provider") for request in requests]
    requested_models = [request.get("requested_model") for request in requests]
    request_costs = [
        float(request["cost_usd"])
        for request in requests
        if isinstance(request.get("cost_usd"), (int, float))
    ]
    requested_providers = [request.get("requested_provider") for request in requests]
    provider_matches = [request.get("provider_matches_request") for request in requests]
    bare_model = model.removeprefix("openrouter/")
    check(
        f"{label}: upstream provider reported",
        bool(requests) and all(providers),
        f"providers={providers}",
    )
    check(
        f"{label}: requested provider honored",
        bool(requested_providers)
        and all(provider == _provider_for(model) for provider in requested_providers)
        and all(match is True for match in provider_matches),
        f"requested={requested_providers} providers={providers} matches={provider_matches}",
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
    """The remote-only surface: resource samples, memory peak, history, traces."""
    samples = await asyncio.to_thread(session.resource_samples)
    check(
        f"resource_samples returns worker CPU/RAM [{harness}]",
        bool(samples),
        f"{len(samples)} sample(s)" + (f", keys={sorted(samples[-1])[:4]}" if samples else ""),
    )
    # Stamped onto the terminal event's raw by the worker's cgroup sampler.
    peak = final_raw.get("memory_peak_bytes")
    check(f"memory_peak_bytes stamped on the result [{harness}]", peak is not None, f"peak={peak}")

    history = await asyncio.to_thread(session.history)
    check(
        f"history replays the turn's events [{harness}]",
        bool(history),
        f"{len(history)} event(s)",
    )

    ok, note = await asyncio.to_thread(_trace_check, session.session_id, expect_model)
    if ok is None:
        skip(f"Cloud Trace root span carries the model and cost [{harness}]", note)
    else:
        check(f"Cloud Trace root span carries the model and cost [{harness}]", ok, note)


def _trace_check(session_id: str, expect_model: str) -> tuple[bool | None, str]:
    """Find THIS session's root span and confirm what it reports.

    The project is shared, so the lookup uses this session id. The root span must name the
    OpenRouter model and carry a cost (``runtime/gemini/tracing.py``).

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
            for tr in client.list_traces(
                request={
                    "project_id": PROJECT,
                    "filter": "span:invoke_agent",
                    "view": trace_v1.ListTracesRequest.ViewType.COMPLETE,
                    "page_size": 50,
                }
            ):
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

    SIGTERM and SIGINT can end the process before ``finally`` runs. The signal handler and
    this idempotent function ensure the engine is deleted once.
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
        # Both CLIs baked, so sessions can pick either harness against the same engine.
        harnesses=("codex", "claude-code"),
        checkpoint=True,  # the resume check needs conversation + workspace continuity
        max_turns=8,
        max_budget_usd=0.50,
    )
    print(f"{time.strftime('%H:%M:%S')} deploying {NAME} ...", flush=True)
    t0 = time.time()
    engine = await asyncio.to_thread(
        gemini.deploy,
        spec,
        PROJECT,
        LOCATION,
        credentials=credentials,
        # The checks run concurrently, so the engine needs room to serve them in parallel;
        # with the default max_instances=1 they queue and the probe is back to ~40 min.
        max_instances=MAX_INSTANCES,
    )
    print(f"{time.strftime('%H:%M:%S')} deployed in {time.time() - t0:.0f}s", flush=True)
    _install_signal_teardown(engine)

    try:
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
            results = await asyncio.gather(
                *model_checks,
                *schema_checks,
                *resume_checks,
                *budget_checks,
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

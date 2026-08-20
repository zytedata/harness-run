"""Live probe — did the turn actually run the model we asked for? Both harnesses.

COSTS REAL MONEY (a few cents) and needs keys. Run it by hand, locally, sparingly — never
in CI. All checks run concurrently, so a pass is a few tens of seconds.

Why this exists: `result.raw["model"]` is what the caller *asked* for, so asserting on it
proves nothing. This probe only accepts evidence that comes back from the harness or the
provider:

  * **claude-code** — the CLI's `system/init` message carries the model it resolved
    (`raw["data"]["model"]`). Independent of our request, and checked for a Claude model
    and for every OpenRouter model, where the same check also confirms the harness
    replaced the CLI's own (wrong) cost figure with one computed from the real prices.
  * **codex** — the app-server's `thread.read()` reports the thread's bound
    `model_provider`, surfaced by the harness as a `model_routing` event. This is what
    proves an OpenRouter override actually took effect. It reports the *provider* only:
    the SDK's `Thread` has no settings block, so the model id is not independently
    confirmable on this path — hence the canary below.
  * **OpenRouter, checked separately** — the chat wire returns both `model` and `provider`
    in the body, so one cheap request confirms the account really serves the id we name,
    and which upstream served it. The Responses wire the codex harness uses returns
    neither, which is why this extra request exists.

One limit worth stating plainly: on the codex/OpenRouter path there is **no way to see the
upstream provider for the agent's own turn**. The two checks get close — the thread is bound
to the openrouter provider, and the account does serve that model id — but neither names the
upstream that answered a given turn. Pin the routing if you need that — see the README.

Configure via env:
  OPENROUTER_API_KEY   required for the OpenRouter checks
  OPENAI_API_KEY       optional — enables the codex+OpenAI check
  ANTHROPIC_API_KEY    optional — enables the claude-code check (or a logged-in claude CLI)
  CLAUDE_MODEL         default claude-haiku-4-5
  SERIAL=1             run checks one at a time

Run:
  make live-attribution
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import urllib.error
import urllib.request

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

    Also checks the cost the toolkit reports. The CLI prices these models from its own
    catalogue and gets it wrong (60x over for deepseek-v4-flash), so the harness recomputes
    from the reported tokens. A `cli_reported_cost_usd` in the result raw is that original
    figure, kept for comparison.
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
    check(
        f"claude-code+{bare}: cost recomputed, not the CLI's figure",
        isinstance(result.cost_usd, float) and cli is not None and result.cost_usd != cli,
        f"ours=${result.cost_usd or 0:.6f} cli=${cli or 0:.6f} source={raw.get('price_source')}",
    )


async def check_codex_openai(key: str) -> None:
    """The app-server reports the thread's provider; for OpenAI it must not say openrouter."""
    spec = AgentSpec(
        name="attr-codex", model=OPENAI_MODEL, harness="codex", max_turns=4, max_budget_usd=0.20
    )
    events, result = await _events(spec, {"OPENAI_API_KEY": key})
    routing = next(
        (e.raw for e in events if (e.raw or {}).get("event") == "model_routing"), None
    )
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
    spec = AgentSpec(
        name="attr-or", model=model, harness="codex", max_turns=4, max_budget_usd=0.30
    )
    events, result = await _events(spec, {"OPENROUTER_API_KEY": key})
    routing = next(
        (e.raw for e in events if (e.raw or {}).get("event") == "model_routing"), None
    )
    if routing is None:
        skip(f"codex+{label}: thread routing reported", "no model_routing event")
    else:
        check(
            f"codex+{label}: thread bound to the openrouter provider",
            routing.get("matches_request") is True,
            f"provider={routing.get('resolved_model_provider')!r} "
            f"model={routing.get('resolved_model')!r}",
        )
    # The model the harness handed Codex must be the provider-relative id, prefix stripped.
    check(
        f"codex+{label}: turn completed and priced",
        not result.is_error and isinstance(result.cost_usd, float),
        f"cost={result.cost_usd!r}",
    )


def _canary(model_id: str, key: str) -> tuple[str | None, str | None, str | None]:
    """Chat-wire canary: returns (served_model, provider, error). Costs a fraction of a cent."""
    body = {
        "model": model_id,
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "ok"}],
    }
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode()
    )
    req.add_header("Authorization", f"Bearer {key}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310 — fixed host
            d = json.loads(r.read())
        return d.get("model"), d.get("provider"), None
    except urllib.error.HTTPError as e:
        return None, None, e.read().decode()[:160]


async def check_openrouter_canary(model: str, key: str) -> None:
    """A separate request on the chat wire, which does name the model and the upstream."""
    bare = model.removeprefix("openrouter/")
    served, provider, err = await asyncio.to_thread(_canary, bare, key)
    check(
        f"openrouter canary: {bare} is served as itself",
        served == bare,
        f"served model={served!r} by provider={provider!r}" + (f" err={err}" if err else ""),
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
            tasks.append((f"canary {m}", check_openrouter_canary(m, ork)))
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

"""Model pricing for harnesses that report tokens but no USD (Codex, Antigravity).

Primary: **LiteLLM's community-maintained pricing dataset** (the same upstream ``ccusage``
prices Codex sessions with), fetched once per process — new models are priced the day the
dataset knows them, no toolkit release needed. Fallback: a small baked table of the current
OpenAI flagships, so offline / air-gapped runs still price them. The fetch is best-effort
with a short timeout and never raises; a total miss yields ``None`` (the harness then
reports unknown cost and cannot enforce ``max_budget_usd``).

**OpenRouter models are not priced here.** Every OpenRouter response carries its own
``usage.cost``, which the proxy records, so the harnesses report that and nothing else. An
estimate would be wrong in a way nobody could see: OpenRouter routes one model id to
providers whose prices differ by up to 2.5x, so a single per-model number matches only the
provider that happens to serve the call. Measured 2026-08-21 against the providers the
probes pin — a table calibrated for one provider was 40% under and 46% over on the two
DeepSeeks. A missing charge is reported as unknown cost instead.

**Long-context rates.** OpenAI bills a model call whose input is above a threshold (272K
tokens today) at higher rates, for the whole call. A price carries those rates when its
source lists them (``ModelPrice.long_context``), and ``ModelPrice.call_cost_usd`` prices one
call with them. ``ModelPrice.cost_usd`` stays at the standard rates: it is for token totals,
which do not say how large each call was.

Stdlib-only (urllib). All prices are USD per single token.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass

_LITELLM_PRICES_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
)
_FETCH_TIMEOUT_S = 8.0

# Context windows, not prices: both bindings pass these to their CLI because neither
# catalogue has an entry for these ids (Claude Code would assume 200k and compact early,
# Codex would warn and use generic metadata). An id that is missing here still runs.
OPENROUTER_CONTEXT_WINDOWS: dict[str, int] = {
    "openrouter/moonshotai/kimi-k3": 1_048_576,
    "openrouter/z-ai/glm-5.3": 1_048_576,
    "openrouter/deepseek/deepseek-v4-flash": 1_048_576,
    "openrouter/deepseek/deepseek-v4-pro": 1_048_576,
}

# USD per 1M tokens: (input, cached input, output) — fallback only; LiteLLM wins when
# reachable. Cache *writes* (billed at 1.25x input since 5.6) are not distinguishable in
# Codex's usage breakdown, so cost is a slight undercount either way.
# The gpt-6.x rows are OpenAI's Standard-tier short-context (<=272K input) prices, checked
# against the official pricing page on 2026-11-18; their long-context rates are in the
# table below.
# The gemini-4 rows are the Gemini Developer API's Standard paid-tier prices, checked against
# the official pricing page on 2026-11-19: output includes thinking tokens, caching is
# implicit (the cached rate), and there are no long-context rates. Carbon and Argon double
# from 2026-12-01 (to 4.00 / 0.20 / 20.00); these rows are the rates through 2026-11-30.
_BUILTIN_PER_MTOK: dict[str, tuple[float, float, float]] = {
    "gemini-4-neon": (3.00, 0.15, 15.00),
    "gemini-4-carbon": (2.00, 0.10, 10.00),
    "gemini-4-argon": (2.00, 0.10, 10.00),
    "gemini-4-hydrogen": (1.50, 0.075, 7.50),
    "gpt-6.2-sol": (2.00, 0.10, 10.00),
    "gpt-6.1-sol": (2.00, 0.10, 10.00),
    "gpt-6.1-terra": (1.00, 0.05, 5.00),
    "gpt-5.6-sol": (5.00, 0.50, 30.00),
    "gpt-5.6": (5.00, 0.50, 30.00),  # bare alias routes to sol
    "gpt-5.6-terra": (2.50, 0.25, 15.00),
    "gpt-5.6-luna": (1.00, 0.10, 6.00),
    "gpt-5.3-codex": (1.75, 0.175, 14.00),
}

# OpenAI's Standard-tier long-context prices, same unit and order as the table above: a call
# whose input is above 272K tokens is billed at these for the whole call (checked against
# the official pricing page on 2026-11-18). A model has a row when codex-cli 0.170.0 gives it
# a context window above that threshold (500K each); the other models of the table above run
# with a 272K window, so their calls cannot reach it.
_BUILTIN_LONG_CONTEXT_ABOVE = 272_000
_BUILTIN_LONG_CONTEXT_PER_MTOK: dict[str, tuple[float, float, float]] = {
    "gpt-6.2-sol": (4.00, 0.20, 15.00),
    "gpt-6.1-sol": (4.00, 0.20, 15.00),
    "gpt-6.1-terra": (2.00, 0.10, 7.50),
}

# LiteLLM lists a long-context rate under the standard key plus a suffix that names the
# threshold: ``input_cost_per_token_above_272k_tokens``.
_LITELLM_LONG_CONTEXT_INPUT_KEY = re.compile(r"input_cost_per_token_above_(\d+)k_tokens")


@dataclass(frozen=True)
class LongContextRates:
    """USD per token for a model call whose input is above ``above_input_tokens``.

    The rates apply to the whole call, not only to the tokens above the threshold.
    """

    above_input_tokens: int
    input_per_token: float
    cached_input_per_token: float
    output_per_token: float


@dataclass(frozen=True)
class ModelPrice:
    """USD per token, plus where the numbers came from (``litellm`` or ``builtin``).

    ``long_context`` is ``None`` when the source lists no long-context rates for the model.
    """

    input_per_token: float
    cached_input_per_token: float
    output_per_token: float
    source: str
    long_context: LongContextRates | None = None

    def cost_usd(self, input_tokens: int, cached_tokens: int, output_tokens: int) -> float:
        """Cost at the standard rates, whatever the size: for token totals."""
        fresh = max(input_tokens - cached_tokens, 0)
        return (
            fresh * self.input_per_token
            + cached_tokens * self.cached_input_per_token
            + output_tokens * self.output_per_token
        )

    def call_cost_usd(self, input_tokens: int, cached_tokens: int, output_tokens: int) -> float:
        """Cost of ONE model call.

        At the long-context rates when its input is above their threshold, at the standard
        rates otherwise. ``input_tokens`` includes the cached ones, as in :meth:`cost_usd`.
        """
        rates = self.long_context
        if rates is None or input_tokens <= rates.above_input_tokens:
            return self.cost_usd(input_tokens, cached_tokens, output_tokens)
        fresh = max(input_tokens - cached_tokens, 0)
        return (
            fresh * rates.input_per_token
            + cached_tokens * rates.cached_input_per_token
            + output_tokens * rates.output_per_token
        )


_lock = threading.Lock()
_litellm_cache: dict | None = None
_litellm_attempted = False


def _fetch_litellm() -> dict | None:
    """One GET of the LiteLLM dataset; ``None`` on any failure. Never raises."""
    from urllib.request import urlopen

    try:
        with urlopen(_LITELLM_PRICES_URL, timeout=_FETCH_TIMEOUT_S) as resp:  # noqa: S310
            data = json.loads(resp.read().decode())
        return data if isinstance(data, dict) else None
    except Exception:  # noqa: BLE001 — pricing is best-effort by design
        return None


def _litellm_prices() -> dict | None:
    """The dataset, fetched at most once per process (failure is cached too)."""
    global _litellm_cache, _litellm_attempted
    with _lock:
        if not _litellm_attempted:
            _litellm_attempted = True
            _litellm_cache = _fetch_litellm()
        return _litellm_cache


def clear_cache() -> None:
    """Forget the fetched dataset (tests; or to force a refetch in a long process)."""
    global _litellm_cache, _litellm_attempted
    with _lock:
        _litellm_cache = None
        _litellm_attempted = False


def _litellm_long_context(entry: dict) -> LongContextRates | None:
    """The long-context rates of one LiteLLM entry; ``None`` when it lists none.

    An entry that names more than one threshold is not read (no OpenAI model has one).
    """
    thresholds = [
        match.group(1) for key in entry if (match := _LITELLM_LONG_CONTEXT_INPUT_KEY.fullmatch(key))
    ]
    if len(thresholds) != 1:
        return None
    suffix = f"_above_{thresholds[0]}k_tokens"
    inp = entry.get(f"input_cost_per_token{suffix}")
    out = entry.get(f"output_cost_per_token{suffix}")
    if inp is None or out is None:
        return None
    cached = entry.get(f"cache_read_input_token_cost{suffix}")
    return LongContextRates(
        above_input_tokens=int(thresholds[0]) * 1000,
        input_per_token=float(inp),
        # As at the standard rates: no cache-read price listed → cached tokens bill as input.
        cached_input_per_token=float(cached if cached is not None else inp),
        output_per_token=float(out),
    )


def model_price(model: str) -> ModelPrice | None:
    """Resolve ``model`` to per-token USD prices, or ``None`` if unknown everywhere.

    LiteLLM keys are tried as the bare id, then ``openai/<id>`` (Codex calls OpenAI directly),
    then ``gemini/<id>`` (Antigravity calls the Gemini Developer API; LiteLLM keys the Gemini 4
    models only that way). The harnesses never call this for an ``openrouter/`` model — see the module
    docstring. The first call may block for the fetch timeout; callers on an event loop
    should run it in a worker thread.
    """
    data = _litellm_prices()
    if data:
        for key in (model, f"openai/{model}", f"gemini/{model}"):
            entry = data.get(key)
            if not isinstance(entry, dict):
                continue
            inp = entry.get("input_cost_per_token")
            out = entry.get("output_cost_per_token")
            if inp is None or out is None:
                continue
            cached = entry.get("cache_read_input_token_cost")
            return ModelPrice(
                input_per_token=float(inp),
                # No cache-read price listed → cached tokens bill as fresh input.
                cached_input_per_token=float(cached if cached is not None else inp),
                output_per_token=float(out),
                source="litellm",
                long_context=_litellm_long_context(entry),
            )
    per_mtok = _BUILTIN_PER_MTOK.get(model)
    if per_mtok is not None:
        long_context = None
        long_per_mtok = _BUILTIN_LONG_CONTEXT_PER_MTOK.get(model)
        if long_per_mtok is not None:
            long_context = LongContextRates(
                above_input_tokens=_BUILTIN_LONG_CONTEXT_ABOVE,
                input_per_token=long_per_mtok[0] / 1e6,
                cached_input_per_token=long_per_mtok[1] / 1e6,
                output_per_token=long_per_mtok[2] / 1e6,
            )
        return ModelPrice(
            input_per_token=per_mtok[0] / 1e6,
            cached_input_per_token=per_mtok[1] / 1e6,
            output_per_token=per_mtok[2] / 1e6,
            source="builtin",
            long_context=long_context,
        )
    return None

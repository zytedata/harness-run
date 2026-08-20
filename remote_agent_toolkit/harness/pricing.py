"""Model pricing for harnesses that report tokens but no USD (Codex).

Codex surfaces token counts only, so ``cost_usd`` and budget enforcement need a price
source. Primary: **LiteLLM's community-maintained pricing dataset** (the same upstream
``ccusage`` prices Codex sessions with), fetched once per process — new models are priced
the day the dataset knows them, no toolkit release needed. Fallback: a small baked table
of the current OpenAI flagships plus the OpenRouter models the Codex binding routes, so
offline / air-gapped runs still price them. The fetch
is best-effort with a short timeout and never raises; a total miss yields ``None`` (the
harness then reports unknown cost and cannot enforce ``max_budget_usd``).

Stdlib-only (urllib). All prices are USD per single token.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass

_LITELLM_PRICES_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)
_FETCH_TIMEOUT_S = 8.0

# USD per 1M tokens: (input, cached input, output) — fallback only; LiteLLM wins when
# reachable. Cache *writes* (billed at 1.25x input since 5.6) are not distinguishable in
# Codex's usage breakdown, so cost is a slight undercount either way.
_BUILTIN_PER_MTOK: dict[str, tuple[float, float, float]] = {
    "gpt-5.6-sol": (5.00, 0.50, 30.00),
    "gpt-5.6": (5.00, 0.50, 30.00),  # bare alias routes to sol
    "gpt-5.6-terra": (2.50, 0.25, 15.00),
    "gpt-5.6-luna": (1.00, 0.10, 6.00),
    "gpt-5.3-codex": (1.75, 0.175, 14.00),
    # OpenRouter models the Codex binding routes (``openrouter/<vendor>/<model>``, see
    # ``harness.codex``). LiteLLM keys OpenRouter the same way, so a dataset entry wins
    # over these as soon as one exists — as of 2026-08-20 it has none for them. Listed
    # OpenRouter prices: it routes a request to one of several upstream providers, whose
    # prices differ slightly, so cost is an estimate even when the dataset is reachable.
    "openrouter/moonshotai/kimi-k3": (3.00, 0.30, 15.00),
    "openrouter/z-ai/glm-5.3": (1.40, 0.26, 4.40),
    "openrouter/deepseek/deepseek-v4-flash": (0.084, 0.0168, 0.168),
    "openrouter/deepseek/deepseek-v4-pro": (1.60, 0.135, 3.20),
}


@dataclass(frozen=True)
class ModelPrice:
    """USD per token, plus where the numbers came from (``litellm`` or ``builtin``)."""

    input_per_token: float
    cached_input_per_token: float
    output_per_token: float
    source: str

    def cost_usd(self, input_tokens: int, cached_tokens: int, output_tokens: int) -> float:
        fresh = max(input_tokens - cached_tokens, 0)
        return (
            fresh * self.input_per_token
            + cached_tokens * self.cached_input_per_token
            + output_tokens * self.output_per_token
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


def model_price(model: str) -> ModelPrice | None:
    """Resolve ``model`` to per-token USD prices, or ``None`` if unknown everywhere.

    LiteLLM keys are tried as the bare id then ``openai/<id>`` (Codex calls OpenAI
    directly). An ``openrouter/<vendor>/<model>`` id needs no special case: that IS
    LiteLLM's key for an OpenRouter model, so the bare-id probe covers it. May block up to the fetch timeout on first call — call it off the event
    loop (``asyncio.to_thread``).
    """
    data = _litellm_prices()
    if data:
        for key in (model, f"openai/{model}"):
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
            )
    per_mtok = _BUILTIN_PER_MTOK.get(model)
    if per_mtok is not None:
        return ModelPrice(
            input_per_token=per_mtok[0] / 1e6,
            cached_input_per_token=per_mtok[1] / 1e6,
            output_per_token=per_mtok[2] / 1e6,
            source="builtin",
        )
    return None

"""List the upstream endpoints OpenRouter can route a model to — free, no model call.

OpenRouter can serve one model id through providers with different quantization, context
limits, prices, and tool support. Use this command to inspect the current choices before
setting ``openrouter_provider``.

This reads the public catalogue plus, with a key, what your account can reach. It makes no
model calls and costs nothing.

Run:
  .venv/bin/python dev/openrouter_endpoints.py                       # the models we ship
  .venv/bin/python dev/openrouter_endpoints.py z-ai/glm-5.3          # a specific model
  .venv/bin/python dev/openrouter_endpoints.py --probe moonshotai/kimi-k3 -n 12

``--probe`` sends N tiny identical completions and reports which provider served each.
**That part costs money** (a fraction of a cent per call). Agent turns also report this
information as `openrouter_request` events.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from collections import Counter

API = "https://openrouter.ai/api/v1"
DEFAULT_MODELS = [
    "moonshotai/kimi-k3",
    "z-ai/glm-5.3",
    "deepseek/deepseek-v4-flash",
    "deepseek/deepseek-v4-pro",
]


def _base_provider_slug(endpoint_slug: object) -> str:
    """Return the provider-wide part of an OpenRouter endpoint slug."""
    return str(endpoint_slug or "").split("/", 1)[0]


def _get(path: str, key: str | None) -> dict:
    req = urllib.request.Request(f"{API}{path}")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310 — fixed host
        return json.loads(r.read())


def _post(body: dict, key: str) -> dict:
    req = urllib.request.Request(f"{API}/chat/completions", data=json.dumps(body).encode())
    req.add_header("Authorization", f"Bearer {key}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310 — fixed host
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": {"message": e.read().decode()[:200]}}


def show_endpoints(model: str, key: str | None) -> None:
    try:
        data = _get(f"/models/{model}/endpoints", key).get("data", {})
    except urllib.error.HTTPError as e:
        print(f"{model}: HTTP {e.code} {e.read().decode()[:120]}")
        return
    eps = data.get("endpoints", [])
    model_price = data.get("pricing") or {}
    print(f"\n{model} — {len(eps)} endpoint(s)")
    if model_price:
        prompt = model_price.get("prompt")
        completion = model_price.get("completion")
        if prompt not in (None, "") and completion not in (None, ""):
            print(
                "  model-list price: "
                f"${float(prompt) * 1e6:.4g}/M input, "
                f"${float(completion) * 1e6:.4g}/M output"
            )
    print(
        f"  {'provider slug':<16} {'exact endpoint':<22} {'provider':<20} "
        f"{'quant':<9} {'context':>10} {'max_out':>10}  "
        f"{'$/M in':>8} {'$/M out':>9} {'cache':>8}  tools  schema"
    )
    for e in sorted(eps, key=lambda x: str(x.get("provider_name"))):
        params = e.get("supported_parameters") or []
        prices = e.get("pricing", {})

        def per_million(name: str) -> str:
            value = prices.get(name)
            return f"{float(value) * 1e6:.4g}" if value not in (None, "") else "?"

        print(
            f"  {_base_provider_slug(e.get('tag')):<16} {str(e.get('tag')):<22} "
            f"{str(e.get('provider_name')):<20} "
            f"{str(e.get('quantization')):<9} "
            f"{str(e.get('context_length')):>10} {str(e.get('max_completion_tokens')):>10}  "
            f"{per_million('prompt'):>8} {per_million('completion'):>9} "
            f"{per_million('input_cache_read'):>8}  "
            f"{'yes' if 'tools' in params else 'NO':<5}  "
            f"{'yes' if 'response_format' in params else 'NO'}"
        )
    quants = {str(e.get("quantization")) for e in eps}
    if len(quants) > 1:
        print(f"  ! mixed quantization across providers: {sorted(quants)}")
    if any("tools" not in (e.get("supported_parameters") or []) for e in eps):
        print("  ! some endpoints do not support tool calling (an agent needs it)")


def probe(model: str, n: int, key: str) -> None:
    """Send N identical tiny completions; report who served them. Costs money."""
    print(f"\nprobing {model} with {n} identical request(s) — this spends real money")
    seen: Counter[str] = Counter()
    for i in range(n):
        d = _post(
            {
                "model": model,
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "ok"}],
            },
            key,
        )
        who = d.get("provider") or f"ERROR {str(d.get('error', {}).get('message'))[:60]}"
        seen[who] += 1
        print(f"  {i + 1:>3}. {who}")
    print(f"  distinct providers: {len(seen)} — {dict(seen)}")
    if len(seen) > 1:
        print("  ! providers varied; set openrouter_provider when runs must use one provider")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("models", nargs="*")
    ap.add_argument("--probe", action="store_true", help="also send N identical calls (COSTS MONEY)")
    ap.add_argument("-n", type=int, default=12, help="probe count (default 12)")
    args = ap.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("note: OPENROUTER_API_KEY unset — showing the public catalogue only\n")
    for model in args.models or DEFAULT_MODELS:
        show_endpoints(model, key)
        if args.probe:
            if not key:
                print("  (skipping probe: no OPENROUTER_API_KEY)")
            else:
                probe(model, args.n, key)
    return 0


if __name__ == "__main__":
    sys.exit(main())

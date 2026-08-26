"""The normalized ``usage`` record — one shape, one meaning, on every harness.

``RunResult.usage`` is a flat dict with a fixed key set, covering the **whole turn**:
subagent sessions and background-task re-invocation segments included. Every key is
always present; ``None`` means the harness did not report that number (never "zero").
The input buckets are mutually disjoint (Anthropic's convention); ``reasoning_output_tokens``
is a share of ``output_tokens``, so total tokens = the three input buckets + ``output_tokens``:

* ``input_tokens``                 — uncached input, not written to a prompt cache
* ``cache_read_input_tokens``      — input served from the prompt cache
* ``cache_creation_input_tokens``  — input written into the prompt cache
* ``output_tokens``                — all generated tokens (reasoning included)
* ``reasoning_output_tokens``      — the reasoning share of ``output_tokens``
                                     (Codex reports it; Anthropic does not → ``None``)

Codex's wire counters use the opposite convention (``input_tokens`` *includes* the
cached and cache-written parts — verified live: input 20068 = 6 fresh + 9905 cached +
10157 written), so :func:`from_codex_counts` converts by subtraction. On OpenAI models
``cache_creation_input_tokens`` is informational — cache writes bill at the plain input
rate, unlike Anthropic's write premium.

The harness's own verbatim records stay on the result event's ``raw``
(``model_usage`` / ``cli_usage`` on claude-code, ``subagent_usage`` on codex).
"""

from __future__ import annotations

USAGE_KEYS = (
    "input_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
)


def merge_usage(*parts: dict | None) -> dict:
    """Key-wise sum of normalized usage dicts; a key every part left ``None`` stays ``None``.

    ``None`` means "not reported", so it must not poison a sum (a subagent that reports
    cache writes still counts when the parent's calls did not) — but it must not become
    a fake ``0`` either when nobody reported the number.
    """
    out: dict = {}
    for key in USAGE_KEYS:
        values = [p[key] for p in parts if p is not None and p.get(key) is not None]
        out[key] = sum(values) if values else None
    return out


def from_claude_model_usage(model_usage: dict | None) -> dict | None:
    """Aggregate the Claude Code CLI's per-model ``model_usage`` into the normalized shape.

    ``model_usage`` is the CLI's complete record — subagent (Agent-tool) sessions
    included, already cumulative across re-invocation segments (verified live; summing
    segments would double-count). Anthropic's buckets are already disjoint, so this is a
    plain per-key sum across models. Anthropic reports no reasoning-token split
    (thinking is inside ``outputTokens``) → ``reasoning_output_tokens`` is ``None``.
    """
    if not model_usage:
        return None
    usage = dict.fromkeys(USAGE_KEYS)
    for source_key, target_key in (
        ("inputTokens", "input_tokens"),
        ("cacheReadInputTokens", "cache_read_input_tokens"),
        ("cacheCreationInputTokens", "cache_creation_input_tokens"),
        ("outputTokens", "output_tokens"),
    ):
        values = [
            entry[source_key]
            for entry in model_usage.values()
            if isinstance(entry, dict) and entry.get(source_key) is not None
        ]
        usage[target_key] = sum(values) if values else None
    return usage


def from_claude_cli_usage(cli_usage: dict | None) -> dict:
    """Fallback mapping from the CLI's flat verbatim ``usage`` dict.

    Only for result messages that carry no ``model_usage`` — the numbers keep that
    dict's narrow scope (main conversation loop, final re-invocation segment).
    """
    cli_usage = cli_usage or {}
    usage = dict.fromkeys(USAGE_KEYS)
    for key in USAGE_KEYS[:4]:  # the CLI has no reasoning split
        value = cli_usage.get(key)
        usage[key] = int(value) if value is not None else None
    return usage


def from_codex_counts(
    input_tokens: int,
    cached_input_tokens: int,
    cache_write_input_tokens: int | None,
    output_tokens: int,
    reasoning_output_tokens: int,
) -> dict:
    """Convert Codex's inclusive counters into the disjoint normalized buckets.

    Codex's ``input_tokens`` includes both the cache-read and cache-written parts;
    ``cache_write_input_tokens`` is optional on the wire (``None`` = not reported,
    e.g. older CLIs — never coerce it to 0).
    """
    return {
        "input_tokens": max(
            input_tokens - cached_input_tokens - (cache_write_input_tokens or 0), 0
        ),
        "cache_read_input_tokens": cached_input_tokens,
        "cache_creation_input_tokens": cache_write_input_tokens,
        "output_tokens": output_tokens,
        "reasoning_output_tokens": reasoning_output_tokens,
    }

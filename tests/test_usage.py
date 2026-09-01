"""The normalized usage record: conversions and merge semantics (harness/_usage.py)."""

from __future__ import annotations

from remote_agent_toolkit.harness._usage import (
    USAGE_KEYS,
    from_claude_cli_usage,
    from_claude_model_usage,
    from_codex_counts,
    merge_usage,
)


def test_codex_counts_convert_inclusive_input_to_disjoint_buckets():
    # Verified live decomposition: wire input 20068 = 6 fresh + 9905 cached + 10157 written.
    usage = from_codex_counts(20068, 9905, 10157, 249, 26)
    assert usage == {
        "input_tokens": 6,
        "cache_read_input_tokens": 9905,
        "cache_creation_input_tokens": 10157,
        "output_tokens": 249,
        "reasoning_output_tokens": 26,
    }


def test_codex_counts_keep_unreported_cache_writes_none_and_clamp_at_zero():
    usage = from_codex_counts(100, 150, None, 10, 0)  # cached > input: never negative
    assert usage["input_tokens"] == 0
    assert usage["cache_creation_input_tokens"] is None


def test_claude_model_usage_sums_across_models_with_reasoning_none():
    usage = from_claude_model_usage(
        {
            "claude-sonnet-5": {
                "inputTokens": 10, "outputTokens": 20, "cacheReadInputTokens": 30,
                "cacheCreationInputTokens": 40, "webSearchRequests": 1,
                "costUSD": 0.5, "contextWindow": 200_000,
            },
            "claude-haiku-4-5": {
                "inputTokens": 1, "outputTokens": 2, "cacheReadInputTokens": 3,
                "cacheCreationInputTokens": 4, "costUSD": 0.01,
            },
        }
    )
    assert usage == {
        "input_tokens": 11,
        "cache_read_input_tokens": 33,
        "cache_creation_input_tokens": 44,
        "output_tokens": 22,
        "reasoning_output_tokens": None,  # Anthropic reports no reasoning split
    }


def test_claude_model_usage_absent_or_empty_is_none():
    assert from_claude_model_usage(None) is None
    assert from_claude_model_usage({}) is None


def test_claude_cli_usage_fallback_maps_flat_keys_and_leaves_gaps_none():
    usage = from_claude_cli_usage({"input_tokens": 5, "output_tokens": 7, "extra": 1})
    assert usage == {
        "input_tokens": 5,
        "cache_read_input_tokens": None,
        "cache_creation_input_tokens": None,
        "output_tokens": 7,
        "reasoning_output_tokens": None,
    }
    assert set(from_claude_cli_usage(None)) == set(USAGE_KEYS)


def test_merge_sums_keywise_and_keeps_all_none_as_none():
    parent = from_codex_counts(1000, 300, None, 100, 10)
    sub = from_codex_counts(500, 0, 50, 20, 0)
    merged = merge_usage(parent, sub, None)  # a None part (no record) is skipped
    assert merged == {
        "input_tokens": 700 + 450,
        "cache_read_input_tokens": 300,
        "cache_creation_input_tokens": 50,  # None + 50: not poisoned, not a fake 0 either
        "output_tokens": 120,
        "reasoning_output_tokens": 10,
    }
    assert merge_usage(parent)["cache_creation_input_tokens"] is None

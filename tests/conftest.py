"""Shared fixtures. The suite must stay fully offline (see .github/workflows/ci.yml)."""

from __future__ import annotations

import pytest

from remote_agent_toolkit.harness import pricing


@pytest.fixture(autouse=True)
def _no_pricing_network(monkeypatch):
    """Keep the LiteLLM pricing fetch off the network: tests price via the baked table.

    Tests that exercise the LiteLLM path monkeypatch ``pricing._litellm_prices``
    themselves (this fixture stubs the underlying fetch, so their override wins).
    """
    pricing.clear_cache()
    monkeypatch.setattr(pricing, "_fetch_litellm", lambda: None)
    yield
    pricing.clear_cache()

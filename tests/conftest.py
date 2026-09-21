"""Shared fixtures. The suite must stay fully offline (see .github/workflows/ci.yml)."""

from __future__ import annotations

import pytest

from agent_run.harness import pricing


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


@pytest.fixture(autouse=True)
def _no_token_minting(monkeypatch):
    """Keep the run's token minting off the network (STS for the GCS token, IAM for the model token).

    ``SandboxSession._submit`` mints both per turn; the offline suite gets fixed fakes
    instead, and the client-side refresh writer/deleter become no-ops. ``test_scoped_gcs.py``
    covers the real functions against fakes.
    """
    from agent_run.runtime.sandbox import model_token, scoped_gcs

    monkeypatch.setattr(model_token, "mint_model_token", lambda *a, **kw: ("fake-model-token", None))

    real_write, real_delete = scoped_gcs.write_run_token, scoped_gcs.delete_run_token

    def offline(real):
        # With an injected store the real function runs (offline); without one it would
        # build a real GCS client, so it becomes a no-op.
        def stub(*a, store=None, **kw):
            return real(*a, store=store, **kw) if store is not None else None
        return stub

    monkeypatch.setattr(scoped_gcs, "mint_run_token", lambda *a, **kw: ("fake-run-token", None))
    monkeypatch.setattr(scoped_gcs, "write_run_token", offline(real_write))
    monkeypatch.setattr(scoped_gcs, "delete_run_token", offline(real_delete))


@pytest.fixture(autouse=True)
def _short_long_polls(monkeypatch):
    """Keep the client's /events long-poll short: ``asyncio.run`` waits for executor threads
    at shutdown, so a cancelled run would otherwise hold a test for the full hold time."""
    from agent_run.runtime.sandbox import backend

    monkeypatch.setattr(backend, "EVENTS_WAIT_S", 0.2)
    monkeypatch.setattr(backend, "EVENTS_CALL_TIMEOUT_S", 5.0)
    # A lost /turn answer is re-asked of the same sandbox for a while; keep that short too.
    monkeypatch.setattr(backend, "DISPATCH_RECONCILE_S", 0.5)
    monkeypatch.setattr(backend, "DISPATCH_RETRY_SLEEP_S", 0.02)

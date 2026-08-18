"""Shared fixtures. The suite must stay fully offline (see .github/workflows/ci.yml)."""

from __future__ import annotations

import pytest

from remote_agent_toolkit.harness import pricing


@pytest.fixture(autouse=True)
def _no_host_resource_sampler(monkeypatch):
    """Keep the worker's CPU/RAM self-sampler off the HOST's cgroups: sampler off by default.

    On a machine whose /sys/fs/cgroup is readable (any cgroup-v2 Linux), ``_run_turn``
    otherwise starts a REAL daemon sampler thread whose first sample races the turn's own
    events — and because these tests monkeypatch the ``CloudLoggingSink`` class, the
    sampler's side-log emits land in the same ``InMemorySink`` the test asserts on, so
    "the last sink event is the result" fails whenever the thread loses the race (the
    long-standing ~50% flake in ``test_run_turn_maps_session_id_and_surfaces_harness_crash``).
    Host cgroup reads are also machine-dependent state the offline suite must not touch.
    Sampler tests opt back in explicitly (their own ``AGENT_RESOURCE_SAMPLE_S`` +
    a fake cgroup ``root``), which overrides this default.
    """
    monkeypatch.setenv("AGENT_RESOURCE_SAMPLE_S", "0")


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

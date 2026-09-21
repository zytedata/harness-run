"""The ``[local]``-extra hint fires on the REAL import path (PR #37 review).

The harness SDKs live in the ``local`` install extra. The guard used to sit only in
``build_options``, but both harnesses' ``_run`` import the SDK a few lines before they
call it, so a base install surfaced the bare ``No module named ...`` instead of the
actionable message. These tests block the SDK import (``None`` in ``sys.modules``, what
a base install looks like to the import system) and drive the harnesses' actual run
path, pinning the message where users hit it.
"""

from __future__ import annotations

import asyncio
import sys

import pytest

from agent_run import AgentSpec
from agent_run.harness._local_sdk import import_local_sdk
from agent_run.harness.context import RunContext


def _ctx(tmp_path, spec):
    return RunContext(
        spec=spec, prompt="go", job_dir=tmp_path / "job", session_id="sid", secrets={}
    )


def test_import_local_sdk_maps_a_missing_sdk_to_the_extra_hint():
    with pytest.raises(ModuleNotFoundError) as ei:
        import_local_sdk("definitely_not_installed_xyz", "some-sdk", "Local X execution")
    assert "some-sdk is not installed" in str(ei.value)
    assert "[local] extra" in str(ei.value)
    assert isinstance(ei.value.__cause__, ModuleNotFoundError)  # original kept for debugging


def test_import_local_sdk_returns_the_module_when_present():
    import json

    assert import_local_sdk("json", "json", "X") is json


def test_import_local_sdk_does_not_mask_a_missing_inner_dependency(tmp_path, monkeypatch):
    """A ModuleNotFoundError raised INSIDE the SDK (one of its own deps missing) is a
    different problem — the extra would not fix it, so it must pass through untranslated."""
    (tmp_path / "fake_local_sdk_pkg.py").write_text("import nonexistent_inner_dep_xyz\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ModuleNotFoundError) as ei:
        import_local_sdk("fake_local_sdk_pkg", "fake-sdk", "X")
    assert ei.value.name == "nonexistent_inner_dep_xyz"
    assert "[local]" not in str(ei.value)


def test_claude_run_path_shows_the_extra_hint_without_the_sdk(tmp_path, monkeypatch):
    from agent_run.harness.claude_code import ClaudeCodeHarness

    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)  # what a base install sees
    spec = AgentSpec(name="a", model="m")

    async def drive():
        return [ev async for ev in ClaudeCodeHarness().run(spec, _ctx(tmp_path, spec))]

    with pytest.raises(
        ModuleNotFoundError,
        match=r"claude-agent-sdk is not installed.*\[local\] extra",
    ):
        asyncio.run(drive())


def test_codex_run_path_shows_the_extra_hint_without_the_sdk(tmp_path, monkeypatch):
    from agent_run.harness.codex import CodexHarness

    monkeypatch.setitem(sys.modules, "openai_codex", None)
    spec = AgentSpec(name="a", model="gpt-5.6-luna", harness="codex")

    async def drive():
        return [ev async for ev in CodexHarness().run(spec, _ctx(tmp_path, spec))]

    with pytest.raises(
        ModuleNotFoundError,
        match=r"openai-codex is not installed.*\[local\] extra",
    ):
        asyncio.run(drive())

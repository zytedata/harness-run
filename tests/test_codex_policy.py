"""Unsupported security restrictions must not produce runnable Codex options."""
import pytest

from harness_run import AgentSpec
from harness_run.harness.codex import CodexHarness
from harness_run.harness.context import RunContext


@pytest.mark.parametrize("settings,field", [
    ({"permission_mode": "unknown-typo"}, "permission_mode"),
    ({"allowed_tools": ["Read"]}, "allowed_tools"),
    ({"allowed_tools": []}, "allowed_tools"),
    ({"disallowed_tools": ["Bash"]}, "disallowed_tools"),
])
def test_unsupported_policy_is_rejected_before_workspace_setup(tmp_path, settings, field):
    spec = AgentSpec(name="audit", model="dummy", harness="codex", **settings)
    ctx = RunContext(spec=spec, prompt="dummy", job_dir=tmp_path, session_id="audit")
    with pytest.raises(ValueError, match=field):
        CodexHarness().build_options(spec, ctx)
    assert not (tmp_path / "codex_home").exists()


@pytest.mark.parametrize("mode", ["default", "acceptEdits", "bypassPermissions"])
def test_supported_modes_remain_explicitly_available(tmp_path, mode):
    spec = AgentSpec(name="audit", model="dummy", harness="codex", permission_mode=mode)
    ctx = RunContext(spec=spec, prompt="dummy", job_dir=tmp_path, session_id="audit")
    opts = CodexHarness().build_options(spec, ctx)
    assert (opts.thread_args["sandbox"].name == "full_access") == (mode == "bypassPermissions")

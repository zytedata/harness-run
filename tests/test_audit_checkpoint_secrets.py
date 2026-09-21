"""A failed credential scrub must never create a credential-bearing snapshot (S11)."""
from pathlib import Path

import pytest

from agent_run import AgentSpec
from agent_run.harness._shared import finalize_checkpoint
from agent_run.harness.context import RunContext
from agent_run.ports.blobstore import LocalBlobStore


@pytest.mark.parametrize("operation", ["read_text", "write_text"])
def test_scrub_failure_prevents_checkpoint(tmp_path, monkeypatch, operation):
    spec = AgentSpec(name="audit", model="dummy", checkpoint=True)
    blobs = LocalBlobStore(str(tmp_path / "blobs"))
    ctx = RunContext(spec=spec, prompt="dummy", job_dir=tmp_path / "job",
                     session_id="audit", blobs=blobs, session_store=object())
    config = ctx.workspace / "repo" / ".git" / "config"
    config.parent.mkdir(parents=True)
    config.write_text('url = https://user:AUDIT_FAKE_TOKEN@audit.invalid/repo.git\n')
    original = getattr(Path, operation)

    def denied(path, *args, **kwargs):
        if path == config:
            raise PermissionError("AUDIT_FAKE_TOKEN must not be echoed")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, operation, denied)
    event = finalize_checkpoint(spec, ctx)
    assert event.raw["event"] == "checkpoint_error"
    assert "AUDIT_FAKE_TOKEN" not in event.summary
    assert not blobs.exists("workspace/audit.tar.gz")

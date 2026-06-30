"""RepoSource serialization + provision_repos (clone before the agent runs), offline."""

from __future__ import annotations

import subprocess

from remote_agent_toolkit import AgentSpec, RepoSource
from remote_agent_toolkit.integrations.git import provision_repos


def test_reposource_and_spec_roundtrip():
    r = RepoSource.git("https://github.com/o/r", ref="v1")
    assert r.url == "https://github.com/o/r" and r.ref == "v1"
    assert RepoSource.from_dict(r.to_dict()) == r
    spec = AgentSpec(name="a", model="m", repos=[RepoSource.git("https://x/y")])
    assert AgentSpec.from_dict(spec.to_dict()) == spec  # repos survive dict round-trip


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True)


def test_provision_repos_clones_into_job_dir(tmp_path):
    # Build a tiny local git repo (no network, no token).
    src = tmp_path / "myrepo"
    src.mkdir()
    _git(src, "init", "-q")
    (src / "hello.txt").write_text("hi")
    _git(src, "add", "-A")
    _git(src, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init")

    job = tmp_path / "job"
    job.mkdir()
    names = provision_repos(job, (RepoSource.git(str(src)),), secrets={})
    assert names == ["myrepo"]
    assert (job / "myrepo" / "hello.txt").read_text() == "hi"


def test_provision_repos_empty_is_noop(tmp_path):
    assert provision_repos(tmp_path, (), secrets={}) == []

"""RepoSource serialization + provision_repos (clone before the agent runs), offline."""

from __future__ import annotations

import subprocess

from agent_run import AgentSpec, RepoSource
from agent_run.integrations.git import (
    _auth_url,
    provision_repos,
    reauth_repos,
    scrub_repo_tokens,
)


def test_reposource_and_spec_roundtrip():
    r = RepoSource.git("https://bitbucket.org/o/r", ref="v1", auth="BB_TOKEN", auth_user="alice")
    assert r.url == "https://bitbucket.org/o/r" and r.ref == "v1"
    assert r.auth == "BB_TOKEN" and r.auth_user == "alice"
    assert RepoSource.from_dict(r.to_dict()) == r
    spec = AgentSpec(harness="claude-code", name="a", model="m", repos=[RepoSource.git("https://x/y")])
    assert AgentSpec.from_dict(spec.to_dict()) == spec  # repos survive dict round-trip


def test_auth_url_is_host_aware():
    # Each host gets the userinfo username git expects with a token over HTTPS.
    assert _auth_url("https://github.com/o/r", "T") == "https://x-access-token:T@github.com/o/r"
    assert _auth_url("https://bitbucket.org/o/r", "T") == "https://x-token-auth:T@bitbucket.org/o/r"
    assert _auth_url("https://gitlab.com/o/r", "T") == "https://oauth2:T@gitlab.com/o/r"
    # Unknown host falls back to the GitHub scheme.
    assert _auth_url("https://git.acme.io/o/r", "T") == "https://x-access-token:T@git.acme.io/o/r"
    # An explicit user overrides the host default — the Bitbucket API-token scheme
    # (`https://<account>:<token>@bitbucket.org/...`).
    assert _auth_url("https://bitbucket.org/o/r", "T", "alice") == "https://alice:T@bitbucket.org/o/r"


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


def _origin(repo_dir):
    return subprocess.run(
        ["git", "remote", "get-url", "origin"], cwd=str(repo_dir),
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def test_scrub_and_reauth_round_trip(tmp_path):
    # A cloned repo whose origin carries a push token (as provision would embed it).
    job = tmp_path / "job"
    repo = job / "spiders"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q")
    url = "https://bitbucket.org/acme/spiders"
    _git(repo, "remote", "add", "origin", _auth_url(url, "SEKRET"))
    assert "SEKRET" in _origin(repo)

    # Scrub before a checkpoint snapshot: the token is stripped from .git/config.
    assert scrub_repo_tokens(job) == 1
    assert "SEKRET" not in _origin(repo)
    assert _origin(repo) == url  # back to a clean, tokenless origin

    # On resume, reauth re-embeds the token from this turn's secrets (host-aware).
    repos = (RepoSource.git(url, auth="BB_TOKEN"),)
    assert reauth_repos(job, repos, {"BB_TOKEN": "SEKRET2"}) == ["spiders"]
    assert _origin(repo) == "https://x-token-auth:SEKRET2@bitbucket.org/acme/spiders"

    # No token supplied → reauth is a no-op (leaves the tokenless origin).
    scrub_repo_tokens(job)
    assert reauth_repos(job, repos, {}) == []
    assert _origin(repo) == url

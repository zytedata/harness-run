"""Git clone + PAT injection (DESIGN.md §8).

Two use cases share the same primitives:

* **Skill sources** (``skills.provision``) clone public skill repos — no auth needed,
  but ``clone`` accepts an optional PAT for private repos.
* **Repo provisioning** (``provision_repo``) clones the target repo into the per-job cwd
  before the agent loop, authenticating with a PAT resolved at runtime from the
  ``SecretResolver`` (never baked into the deployed engine — DESIGN.md §3.5) and
  configuring a commit identity so the agent can branch/commit/push.

The PAT is embedded in the ``origin`` remote so the agent's ``git push`` works without
handling credentials itself — acceptable because the job dir is isolated and ephemeral.
The PAT is redacted from any surfaced output. Uses stdlib ``subprocess``; no third-party
import. Lifted & generalized from the PoC ``git_repo.py`` (Zyte-specific defaults dropped).
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


def _inject_pat(url: str, pat: str) -> str:
    """https://github.com/org/repo(.git) -> https://x-access-token:<pat>@github.com/..."""
    if not url.startswith("https://"):
        raise ValueError(f"only https git URLs supported, got: {url}")
    return "https://x-access-token:" + pat + "@" + url[len("https://"):]


def redact(text: str, pat: str | None) -> str:
    return text.replace(pat, "<PAT>") if pat else text


def repo_name(repo_url: str) -> str:
    name = repo_url.rstrip("/").split("/")[-1]
    return name[:-4] if name.endswith(".git") else name


def clone(
    url: str,
    ref: str | None = None,
    dest: Path | None = None,
    pat: str | None = None,
) -> Path:
    """Plain ``git clone`` of ``url`` (optionally at ``ref``) into ``dest``; return the dir.

    If ``dest`` is ``None`` a fresh temp directory is created under the system temp. With
    ``ref`` the clone is shallow (``--depth 1``). No PAT is needed for public repos; pass
    ``pat`` to inject token-authenticated HTTPS for private ones. The PAT is redacted from
    any error message.
    """
    dest = Path(dest) if dest is not None else Path(tempfile.mkdtemp(prefix="rat-clone-"))
    target = _inject_pat(url, pat) if pat else url
    cmd = ["git", "clone"]
    if ref:
        cmd += ["--branch", ref, "--depth", "1"]
    cmd += [target, str(dest)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("git clone failed: " + redact(proc.stderr, pat)[:500])
    return dest


def provision_repo(
    job_dir: Path,
    repo_url: str,
    pat: str,
    *,
    user_name: str = "Agent",
    user_email: str = "agent@example.com",
) -> Path:
    """Clone ``repo_url`` into ``job_dir/<name>`` with PAT auth + commit/push config.

    Returns the cloned repo directory. Raises on clone failure (PAT redacted from the error).
    """
    repo_dir = Path(job_dir) / repo_name(repo_url)
    authed = _inject_pat(repo_url, pat)
    clone_proc = subprocess.run(
        ["git", "clone", authed, str(repo_dir)], capture_output=True, text=True
    )
    if clone_proc.returncode != 0:
        raise RuntimeError("git clone failed: " + redact(clone_proc.stderr, pat)[:500])

    # Identity for commits (the embedded-PAT origin already enables push).
    for cfg_key, val in (("user.name", user_name), ("user.email", user_email)):
        subprocess.run(["git", "config", cfg_key, val], cwd=str(repo_dir), check=False)
    return repo_dir


# Conventional GitHub token names looked up in resolved secrets, in priority order.
_GH_TOKEN_KEYS = ("GH_TOKEN", "GITHUB_TOKEN", "GH_PAT")


def provision_repos(
    job_dir: Path,
    repos,
    secrets: dict | None = None,
    *,
    user_name: str = "Agent",
    user_email: str = "agent@example.com",
) -> list[str]:
    """Clone each ``RepoSource`` in ``repos`` into ``job_dir`` before the agent runs.

    If ``secrets`` carries a GitHub token (``GH_TOKEN`` / ``GITHUB_TOKEN`` / ``GH_PAT``) and
    the URL is HTTPS, the clone is token-authenticated (push-ready, with a commit identity);
    otherwise it's a plain read-only clone (fine for public repos). Honors each source's
    ``ref``. Returns the cloned directory names (under ``job_dir``).
    """
    if not repos:
        return []
    secrets = secrets or {}
    token = next((secrets[k] for k in _GH_TOKEN_KEYS if secrets.get(k)), None)
    names: list[str] = []
    for repo in repos:
        dest = Path(job_dir) / repo_name(repo.url)
        use_pat = token if (token and repo.url.startswith("https://")) else None
        clone(repo.url, ref=repo.ref, dest=dest, pat=use_pat)
        if use_pat:  # configure a commit identity so the agent can commit + push
            for cfg_key, val in (("user.name", user_name), ("user.email", user_email)):
                subprocess.run(["git", "config", cfg_key, val], cwd=str(dest), check=False)
        names.append(dest.name)
    return names

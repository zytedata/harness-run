"""Git clone + token injection (DESIGN.md §8).

Two use cases share the same primitives:

* **Skill sources** (``skills.provision``) clone public skill repos — no auth needed,
  but ``clone`` accepts an optional token for private repos.
* **Repo provisioning** (``provision_repos``) clones the target repos into the per-job cwd
  before the agent loop, authenticating with a token named by ``RepoSource.auth`` and resolved
  at runtime from the per-invocation ``secrets`` (never baked into the deployed engine —
  DESIGN.md §3.5), and configuring a commit identity so the agent can branch/commit/push.

The token is embedded in the ``origin`` remote so the agent's ``git push`` works without
handling credentials itself — acceptable because the job dir is isolated and ephemeral, and
because the real defense is a *scoped* token, not hiding (the agent can read ``.git/config``).
Injection is **host-aware**: GitHub uses ``x-access-token``, Bitbucket ``x-token-auth``, GitLab
``oauth2`` (default ``x-access-token``). The token is redacted from any surfaced output, is not
placed in the agent's environment, and is scrubbed from ``.git/config`` before a checkpoint
snapshot (:func:`scrub_repo_tokens`) then re-embedded on resume (:func:`reauth_repos`). Uses
stdlib ``subprocess``; no third-party import.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

# Host substring -> the userinfo *username* git wants alongside a token over HTTPS. The token
# is the password. Default (unmatched host) is GitHub's scheme, the most common.
_HOST_AUTH_USER = (
    ("bitbucket.org", "x-token-auth"),
    ("gitlab", "oauth2"),
    ("github", "x-access-token"),
)
_DEFAULT_AUTH_USER = "x-access-token"

# Strips any ``user[:pass]@`` userinfo from an https URL (redaction / credential scrubbing).
_USERINFO_RE = re.compile(r"(https://)[^/@\s]+@")


def _auth_user_for(url: str) -> str:
    host = url[len("https://"):].split("/", 1)[0].lower()
    for needle, user in _HOST_AUTH_USER:
        if needle in host:
            return user
    return _DEFAULT_AUTH_USER


def _auth_url(url: str, token: str) -> str:
    """Embed ``token`` into an https git ``url`` using the host-appropriate userinfo scheme."""
    if not url.startswith("https://"):
        raise ValueError(f"only https git URLs supported for token auth, got: {url}")
    user = _auth_user_for(url)
    return f"https://{user}:{token}@{url[len('https://'):]}"


def redact(text: str, token: str | None) -> str:
    return text.replace(token, "<TOKEN>") if token else text


def repo_name(repo_url: str) -> str:
    name = repo_url.rstrip("/").split("/")[-1]
    return name[:-4] if name.endswith(".git") else name


def _set_commit_identity(repo_dir: Path, user_name: str, user_email: str) -> None:
    for cfg_key, val in (("user.name", user_name), ("user.email", user_email)):
        subprocess.run(["git", "config", cfg_key, val], cwd=str(repo_dir), check=False)


def clone(
    url: str,
    ref: str | None = None,
    dest: Path | None = None,
    token: str | None = None,
) -> Path:
    """Plain ``git clone`` of ``url`` (optionally at ``ref``) into ``dest``; return the dir.

    If ``dest`` is ``None`` a fresh temp directory is created under the system temp. With
    ``ref`` the clone is shallow (``--depth 1``). No token is needed for public repos; pass
    ``token`` to inject host-aware token-authenticated HTTPS for private ones. The token is
    redacted from any error message.
    """
    dest = Path(dest) if dest is not None else Path(tempfile.mkdtemp(prefix="rat-clone-"))
    target = _auth_url(url, token) if token else url
    cmd = ["git", "clone"]
    if ref:
        cmd += ["--branch", ref, "--depth", "1"]
    cmd += [target, str(dest)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("git clone failed: " + redact(proc.stderr, token)[:500])
    return dest


def provision_repo(
    job_dir: Path,
    repo_url: str,
    token: str,
    *,
    user_name: str = "Agent",
    user_email: str = "agent@example.com",
) -> Path:
    """Clone ``repo_url`` into ``job_dir/<name>`` with token auth + commit/push config.

    Returns the cloned repo directory. Raises on clone failure (token redacted from the error).
    """
    repo_dir = Path(job_dir) / repo_name(repo_url)
    clone(repo_url, dest=repo_dir, token=token)
    _set_commit_identity(repo_dir, user_name, user_email)  # embedded-token origin enables push
    return repo_dir


def provision_repos(
    job_dir: Path,
    repos,
    secrets: dict | None = None,
    *,
    user_name: str = "Agent",
    user_email: str = "agent@example.com",
) -> list[str]:
    """Clone each ``RepoSource`` in ``repos`` into ``job_dir`` before the agent runs.

    Each repo's ``auth`` names a per-invocation secret; when that secret is present and the URL
    is HTTPS, the clone is token-authenticated (push-ready, host-aware, with a commit identity),
    otherwise it's a plain read-only clone (fine for public repos). Honors each source's ``ref``.
    Returns the cloned directory names (under ``job_dir``).
    """
    if not repos:
        return []
    secrets = secrets or {}
    names: list[str] = []
    for repo in repos:
        dest = Path(job_dir) / repo_name(repo.url)
        token = secrets.get(repo.auth) if getattr(repo, "auth", None) else None
        use_token = token if (token and repo.url.startswith("https://")) else None
        clone(repo.url, ref=repo.ref, dest=dest, token=use_token)
        if use_token:  # configure a commit identity so the agent can commit + push
            _set_commit_identity(dest, user_name, user_email)
        names.append(dest.name)
    return names


def reauth_repos(job_dir: Path, repos, secrets: dict | None = None) -> list[str]:
    """Re-embed push tokens into restored repos' ``origin`` (resume after a checkpoint).

    Checkpoint snapshots are credential-scrubbed (:func:`scrub_repo_tokens`), so a restored
    workspace has a tokenless ``origin``. On resume we re-embed the token from the freshly
    supplied per-invocation ``secrets`` (host-aware) so the agent can push again. Repos without
    an ``auth`` secret, or whose dir is missing, are skipped. Returns the re-authed dir names.
    """
    if not repos:
        return []
    secrets = secrets or {}
    names: list[str] = []
    for repo in repos:
        token = secrets.get(repo.auth) if getattr(repo, "auth", None) else None
        if not (token and repo.url.startswith("https://")):
            continue
        dest = Path(job_dir) / repo_name(repo.url)
        if not (dest / ".git").is_dir():
            continue
        subprocess.run(
            ["git", "remote", "set-url", "origin", _auth_url(repo.url, token)],
            cwd=str(dest), check=False,
        )
        names.append(dest.name)
    return names


def scrub_repo_tokens(job_dir) -> int:
    """Strip embedded credentials from every ``.git/config`` under ``job_dir`` in place.

    Rewrites any ``https://<user>:<token>@host/...`` remote URL to ``https://host/...`` so a
    checkpoint snapshot never captures a token. Returns the number of config files changed.
    Best-effort per file (an unreadable/unwritable config is skipped). Idempotent.
    """
    changed = 0
    for cfg in Path(job_dir).glob("**/.git/config"):
        try:
            text = cfg.read_text()
        except Exception:  # noqa: BLE001 — unreadable config; skip
            continue
        scrubbed = _USERINFO_RE.sub(r"\1", text)
        if scrubbed != text:
            try:
                cfg.write_text(scrubbed)
                changed += 1
            except Exception:  # noqa: BLE001 — unwritable config; skip
                continue
    return changed

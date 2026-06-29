"""Git clone + PAT injection (DESIGN.md §8) — P1.

Provisions a repo before the agent loop, authenticating with a PAT resolved at runtime
from the ``SecretResolver`` (never baked into the deployed engine — DESIGN.md §3.5).
Lifted from the PoC ``git_repo.py``. Uses stdlib ``subprocess``; no third-party import.
P0: stub.
"""

from __future__ import annotations


def clone(url: str, dest: str, ref: str | None = None, token: str | None = None) -> str:
    """Clone ``url`` into ``dest`` (optionally at ``ref``), injecting ``token`` as auth (P1).

    Returns the checked-out directory path. ``token`` is a runtime-resolved value, not a
    secret name.
    """
    raise NotImplementedError(
        "P1: git clone url@ref into dest with token-authenticated HTTPS; return dest."
    )

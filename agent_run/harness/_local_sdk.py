"""Harness-SDK import with the ``[local]`` install-extra hint.

The harness SDKs (``claude-agent-sdk``, ``openai-codex``) live in the ``local`` install
extra; the base install covers only the remote path. Every lazy SDK import in the
harnesses goes through :func:`import_local_sdk`, so a missing SDK fails with the
actionable message on whichever path imports it first — the ``_run`` imports fire before
``build_options``, so guarding only the latter showed users the bare
``No module named ...`` (PR #37 review).
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


def import_local_sdk(module: str, dist: str, what: str) -> Any:
    """Import harness SDK ``module``, mapping *its* absence to the ``[local]``-extra hint.

    Only the SDK itself missing is translated (``exc.name == module``): a
    ``ModuleNotFoundError`` raised from inside the SDK for one of its own dependencies
    is a different problem, and installing the extra would not fix it.
    """
    try:
        return import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name != module:
            raise
        raise ModuleNotFoundError(
            f"{dist} is not installed. {what} needs the harness SDKs: "
            f"install agent-run with the [local] extra."
        ) from exc

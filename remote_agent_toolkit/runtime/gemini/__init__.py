"""The ``gemini.*`` public namespace — Gemini Agent Runtime backend (DESIGN.md §5).

``gemini.deploy`` is an ops/CI action (rare); ``gemini.get_engine`` /
``gemini.list_engines`` are the app-code hot path (lookup-and-run). The underlying
Google SDK (``agentplatform`` / ``google-cloud-aiplatform``) is an internal detail imported
lazily inside ``.backend``.

No submodule here may be named after a re-export below: importing a submodule binds it as
an attribute of this package, so a ``deploy.py`` would silently overwrite the ``deploy``
function on the first call that lazily imports it (hence ``_deploy.py``).
"""

from __future__ import annotations

from .backend import deploy, get_engine, list_engines

__all__ = ["deploy", "get_engine", "list_engines"]

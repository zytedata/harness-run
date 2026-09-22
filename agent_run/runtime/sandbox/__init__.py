"""The ``sandbox.*`` public namespace — the sandbox runtime backend (DESIGN.md §13).

``sandbox.deploy`` is an ops/CI action (rare); ``sandbox.get_engine`` /
``sandbox.list_engines`` are the app-code hot path (lookup-and-run). The underlying
platform client (``google-cloud-agentplatform``) is an internal detail imported lazily
inside ``.provider``.

No submodule here may be named after a re-export below: importing a submodule binds it as
an attribute of this package, so a ``deploy.py`` would silently overwrite the ``deploy``
function on the first call that lazily imports it (hence ``_image.py`` and friends).
"""

from __future__ import annotations

from .backend import deploy, get_engine, list_engines

__all__ = ["deploy", "get_engine", "list_engines"]

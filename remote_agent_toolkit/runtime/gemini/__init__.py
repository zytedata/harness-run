"""The ``gemini.*`` public namespace — Gemini Agent Runtime backend (DESIGN.md §5).

``gemini.deploy`` is an ops/CI action (rare); ``gemini.get_engine`` /
``gemini.list_engines`` are the app-code hot path (lookup-and-run). The underlying
Google SDK (``vertexai`` / ``google-cloud-aiplatform``) is an internal detail imported
lazily inside ``.backend``.
"""

from __future__ import annotations

from .backend import deploy, get_engine, list_engines

__all__ = ["deploy", "get_engine", "list_engines"]

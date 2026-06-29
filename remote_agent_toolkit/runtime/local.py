"""Local, in-process run plane (DESIGN.md §3.2, §3.4) — P1.

``local.deploy(spec)`` returns a ``LocalEngine`` exposing the *same* Engine/Session/Run
surface as ``gemini``, backed by the in-process ADK harness and filesystem/in-memory
port adapters (no GCP beyond model access). This is the first-class dev loop.

P0: signatures only. Bodies raise ``NotImplementedError``; P1 fills them in.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from ..events import RunResult

if TYPE_CHECKING:
    from ..spec import AgentSpec
    from .base import Engine


def deploy(spec: AgentSpec, **kw: Any) -> Engine:
    """Stand up an in-process engine for ``spec`` (in-process ADK; P1).

    Mirrors ``gemini.deploy`` exactly so the dev loop and prod loop are the same code.
    Wires the local/in-memory port adapters (``LocalBlobStore``, ``InMemorySink``,
    ``InMemoryDispatch``, ``EnvSecretResolver``) and the ``ClaudeCodeHarness``.

    The Claude Agent SDK / ADK are imported lazily here (not at module import time).
    """
    raise NotImplementedError(
        "local.deploy is a P1 capability: build a LocalEngine from the spec backed by "
        "the in-process ADK ClaudeCodeHarness and local/in-memory port adapters."
    )


def run(spec: AgentSpec, message: str, **kw: Any) -> RunResult:
    """Convenience: ``deploy(spec)`` → ``start_session()`` → ``run(message)`` (P1).

    Returns the awaited :class:`RunResult`. For streaming/polling, use ``deploy`` and
    drive the ``Session``/``Run`` directly.
    """
    raise NotImplementedError(
        "local.run is a P1 convenience over deploy + start_session + run; awaits the "
        "Run and returns its RunResult."
    )

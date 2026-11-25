"""The harness seam (DESIGN.md §1, §7): the abstraction over "a coding agent loop".

Two bindings ship: Claude Code (``claude_code``) and Codex (``codex``). Neither is a default.
``resolve_harness`` is the single selection point — both runtimes call it instead of
hardcoding a binding, so ``AgentSpec.harness`` picks the loop on either backend.
Bindings are imported lazily inside the factory to keep the package import zero-dep.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Harness

if TYPE_CHECKING:
    from ..spec import AgentSpec

__all__ = ["Harness", "model_calls_use_vertex", "resolve_harness"]


def resolve_harness(spec: AgentSpec) -> Harness:
    """Instantiate the harness named by ``spec.harness`` (both runtimes' single seam)."""
    name = getattr(spec, "harness", None)
    if not name:
        raise ValueError(
            "AgentSpec.harness is not set — name the coding-agent loop explicitly "
            "('claude-code' or 'codex'); there is no default"
        )
    if name == "claude-code":
        from .claude_code import ClaudeCodeHarness

        return ClaudeCodeHarness()
    if name == "codex":
        from .codex import CodexHarness

        return CodexHarness()
    raise ValueError(f"unknown harness: {name!r} (expected 'claude-code' or 'codex')")


def model_calls_use_vertex(spec: AgentSpec) -> bool:
    """Whether a turn with this effective spec sends its model calls to Vertex.

    This decides which turns of a Vertex engine are given the model token and the Vertex
    environment (the sandbox client mints them, the worker hands them to the CLI): only a
    turn that calls its model through Vertex needs them, and anything a turn is given the
    agent's shell can use. Today that is Claude Code on a Claude model. A turn on an
    ``openrouter/`` model calls OpenRouter with the caller's key, and Codex calls OpenAI
    with the caller's key, so neither gets a token. The question is where the model call
    goes, not which harness runs: a harness that learns to call its model through Vertex
    belongs here too.
    """
    from ._shared import OPENROUTER_PREFIX

    if (getattr(spec, "model", None) or "").startswith(OPENROUTER_PREFIX):
        return False
    return getattr(spec, "harness", None) == "claude-code"

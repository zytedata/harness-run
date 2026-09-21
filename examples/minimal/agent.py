"""Minimal generic example — a tiny coding agent run locally, in-process.

Defines an :class:`AgentSpec` (nothing project-specific), deploys it with ``local.deploy``, and
runs one task — streaming progress events, then reading the structured result. The same
spec + ``Engine``/``Session`` API runs remotely, in a Google Agent Sandbox, via ``sandbox``
(swap ``local`` → ``sandbox.deploy`` / ``sandbox.get_engine`` — see the top-level README).

Prerequisites and how to run: see this directory's ``README.md``. In short (Python 3.12,
toolkit installed, Claude Code auth available in your environment)::

    python examples/minimal/agent.py
"""

from __future__ import annotations

import asyncio

import pydantic

from agent_run import AgentSpec, SystemPrompt, local


class PrimeReport(pydantic.BaseModel):
    """Structured result the agent is asked to emit as its final message."""

    checked: list[int]
    primes: list[int]


SPEC = AgentSpec(
    harness="claude-code",
    name="hello-coder",
    # Haiku is cheap for a demo; bump to "claude-sonnet-4-6" for real work.
    model="claude-haiku-4-5",
    system_prompt=SystemPrompt.inherit(
        append="You are a concise coding assistant. Verify your work by running code."
    ),
    # permission_mode defaults to "bypassPermissions" — safe here because each run gets an
    # isolated, throwaway cwd and there's no human to answer prompts.
    max_turns=12,
    max_budget_usd=1.0,
    output_schema=PrimeReport,
)

TASK = (
    "Write a Python function is_prime(n) to primes.py, then use it to check the numbers "
    "2 through 10. Report your result as a JSON object with keys 'checked' (the numbers "
    "you checked) and 'primes' (those that are prime)."
)


def _oneline(text: str, limit: int = 100) -> str:
    return " ".join((text or "").split())[:limit]


async def main() -> None:
    engine = local.deploy(SPEC)
    session = engine.start_session()

    run = session.run(TASK)
    # Stream progress as it happens (kind ∈ thinking|tool_use|tool_result|message|status|result).
    async for event in run:
        print(f"  [{event.kind:11}] {_oneline(event.summary)}")

    result = run.result
    print("\n--- result ---")
    print("text:        ", _oneline(result.text or "", 300))
    print("structured:  ", result.structured_output)
    print(f"turns={result.num_turns}  cost=${result.cost_usd:.4f}  error={result.is_error}")


if __name__ == "__main__":
    asyncio.run(main())

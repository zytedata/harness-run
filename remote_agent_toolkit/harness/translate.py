"""Claude SDK message → generic ``AgentEvent`` translation (DESIGN.md §7, §8) — P1.

GENERIC mapping lifted from the PoC ``event_mapping.py``. Kept separate from
``claude_code`` so the event vocabulary is reusable and testable. ``claude_agent_sdk``
types are imported lazily. P0: signature/stub only.
"""

from __future__ import annotations

from typing import Any

from ..events import AgentEvent


def to_event(message: Any) -> AgentEvent | None:
    """Translate one Claude SDK message into an :class:`AgentEvent` (or ``None`` to drop).

    Maps SDK message types onto the generic ``kind`` vocabulary
    (``thinking | tool_use | tool_result | message | status | result``); carries
    ``cost_usd``/``usage`` on the terminal ``result`` event. P1.
    """
    raise NotImplementedError(
        "P1: map a claude_agent_sdk message (assistant/tool/result/etc.) to an "
        "AgentEvent, attaching cost_usd/usage on the terminal result event."
    )

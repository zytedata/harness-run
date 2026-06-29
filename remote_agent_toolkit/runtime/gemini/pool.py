"""Warm-pool worker side (DESIGN.md §6) — P2.

The way to get fast *and* long on Gemini Agent Runtime: pre-warmed async jobs blocked
on an inbound channel.

* A job submitted with a ``__POOL_WAIT__`` sentinel blocks pulling a shared Pub/Sub
  subscription (competing-consumers = atomic claim, via the ``DispatchTransport`` port).
* The worker emits heartbeats while idle (the async executor kills silent jobs) and
  pre-warms during the wait (provision skills, establish the GCS channel) so
  post-assignment setup is ~0.
* On claim, the control plane refills the pool so a warm worker is ready for the next turn.

Lifted & generalized from the PoC ``pool.py`` + worker loop (DESIGN.md §8). The Google
SDK / DispatchTransport adapter are imported lazily. P0: signatures/stubs only.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ...ports.dispatch import DispatchTransport
    from ...spec import AgentSpec

POOL_WAIT_SENTINEL = "__POOL_WAIT__"


async def worker_loop(spec: AgentSpec, dispatch: DispatchTransport, **kw: Any) -> None:
    """Run the warm-pool worker: pre-warm, heartbeat, block on claim, process turn (P2)."""
    raise NotImplementedError(
        "P2: implement the warm-pool worker loop — pre-warm during the wait, emit "
        "heartbeats, atomically claim a dispatched message, and process it as a turn."
    )

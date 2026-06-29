"""Conformance suites for swappable ports (DESIGN.md §2, §7) — stdlib-only.

A team implementing its own ``SessionStore`` (or other port) runs the matching
conformance suite against its backend to prove it satisfies the contract. P1 fills in
the body; the signature is real now.
"""

from __future__ import annotations

from typing import Any, Callable


async def run_session_store_conformance(
    store_factory: Callable[[], Any],
) -> None:
    """Validate a Claude SDK ``SessionStore`` implementation (DESIGN.md §6, §7).

    ``store_factory`` is a zero-arg callable returning a fresh store instance. The suite
    asserts the contract the ``GcsSessionStore`` relies on:

    * **Round-trip** — ``append`` then ``load`` returns the appended messages, in order,
      byte-for-byte.
    * **Ordering** — concurrent/interleaved appends preserve per-session append order.
    * **Keying** — sessions are isolated by ``session_id`` alone (``project_key`` is
      ignored), so any worker/cwd can resume (DESIGN.md §6).
    * **Cascade delete** — deleting a session removes all its subkeys; ``list_subkeys``
      then returns empty and ``load`` yields nothing.
    * **Missing session** — ``load`` of an unknown ``session_id`` returns empty, not an
      error.

    Raises ``AssertionError`` on the first violated invariant; returns ``None`` on success.
    """
    raise NotImplementedError(
        "P1: implement the SessionStore conformance checks (round-trip, ordering, "
        "session_id-only keying, cascade delete, missing-session) against the store "
        "produced by store_factory."
    )

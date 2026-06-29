"""Conformance suites for swappable ports (DESIGN.md §2, §7) — stdlib-only.

A team implementing its own ``SessionStore`` (or other port) runs the matching conformance
suite against its backend to prove it satisfies the contract. Self-contained: no pytest
dependency — it's a plain async callable that raises ``AssertionError`` on the first
violated invariant.
"""

from __future__ import annotations

from typing import Any, Callable


async def run_session_store_conformance(
    store_factory: Callable[[], Any],
) -> None:
    """Validate a Claude SDK ``SessionStore`` implementation (DESIGN.md §6, §7).

    ``store_factory`` is a zero-arg callable returning a fresh store instance. The suite
    asserts the contract the ``BlobSessionStore`` relies on:

    * **Round-trip** — ``append`` then ``load`` returns the appended entries, in order.
    * **Ordering** — multiple appends preserve order; lexical key order == chronological.
    * **Keying** — sessions are isolated by ``session_id`` alone; the store API takes no
      ``project_key``, so two keys with the same ``session_id`` but different cwd/project
      context share data (the whole point — any worker/cwd can resume, DESIGN.md §6).
    * **uuid dedup** — appending the same entry (same ``uuid``) twice → ``load`` returns it
      once.
    * **Missing session** — ``load`` of an unknown ``session_id`` returns ``None``, not an
      error.
    * **Subkeys** — appending under a ``subpath`` makes ``list_subkeys`` report it; a
      main-only session reports ``[]``.

    Raises ``AssertionError`` on the first violated invariant; returns ``None`` on success.
    """
    # -- Round-trip + ordering ----------------------------------------------
    store = store_factory()
    sid = "session-roundtrip"
    key = {"session_id": sid}
    e1 = {"uuid": "u1", "n": 1}
    e2 = {"uuid": "u2", "n": 2}
    e3 = {"uuid": "u3", "n": 3}
    await store.append(key, [e1, e2])
    await store.append(key, [e3])
    loaded = await store.load(key)
    assert loaded == [e1, e2, e3], f"round-trip/ordering failed: {loaded!r}"

    # -- Keying by session_id alone (no project_key in the API) -------------
    # A second store instance with no shared cwd/project context, same session_id,
    # observes the same data. The key dict carries only session_id (+ optional subpath).
    other = store_factory()
    assert await other.load({"session_id": sid}) == [e1, e2, e3], (
        "session_id-only keying failed: a fresh store with the same session_id did not "
        "see the data (the store must not key on cwd/project)"
    )

    # -- uuid dedup ----------------------------------------------------------
    store = store_factory()
    key = {"session_id": "session-dedup"}
    await store.append(key, [e1])
    await store.append(key, [e1])  # same uuid again
    loaded = await store.load(key)
    assert loaded == [e1], f"uuid dedup failed: {loaded!r}"

    # -- Missing session -----------------------------------------------------
    store = store_factory()
    assert await store.load({"session_id": "nonexistent"}) is None, (
        "missing session must load as None, not error/empty-list"
    )

    # -- Subkeys -------------------------------------------------------------
    store = store_factory()
    sid = "session-subkeys"
    await store.append({"session_id": sid}, [e1])  # main only so far
    assert await store.list_subkeys({"session_id": sid}) == [], (
        "main-only session must report no subkeys"
    )
    await store.append({"session_id": sid, "subpath": "subagents/agent-X"}, [e2])
    subs = await store.list_subkeys({"session_id": sid})
    assert subs == ["subagents/agent-X"], f"subkeys failed: {subs!r}"

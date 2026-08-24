"""Conformance suites for swappable ports (DESIGN.md §2, §7) — stdlib-only.

A team implementing its own ``SessionStore`` (or other port) runs the matching conformance
suite against its backend to prove it satisfies the contract. Self-contained: no pytest
dependency — it's a plain async callable that raises ``AssertionError`` on the first
violated invariant.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Callable


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


def _check_result_contract(events: list[Any]) -> Any:
    """Assert the terminal-result contract shared by every ``Harness`` backend; return the init event."""
    assert events, "the turn yielded no events"

    inits = [
        e for e in events if e.kind == "status" and (e.raw or {}).get("subtype") == "init"
    ]
    assert inits, "no status event with raw['subtype'] == 'init'"

    tail = list(reversed(events))
    last_result = next((i for i, e in enumerate(tail) if e.kind == "result"), None)
    assert last_result is not None, "the turn yielded no result event"
    assert all(e.kind == "status" for e in tail[:last_result]), (
        "only status events may follow the terminal result: "
        f"{[e.kind for e in reversed(tail[:last_result])]}"
    )
    final = tail[last_result]
    assert final.cost_usd is None or isinstance(final.cost_usd, float), (
        f"result cost_usd must be a float or None: {final.cost_usd!r}"
    )
    assert isinstance(final.usage, dict), f"result usage must be a dict: {final.usage!r}"
    raw = final.raw or {}
    assert isinstance(raw.get("num_turns"), int), f"result raw['num_turns'] must be an int: {raw!r}"
    assert isinstance(raw.get("is_error"), bool), f"result raw['is_error'] must be a bool: {raw!r}"
    return inits[0]


async def run_harness_conformance(
    run_turn: Callable[[], AsyncIterator[Any]],
) -> None:
    """Validate a :class:`~remote_agent_toolkit.harness.base.Harness` implementation (DESIGN.md §7).

    *run_turn* is a zero-arg callable returning the event iterator of ONE completed turn —
    typically ``lambda: harness.run(spec, ctx)``, against a live backend or a scripted fake.
    Backend-agnostic: holds for Claude Code, Codex, and any future ``Harness``. The suite
    asserts what embedders read beyond ``AgentEvent``'s own fields, since ``raw`` is
    otherwise a pass-through with no promise attached:

    * **Init event** — a ``status`` event carries ``raw["subtype"] == "init"``.
    * **Terminal result** — the turn ends at a ``result`` event (only ``status`` events, such
      as a checkpoint notice, may follow it) whose spend is on ``cost_usd`` (a float, or
      ``None`` when the backend cannot price the run), whose token accounting is a ``usage``
      dict, and whose ``raw`` carries ``num_turns`` (an int) and ``is_error`` (a bool).

    :func:`run_claude_code_harness_conformance` layers Claude Code's stricter init-payload
    contract on top of this one.

    Raises ``AssertionError`` on the first violated invariant; returns ``None`` on success.
    """
    events = [event async for event in run_turn()]
    _check_result_contract(events)


async def run_claude_code_harness_conformance(
    run_turn: Callable[[], AsyncIterator[Any]],
) -> None:
    """Validate :class:`~remote_agent_toolkit.harness.claude_code.ClaudeCodeHarness` (DESIGN.md §7).

    Everything :func:`run_harness_conformance` checks, plus the init payload only Claude
    Code's SDK makes: the ``init`` status event's ``raw["data"]`` is the backend's own
    payload, verbatim, including the session id at ``raw["data"]["session_id"]`` — that is
    where a caller reads the id the backend actually used (and the resolved
    ``mcp_servers``). Codex's init event carries no such payload, so this check does not
    hold for :class:`~remote_agent_toolkit.harness.codex.CodexHarness`.

    Raises ``AssertionError`` on the first violated invariant; returns ``None`` on success.
    """
    events = [event async for event in run_turn()]
    init = _check_result_contract(events)
    data = (init.raw or {}).get("data")
    assert isinstance(data, dict), f"init raw['data'] must be the backend payload dict: {data!r}"
    sid = data.get("session_id")
    assert isinstance(sid, str) and sid, f"init raw['data']['session_id'] missing: {data!r}"

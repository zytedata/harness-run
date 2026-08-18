"""Engine / Session / Run protocols + the session state machine (DESIGN.md §4).

Stdlib-only: these are ``typing.Protocol`` contracts that ``local`` and ``gemini``
both implement identically, so app code is backend-agnostic.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Protocol, TYPE_CHECKING, runtime_checkable

from ..events import AgentEvent, RunResult, RunStatus, StopReason

if TYPE_CHECKING:
    from ..config import SessionConfig, TurnConfig


@runtime_checkable
class Run(Protocol):
    """Handle returned by :meth:`Session.run` / :meth:`Session.send`.

    Consumable three ways (DESIGN.md §5):

    * **await** it       → wait for completion, returns :class:`RunResult`.
    * **async-iterate**  → stream :class:`AgentEvent`s as they happen.
    * **poll**           → check ``done`` / ``status``, read ``result`` when done.

    For ``gemini``: "stream" = tail the EventSink; "await" = wait for the terminal
    result event; "poll" = read the latest logged status.
    """

    def __await__(self):  # -> Generator[Any, None, RunResult]
        """Await the run to completion, yielding a :class:`RunResult`."""
        ...

    def __aiter__(self) -> AsyncIterator[AgentEvent]:
        """Async-iterate over :class:`AgentEvent`s for the run."""
        ...

    @property
    def done(self) -> bool:
        """Whether the run has reached a terminal state."""
        ...

    @property
    def status(self) -> RunStatus:
        """Current run status."""
        ...

    @property
    def result(self) -> RunResult | None:
        """The terminal result once ``done``, else ``None``."""
        ...


@runtime_checkable
class Session(Protocol):
    """A run-plane session with the CMA-style lifecycle.

    ``pending → running ↔ idle(stop_reason) → terminated`` (DESIGN.md §4). A session
    is addressable by ``session_id``, so another process can re-attach via
    :meth:`Engine.get_session` and poll / continue it.
    """

    def run(
        self,
        message: str,
        *,
        secrets: dict[str, str] | None = None,
        config: TurnConfig | None = None,
        hooks: Any | None = None,
    ) -> Run:
        """Start a run from ``message`` (kicks off a fresh turn).

        ``secrets`` is a per-invocation name → value map (the agent's own keys, any repo
        ``auth`` / GitHub MCP token). Values are never baked into the spec or logged.
        ``config`` is this turn's :class:`~remote_agent_toolkit.config.TurnConfig` — a
        sparse overlay of the invocation knobs on the session's effective spec. *hooks*
        are Claude Agent SDK hook callbacks (``{HookEvent: [HookMatcher, ...]}``) — live
        callables in the caller's process, so ``local`` only; a backend that runs the turn
        elsewhere (``gemini``) rejects them.
        """
        ...

    def send(
        self,
        message: str,
        *,
        secrets: dict[str, str] | None = None,
        config: TurnConfig | None = None,
        hooks: Any | None = None,
    ) -> Run:
        """Resume an idle session with ``message`` (e.g. answer a ``needs_input`` pause).

        Resumes via checkpoint on a warm worker (DESIGN.md §3.7). Pass ``secrets`` again — they
        are not persisted across turns, so repo push auth is re-embedded on resume. ``config``
        is a per-turn :class:`~remote_agent_toolkit.config.TurnConfig`; the SESSION config
        cannot change here (bound at :meth:`Engine.start_session`, world snapshot-restored).
        *hooks* are per-turn like ``secrets`` (see :meth:`run`).
        """
        ...

    async def interrupt(self) -> None:
        """Interrupt the in-flight run, transitioning the session toward idle."""
        ...

    @property
    def status(self) -> RunStatus:
        """Current session status."""
        ...

    @property
    def stop_reason(self) -> StopReason | None:
        """Why the session is idle, or ``None`` while running/pending."""
        ...

    @property
    def last_result(self) -> RunResult | None:
        """The most recent run's result, or ``None`` if no run completed yet."""
        ...

    @property
    def session_id(self) -> str:
        """Stable id for re-attach / resume."""
        ...

    @property
    def workspace(self):  # -> pathlib.Path
        """The agent's working directory — always a leaf named ``workspace``.

        Cross-backend contract: the agent runs in ``.../<session>/workspace`` (never a
        bare ``jobs/<uuid>`` dir, whose anonymous-temp look invites weaker models to
        ``cd`` away), with session bookkeeping kept outside the visible cwd. On ``local``
        this is a host :class:`~pathlib.Path` — seed input files into it before
        :meth:`run`, collect artifacts from it after. Backends whose filesystem is
        remote (``gemini``) raise ``NotImplementedError``.
        """
        ...

    def history(self) -> list[AgentEvent]:
        """All persisted events of this session, oldest first.

        Works for re-attached sessions long after the run (``gemini``: mirrored GCS events →
        platform job output → Cloud Logging). Empty when nothing was persisted.
        """
        ...

    def fork(self) -> Session:
        """Fork this session into an independent branch sharing prior history."""
        ...


@runtime_checkable
class Engine(Protocol):
    """A deployed (or local) agent handle (DESIGN.md §4).

    ``gemini.deploy(spec)`` / ``local.deploy(spec)`` return an ``Engine``;
    ``gemini.get_engine(name)`` looks one up without deploying.
    """

    def start_session(self, config: SessionConfig | None = None) -> Session:
        """Begin a new session against this engine.

        ``config`` binds the session's :class:`~remote_agent_toolkit.config.SessionConfig`
        (a sparse overlay over the deployed spec: repos, skills, prompt, model, ...) for
        the session's whole life — its world is created on the first turn and
        snapshot-restored after, so it cannot change mid-conversation.
        """
        ...

    def get_session(self, session_id: str) -> Session:
        """Re-attach to an existing session by id (poll / continue).

        Deliberately takes NO config: the session runs under the config bound at
        :meth:`start_session` (persisted with the session's records and read back here),
        so a re-attaching process can never substitute a different world mid-conversation.
        """
        ...

    def list_sessions(self) -> list[dict]:
        """Enumerate known past sessions, newest first.

        Each entry carries at least ``session_id``; feed it to :meth:`get_session` and read
        :meth:`Session.history` / ``last_result``.
        """
        ...

    def versions(self) -> list[str]:
        """List the deployed versions of this engine, newest first.

        A version is whatever the backend versions a deployment by: on ``gemini`` an Agent
        Runtime **runtime revision** (each ``deploy`` of an existing name mints one); on
        ``local`` the single pseudo-version ``"local"``.
        """
        ...

    @property
    def name(self) -> str:
        """The agent name (maps to the engine display name)."""
        ...

    @property
    def version(self) -> str:
        """The resolved engine version — the one this handle's runs execute on."""
        ...

    @property
    def resource(self) -> str:
        """The underlying resource name (e.g. the Agent Engine resource path)."""
        ...

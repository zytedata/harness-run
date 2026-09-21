"""Session/turn configuration overlays over a deployed :class:`~agent_run.spec.AgentSpec`.

Configuration has three scopes (README "Deploy / session / turn"):

* **Deploy** — what is physically in the engine (``AgentSpec``: identity, ``packages``,
  engine ``env``, baked harness CLIs) plus the *defaults* for everything below.
* **Session** — the world a conversation lives in. A session's first turn clones
  ``repos`` and provisions ``skills`` into a fresh workspace; every later turn restores a
  snapshot of the previous turn's workspace and never re-reads those fields, so they are
  settable **once, at session start**: :class:`SessionConfig`, passed to
  ``engine.start_session(config=...)``.
* **Turn** — knobs the harness re-reads on every invocation (model, budgets, tool
  policy, output schema): :class:`TurnConfig`, passed to ``session.run(config=...)`` /
  ``send(config=...)``.

Both config types are **sparse overlays**: a field left at :data:`INHERIT` keeps the
value from the layer below (turn ← session ← deployed spec); a set field replaces it
wholesale (no deep merges — the one exception is ``SessionConfig.extra_env``, which by
definition *adds* to ``spec.env``). Deploy-only facts (``name``, ``packages``, engine
``env``) have no field here, so a per-run change of them is unrepresentable rather than
silently ignored.

Like the spec, configs are *data*: no secret values, ever. A ``RepoSource`` with
credentials embedded in its URL is rejected at construction (name the token via
``auth=``/``auth_user=`` and pass its value in ``run(secrets=...)``): staged configs are
referenced from persisted invocation payloads, so they must be credential-free.

Stdlib-only, like ``spec.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping, Sequence

from .spec import (
    AgentSpec,
    McpServer,
    RepoSource,
    SkillSource,
    SystemPrompt,
    _assert_non_secret_env,
    _output_schema_to_dict,
)


class _Inherit:
    """Sentinel type: "this field inherits the value from the layer below".

    A dedicated sentinel (not ``None``) because ``None`` is itself a meaningful setting
    for several fields — ``system_prompt=None`` means "use the harness's default prompt",
    not "inherit the deployed prompt". Singleton; test with ``is INHERIT``.
    """

    _instance: _Inherit | None = None

    def __new__(cls) -> _Inherit:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "INHERIT"


INHERIT: Any = _Inherit()


def _assert_repos_credential_free(repos: Sequence[RepoSource]) -> None:
    """Reject a ``RepoSource`` whose URL embeds ``user:token@`` credentials.

    The worker gets configs in the turn body, and the client also writes them to GCS as
    the 30-day post-mortem record — so they must never carry secret values.
    """
    from urllib.parse import urlsplit

    for repo in repos:
        parts = urlsplit(repo.url)
        if parts.username or parts.password:
            raise ValueError(
                f"RepoSource.url for {parts.hostname or repo.url!r} embeds credentials; "
                "configs are staged per invocation and kept for debugging, so they must "
                "be credential-free. Name the token via RepoSource(auth=..., "
                "auth_user=...) and pass its value in run(secrets={...}) instead."
            )


def _tuple_or_inherit(value: Any) -> Any:
    """Coerce a set sequence field to a tuple (hashable/frozen), passing INHERIT/None through."""
    if value is INHERIT or value is None:
        return value
    return tuple(value)


# Turn-scoped fields: what the harness re-reads on every invocation. TurnConfig is exactly
# these; SessionConfig has them too (a session-wide default for its turns).
_KNOB_FIELDS = (
    "model",
    "reasoning_effort",
    "max_turns",
    "max_budget_usd",
    "max_buffer_size",
    "background_task_timeout",
    "permission_mode",
    "allowed_tools",
    "disallowed_tools",
    "output_schema",
    "openrouter_provider",
    "openrouter_routing",
)

# Session-scoped fields: the conversation's world (created on turn 1, snapshot-restored
# after) plus session-lifecycle switches. ``extra_env`` is handled separately (additive).
_WORLD_FIELDS = (
    "repos",
    "skills",
    "mcp_servers",
    "system_prompt",
    "harness",
    "checkpoint",
    "interactive",
)

_SEQUENCE_FIELDS = ("repos", "skills", "mcp_servers", "allowed_tools", "disallowed_tools")


def _encode_field(name: str, value: Any) -> Any:
    """Serialize one config field value with the same encoding ``AgentSpec.to_dict`` uses."""
    if name == "system_prompt" and isinstance(value, SystemPrompt):
        return value.to_dict()
    if name in ("repos", "skills", "mcp_servers") and value is not None:
        return [item.to_dict() for item in value]
    if name in ("allowed_tools", "disallowed_tools") and value is not None:
        return list(value)
    if name == "output_schema" and value is not None:
        return _output_schema_to_dict(value)
    if name == "extra_env" and value is not None:
        _assert_non_secret_env(value)
        return dict(value)
    return value


def _decode_field(name: str, value: Any) -> Any:
    """Inverse of :func:`_encode_field` (mirrors ``AgentSpec.from_dict``)."""
    if name == "system_prompt" and isinstance(value, Mapping):
        return SystemPrompt.from_dict(value)
    if name == "repos" and value is not None:
        return tuple(RepoSource.from_dict(r) for r in value)
    if name == "skills" and value is not None:
        return tuple(SkillSource.from_dict(s) for s in value)
    if name == "mcp_servers" and value is not None:
        return tuple(McpServer.from_dict(m) for m in value)
    if name in ("allowed_tools", "disallowed_tools") and value is not None:
        return tuple(value)
    return value


class _ConfigBase:
    """Shared sparse-overlay behavior: set-field iteration + to/from dict."""

    def set_fields(self) -> dict[str, Any]:
        """The fields explicitly set on this config (everything not INHERIT)."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)  # type: ignore[arg-type]
            if getattr(self, f.name) is not INHERIT
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the SET fields only (sparse). An explicitly-set ``None`` survives as null."""
        return {name: _encode_field(name, value) for name, value in self.set_fields().items()}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]):
        """Reconstruct from :meth:`to_dict` output: absent key = INHERIT, null = explicit None."""
        known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
        unknown = set(d) - known
        if unknown:
            # Fail closed: an unknown field means a newer client staged something this
            # revision cannot honor — silently dropping it would run the wrong config
            # (the client/engine same-revision rule, README "Deploy / session / turn").
            raise ValueError(
                f"{cls.__name__} does not understand field(s) {sorted(unknown)}; "
                "client and engine must run the same toolkit revision"
            )
        return cls(**{name: _decode_field(name, value) for name, value in d.items()})


@dataclass(frozen=True)
class SessionConfig(_ConfigBase):
    """Sparse overlay over the deployed spec, bound **once** at session start.

    Pass to ``engine.start_session(config=...)``. Every field left at :data:`INHERIT`
    keeps the deployed spec's value; a set field replaces it wholesale for this session
    (``extra_env`` is the one additive exception). ``send()`` takes no session config —
    later turns restore the first turn's workspace snapshot, so the session's world
    cannot change mid-conversation (and re-attaching by id re-reads the config the opener
    persisted; it cannot be substituted).

    Attributes:
        repos: The repos cloned into the session's workspace on its first turn.
        skills: Skill sources for the session. REPLACES the deployed declaration (no
            merging); on gemini, the deploy-baked skills stay a staging fast path only
            while this declaration matches them.
        mcp_servers: MCP servers attached to the agent for this session.
        system_prompt: The conversation's framing, re-applied to the harness on every
            turn. Same semantics as the spec: a plain ``str`` replaces the harness's
            built-in prompt, :class:`SystemPrompt` inherits and appends, ``None`` keeps
            the harness default.
        harness: Selects among the harness CLIs baked into the engine image
            (``AgentSpec.harnesses``); asking for one the image lacks fails the first
            turn loudly.
        checkpoint / interactive: Session-lifecycle switches (checkpointing needs the
            engine deployed with an output bucket).
        extra_env: Extra non-secret env vars for the agent subprocess, ADDED on top of
            the deployed ``spec.env`` (deploy-only). Non-secret only, like ``spec.env``.
        model / reasoning_effort / max_turns / max_budget_usd / max_buffer_size /
            background_task_timeout / permission_mode / allowed_tools / disallowed_tools /
            output_schema / openrouter_provider / openrouter_routing:
            session-wide defaults for the per-turn knobs (see :class:`TurnConfig`).
            ``output_schema`` also drives client-side structured parsing.
    """

    # -- session world (created on turn 1; later turns restore the snapshot) -----
    repos: Sequence[RepoSource] | Any = INHERIT
    skills: Sequence[SkillSource] | Any = INHERIT
    mcp_servers: Sequence[McpServer] | Any = INHERIT
    system_prompt: str | SystemPrompt | None | Any = INHERIT
    harness: str | Any = INHERIT
    checkpoint: bool | Any = INHERIT
    interactive: bool | None | Any = INHERIT
    extra_env: Mapping[str, str] | Any = INHERIT

    # -- invocation knobs (session-wide defaults; TurnConfig overrides per turn) --
    model: str | Any = INHERIT
    reasoning_effort: str | None | Any = INHERIT
    max_turns: int | Any = INHERIT
    max_budget_usd: float | Any = INHERIT
    max_buffer_size: int | Any = INHERIT
    background_task_timeout: float | Any = INHERIT
    permission_mode: str | Any = INHERIT
    allowed_tools: Sequence[str] | None | Any = INHERIT
    disallowed_tools: Sequence[str] | None | Any = INHERIT
    output_schema: Any = INHERIT
    openrouter_provider: str | None | Any = INHERIT
    openrouter_routing: Mapping[str, Any] | None | Any = INHERIT

    def __post_init__(self) -> None:
        for name in _SEQUENCE_FIELDS:
            object.__setattr__(self, name, _tuple_or_inherit(getattr(self, name)))
        if self.repos is not INHERIT and self.repos is not None:
            _assert_repos_credential_free(self.repos)
        if self.extra_env is not INHERIT and self.extra_env is not None:
            _assert_non_secret_env(self.extra_env)
            object.__setattr__(self, "extra_env", dict(self.extra_env))


@dataclass(frozen=True)
class TurnConfig(_ConfigBase):
    """Sparse overlay for ONE turn: the knobs the harness re-reads on every invocation.

    Pass to ``session.run(config=...)`` / ``send(config=...)``. Layered on top of the
    session (which is layered on the deployed spec), for that turn only — e.g. escalate
    the model for a hard follow-up turn, or require structured output only on the final
    turn. Deliberately has NO workspace-shaping fields (``repos``, ``skills``, ...):
    those are created on the session's first turn and snapshot-restored after, so a
    per-turn change could not be honored (see :class:`SessionConfig`).
    """

    model: str | Any = INHERIT
    reasoning_effort: str | None | Any = INHERIT
    max_turns: int | Any = INHERIT
    max_budget_usd: float | Any = INHERIT
    max_buffer_size: int | Any = INHERIT
    background_task_timeout: float | Any = INHERIT
    permission_mode: str | Any = INHERIT
    allowed_tools: Sequence[str] | None | Any = INHERIT
    disallowed_tools: Sequence[str] | None | Any = INHERIT
    output_schema: Any = INHERIT
    openrouter_provider: str | None | Any = INHERIT
    openrouter_routing: Mapping[str, Any] | None | Any = INHERIT

    def __post_init__(self) -> None:
        for name in ("allowed_tools", "disallowed_tools"):
            object.__setattr__(self, name, _tuple_or_inherit(getattr(self, name)))


def apply_session_config(spec: AgentSpec, config: SessionConfig | None) -> AgentSpec:
    """The effective spec of a session: ``config``'s set fields over ``spec``.

    Pure field-level replacement, except ``extra_env`` which merges ON TOP of
    ``spec.env``. Deploy-only fields (``name``, ``packages``, ``env`` as a whole,
    ``harnesses``) pass through untouched — ``SessionConfig`` has no way to express them.
    """
    import dataclasses

    if config is None:
        return spec
    changes = config.set_fields()
    extra_env = changes.pop("extra_env", None)
    if extra_env:
        changes["env"] = {**(spec.env or {}), **extra_env}
    return dataclasses.replace(spec, **changes) if changes else spec


def apply_turn_config(spec: AgentSpec, config: TurnConfig | None) -> AgentSpec:
    """The effective spec of one turn: ``config``'s set fields over the session's spec."""
    import dataclasses

    if config is None:
        return spec
    changes = config.set_fields()
    return dataclasses.replace(spec, **changes) if changes else spec


def validate_harness_choice(baked: AgentSpec, effective: AgentSpec) -> None:
    """Fail loudly when a config selects a harness whose CLI is not in the engine image.

    Harness *availability* is a deploy-time fact (the CLI binaries are baked into the
    image per ``AgentSpec.harnesses``); a session config only selects among what is
    baked. Raises ``ValueError`` — the worker turns it into a terminal error result.
    """
    if effective.harness not in baked.baked_harnesses:
        raise ValueError(
            f"harness {effective.harness!r} is not baked into this engine "
            f"(available: {list(baked.baked_harnesses)}); deploy with "
            f"AgentSpec(harnesses=(..., {effective.harness!r})) to offer it"
        )

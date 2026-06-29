"""The declarative, serializable agent definition (DESIGN.md §5).

An :class:`AgentSpec` is *data*: it can live in code or YAML, be version-controlled,
diffed in CI, and deployed reproducibly. It carries *references* (secret names, skill
sources) never values — secrets and absolute paths resolve at runtime, never pickled
into a deployed engine (DESIGN.md §3.5).

This module is stdlib-only. ``yaml`` is imported lazily inside ``to_yaml``/``from_yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


@dataclass(frozen=True)
class SystemPrompt:
    """Inherit the harness's built-in system prompt and (optionally) append to it.

    Using a plain ``str`` as ``AgentSpec.system_prompt`` means "replace entirely".
    Using a ``SystemPrompt`` means "inherit the harness's built-in prompt, then append".

    Attributes:
        append: Text appended after the inherited prompt, or ``None`` for no addition.
        exclude_dynamic_sections: Drop harness-injected dynamic sections (e.g. env/date
            context) from the inherited prompt.
    """

    append: str | None = None
    exclude_dynamic_sections: bool = False

    @classmethod
    def inherit(
        cls,
        append: str | None = None,
        exclude_dynamic_sections: bool = False,
    ) -> SystemPrompt:
        """Inherit the harness's built-in system prompt and append ``append``."""
        return cls(append=append, exclude_dynamic_sections=exclude_dynamic_sections)

    def to_dict(self) -> dict[str, Any]:
        return {
            "append": self.append,
            "exclude_dynamic_sections": self.exclude_dynamic_sections,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> SystemPrompt:
        return cls(
            append=d.get("append"),
            exclude_dynamic_sections=bool(d.get("exclude_dynamic_sections", False)),
        )


@dataclass(frozen=True)
class SkillSource:
    """A source of agent skills, resolved & staged at deploy/run time.

    A tagged union over ``kind``:

    * ``"git"``   — clone ``url`` (optionally at ``ref``), read skills from ``subdir``.
    * ``"path"``  — a local directory ``path``.
    * ``"builtin"`` — a named, library-provided skill bundle ``name``.

    A list of sources allows a base + extra sources (multi-source skills, DESIGN.md §12).
    """

    kind: Literal["git", "path", "builtin"]
    url: str | None = None
    ref: str | None = None
    path: str | None = None
    name: str | None = None
    subdir: str = "skills"

    # NOTE: the ``path`` builder (DESIGN names it ``SkillSource.path``) collides with
    # the ``path`` field, so it is defined as ``_path`` and bound as ``SkillSource.path``
    # after ``@dataclass`` runs (see below) — defining it inline would clobber the
    # field's default with the classmethod object.

    @classmethod
    def git(cls, url: str, ref: str | None = None, subdir: str = "skills") -> SkillSource:
        """Skills from a git repository ``url`` (optionally pinned to ``ref``)."""
        return cls(kind="git", url=url, ref=ref, subdir=subdir)

    @classmethod
    def _path(cls, dir: str) -> SkillSource:  # noqa: A002 - matches DESIGN signature
        """Skills from a local directory ``dir``."""
        return cls(kind="path", path=dir)

    @classmethod
    def builtin(cls, name: str) -> SkillSource:
        """A named, library-provided builtin skill bundle."""
        return cls(kind="builtin", name=name)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"kind": self.kind, "subdir": self.subdir}
        for f in ("url", "ref", "path", "name"):
            v = getattr(self, f)
            if v is not None:
                d[f] = v
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> SkillSource:
        return cls(
            kind=d["kind"],
            url=d.get("url"),
            ref=d.get("ref"),
            path=d.get("path"),
            name=d.get("name"),
            subdir=d.get("subdir", "skills"),
        )


# Bind the ``path`` builder under its DESIGN name without shadowing the ``path`` field
# at class-definition time (which would clobber the field default).
SkillSource.path = SkillSource._path  # type: ignore[assignment]


@dataclass(frozen=True)
class McpServer:
    """An MCP server attached to the agent.

    A tagged union over ``kind``:

    * ``"github"`` — the GitHub MCP server.
    * ``"url"``    — a remote MCP server at ``url`` with optional static ``headers``.
    * ``"stdio"``  — a local subprocess MCP server (``command`` + ``args``).

    Credentials are NOT carried here. Reference secret *names* via
    ``AgentSpec.secrets``; they are resolved from Secret Manager at runtime and
    injected into the server's environment/headers by the harness.
    """

    kind: Literal["github", "url", "stdio"]
    name: str | None = None
    url: str | None = None
    headers: Mapping[str, str] | None = None
    command: str | None = None
    args: tuple[str, ...] | None = None

    @classmethod
    def github(cls) -> McpServer:
        """The GitHub MCP server. Auth token comes from ``AgentSpec.secrets``."""
        return cls(kind="github", name="github")

    # NOTE: the ``url`` builder (DESIGN names it ``McpServer.url``) collides with the
    # ``url`` field; defined as ``_url`` and bound as ``McpServer.url`` after
    # ``@dataclass`` (see below).

    @classmethod
    def _url(
        cls,
        name: str,
        url: str,  # noqa: A002 - matches DESIGN signature
        headers: Mapping[str, str] | None = None,
    ) -> McpServer:
        """A remote MCP server. Secrets are referenced via ``AgentSpec.secrets``."""
        return cls(
            kind="url",
            name=name,
            url=url,
            headers=dict(headers) if headers is not None else None,
        )

    @classmethod
    def stdio(cls, name: str, command: str, args: list[str] | None = None) -> McpServer:
        """A local subprocess MCP server. Secrets are referenced via ``AgentSpec.secrets``."""
        return cls(
            kind="stdio",
            name=name,
            command=command,
            args=tuple(args) if args is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"kind": self.kind}
        for f in ("name", "url", "command"):
            v = getattr(self, f)
            if v is not None:
                d[f] = v
        if self.headers is not None:
            d["headers"] = dict(self.headers)
        if self.args is not None:
            d["args"] = list(self.args)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> McpServer:
        args = d.get("args")
        headers = d.get("headers")
        return cls(
            kind=d["kind"],
            name=d.get("name"),
            url=d.get("url"),
            headers=dict(headers) if headers is not None else None,
            command=d.get("command"),
            args=tuple(args) if args is not None else None,
        )


# Bind the ``url`` builder under its DESIGN name (see the SkillSource.path note above).
McpServer.url = McpServer._url  # type: ignore[assignment]


@dataclass(frozen=True)
class AgentSpec:
    """The declarative agent definition.

    Frozen and hashable: collection fields default to tuples. The constructor coerces
    lists to tuples (via ``__post_init__``) so callers may pass lists for convenience,
    as the DESIGN.md §5 example does.

    Attributes:
        name: Stable agent name. Maps to a deployed engine's display name (§5).
        model: Model id, e.g. ``"claude-sonnet-4-6"``.
        system_prompt: A ``SystemPrompt`` (inherit + append) or a plain ``str``
            (replace entirely) or ``None`` (harness default).
        skills: Skill sources, resolved & staged at deploy/run time.
        mcp_servers: MCP servers attached to the agent.
        allowed_tools / disallowed_tools: Tool allow/deny lists, or ``None`` for default.
        secrets: Secret *names* resolved from Secret Manager at runtime (never pickled).
        permission_mode: Claude Code permission mode (e.g. ``"bypassPermissions"``).
        max_turns: Hard cap on agent turns.
        max_budget_usd: Hard cap on spend.
        checkpoint: Enable checkpoint/resume (interactive pauses).
        output_schema: Optional structured-output schema (pydantic model or JSON schema).
        env: Extra environment variables for the runtime.
    """

    name: str
    model: str
    system_prompt: str | SystemPrompt | None = None
    skills: tuple[SkillSource, ...] = ()
    mcp_servers: tuple[McpServer, ...] = ()
    allowed_tools: tuple[str, ...] | None = None
    disallowed_tools: tuple[str, ...] | None = None
    secrets: tuple[str, ...] = ()
    permission_mode: str = "default"
    max_turns: int = 120
    max_budget_usd: float = 10.0
    checkpoint: bool = False
    output_schema: Any = None
    env: Mapping[str, str] | None = field(default=None)

    def __post_init__(self) -> None:
        # Coerce list args to frozen-hashable tuples without breaking frozen-ness.
        object.__setattr__(self, "skills", tuple(self.skills))
        object.__setattr__(self, "mcp_servers", tuple(self.mcp_servers))
        object.__setattr__(self, "secrets", tuple(self.secrets))
        if self.allowed_tools is not None:
            object.__setattr__(self, "allowed_tools", tuple(self.allowed_tools))
        if self.disallowed_tools is not None:
            object.__setattr__(self, "disallowed_tools", tuple(self.disallowed_tools))

    # -- serialization ---------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain, JSON/YAML-friendly dict."""
        d: dict[str, Any] = {
            "name": self.name,
            "model": self.model,
            "skills": [s.to_dict() for s in self.skills],
            "mcp_servers": [m.to_dict() for m in self.mcp_servers],
            "secrets": list(self.secrets),
            "permission_mode": self.permission_mode,
            "max_turns": self.max_turns,
            "max_budget_usd": self.max_budget_usd,
            "checkpoint": self.checkpoint,
        }
        if isinstance(self.system_prompt, SystemPrompt):
            d["system_prompt"] = self.system_prompt.to_dict()
        elif self.system_prompt is not None:
            d["system_prompt"] = self.system_prompt
        if self.allowed_tools is not None:
            d["allowed_tools"] = list(self.allowed_tools)
        if self.disallowed_tools is not None:
            d["disallowed_tools"] = list(self.disallowed_tools)
        if self.env is not None:
            d["env"] = dict(self.env)
        if self.output_schema is not None:
            d["output_schema"] = _output_schema_to_dict(self.output_schema)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> AgentSpec:
        """Reconstruct from :meth:`to_dict` output.

        ``output_schema`` round-trips as the JSON schema dict it was serialized to;
        a pydantic model passed in is not reconstructed as a class (see ``# TODO``).
        """
        sp_raw = d.get("system_prompt")
        if isinstance(sp_raw, Mapping):
            system_prompt: str | SystemPrompt | None = SystemPrompt.from_dict(sp_raw)
        else:
            system_prompt = sp_raw  # str or None

        allowed = d.get("allowed_tools")
        disallowed = d.get("disallowed_tools")
        return cls(
            name=d["name"],
            model=d["model"],
            system_prompt=system_prompt,
            skills=tuple(SkillSource.from_dict(s) for s in d.get("skills", ())),
            mcp_servers=tuple(McpServer.from_dict(m) for m in d.get("mcp_servers", ())),
            allowed_tools=tuple(allowed) if allowed is not None else None,
            disallowed_tools=tuple(disallowed) if disallowed is not None else None,
            secrets=tuple(d.get("secrets", ())),
            permission_mode=d.get("permission_mode", "default"),
            max_turns=int(d.get("max_turns", 120)),
            max_budget_usd=float(d.get("max_budget_usd", 10.0)),
            checkpoint=bool(d.get("checkpoint", False)),
            output_schema=d.get("output_schema"),
            env=dict(d["env"]) if d.get("env") is not None else None,
        )

    def to_yaml(self) -> str:
        """Serialize to YAML. Requires the ``yaml`` package (imported lazily)."""
        import yaml  # lazy: keep the module stdlib-only at import time

        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    @classmethod
    def from_yaml(cls, text: str) -> AgentSpec:
        """Parse a YAML document into an ``AgentSpec``. Requires ``yaml`` (lazy)."""
        import yaml  # lazy

        return cls.from_dict(yaml.safe_load(text))


def _output_schema_to_dict(schema: Any) -> Any:
    """Best-effort serialization of an output schema to a JSON-schema dict.

    * pydantic models expose ``model_json_schema()`` → use it.
    * a plain ``dict`` is assumed to already be a JSON schema → pass through.
    * anything else is stored as-is.

    TODO(P1): round-tripping back to a pydantic *class* is not supported; ``from_dict``
    returns the JSON-schema dict. Decide whether to reconstruct via ``jsonschema``/a
    registry once structured outputs land.
    """
    model_json_schema = getattr(schema, "model_json_schema", None)
    if callable(model_json_schema):
        return model_json_schema()
    if isinstance(schema, dict):
        return schema
    return schema

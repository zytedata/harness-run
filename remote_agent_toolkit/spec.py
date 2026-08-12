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
    """

    append: str | None = None

    @classmethod
    def inherit(cls, append: str | None = None) -> SystemPrompt:
        """Inherit the harness's built-in system prompt and append ``append``."""
        return cls(append=append)

    def to_dict(self) -> dict[str, Any]:
        return {"append": self.append}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> SystemPrompt:
        return cls(append=d.get("append"))


@dataclass(frozen=True)
class SkillSource:
    """A source of agent skills, resolved & staged at deploy/run time.

    A tagged union over ``kind``:

    * ``"git"``     — clone ``url`` (optionally at ``ref``), read skills from ``subdir``.
    * ``"local"``   — a local directory ``path``.
    * ``"builtin"`` — a named, library-provided skill bundle ``name``.

    A list of sources allows a base + extra sources (multi-source skills, DESIGN.md §12).
    """

    kind: Literal["git", "local", "builtin"]
    url: str | None = None
    ref: str | None = None
    path: str | None = None
    name: str | None = None
    subdir: str = "skills"

    @classmethod
    def git(cls, url: str, ref: str | None = None, subdir: str = "skills") -> SkillSource:
        """Skills from a git repository ``url`` (optionally pinned to ``ref``)."""
        return cls(kind="git", url=url, ref=ref, subdir=subdir)

    @classmethod
    def local(cls, path: str) -> SkillSource:
        """Skills from a local directory ``path``."""
        return cls(kind="local", path=path)

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


@dataclass(frozen=True)
class McpServer:
    """An MCP server attached to the agent.

    A tagged union over ``kind``:

    * ``"github"`` — the GitHub MCP server.
    * ``"remote"`` — a remote MCP server at ``url`` with optional static ``headers``.
    * ``"stdio"``  — a local subprocess MCP server (``command`` + ``args``).

    Credentials are NOT carried here. A ``github`` server's token is resolved at runtime
    from the per-invocation ``secrets`` (conventional names ``GH_TOKEN`` / ``GITHUB_TOKEN``
    / ``GH_PAT``) and injected into the server's headers by the harness — never into the
    agent's own environment. A ``remote`` server may carry static ``headers``.
    """

    kind: Literal["github", "remote", "stdio"]
    name: str | None = None
    url: str | None = None
    headers: Mapping[str, str] | None = None
    command: str | None = None
    args: tuple[str, ...] | None = None

    @classmethod
    def github(cls) -> McpServer:
        """The GitHub MCP server. Auth token comes from the per-invocation ``secrets``."""
        return cls(kind="github", name="github")

    @classmethod
    def remote(
        cls,
        name: str,
        url: str,
        headers: Mapping[str, str] | None = None,
    ) -> McpServer:
        """A remote MCP server with optional static ``headers``."""
        return cls(
            kind="remote",
            name=name,
            url=url,
            headers=dict(headers) if headers is not None else None,
        )

    @classmethod
    def stdio(cls, name: str, command: str, args: list[str] | None = None) -> McpServer:
        """A local subprocess MCP server (``command`` + ``args``)."""
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


@dataclass(frozen=True)
class RepoSource:
    """A git repository cloned into the agent's working directory *before* it runs.

    A common setup step: clone a repo so the agent can read/modify it, then commit and push.

    Auth (for private repos / pushing) is a token *named* by ``auth`` and resolved from the
    per-invocation ``secrets`` passed to ``run``/``send`` — never carried here, never baked
    into the deployed engine. The token is embedded into the clone's ``origin`` as
    ``https://<user>:<token>@host/...`` so the agent can ``git push`` without handling it, and
    is *not* placed in the agent's environment. The userinfo ``<user>`` defaults per host
    (GitHub ``x-access-token``, Bitbucket ``x-token-auth``, GitLab ``oauth2``); set ``auth_user``
    for schemes that pair the token with a real account name — e.g. a Bitbucket **API token**,
    which clones as ``https://<account>:<token>@bitbucket.org/...``. Without ``auth`` (or if the
    caller doesn't supply that secret) the repo is cloned read-only, which is fine for public repos.

    NB: to let the agent push, the token must be reachable by the agent (it can read
    ``.git/config``); the defense is a *scoped, short-lived* token (e.g. a GitHub App
    installation token), not hiding. See the README security section.
    """

    url: str
    ref: str | None = None
    auth: str | None = None
    auth_user: str | None = None

    @classmethod
    def git(
        cls,
        url: str,
        ref: str | None = None,
        auth: str | None = None,
        auth_user: str | None = None,
    ) -> RepoSource:
        """Clone ``url`` (optionally at ``ref``) into the agent's cwd before it runs.

        ``auth`` names the per-invocation secret holding the token used to authenticate the
        clone and enable ``git push``. ``auth_user`` overrides the userinfo username (default:
        host-based) — set it to your account name for a Bitbucket API token. Omit both for
        public, read-only clones.
        """
        return cls(url=url, ref=ref, auth=auth, auth_user=auth_user)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"url": self.url}
        if self.ref is not None:
            d["ref"] = self.ref
        if self.auth is not None:
            d["auth"] = self.auth
        if self.auth_user is not None:
            d["auth_user"] = self.auth_user
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> RepoSource:
        return cls(url=d["url"], ref=d.get("ref"), auth=d.get("auth"), auth_user=d.get("auth_user"))


@dataclass(frozen=True)
class AgentSpec:
    """The declarative agent definition.

    Frozen and hashable: collection fields default to tuples. The constructor coerces
    lists to tuples (via ``__post_init__``) so callers may pass lists for convenience,
    as the DESIGN.md §5 example does.

    Attributes:
        name: Stable agent name. Maps to a deployed engine's display name (§5).
        model: Model id the harness runs, e.g. ``"claude-sonnet-4-6"`` (Claude Code) or
            a GPT model id (Codex).
        harness: The coding-agent loop to run: ``"claude-code"`` (default) or ``"codex"``.
            The spec's harness-shaped fields (``permission_mode``, tool lists, skills)
            are translated by each binding; see the harness module docstrings for the
            mapping and any parity caveats.
        system_prompt: A ``SystemPrompt`` (inherit + append) or a plain ``str``
            (replace entirely) or ``None`` — the harness's built-in prompt, i.e. the
            same agent ``SystemPrompt.inherit()`` asks for.
        skills: Skill sources, resolved & staged at deploy/run time.
        repos: Git repositories cloned into the agent's cwd before it runs (with push auth
            from a GitHub token in ``secrets`` when present).
        mcp_servers: MCP servers attached to the agent.
        allowed_tools / disallowed_tools: Tool allow/deny lists, or ``None`` for default.
        permission_mode: Claude Code permission mode. Defaults to ``"bypassPermissions"``
            (unattended runs): these agents run in an isolated, throwaway working
            directory, so there is no human present to answer prompts. Set ``"default"``
            (or ``"acceptEdits"``) when running somewhere a prompt could actually be
            answered, or when the cwd is not disposable.
        max_turns: Hard cap on agent turns. Applies to each model invocation: when a
            background task (Bash ``run_in_background``, Monitor) completes, the CLI
            re-invokes the model with a fresh turn count, so a run that waits on
            background tasks may consume more turns in total — ``RunResult.num_turns``
            reports the cumulative count. ``max_budget_usd`` is cumulative regardless.
        max_budget_usd: Hard cap on spend.
        reasoning_effort: How much reasoning the model spends per response, or ``None``
            for the harness default. The union of both harnesses' vocabularies is
            accepted — ``"none" | "minimal" | "low" | "medium" | "high" | "xhigh" |
            "max"`` — and each harness maps levels only the other side supports to its
            nearest own: Codex maps ``max → xhigh`` (with a ``spec_warning`` status);
            Claude Code maps ``minimal``/``none`` ``→ low``. Unknown strings pass
            through to the SDK untouched.
        background_task_timeout: Seconds to keep a turn open waiting for the agent's
            still-running background tasks after the model ends its turn (event-driven
            waiting: the harness holds the stream open and the CLI re-invokes the model
            when a task completes). On expiry the turn finalizes with the result already
            produced, plus a ``task_wait_timeout`` status event. Size it to the longest
            background job the agent legitimately waits on (e.g. a verification crawl).
        checkpoint: Enable checkpoint/resume (interactive pauses).
        interactive: Append the "stop and await the operator" guidance to the system
            prompt. ``None`` (default) follows ``checkpoint`` — the historical coupling.
            Set ``False`` to checkpoint an autonomous loop without pause guidance, or
            ``True`` for the guidance without checkpointing.
        output_schema: Optional structured-output schema (pydantic model or JSON schema).
        env: Extra environment variables for the agent. **Non-secret only** — values live in
            the spec and are baked into the deployed engine image, so they are visible to
            anyone who can read the deployment. Secrets are NOT declared here; they are passed
            per-invocation to ``run``/``send`` (see ``secrets=`` on the run plane) so nothing
            sensitive is ever baked or shared across runs.
        packages: Python package requirement specifiers (e.g. ``"pandas==2.2.*"``) the agent
            starts with, on BOTH backends: ``gemini.deploy`` bakes them into the engine image;
            ``local.deploy`` resolves them into a per-engine venv (via ``uv``, Python pinned to
            the engine contract's 3.12) activated in the agent's environment. Either way the
            agent can still install more at runtime via ``uv``.
    """

    name: str
    model: str
    harness: str = "claude-code"
    system_prompt: str | SystemPrompt | None = None
    skills: tuple[SkillSource, ...] = ()
    repos: tuple[RepoSource, ...] = ()
    mcp_servers: tuple[McpServer, ...] = ()
    allowed_tools: tuple[str, ...] | None = None
    disallowed_tools: tuple[str, ...] | None = None
    permission_mode: str = "bypassPermissions"
    max_turns: int = 120
    max_budget_usd: float = 10.0
    reasoning_effort: str | None = None
    background_task_timeout: float = 3600.0
    checkpoint: bool = False
    interactive: bool | None = None
    output_schema: Any = None
    env: Mapping[str, str] | None = field(default=None)
    packages: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Coerce list args to frozen-hashable tuples without breaking frozen-ness.
        object.__setattr__(self, "skills", tuple(self.skills))
        object.__setattr__(self, "repos", tuple(self.repos))
        object.__setattr__(self, "mcp_servers", tuple(self.mcp_servers))
        object.__setattr__(self, "packages", tuple(self.packages))
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
            "harness": self.harness,
            "skills": [s.to_dict() for s in self.skills],
            "repos": [r.to_dict() for r in self.repos],
            "mcp_servers": [m.to_dict() for m in self.mcp_servers],
            "permission_mode": self.permission_mode,
            "max_turns": self.max_turns,
            "max_budget_usd": self.max_budget_usd,
            "background_task_timeout": self.background_task_timeout,
            "checkpoint": self.checkpoint,
            "packages": list(self.packages),
        }
        if isinstance(self.system_prompt, SystemPrompt):
            d["system_prompt"] = self.system_prompt.to_dict()
        elif self.system_prompt is not None:
            d["system_prompt"] = self.system_prompt
        if self.allowed_tools is not None:
            d["allowed_tools"] = list(self.allowed_tools)
        if self.disallowed_tools is not None:
            d["disallowed_tools"] = list(self.disallowed_tools)
        if self.reasoning_effort is not None:
            d["reasoning_effort"] = self.reasoning_effort
        if self.interactive is not None:
            d["interactive"] = self.interactive
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
            harness=d.get("harness", "claude-code"),
            system_prompt=system_prompt,
            skills=tuple(SkillSource.from_dict(s) for s in d.get("skills", ())),
            repos=tuple(RepoSource.from_dict(r) for r in d.get("repos", ())),
            mcp_servers=tuple(McpServer.from_dict(m) for m in d.get("mcp_servers", ())),
            allowed_tools=tuple(allowed) if allowed is not None else None,
            disallowed_tools=tuple(disallowed) if disallowed is not None else None,
            permission_mode=d.get("permission_mode", "bypassPermissions"),
            max_turns=int(d.get("max_turns", 120)),
            max_budget_usd=float(d.get("max_budget_usd", 10.0)),
            reasoning_effort=d.get("reasoning_effort"),
            background_task_timeout=float(d.get("background_task_timeout", 3600.0)),
            checkpoint=bool(d.get("checkpoint", False)),
            interactive=None if d.get("interactive") is None else bool(d["interactive"]),
            output_schema=d.get("output_schema"),
            env=dict(d["env"]) if d.get("env") is not None else None,
            packages=tuple(d.get("packages", ())),
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

    Note: round-tripping back to a pydantic *class* is not supported; ``from_dict`` returns
    the JSON-schema dict (which the parser validates against just the same).
    """
    model_json_schema = getattr(schema, "model_json_schema", None)
    if callable(model_json_schema):
        return model_json_schema()
    if isinstance(schema, dict):
        return schema
    return schema

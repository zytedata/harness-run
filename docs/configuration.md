# Configuration

An agent is configured at three scopes: the `AgentSpec` it is deployed with, a `SessionConfig` bound
when a session starts, and a `TurnConfig` passed to each turn. This page covers the scopes, then the
spec fields that shape the agent's environment: tools, budgets, environment variables, MCP servers,
skills, git repositories, and pre-installed packages.

## Deploy / session / turn: the three configuration scopes

An engine is deployed per agent *role*, but much of its configuration is naturally
per-**session** (which repo/ref this conversation operates on, the prompt or model variant
of an A/B experiment) or per-**turn** (budgets, structured output for the final turn) —
and requiring an image rebuild per combination would be unworkable. Configuration
therefore has three scopes, each with its own type, named by the lifetime of what it
configures:

| Scope | Type | Bound at | What belongs here |
|---|---|---|---|
| deploy | `AgentSpec` | `sandbox.deploy(spec)` | identity (`name`), image contents (`packages`, the agent `env`, the harness CLIs — `harnesses=(...)` bakes several), and the *defaults* for everything below |
| session | `SessionConfig` | `engine.start_session(config=...)` | the conversation's world: `repos`, `skills`, `mcp_servers`, `system_prompt`, `harness` (selects among the baked CLIs), `checkpoint`/`interactive`, `extra_env` — plus session-wide defaults for the turn knobs |
| turn | `TurnConfig` | `session.run(config=...)` / `send(config=...)` | the knobs the harness re-reads every invocation: `model`, `openrouter_provider`, `openrouter_routing`, `reasoning_effort`, `max_turns`, `max_budget_usd`, `max_buffer_size`, `background_task_timeout`, `permission_mode`, tool lists, `output_schema` |

Both config types are **sparse overlays**: a field left at `INHERIT` (the default) keeps
the value from the layer below; a set field replaces it wholesale (`extra_env` is the one
additive exception — it adds onto the deployed `env`). Deploy-only facts have no config
field at all, so "different `packages` per run" is a `TypeError`, not a silent no-op.

```python
from harness_run import RepoSource, SessionConfig, TurnConfig

engine = sandbox.get_engine("code-agent", project=..., location=...)

session = engine.start_session(config=SessionConfig(          # bound ONCE, for good
    repos=[RepoSource.git("https://bitbucket.org/o/store", ref="fix/issue-123",
                          auth="bb-token")],                   # named secret, never inline
    model="claude-sonnet-4-6",                                 # overrides the deployed default
))
run  = await session.run("Fix the failing test", secrets={"bb-token": tok})
run2 = await session.send(                                     # same session config; per-turn overlay:
    "Summarize what you changed as JSON",
    secrets={"bb-token": tok},
    config=TurnConfig(output_schema=ChangeReport, reasoning_effort="low"),
)
```

**Why the session config binds once.** A session's first turn clones `repos` and provisions
`skills` into a fresh workspace; every later turn restores a snapshot of the previous
turn's workspace (with the agent's uncommitted work) and never re-reads those fields. A
mid-conversation change could not be honored — so the API refuses to express it: `send()`
takes no session config, and `get_session(sid)` re-attach reads back the config the opener
persisted rather than accepting one. Only a definite "no such object" from the bucket means the
session has no config: a permission error, a timeout or a malformed record fails the re-attach
instead of silently running the baked spec, and the next attempt re-reads.

The contracts behind this:

* **The worker executes the merged spec** (deploy-baked ← session ← turn) and **echoes it
  as an `effective_spec` event** — the durable ground-truth record of what actually ran
  (drive post-mortems and replays from it, not from what you think you passed). The configs
  ride the turn's HTTP body; copies are persisted in GCS next to the session's records
  (30-day lifecycle) for the same reason, and `get_session` re-attach reads the session's back.
* **Fail closed**: a config field this image's library doesn't know, or a `harness` the
  image doesn't bake, FAILS the turn with a terminal error — never a silent fall-back to the
  baked spec (a wrong-configuration run is exactly what this exists to prevent). Client and
  image must come from the same library revision (the turn body is a wire contract).
* **No config → nothing recorded**: a plain `start_session()`/`run()` is byte-for-byte the
  pre-config behavior; the turn runs the deploy-baked spec with zero extra moving parts.
* **Deploy-time fields stay deploy-time**: `packages` and harness *availability* come from
  the image; a session selects among what is baked (`AgentSpec(harnesses=("claude-code",
  "codex"))` bakes both CLIs so sessions can pick either). Baked skills are reused as a
  staging fast path when the effective `skills` match the deployed ones; a differing
  declaration resolves from its own sources at run time, as does every session of an engine
  deployed with `skills_resolution="session"` (see [Skills](#skills)).
* **No credentials in configs**: config objects are persisted and kept for debugging, so
  `SessionConfig` refuses a `RepoSource.url` embedding
  `user:token@` at construction — name the token via `RepoSource(auth=..., auth_user=...)`
  and pass the value in `run(secrets=...)`.

**Managing deployed engines** (control plane):

## Customizing the agent's environment

Beyond skills, several `AgentSpec` fields shape what the agent can do and the environment its tools run in:

- **`model`** — the model id for the chosen harness: a Claude id (`"claude-sonnet-4-6"`), an OpenAI
  id (`"gpt-5.6-luna"`), or an OpenRouter id (`"openrouter/z-ai/glm-5.3"`). See
  [Harnesses and models](harnesses-and-models.md).
- **`openrouter_provider`** — pin one OpenRouter provider for an `openrouter/` model, so every
  request goes to the same one (see [OpenRouter models](openrouter.md)). Also
  settable per session and per turn.
- **`openrouter_routing`** — OpenRouter's whole `provider` object instead of one pinned slug:
  several providers, a deny list, fallbacks on, a price or throughput sort (see
  [Full provider routing control](openrouter.md#full-provider-routing-control)). Also settable per session and
  per turn.
- **`allowed_tools` / `disallowed_tools`** — tool allow/deny lists. Unset `allowed_tools` defaults to a
  coding-agent set (`Read, Write, Edit, Bash, Glob, Grep, TodoWrite`); the `Skill` tool is enabled
  automatically when skills are present.
- **`max_turns` / `max_budget_usd`** — hard caps on loop length and spend (the run ends with the matching
  stop reason).
- **`max_buffer_size`** — bytes allowed in a single message read from the Claude Code CLI's stdout
  (default 32 MiB; `claude-code` only). One oversized message — a `Read` of a large image arrives
  base64-encoded, and a big tool result is one line — kills the turn with no result at all, so raise this
  for an agent whose tool outputs are legitimately enormous rather than discovering the ceiling in
  production. (The SDK's own default is 1 MiB; the library's is deliberately far above it.)
- **`reasoning_effort`** — how much reasoning the model spends per response
  (`"none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max"`; unset = harness default). The
  vocabulary is the union of both harnesses'; each maps levels only the other supports to its nearest
  own (Claude: `minimal`/`none` → `low`; Codex: `max` → `xhigh`, with a status warning).
- **`env`** — extra **non-secret** environment variables for the agent's tool subprocess (config flags,
  etc.). Values live in the spec and are baked into the deployed engine, so never put secrets here — pass
  those per-invocation (see [Secrets & security](secrets-and-security.md)). A `None` value (YAML `null`)
  removes the variable from the environment of the agent's shell commands, whether the host environment
  or a per-run secret set it. That keeps the agent's environment predictable, e.g. to hide an API key
  during an evaluation. It is not a security boundary: the harness process still holds the value, and
  code the agent runs could still find it there or on disk. A value the agent must never see should not
  be on the machine or passed to the run at all.
- **`mcp_servers`** — `McpServer.github()`, `.remote(name, url, headers=...)`, `.stdio(name, command, args=...)`.
  A `github()` server's token is supplied per-invocation under a conventional name (`GH_TOKEN` /
  `GITHUB_TOKEN` / `GH_PAT`) and injected into its headers, not the agent's env.
- **`codex_config`** — extra Codex `config.toml` settings as a `{dotted.key: value}` mapping (strings,
  booleans, numbers, lists), for the `codex` harness only; `claude-code` ignores it with a status
  warning. See [Codex](harnesses-and-models.md#codex).

Credentials (the agent's own API keys, repo push tokens, MCP tokens) are **not** spec fields — they are
passed at call time to `run`/`send`. See [Secrets & security](secrets-and-security.md).

**Python libraries & CLI tools.** The agent's `Bash` tool runs in a real shell with **`uv`** and **`git`** on
`PATH`, so it can fetch and run dependencies on the fly — e.g. `uv run --with httpx script.py`, or
`uv pip install …` inside its working directory. That covers most "use library X" needs with no change to
the deployment. To pin libraries into the deployed engine instead, see
[Pre-baked engine dependencies](#pre-baked-engine-dependencies) below.

**Long-running commands.** The Bash tool's timeouts are Claude Code defaults, not platform limits: a command
is killed after **120 s** unless the agent passes a per-call `timeout`, which is itself capped at **10 min**.
For agents that legitimately run long commands (test suites, builds), raise the caps via `env` — they are ordinary
Claude Code environment settings:

```python
spec = AgentSpec(
    name="builder",
    model="claude-sonnet-4-6",
    env={"BASH_DEFAULT_TIMEOUT_MS": "600000", "BASH_MAX_TIMEOUT_MS": "3600000"},  # 10 min / 1 h
)
```

(Alternatively the agent can start long work with the Bash tool's `run_in_background` and poll it.) Also
tunable: `BASH_MAX_OUTPUT_LENGTH` for commands with very large output.


## Skills

Skills are `<name>/SKILL.md` folders ([Agent Skills](https://agentskills.io)) staged into the
agent's working directory where its harness discovers them (`.claude/skills` for Claude Code,
`.agents/skills` for Codex). Declare where they come from on the spec, as a list so a base set
and overrides can be layered (later sources win on a name collision):

```python
from harness_run import AgentSpec, SkillSource

spec = AgentSpec(
    name="entry-builder",
    model="claude-sonnet-4-6",
    skills=[
        SkillSource.git("https://github.com/zytedata/claude-skills", ref="0.2.0"),    # public, pinned
        SkillSource.git("https://git.example/acme/tools", ref="main", subdir="skills",
                        auth="SKILLS_TOKEN"),                                         # private, tracking a branch
    ],
    skills_resolution="session",
)
```

- **`SkillSource.git(url, ref=None, subdir="skills", auth=None, auth_user=None)`** clones `url`
  (at `ref`: a branch, tag or commit; the default branch when unset) and reads the skill folders
  under `subdir`. `auth` names the per-invocation secret holding the token for a private
  repository, with the same contract as `RepoSource.auth`: the value comes from
  `run(secrets={"SKILLS_TOKEN": ...})` on every turn, never from the spec. The token is used for
  the clone only; the clone is discarded once the folders are copied out, so neither the agent's
  workspace nor its environment carries it. A named secret that a turn does not supply fails
  the turn, with no anonymous retry. `SkillSource.local(path)` reads a directory on the machine
  that resolves the source.
- **`skills_resolution`** (spec, `sandbox` only) says when the sources are resolved.
  `"deploy"` (default) resolves them on the deploying machine, with its git credentials, and
  bakes the result into the image: a skill change is a redeploy, and every session runs the
  deploy-time snapshot (a `SessionConfig(skills=...)` that differs from the deployed declaration
  is the exception; the worker resolves it on the session's first turn). `"session"` bakes no
  skills: every session's first turn resolves the effective sources on the worker, so pushing to
  the skill repository changes what the next session runs, with no redeploy, and a session can
  pin its own ref through `SessionConfig(skills=[...])`. Git sources only (`deploy` rejects a
  `local` one, which the sandbox could not read). Later turns of the session run the skills its
  first turn resolved: the workspace, skills included, is restored from the checkpoint. The
  `local` runtime always resolves per run and ignores the field.
- **Provenance.** Whatever resolved the skills records what each source resolved to, with the
  commit of a git source: on the `workspace_ready` status event (`skill_sources`), on the
  terminal result (`RunResult.skill_sources`, also reconstructed by `session.last_result` and
  from `session.history()`), and in `.harness-run-sources.json` next to the staged folders, so a
  restored workspace reports the resolution its first turn made. With a floating ref this is
  how a run's skill version is known afterwards:

  ```python
  result = await session.run("build the entry", secrets={"SKILLS_TOKEN": token})
  for source in result.skill_sources or ():
      print(source["url"], source["ref"], source["commit"], source["skills"])
  ```

A source that stages nothing (a wrong `subdir`, an empty repository), a clone that fails and a
missing `auth` secret all fail the turn as configuration errors; skills are never silently
skipped or served stale. On a ready-pool sandbox a session-time resolution adds the clone to the
first turn (a few seconds for a small repository); the baked fast path costs nothing.

## Cloning a git repo

A common setup is to clone a repo into the agent's working directory **before it runs** — so it can read and
modify the code, then commit and push. Declare repos on the spec, naming (via `auth`) the per-invocation
secret that authenticates each one:

```python
from harness_run import AgentSpec, RepoSource

spec = AgentSpec(
    name="repo-fixer",
    model="claude-sonnet-4-6",
    repos=[
        RepoSource.git("https://github.com/acme/some-repo", ref="main", auth="GH_TOKEN"),
        # Bitbucket API token authenticates as <account>:<token> — name the account via auth_user:
        RepoSource.git("https://bitbucket.org/acme/backend", auth="BITBUCKET_API_TOKEN",
                       auth_user="my-bitbucket-account"),
    ],
)

# The token values are supplied at call time, never baked into the spec/engine:
result = await session.run(
    "fix the failing test and push a branch",
    secrets={"GH_TOKEN": gh, "BITBUCKET_API_TOKEN": bb},
)
```

Each repo is cloned into the agent's cwd before the loop starts. When the `auth` secret is supplied, the
clone is **authenticated and push-ready**: the token is embedded into `origin` as
`https://<user>:<token>@host/...` and a commit identity is configured, so the agent can `git push`. The
userinfo `<user>` defaults per host (`x-access-token` for GitHub, `x-token-auth` for Bitbucket access tokens,
`oauth2` for GitLab); set `auth_user` for schemes that pair the token with a real account name — e.g. a
Bitbucket **API token** (`https://<account>:<token>@bitbucket.org/...`). Without `auth` (or for a repo with no
`auth`) the clone is read-only, which is fine for public repos. The push token is **not** placed in the agent's environment, and it is
scrubbed from `.git/config` before any checkpoint snapshot (then re-embedded on resume from the freshly
supplied secret). To let the agent open PRs, also add `McpServer.github()`.

> **Push auth is reachable by the agent — scope it, don't rely on hiding it.** To `git push`, the token
> must be usable by the agent (it can read `.git/config`). The real defense is a **scoped, short-lived**
> credential — a GitHub App installation token limited to the target repo, a Bitbucket repository/workspace
> access token — not a broad personal token. See [Secrets & security](secrets-and-security.md).


## Pre-baked engine dependencies

Runtime `uv` (above) is great for experimentation, but for **pinned versions, a private index, or packages
you don't want re-fetched on every run**, declare them on the spec. `sandbox.deploy` bakes them into the
sandbox image (a `pip install` layer of the generated Dockerfile), so a deployed agent starts with them
already installed:

```python
spec = AgentSpec(
    name="data-agent",
    model="claude-sonnet-4-6",
    packages=["pandas==2.2.*", "httpx>=0.27"],   # pip/uv requirement specifiers, baked at deploy time
)
```

**The image's base layer is pinned in-tree.** `runtime/sandbox/_image.py` lists what every sandbox image
installs besides the library (the `claude-agent-sdk` pin that ships the `claude` binary, `openai-codex` when
the Codex harness is baked, `google-cloud-storage`, `uv`, …). Your `packages` merge with these — a pin that
contradicts the base fails fast at deploy, before the build. The image tag is a content digest of the whole
build context (the library's source, the spec, the requirements, the Dockerfile), so a deploy of an unchanged
spec from an unchanged library finds its image in the registry and skips the build.

**`local` honors `packages` too**: `local.deploy` resolves them into a per-engine venv (via `uv`, with the
engine contract's Python 3.12 — uv provisions the interpreter if your machine lacks it) and activates it in
the agent's environment. Same spec, same starting packages on both backends — and since `uv` hardlinks from
its global cache, a warm local deploy takes seconds, so iteration stays fast. The venv is per-engine, so an
agent installing extras mid-run never leaks into other runs. Note this covers *Python package* parity;
OS-level parity (glibc, system libs) is what the [`dev/` parity image](../dev/) is for. The image build runs on
your machine with Docker (~30 s warm, a few minutes cold) and every package adds pull time when a sandbox
starts, so pin only what genuinely needs to be pre-installed and lean on runtime `uv` for the rest.

> **Troubleshooting install issues locally.** A dev image under [`dev/`](../dev/) mirrors the engine's install
> contract (Debian/glibc, Python 3.12, `uv`, `git`, your `packages`) so dependency failures surface with a
> shell at hand: `make parity-build PACKAGES="pandas==2.2.*"`, then `make parity-check` (or `parity-shell` to
> run your agent inside it). It reproduces the *install* environment, not the sandbox.

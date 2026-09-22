# Secrets and security

Secrets are **passed per-invocation**, never declared on the spec and never baked into the deployed engine.
You hand `run`/`send` a `name → value` map; the values live only for that run:

```python
result = await session.run(
    "run the nightly report and push results",
    secrets={
        "SERVICE_API_KEY": service_key,             # the agent's own API key (its code/tools read it)
        "GH_TOKEN": gh_installation_tok, # a repo's auth= token (git push) — see "Cloning a git repo"
    },
)
```

**How each secret is routed.** The library puts a secret where its *consumer* needs it, and nowhere else:

| Secret | Consumer | Placed in the agent's env? |
|---|---|---|
| the agent's own keys (`SERVICE_API_KEY`, …) | the agent's code/tools | **yes** — that's the point |
| a repo's `auth=` token | embedded in `git` `origin` for clone/push | **no** |
| a `McpServer.github()` token (`GH_TOKEN`/`GITHUB_TOKEN`/`GH_PAT`) | injected into MCP request headers | **no** |
| `OPENROUTER_API_KEY` | the local proxy, which is what calls OpenRouter | **no** — the CLI gets a per-run proxy token |

**The blast-radius model — the honest part.** A background agent can be steered by hostile input (a web page it
reads, a file in a repo) into revealing whatever it can reach: environment variables, `.git/config`, a
token it was handed. The library does **not** pretend to hide credentials from the agent. Instead it shrinks
the blast radius:

1. **Per-invocation, not baked/shared** — a run only ever has the secrets that *that* call passed, and
   (since the run-scoped GCS tokens below) can reach only its own staged objects, so one deployed engine
   serves many tenants without any of them sharing credentials.
2. **Scope the credentials** — prefer short-lived, narrowly-scoped tokens (a GitHub **App installation
   token** for one repo, a Bitbucket **repository access token**) over broad personal tokens. If a run is
   coerced, the exposure is limited to that one scoped token for that one run.
3. **Least secrets per run** — pass only what the task needs.

**LLM API key.** By default `sandbox` routes the model through **Vertex** with **hourly tokens** the client
mints from a predict-only service account (`harness-run-model@<project>`, created by `harness-run-gcp-setup`) and the
sandbox worker serves to Claude Code from a loopback metadata server (the client pushes a fresh one every
25 minutes while the turn runs, so long turns just work — see [Long turns](gcp-setup.md)).
The token is not in the agent's environment, but the agent's shell can fetch it the way the CLI does — the
sandbox has no other identity — and it is worth `aiplatform.endpoints.predict` for at most an hour and
nothing else; see
[What the agent's shell can reach](#what-the-agents-shell-can-reach) below. `deploy(..., use_vertex=False)`
switches to API-key mode, where you pass `ANTHROPIC_API_KEY` as a per-invocation secret — then the agent can
read a long-lived key, so prefer Vertex for anything exposed to untrusted input. Locally the agent likewise inherits your shell's environment (including
your own `ANTHROPIC_API_KEY`, or — with no key set — your `claude.ai` subscription login: see
[which credential pays for a local run](local-runtime.md)); local is a trusted-dev context.
An [OpenRouter model](openrouter.md) is the one case where a model key is never
in the agent's environment even in API-key mode.

**In transit & at rest.** Secret values travel in exactly one place: the HTTPS body of the turn the client
posts to the sandbox's worker through the platform's authenticated proxy. Nothing is persisted by the
platform — no job input, no message queue, no staged object. The library **never** writes secret values to
an `AgentEvent`, to the event mirror or to a checkpoint — push tokens are scrubbed from `.git/config` before
a workspace snapshot and re-embedded on resume. Don't log the `secrets` dict yourself.

### What the agent's shell can reach

The agent runs inside a **gVisor sandbox with no Google identity**: the metadata server answers with a
tenant-project workload identity that is 403 on everything in your project (verified live 2026-09-10 and
re-checked by `make live-smoke`'s isolation check on every run). Whatever a prompt-injected agent finds in
its shell is therefore what the turn itself brought along, and the library keeps that small:

- **Run-scoped GCS token.** The worker's GCS work (the event mirror, checkpoints, transcripts, artifacts)
  runs on a short-lived, downscoped token the client mints per turn — this bucket, only this run's object
  prefixes (`runtime/sandbox/scoped_gcs.py`). Another run's records are unreachable with it. The client
  refreshes it for turns longer than an hour through an object under the run's own prefix.
- **Model token.** An hourly Vertex token for the predict-only model service account (above), served by
  the worker's loopback metadata server (`GCE_METADATA_HOST` in the agent's env points at it): model calls,
  nothing else. Spend past the run's `max_budget_usd` is the only thing a coerced agent could add.
- **The turn's own secrets** — those the caller passed, routed as the table above says.

Sandboxes are created per turn and deleted at its end, so nothing a run leaves behind survives into the
next one; multi-turn continuity rides the checkpoint (a credential-scrubbed tar in your bucket). There is
no runtime service account, no per-worker message channel and no platform-persisted job input
(DESIGN.md §6 and §13).


## Runtime secret references in persistent configuration

`AgentSpec`, `SessionConfig`, and effective-spec events are persistent configuration,
not secret stores. Pass values through `run(secrets=...)` on every turn, including resume.
Ordinary non-secret environment values and static MCP headers remain supported.

Authenticated remote MCP servers can now declare references:

```python
server = McpServer.remote(
    "private-api", "https://mcp.example.invalid",
    headers={"Accept": "application/json"},
    header_secrets={"Authorization": "MCP_AUTH", "X-API-Key": "MCP_KEY"},
)
config = SessionConfig(mcp_servers=[server], extra_env={"MODE": "analysis"})
# Supply real values from your secret manager, not from source/config:
# session.run(prompt, secrets={"MCP_AUTH": auth_header, "MCP_KEY": api_key})
```

Each reference names a per-invocation secret containing the **complete header value**;
the library does not prepend `Bearer` or another scheme. A missing/empty reference fails
the run, without reading ambient environment as a fallback. Values are resolved only
when building harness options, never merged back into the serializable spec.

Claude receives the resolved headers through its runtime SDK config. Codex receives
environment-backed headers: only generated environment-variable names enter its
`env_http_headers` overrides. Those variables and the original secret names are excluded
from Codex's shell environment. This uses the documented
[Codex environment-backed HTTP header configuration](https://learn.chatgpt.com/docs/extend/mcp?surface=cli).
It is not OS isolation: arbitrary same-UID code may inspect process state. Use scoped
credentials and separate trust domains for mutually untrusted workloads.

### Compatibility and validation boundaries

Static `Authorization`, `Proxy-Authorization`, `Cookie`, `X-API-Key`, `Api-Key`, and
`X-Auth-Token` headers (case-insensitive) are rejected; move them to `header_secrets`.
Static and referenced headers may not overlap. Existing serialized configs containing
these fields must be migrated before this revision can load them.

Durable `env`/`extra_env` reject common credential names (`API_KEY`, `TOKEN`, `SECRET`,
`PASSWORD`, `CREDENTIALS`, and names ending in the corresponding underscore-prefixed
suffix). Pass those through runtime secrets instead. This is deliberately conservative:
some credential-file settings also need to be supplied at runtime. Values are not echoed
in validation errors. Construction and serialization validate these fields; runtime
options recheck mutable mappings.

This is **not a universal secret detector**. Custom header/env names, URLs, instructions,
tool output, repository contents, and arbitrary strings can still contain sensitive data.
Callers must keep those fields non-secret and review retention/access to artifacts. The
separate Git checkpoint-scrubbing fix is still needed. No automatic rewriting/deletion
of already persisted configuration or history is performed by this change.

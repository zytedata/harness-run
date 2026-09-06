# Runtime secrets and persistent configuration

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

## Compatibility and validation boundaries

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

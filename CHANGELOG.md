# Changelog

All notable changes to `agent-run` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning is [SemVer](https://semver.org/)-style with the usual 0.x caveat:
**minor releases (0.X) may contain backwards-incompatible changes**; patch
releases (0.X.Y) do not. Every incompatible change is listed under a
**Backwards-incompatible** heading together with update notes describing how to
migrate.

Releases are git tags (`vX.Y.Z`) on `main`, published to [PyPI](https://pypi.org/project/agent-run/)
by `.github/workflows/publish.yml` through PyPI's Trusted Publishing; install a specific release with

```
pip install "agent-run==X.Y.Z"
```

Release checklist: update this file (move the `Unreleased` section into a new
version heading with the date), bump `version` in `pyproject.toml`, merge that
commit to `main` (PR), tag the merge commit `vX.Y.Z` and push the tag. The
publish workflow refuses a tag whose version differs from `pyproject.toml`; once
it has uploaded the release, create the GitHub Release for the tag with this
file's section as the notes.

## 0.4.0 — 2026-09-23

First public release. `agent-run` was developed as an internal library (`remote-agent-toolkit`,
versions 0.1.0–0.3.1); the notes below describe this release relative to 0.3.1 for the teams
moving over, and the PR references point at that private repository. Earlier history is in the
git log.

### Backwards-incompatible

- **`AgentSpec.harness` is required; there is no default harness.** It used to default to
  `"claude-code"`, which quietly made Claude Code the norm and Codex the opt-in — the wrong
  signal from a library whose point is that both loops run the same agent. `AgentSpec(name=…,
  model=…)` now raises `TypeError`; name the loop: `harness="claude-code"` or
  `harness="codex"`. **Update note**: add `harness=` to every `AgentSpec(...)` you construct.
  Nothing else changes — a spec that already named its harness behaves exactly as before.

  The same assumption is gone from three further places: `AgentSpec.from_dict` requires a
  `harness` key instead of substituting `"claude-code"` (every dict `to_dict` has ever
  written carries one); `resolve_harness` raises a named error instead of falling back; and
  `skills.skills_subdir` raises for an unknown harness instead of returning Claude's
  `.claude/skills`, which would have staged an agent's skills where its CLI never looks and
  started it with none of them.

- **The library is renamed `remote-agent-toolkit` -> `agent-run`**, ahead of the open-source
  release. The distribution is `agent-run`, the import root is `agent_run`, the console script
  is `agent-run-gcp-setup`, and the `RATK_*` environment variables are now `AGENT_RUN_*`
  (`AGENT_RUN_RESOURCE_SAMPLE_S`, `AGENT_RUN_GITHUB_MCP_TOKEN`, `AGENT_RUN_OPENROUTER_PROXY_TOKEN`,
  `AGENT_RUN_MCP_HEADER_*`, `AGENT_RUN_METADATA_PORT`, `AGENT_RUN_WORKSPACE_ROOT`,
  `AGENT_RUN_BAKED_SPEC`). **Update note**: change your imports (`import remote_agent_toolkit`
  -> `import agent_run`), rename any `RATK_*` variable you set, and re-run
  `agent-run-gcp-setup --project <id>` once per project — the GCP resources the tool creates are
  renamed with it and the new ones do not exist yet:
  - the Artifact Registry Docker repo `ratk` -> `agent-run` (deploy's default `image_repo`),
  - the model service account `ratk-model@` -> `agent-run-model@`,
  - its custom role `ratkRuntimePredict` -> `agentRunPredict`.

  The old repo, account and role are left in place; delete them once nothing deploys against
  them. Images already pushed to the `ratk` repo stay valid — pass `sandbox.deploy(image=...)`
  or `image_repo=...` to keep using them.

  Sandbox display names now start with `agent-run-<agent>-` instead of `ratk-<agent>-`, and the
  sandbox host is `agent-run-sandbox-host`. The display-name prefix is how a client finds and
  sweeps its own sandboxes, so sandboxes created by an older version are not swept by this one:
  they are not adopted, and they expire on their own TTL (`max_turn_s`, default 8 h) instead.
  Drain or delete them before upgrading if you would rather not pay for the overlap.

  One identifier is deliberately **not** renamed: the `ratk-session:` uuid5 salt in
  `agent_run/checkpoint/session_store.py`. It is a hash input, never displayed, and its output
  keys both the transcript blob prefix and the run-scoped GCS credential's scope, so renaming it
  would silently orphan every stored session (uuid5 is one-way — they could not be found again).
  See the note in that function if it ever has to change.

- **The `gemini` run-plane namespace is renamed `sandbox`**, so the two backends are now
  `local` and `sandbox` — named for where a turn runs rather than for the vendor that hosts
  it. `agent_run.runtime.gemini` -> `agent_run.runtime.sandbox`, `gemini.deploy` /
  `gemini.get_engine` / `gemini.list_engines` -> `sandbox.*`, and `GeminiEngine` /
  `GeminiSession` -> `SandboxEngine` / `SandboxSession`. The example moves with it
  (`examples/gemini/` -> `examples/sandbox/`). **Update note**: `from agent_run.runtime import
  gemini` becomes `from agent_run.runtime import sandbox`; the Engine/Session/Run surface is
  unchanged, so nothing else in calling code moves.

  Nothing persisted carries the name — deploy records have no backend field — so there is no
  stored state to migrate for this half of the rename. Google's product keeps its own name
  throughout the docs: the backend is `sandbox`, the platform it runs on is still Gemini
  Enterprise Agent Platform.

- **The `sandbox` backend runs on Agent Sandbox instead of Agent Runtime query jobs** (zytedata/remote-agent-toolkit#84,
  DESIGN.md §13). `sandbox.deploy` now builds the agent's container image with Docker, pushes it
  to Artifact Registry and creates an immutable sandbox **template** (a template is a version);
  a turn claims a ready sandbox off the roster or creates one, hands it the turn over HTTP
  through the platform's proxy, streams events straight from the sandbox (the GCS mirror stays
  the durable record and the fallback) and deletes the sandbox at the terminal event. The
  sandbox has no Google identity: GCS access runs on the run-scoped token, model calls on
  hourly tokens minted from a predict-only service account and served to Claude Code by the
  worker's loopback metadata server. Measured against the warm pool it
  replaces: 1.3 s to the first event and ~6 s to the result of a one-tool Haiku turn (4.2 s /
  14.5 s before); a turn with no ready sandbox takes ~20 s instead of ~150 s. Module, `deploy` /
  `get_engine` / `list_engines`, `Engine` / `Session` / `Run` keep their names, so app code that
  looks an engine up and runs turns keeps working. What breaks:
  - `sandbox.deploy` drops `service_account`, `scoped_gcs`, `min_instances`, `max_instances`,
    `staging_bucket`, `new_engine`; adds `image_repo` (Artifact Registry Docker repo; default
    `<location>-docker.pkg.dev/<project>/agent-run`), `image` (skip the build, use a pushed image),
    `model_service_account` (default `agent-run-model@<project>`), `max_turn_s` (a turn's ceiling on
    a sandbox, default 8 h), `internet_access`, `log`. `use_vertex`, `resource_limits`
    (`cpu` now 1–8), `warm_pool` / `pool_size` / `pool_max_wait_s`, `output_bucket`,
    `credentials` stay. Deploying needs the Docker CLI logged into the registry.
  - `get_engine` drops `scoped_gcs`; `warm_pool` defaults to what the deploy recorded. A
    **deploy record** under the output bucket (`deploys/<name>/<template>.json`) gives a
    looked-up handle the baked spec and the model-credential settings.
  - Versions: `engine.versions()` lists template ids newest first, `revisions()` their
    metadata (`image`, `current`); `get_engine(name, version=)` **routes** to that template
    (it used to assert on the serving revision); `set_traffic()` is gone ("serving" = newest
    template); `delete_version()` deletes a template and its ready sandboxes; `engine.resource`
    is a template; `engine.delete()` takes no `delete_pool_resources` (it always deletes the
    sandboxes and every version's template).
  - Session ids are always client-minted UUIDs; `list_sessions()` reads the event mirror only;
    `history()` has the mirror layer only (no platform job output, no Cloud Logging).
  - CPU/RAM self-sampling stays but moves: the worker samples its cgroup (gVisor mounts v1
    accounting) every 20 s (`AGENT_RUN_RESOURCE_SAMPLE_S` replaces `AGENT_RESOURCE_SAMPLE_S`) and
    writes the samples to the session's **event mirror only** instead of the
    `agent_run_resources` Cloud Logging log — `session.resource_samples()` reads
    them from there (rows as before: `time` + `memory_current_bytes` / `memory_limit_bytes` /
    `memory_peak_bytes` / `cpu_usec`), `session.history()` leaves them out unless
    `include_samples=True`. The memory-pressure status event and the `memory_peak_bytes` /
    `memory_limit_bytes` / `cpu_usec` keys on the result's `raw` are unchanged, and the same
    three now come as `RunResult.resources` (`None` when nothing was sampled, e.g. `local`).
    Cloud Trace spans and the `agent_run_steps` Cloud Logging log stop;
    `session.history()` is the record.
  - Events: `turn_started` carries the sandbox id as `worker` and `warm` (from the ready pool);
    `control_ready` reports the HTTP channel; `workspace_ready.scoped_gcs` is always true. A
    run whose sandbox dies mid-turn ends with an explained error result (`sandbox_unreachable`)
    instead of a platform retry.
  - `Session.send()` into a running turn raises `ControlUnavailable` while the turn is still
    being dispatched (before its first event) — wait for the first event and send again.
  - Dependencies: `google-cloud-agentplatform>=2.1` replaces `google-cloud-aiplatform<2`,
    `google-adk`, `google-cloud-logging`, `google-cloud-pubsub` and `a2a-sdk`;
    `runtime/sandbox/constraints.txt` is gone (no pickle coupling to a deploy venv).
  - IAM: the runtime service account, its custom role and the conditional bucket bindings are
    no longer needed. The client identity needs `roles/aiplatform.user` (sandbox + template
    permissions), storage on the output bucket, `artifactregistry.writer` on the image repo and
    `iam.serviceAccountTokenCreator` on the model service account; the Agent Sandbox service
    agent (`service-<number>@gcp-sa-vertex-sandbox`) needs `artifactregistry.reader` on the
    repo. `agent-run-gcp-setup` sets exactly this up (`--model-sa`, `--repo` replace
    `--runtime-sa`, `--staging-bucket`). The `ports.dispatch` and `ports.eventsink` modules are
    removed.
  - **Turns longer than an hour keep model access, with no setup.** The model token is not
    in the agent's environment any more: the worker serves it from a loopback **metadata
    server** and points Claude Code at it (`GCE_METADATA_HOST`), so the CLI fetches the token
    as it would on a VM and refreshes it itself when it nears expiry; the client re-mints an
    hourly token every 25 minutes and pushes it to the worker (`/token` now takes
    `{access_token, expires_at}`) for as long as it holds the running turn — a session
    re-attached from another process does the same. The organization-policy route
    (`constraints/iam.allowServiceAccountCredentialLifetimeExtension`, tokens minted for
    `max_turn_s`) is gone; `max_turn_s` remains the sandbox's lifetime bound. The condition
    that remains: some client process must hold the session while the turn runs — the sandbox
    cannot mint tokens, so a turn whose last client exits loses model access within the hour.
  - **Update notes for adopters (at the pin bump):**
    1. *Deploy prerequisites:* the Docker CLI logged into Artifact Registry
       (`docker login -u oauth2accesstoken --password-stdin <region>-docker.pkg.dev`); rerun
       `agent-run-gcp-setup --model-sa ... --repo ...` (it creates `agent-run-model@` with the predict-only
       role and grants the sandbox service agent read on the repo); `roles/iam.serviceAccountTokenCreator`
       on `agent-run-model@` for the deployer **and every identity that runs turns** (each client mints
       the model token). The platform caps a template at 8 vCPU; 16 GiB is verified.
    2. *Removed `deploy()` arguments now raise* (`service_account`, `scoped_gcs`, `min_instances`,
       `max_instances`, `staging_bucket`, `new_engine`), as do unknown ones — a deploy script that
       still passes them fails at once instead of silently deploying without them. `get_engine`
       takes only `warm_pool`, `version`, `credentials`, `output_bucket`; `engine.delete()` takes
       no arguments.
    3. *Re-attach from another process:* `current_run`, `busy`, `last_result`, `send()`,
       `interrupt()`, `exec()` and `run()` all adopt a turn still running there (tokens refreshed,
       sandbox released at the end); a poller may use any of them. `history()` is a plain record
       read and adopts nothing.
    4. *Validators that now raise* — grep your specs and configs: `AgentSpec.env` /
       `SessionConfig.extra_env` keys named like credentials (`*_TOKEN`, `*_API_KEY`, `*_SECRET`,
       `*_PASSWORD`, `*_CREDENTIALS`) are rejected (pass them as per-turn `secrets`); static
       `Authorization`-class MCP headers are rejected; the Codex harness raises on `allowed_tools`
       and unknown `permission_mode`; a Codex resume raises on an unreadable checkpoint instead of
       starting fresh (handle the failed run).
- **Codex resume fails explicitly for unreadable or corrupt existing checkpoints**
  (zytedata/remote-agent-toolkit#64). Denied metadata reads, malformed metadata/thread IDs, and missing/unreadable
  rollouts no longer silently start a fresh conversation. Errors omit storage exception
  text. **Update note:** handle the failed run and repair the checkpoint or deliberately
  start a new session. Absent metadata and unsafe restore destinations retain their
  existing fresh-conversation behavior. Checkpoint permissions are unchanged.
- **Every event of a turn carries its `turn_id`** (zytedata/remote-agent-toolkit#80): live streams and history recovery
  select a turn by identity, never by clock, so a re-attached session's back-to-back turns
  cannot replay each other's results.

### Added

- **The project is licensed under Apache-2.0** (`LICENSE`), declared as a PEP 639 SPDX
  expression in `pyproject.toml` and shipped inside the wheel. Trove classifiers and
  keywords are declared too. The licence text is verbatim, with no copyright line and no
  `NOTICE` file, matching the other Zyte open-source repositories.

- **`AgentSpec.codex_config`**: extra Codex `config.toml` settings, as a mapping of dotted
  key to value (strings, booleans, numbers, lists), passed to the `codex` harness as
  `--config` overrides after the harness's own so they win, e.g.
  `{"sandbox_workspace_write.network_access": True, "web_search": "disabled"}`.
  Round-trips through `to_dict`/YAML and is a knob on `SessionConfig` and `TurnConfig`.
  The `claude-code` harness ignores it with a `spec_warning` status event. Existing specs
  are unaffected. (zytedata/remote-agent-toolkit#83)
- **`Session.exec(command, *, cwd=None, timeout=None) -> ExecResult`** (zytedata/remote-agent-toolkit#85): a read-only shell
  probe of the running turn's workspace, for showing the agent's work while it works (agentic-scraping
  renders the workspace's `git diff` every ~10 s in its change panel; the events cannot give that —
  Codex reports paths only, Claude Code old/new strings without context, and shell edits escape
  both). The command runs under `/bin/bash -c` in the agent's cwd (`cwd=` relative to it), on
  `sandbox` through the sandbox worker's existing `/exec` endpoint and on `local` on the host; both
  share one implementation (`control.run_shell`): output capped at 500 000 characters per stream
  with `truncated` flagged instead of failing, a timeout (default 60 s; at most 240 s on `sandbox`,
  the platform proxy cuts a call at ~300 s) that kills the command and reports `returncode` 124.
  Only a running turn has a workspace to probe: called before the sandbox is known `exec()` waits
  for dispatch, called with no turn running or after the turn ended it raises `ControlUnavailable`
  (`local` keeps the same rule for parity). `ExecResult` is exported from the package root. The
  worker's `/exec` now defaults its cwd to the running turn's workspace (was the workspace root)
  and takes an optional `turn_id` it checks against the running turn.
- **A session re-attached in another process adopts its running turn** (`sandbox`; zytedata/remote-agent-toolkit#86 review). A
  session held only by id (`engine.get_session(id)` in a fresh process — a poller, or a worker
  adopting a job whose owner died mid-turn) had no run, so `exec()` raised `ControlUnavailable`
  while the turn was still running, `interrupt()` was a no-op and `send()` started a **second**
  concurrent turn — the thing the `Session` contract promises never happens. Every mirrored event
  names its turn and sandbox (`raw.turn_id` / `raw.worker`, zytedata/remote-agent-toolkit#80), so the first `send()`, `interrupt()`,
  `exec()` or `run()` on such a session now reads the mirror once and, if the last `turn_started`
  has no `result` after it, adopts that turn as the session's current run: events replay from the
  worker then flow live, `send()` steers, `interrupt()` stops, `exec()` probes, `run()` raises,
  `last_result` lands at the end. **`Session.current_run`** (new on the protocol, with `busy`) is
  the running turn's `Run` or `None` — on such a session it hands the adopter that run, so it
  consumes the events as the owner would instead of polling `last_result`; on `local` it is the
  in-process run. The adopter also runs the run-scoped storage token refresh (the
  owner may be gone) and deletes the sandbox at the result (10 s late, in case the owner is still
  reading); an adopter whose event loop merely shuts down leaves the turn to its owner. `Run` gained
  `cancelled` / `cancel_note`. Live: a fresh interpreter knowing only the engine name and session
  id probed, steered (acknowledged on the owner's stream, shaping the reply) and interrupted (the
  owner's turn ended `INTERRUPTED`) turns running under another process. `dev/live_smoke.py` runs
  those checks and takes `CPU` / `MEMORY` for the template's size (a 1 CPU template provisions in
  ~30 s where 4 CPU ones hit the platform's 30-minute deadline on 2026-09-14/15).

### Changed

- **`permission_mode="plan"` on the `codex` harness denies escalations**: the read-only
  sandbox now pairs with `deny_all` instead of `auto_review`, matching
  `codex exec --ask-for-approval never`. A `plan` run that asked to write or reach the
  network could previously have that granted by Codex's auto-reviewer; now the request
  is refused and the run stays read-only. Runs that need auto-reviewed escalations
  should use `permission_mode="default"`. (zytedata/remote-agent-toolkit#83)

### Fixed

- **A `/turn` call whose answer was lost no longer re-dispatches the turn to a second sandbox**
  (PR zytedata/remote-agent-toolkit#84 review, A1). The worker accepts a turn by starting its thread and answering at once,
  so a proxy timeout or 5xx on the call may hide an acceptance — and the dispatch replayed the
  same body (same `turn_id`, prompt and secrets) on another sandbox while the first was running
  the agent, repeating its side effects. Acceptance is now idempotent per turn id on the worker
  (`/turn` re-acknowledges a turn it knows, running or finished, with `already_accepted`), and
  a lost answer is settled with the **same** sandbox before anything else: the dispatch moves
  on only from a sandbox that definitely never accepted the turn (gone before the call reached
  it, or refusing). When the sandbox cannot be asked any more (it vanished, or stayed silent for
  `DISPATCH_RECONCILE_S`, 2 min) the run ends with a `dispatch_uncertain` error result naming
  the sandbox, which is deleted — stopping whatever it started — instead of a silent replay.
  A worker from an image predating the idempotent answer is understood too (its "turn_id
  already used" refusal names the turn as its own).
- **`interrupt()` while the turn is still being dispatched no longer leaves the sandbox running
  the turn unowned** (PR zytedata/remote-agent-toolkit#84 review, A2). The hand-over runs on a thread that cannot be
  cancelled; the cancelled run's completion saw no sandbox yet and deleted nothing, so the
  worker accepted the turn afterwards and ran it — with the turn's secrets, on a sandbox no
  session owned, billing until the platform TTL. The dispatch now records the sandbox it holds
  at every step; a run cancelled meanwhile marks the hand-over abandoned, nothing more is
  posted, and the sandbox the dispatch lands on (claimed, or already running the turn) is
  deleted the moment it lands. The completion decides who owns the sandbox in one step under
  the dispatch lock before it reads the sandbox, so a hand-over landing between those two
  reads (follow-up of 2026-09-21) can no longer slip past both paths.
- **The terminal result is in the mirror before the client can see it — and before the sandbox
  is deleted** (PR zytedata/remote-agent-toolkit#84 review, B). The worker appended the result to its live record first and
  queued the mirror write behind it, while the client deleted the sandbox at the result it saw;
  a reader other than the running process (`history()`, a re-attached `last_result`,
  `_find_running_turn`) could find a turn that started and never ended, adopt a deleted
  sandbox and synthesize a `sandbox_unreachable` error for a turn that succeeded. The worker
  now writes the result to the mirror and waits for the write (bounded, 30 s) before serving it
  on `/events`; the mirror copy carries `raw.mirror = "worker"`. When the write fails (a dead
  run-scoped token, a storage outage, the timeout) the live copy says `raw.mirror = "failed"`
  and the client writes the line itself with its own credentials (`raw.mirror = "client"`)
  before releasing the sandbox; only if that fails too is the gap logged as an error.
  `MirrorStream.append(event, wait_s=)` and `write_turn_mirror` now report the write's outcome.
- **`deploy(..., internet_access=False)` on the image the newest template already serves no
  longer reuses that template silently** (PR zytedata/remote-agent-toolkit#84 review, C). The "no new version" check compared
  image, state, CPU and memory only, so a redeploy asking for no internet access returned the
  existing template with egress on and reported a no-op. The template listing now carries the
  platform's `internet_access` flag, the check compares it (a listing that does not report the
  flag is trusted for the default, on, only — asking for no egress then always creates a
  template known to have none), and the deploy record stores what was asked.
- **`busy` and `last_result` on a re-attached session adopt the turn running under another
  process, like `current_run`** (PR zytedata/remote-agent-toolkit#84 review, E4). Only `run()` / `send()` / `interrupt()` /
  `exec()` / `current_run` adopted; a poller that re-attached with `get_session` and asked
  `last_result` alone (self-healing's re-attach path) refreshed no tokens and released no
  sandbox, so a turn longer than an hour under a dead owner lost model access and its sandbox
  billed until the TTL. Both accessors now adopt on their first access, at no extra storage
  read (`last_result` already read the record). Adoption drives the turn on the running event
  loop; without one, `last_result` keeps re-reading the record until the result appears and
  then stops its refresher.
- **`deploy()` rejects unknown and removed keyword arguments** (PR zytedata/remote-agent-toolkit#84 review, E2). It swallowed
  `**kwargs` while `get_engine()` rejected them, so deploy scripts migrated from the Agent
  Runtime era that still pass `service_account=` or `new_engine=` deployed without them and
  believed otherwise. The removed names raise a `TypeError` that says so; any other unknown
  name raises like `get_engine`. Validation happens before any side effect.
- **`make live-smoke` mints the model token from the toolkit's predict-only account and
  checks what that token can reach** (PR zytedata/remote-agent-toolkit#84 review, D). `MODEL_SA` defaulted to the
  operator account, whose impersonated token — served to the agent by the worker's loopback
  metadata server — carried that account's full project permissions, and the isolation check
  probed only the platform's metadata server, so it could not notice. The default is now
  `agent-run-model@<project>` (`default_model_service_account`), and a new **model-token** check
  fetches the loopback token mid-turn through `session.exec()` and requires it refused on
  listing the project's reasoning engines, the output bucket and its service accounts (the
  turn answering proves it is good for the model). `OUTPUT_BUCKET` is configurable too.
- **The worker's run-scoped GCS token is refreshed while it is still alive, and its death is
  reported once** (ported from zytedata/remote-agent-toolkit#87 to the sandbox worker). The worker fetches its replacement
  token WITH the current one, so the swap has to happen before the current token expires: the
  credential's expiry is now clamped to a ten-minute refresh horizon on top of the expiry the
  client sends with the turn (a token object reporting a far-future expiry can no longer park
  the worker on one token past the client's next re-mint). A refresh failure is logged once,
  with the reason, instead of on every retry, and after three consecutive failures the token
  is reported dead with a single error; the turn keeps running and a client on the live
  `/events` channel still gets its result, but the durable mirror stops there, so a session
  re-attached later would find no record of the rest of the turn. The other half of zytedata/remote-agent-toolkit#87
  (`history(require_result=)` skipping a record without a terminal result) has no counterpart
  here: the sandbox `history()` has the mirror layer only, and a turn's outcome comes from the
  worker's `/events` channel, then the mirror tail, then a synthetic error after a grace period.
- **`engine.delete()` no longer deletes the ready sandboxes of an engine whose name extends
  this one's** (zytedata/remote-agent-toolkit#84 field report). The orphan sweep matched sandboxes by display-name *prefix*, and
  `agent-run-x-` is a prefix of `agent-run-x-b040d27-…` — so tearing down `x` next to the revision-suffixed
  `x-b040d27` (the naming the README recommends for side-by-side toolkit revisions) emptied the
  latter's pool, whose next turn then hit `FAILED_PRECONDITION` on each stale entry and fell back
  to a cold sandbox. The sweep now matches on the sandbox's **template** (one of this engine's
  versions), with the exact `<prefix><8 hex>` name shape as the fallback for a listing row without
  a template.
- **A dispatch that consumed stale pool entries refills them.** When rostered sandboxes turned
  out dead and the turn fell back to a created sandbox, no refill was scheduled (the refill was tied
  to the turn *using* a pool sandbox), so a batch of stale entries drained the pool to empty on a
  turn that never used it. Every entry taken off the roster by a dispatch — used or dead — is now
  replaced off the critical path.
- **`wait_until_warm()` verifies the sandboxes it reports as ready.** Rostered entries that no
  longer answer `/health` (deleted by another action, OOM, reclaimed) are dropped from the roster
  and deleted, so `deploy --warm N` cannot print `ready` for a pool another action has emptied.
- **`agent-run-gcp-setup` discovers the ADC principal on a plain `gcloud auth application-default
  login`.** It asked Google's userinfo endpoint through the project-quota session, whose
  `x-goog-user-project` header made the endpoint answer 403 for a user without
  `serviceusage.serviceUsageConsumer` on the project, so the audit demanded `--impersonator` for
  the caller themself. It now asks with a bare bearer token (tokeninfo as the fallback).
- **Resuming a session on another worker failed when the workspace held a virtualenv** (zytedata/remote-agent-toolkit#74).
  Every `.venv` has `bin/python` as a symlink to an absolute path, and the restore extracted
  the snapshot with `tarfile`'s `data` filter, which refuses such a link — after writing every
  member before it. The runtime swallowed the error, treated the workspace as fresh and ran
  `provision_repos`, whose `git clone` then failed on the half-restored directory ("destination
  path ... already exists and is not an empty directory"), ending the turn as `harness_error`.
  Restore now keeps the `data` filter's protections (member paths stay inside the workspace, a
  member that would write *through* a symlink is still refused, ownership and mode bits are
  stripped) but recreates symlinks whatever they point at, which is safe because a symlink is
  only followed at use, never at extraction. Any other member the filter refuses (a FIFO, a
  device node, a hard link outside the tree) is skipped and reported instead of aborting the
  restore. `restore()` removes a directory it created or found empty if the extraction fails,
  so the fresh-workspace fallback never stages into leftovers; the failure is logged with its
  traceback; and `workspace_ready` gains `restore_error` (the summarized exception, `None` when
  nothing went wrong) and `skipped` (member names left out), so a caller can tell a fresh start
  from a resume whose snapshot could not be restored.
- **Claude resume no longer ends on a stopped background task's empty result.** A fresh
  CLI may emit an old task notification and a zero-turn success before handling the
  queued user prompt. The harness waits for that prompt without sending it twice, and
  reports an explicit error if the CLI times out, exits, or crashes without answering.
  Normal results, usage and errors keep their existing behavior.
- **Checkpoint status reaches consumers before the terminal result on both harnesses**
  (zytedata/remote-agent-toolkit#64). The snapshot still runs inline, including on interruption; a trailing status
  is no longer lost when a consumer stops at the result or exposed to the next turn.
  Codex reports conversation-save failures as `checkpoint_error` while preserving a
  successfully saved `workspace_key` and the completed/interrupted turn's result.

- **Sandbox events are scoped to the submitted turn and addressed worker.** A previous
  turn's trailing checkpoint cannot retire a booting worker's subscription. Pickup
  requires the matching `turn_started` and unrelated events cannot extend its deadline.
  Live tails and all history-recovery fallbacks reject earlier turns and abandoned
  workers, including after reattachment. Early config errors carry the same identity.
  `Session.history()` still returns the complete session.

### Changed

- **Documentation restructured for the public release.** The README is a short front page: what the
  library is, install, one local and one remote example, the feature list, and links. The reference
  it used to carry moved, section by section, into `docs/`: `getting-started`, `harnesses-and-models`,
  `configuration`, `runs-and-sessions`, `local-runtime`, `sandbox-runtime`, `structured-output`,
  `openrouter`, `secrets-and-security` (which absorbs the former `docs/secret-configuration.md`),
  `observability` and `gcp-setup`. Examples no longer refer to internal projects, repositories or
  domains; the live dev probes and `examples/sandbox` require `PROJECT` instead of defaulting to a
  shared test project. `requires-python` is `>=3.12` with no upper bound: the `<3.14` cap dated from
  the first scaffold and no dependency needs it (the sandbox image and local package venvs pin their
  own Python 3.12 independently of the client's interpreter).

- **`send()` into a turn that is still starting is queued instead of refused** (zytedata/remote-agent-toolkit#84, from
  agentic-scraping's integration). Between `run()` and the worker's `control_ready` event (~1 s
  from the ready pool, 15–25 s when a sandbox is created) `send()` used to raise
  `ControlUnavailable`, so every consumer needed a retry loop around the dispatch window. The
  session now holds the message and posts it the moment the worker announces its channel — in
  order, acknowledged by the same `user` event, on the same `Run`. An `interrupt=True` message
  queued that early interrupts the harness right after it starts. Messages still queued when the
  turn ends (dispatch failed, sandbox died) are dropped and counted on `RunResult.warning`.
  `ControlUnavailable` is now raised only when a ready worker did not take the message.
- **The default sandbox size is 4 CPU / 4 GiB** (`sandbox.deploy(resource_limits=)` default was
  `{"cpu": "4", "memory": "8Gi"}`). A deploy of an unchanged spec with the default creates a new
  version, since the template's resources changed; pass `resource_limits` explicitly to keep 8 GiB.
- **`deploy` no longer blocks on the template's long-running operation** (zytedata/remote-agent-toolkit#84 field note): the
  template can list as ACTIVE minutes before the operation completes (nine minutes, measured), and
  the command sat on "creating template …" with nothing to look at. `create_template` now starts
  the create without waiting, polls the template listing and returns at ACTIVE, printing a
  progress line through `deploy(log=)` every 30 s; a template that ends FAILED raises with the
  operation's error message, and a template still PROVISIONING after 30 minutes raises with the
  operation name instead of hanging. `SandboxProvider.create_template` gained `log=`.
- **A stuck template blocks the next one; `deploy` now clears them.** Five 4 CPU template
  creates in two days sat PROVISIONING until the platform's own deadline (30 min 31 s, every time)
  and ended `INTERNAL` with no reason — with the same image and config as templates that came up
  in 90 s, and while a 1 CPU one came up in 33 s. Each stall followed an earlier stuck or FAILED
  template, and a stalled create completed 34 s after that FAILED template was deleted: a failed
  template appears to hold its warm-pool capacity until it is deleted. `create_template` now
  deletes every FAILED template under the host instance before creating (any engine's — a FAILED
  template is never usable), deletes the new template when it ends FAILED (the error still carries
  the platform's reason), and waits 32 minutes — past that deadline — before giving up on a
  template still PROVISIONING, deleting it too so it does not hold up the re-run. Each deletion is
  reported through `log=`; one the platform refuses is named in the error instead.
- **`get_engine` resolves to the newest ACTIVE template**, skipping FAILED and PROVISIONING ones
  with a warning (it used to address whatever was newest — a FAILED deploy attempt included);
  when no version is ACTIVE it raises a `LookupError` that lists each version's state.
  `revisions()` keeps FAILED templates visible with their `state`; the provider's listing hides
  DEPROVISIONING templates along with DELETED ones.
- **`get_engine` warns when the resolved template has no deploy record** under the output bucket
  (the handle then only addresses the template and a Vertex-routed turn fails at dispatch). The
  usual cause is a `deploy` interrupted while waiting for the template to be created; the warning
  says to re-run `deploy`, which is idempotent (image and template reused, record written).
- `agent_run.checkpoint.restore()` returns a `RestoreResult` (truthy iff a snapshot
  existed; `.found`, `.skipped`) instead of a bare `bool`, and `BlobStore.get_tree()` returns the
  list of skipped member names instead of `None`. Truthiness checks keep working; `is True`
  comparisons do not.

# Changelog

All notable changes to `agent-run` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning is [SemVer](https://semver.org/)-style with the usual 0.x caveat:
**minor releases (0.X) may contain backwards-incompatible changes**; patch
releases (0.X.Y) do not. Every incompatible change is listed under a
**Backwards-incompatible** heading together with update notes describing how to
migrate.

Releases are git tags (`vX.Y.Z`) on `main`, and every tag is published to Zyte's
internal PyPI (https://pypi.internal.example/simple/) by `.github/workflows/publish.yml`;
install a specific release with

```
uv pip install "agent-run==X.Y.Z" --extra-index-url "https://<user>:<password>@pypi.internal.example/simple/"
uv pip install "agent-run @ git+https://github.com/zytedata/remote-agent-toolkit@vX.Y.Z"
```

Release checklist: update this file (move the `Unreleased` section into a new
version heading with the date), bump `version` in `pyproject.toml`, merge that
commit to `main` (PR), tag the merge commit `vX.Y.Z` and push the tag (CI refuses
a tag whose version differs from `pyproject.toml`), then create the GitHub Release
for the tag with this file's section as the notes.

## Unreleased

### Backwards-incompatible

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

- **The `sandbox` backend runs on Agent Sandbox instead of Agent Runtime query jobs** (#84,
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
  (#64). Denied metadata reads, malformed metadata/thread IDs, and missing/unreadable
  rollouts no longer silently start a fresh conversation. Errors omit storage exception
  text. **Update note:** handle the failed run and repair the checkpoint or deliberately
  start a new session. Absent metadata and unsafe restore destinations retain their
  existing fresh-conversation behavior. Checkpoint permissions are unchanged.
- **Every event of a turn carries its `turn_id`** (#80): live streams and history recovery
  select a turn by identity, never by clock, so a re-attached session's back-to-back turns
  cannot replay each other's results.

### Added

- **`AgentSpec.codex_config`**: extra Codex `config.toml` settings, as a mapping of dotted
  key to value (strings, booleans, numbers, lists), passed to the `codex` harness as
  `--config` overrides after the harness's own so they win, e.g.
  `{"sandbox_workspace_write.network_access": True, "web_search": "disabled"}`.
  Round-trips through `to_dict`/YAML and is a knob on `SessionConfig` and `TurnConfig`.
  The `claude-code` harness ignores it with a `spec_warning` status event. Existing specs
  are unaffected. ([#83])
- **`Session.exec(command, *, cwd=None, timeout=None) -> ExecResult`** (#85): a read-only shell
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
- **A session re-attached in another process adopts its running turn** (`sandbox`; #86 review). A
  session held only by id (`engine.get_session(id)` in a fresh process — a poller, or a worker
  adopting a job whose owner died mid-turn) had no run, so `exec()` raised `ControlUnavailable`
  while the turn was still running, `interrupt()` was a no-op and `send()` started a **second**
  concurrent turn — the thing the `Session` contract promises never happens. Every mirrored event
  names its turn and sandbox (`raw.turn_id` / `raw.worker`, #80), so the first `send()`, `interrupt()`,
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
  should use `permission_mode="default"`. ([#83])

### Fixed

- **A `/turn` call whose answer was lost no longer re-dispatches the turn to a second sandbox**
  (PR #84 review, A1). The worker accepts a turn by starting its thread and answering at once,
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
  the turn unowned** (PR #84 review, A2). The hand-over runs on a thread that cannot be
  cancelled; the cancelled run's completion saw no sandbox yet and deleted nothing, so the
  worker accepted the turn afterwards and ran it — with the turn's secrets, on a sandbox no
  session owned, billing until the platform TTL. The dispatch now records the sandbox it holds
  at every step; a run cancelled meanwhile marks the hand-over abandoned, nothing more is
  posted, and the sandbox the dispatch lands on (claimed, or already running the turn) is
  deleted the moment it lands. The completion decides who owns the sandbox in one step under
  the dispatch lock before it reads the sandbox, so a hand-over landing between those two
  reads (follow-up of 2026-09-21) can no longer slip past both paths.
- **The terminal result is in the mirror before the client can see it — and before the sandbox
  is deleted** (PR #84 review, B). The worker appended the result to its live record first and
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
  longer reuses that template silently** (PR #84 review, C). The "no new version" check compared
  image, state, CPU and memory only, so a redeploy asking for no internet access returned the
  existing template with egress on and reported a no-op. The template listing now carries the
  platform's `internet_access` flag, the check compares it (a listing that does not report the
  flag is trusted for the default, on, only — asking for no egress then always creates a
  template known to have none), and the deploy record stores what was asked.
- **`busy` and `last_result` on a re-attached session adopt the turn running under another
  process, like `current_run`** (PR #84 review, E4). Only `run()` / `send()` / `interrupt()` /
  `exec()` / `current_run` adopted; a poller that re-attached with `get_session` and asked
  `last_result` alone (self-healing's re-attach path) refreshed no tokens and released no
  sandbox, so a turn longer than an hour under a dead owner lost model access and its sandbox
  billed until the TTL. Both accessors now adopt on their first access, at no extra storage
  read (`last_result` already read the record). Adoption drives the turn on the running event
  loop; without one, `last_result` keeps re-reading the record until the result appears and
  then stops its refresher.
- **`deploy()` rejects unknown and removed keyword arguments** (PR #84 review, E2). It swallowed
  `**kwargs` while `get_engine()` rejected them, so deploy scripts migrated from the Agent
  Runtime era that still pass `service_account=` or `new_engine=` deployed without them and
  believed otherwise. The removed names raise a `TypeError` that says so; any other unknown
  name raises like `get_engine`. Validation happens before any side effect.
- **`make live-smoke` mints the model token from the toolkit's predict-only account and
  checks what that token can reach** (PR #84 review, D). `MODEL_SA` defaulted to the
  operator account, whose impersonated token — served to the agent by the worker's loopback
  metadata server — carried that account's full project permissions, and the isolation check
  probed only the platform's metadata server, so it could not notice. The default is now
  `agent-run-model@<project>` (`default_model_service_account`), and a new **model-token** check
  fetches the loopback token mid-turn through `session.exec()` and requires it refused on
  listing the project's reasoning engines, the output bucket and its service accounts (the
  turn answering proves it is good for the model). `OUTPUT_BUCKET` is configurable too.
- **The worker's run-scoped GCS token is refreshed while it is still alive, and its death is
  reported once** (ported from #87 to the sandbox worker). The worker fetches its replacement
  token WITH the current one, so the swap has to happen before the current token expires: the
  credential's expiry is now clamped to a ten-minute refresh horizon on top of the expiry the
  client sends with the turn (a token object reporting a far-future expiry can no longer park
  the worker on one token past the client's next re-mint). A refresh failure is logged once,
  with the reason, instead of on every retry, and after three consecutive failures the token
  is reported dead with a single error; the turn keeps running and a client on the live
  `/events` channel still gets its result, but the durable mirror stops there, so a session
  re-attached later would find no record of the rest of the turn. The other half of #87
  (`history(require_result=)` skipping a record without a terminal result) has no counterpart
  here: the sandbox `history()` has the mirror layer only, and a turn's outcome comes from the
  worker's `/events` channel, then the mirror tail, then a synthetic error after a grace period.
- **`engine.delete()` no longer deletes the ready sandboxes of an engine whose name extends
  this one's** (#84 field report). The orphan sweep matched sandboxes by display-name *prefix*, and
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
- **Resuming a session on another worker failed when the workspace held a virtualenv** (#74).
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
  (#64). The snapshot still runs inline, including on interruption; a trailing status
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

- **`send()` into a turn that is still starting is queued instead of refused** (#84, from
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
- **`deploy` no longer blocks on the template's long-running operation** (#84 field note): the
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

[#83]: https://github.com/zytedata/remote-agent-toolkit/pull/83


## 0.3.1 — 2026-09-08

### Fixed

- **Run-scoped GCS tokens could not be minted by a service-account client** (`invalid_request:
  Invalid arguments provided in the request.` from STS). STS caps the size of a downscoped
  token, and the cap includes the caller's own base token: the boundary's nine rules, each
  carrying an object-name AND an `objectListPrefix` list clause, minted to ~10.1k chars — under
  the cap behind a ~250-char user token (every developer login), over it behind a ~1.1k-char
  service-account token (every production caller). The list clause is now emitted only for
  the two prefixes the worker actually lists with the run token (the control inbox and the
  checkpoint transcript directory, `scoped_gcs._LISTED_PREFIXES`); the token drops to ~6.5k
  chars. Client-side only: no engine redeploy, and a patched client works against engines
  deployed from earlier revisions. The mint error no longer blames bucket IAM, which STS does
  not check at mint time (it minted for a nonexistent bucket during the investigation).
  Measured 2026-09-08 against `my-project-agent-output` with a user and a
  service-account caller.
- **The Codex conversation checkpoint was outside the run token's prefixes.** The Codex
  harness persists its thread under `checkpoints/codex-threads/<session>/` (restored on the
  next turn), and `run_object_prefixes` did not list it, so with run-scoped tokens the
  persist failed silently (checkpoint writes are best-effort) and every second turn of a
  Codex session started without its conversation. The prefix is now in the boundary (a
  tenth rule; minted from a service account at ~8.3k chars, verified). A new offline test
  drives a two-turn Codex resume, the transcript store and the workspace snapshot through a
  recording store and fails on any key the token could not reach, or any listed prefix
  without a list clause, so the prefix list follows the writers from now on.
  Because the run's own token can now write `codex-threads/<session>/meta.json`, the restore
  validates the `relpath` it reads from it: a path outside `CODEX_HOME/sessions` (absolute,
  `..`, a symlink escape, a non-`.jsonl` name) is refused and the session starts fresh, where
  before the next worker wrote wherever the file said (the same check as #64).

## 0.3.0 — 2026-09-07

### Backwards-incompatible

- **`sandbox.deploy` always runs the engine as a service account you own.** With
  `service_account=` omitted (or `None`) the engine now runs as
  `agent-run-runtime@<project>.iam.gserviceaccount.com`, the account `agent-run-gcp-setup` creates
  with the README role set. There is no way to deploy as the platform default (the Agent
  Runtime service agent, the identity behind the security finding below), and the deploy
  checks the account exists before any side effect, failing with the fix
  (`agent-run-gcp-setup --project <id>`) when it does not. `agent-run-gcp-setup --no-runtime-sa` is
  gone for the same reason. **Update note**: run `agent-run-gcp-setup --project <id>` once per
  project (it only adds), then a plain redeploy from this revision moves each engine off
  the default identity. Pass `service_account=` only for an account of your own (the README
  role set plus what your agent needs). The deployer needs `roles/iam.serviceAccountUser`
  on the account (project owners and editors already have it).

### Security

- **The Agent Runtime service agent was reachable from the agent's shell, and with it
  every run's staged secrets in the shared output bucket** (found and verified
  2026-09-02; README "The runtime identity is reachable by the agent"). Fixed by
  **run-scoped GCS tokens**: the client mints a downscoped token per turn, limited to
  the run's own object prefixes, and the worker does all of its GCS work with it
  (`runtime/sandbox/scoped_gcs.py`; `GcsBlobStore(credentials=...)` and a process
  default the worker sets at turn start). The token rides the cold path as the last
  `AGENT_GCS_TOKEN=` directive and the warm path as `gcs_token` in the payload; the
  client refreshes it while the run lives. On by default (`get_engine(...,
  scoped_gcs=True)`); a minting failure fails the turn. **Update note**: create a runtime
  service account with the roles in the README IAM table (the default Agent Runtime
  service agent cannot be locked out of the bucket: its Google-managed
  `reasoningEngineServiceAgent` role reads every bucket in the engine project), give it
  `objectCreator` + `legacyObjectReader` on `jobs/` in the output bucket (the platform
  writes the job output and reads the job input by name), redeploy every engine from
  this revision with the new `sandbox.deploy(..., service_account=...)`, and only then
  remove the service agent's `objectAdmin` on the bucket and its `roles/aiplatform.user`
  on the project (README, "Migration"). No date: the removal waits until no engine runs
  as the service agent. Mixed client/engine versions keep working on the runtime
  identity, which is the unfixed state. New dev scripts:
  `dev/live_isolation_probe.py` (the finding) and `dev/live_scoped_gcs.py` (the fix:
  `RUNTIME_SA=<email>` for the custom runtime identity, unset for a bucket in another
  project).
- `sandbox.deploy(..., service_account=)` sets the engine's runtime identity (forwarded to
  `AgentEngineConfig.service_account`); omitted, the project's `agent-run-runtime@` (see
  Backwards-incompatible above). The account's Vertex role is a custom role with only
  `aiplatform.endpoints.predict` (README IAM table), never `roles/aiplatform.user`.
- **Per-worker dispatch for warm pools.** Every warm worker of an engine used to pull one
  shared Pub/Sub subscription as the same identity, so a shell in one worker could take
  another run's turn — its pointers and, with run-scoped tokens, its token — and ack it so
  that run never started. Now `fill_pool` gives each worker its own randomly named,
  filtered subscription (named only in that worker's job input; `roles/pubsub.subscriber`
  can consume by name but not list) and records it in the pool's roster (client-owned GCS
  objects under `pool/`, `runtime/sandbox/roster.py`); each turn claims one idle worker off
  the roster atomically and is published addressed to that worker alone, whose channel the
  client deletes as soon as the worker has started the turn. Along the way: a turn that
  finds no idle worker spawns one for itself instead of stranding (an empty pool now
  degrades to cold latency), a worker that never starts its turn is replaced by a
  re-dispatch (pickup watchdog), `interrupt()` cancels a warm turn's worker job, `delete()`
  cancels idle workers other processes submitted, and every turn now opens with a
  `turn_started` status event carrying the worker id. The engine env bakes
  `AGENT_POOL_TOPIC` (the deploy's topic) instead of `AGENT_POOL_SUBSCRIPTION`. **Update
  note**: redeploy warm engines. `get_engine(warm_pool=True)` on an engine deployed before
  this change warns and keeps driving its shared subscription. Never grant the runtime
  identity anything under the output bucket's `pool/` prefix: the roster decides where
  turns go.

### Backwards-incompatible

- The harness SDKs (`claude-agent-sdk`, `openai-codex`) moved from the base
  install to a new `local` install extra. The remote path (deploying and
  driving Gemini Agent Runtime engines) is unaffected — engines install the
  SDKs from their own baked requirements. **Update note**: if you run agents
  locally (`local.deploy`), install `agent-run[local]`; a missing
  SDK now fails with an error message pointing at the extra. Motivation:
  `claude-agent-sdk` pins `mcp<2`, which blocked remote-only clients from
  using the mcp 2.x SDK.
- `RunResult.usage` is now a toolkit-normalized token record: the same flat dict on
  every harness, whose five keys are always present — `input_tokens`,
  `cache_read_input_tokens`, `cache_creation_input_tokens`, `output_tokens`,
  `reasoning_output_tokens` — with `None` for a number the harness does not report
  (never a fake `0`). The three input buckets are disjoint; `reasoning_output_tokens`
  is the reasoning share of `output_tokens`. It also became complete: it covers the
  whole turn, subagent sessions and background-task re-invocation segments included.
  Previously it was each CLI's own dict, verbatim — on Claude Code that meant main
  conversation loop only and, on re-invoked turns, the final segment only (up to a
  ~2x undercount on subagent-heavy runs); on Codex it meant a different shape
  (inclusive `input_tokens`, `cached_input_tokens`, `total_tokens`) that missed
  collab-subagent threads entirely, since Codex reports no subagent usage on its
  wire ([openai/codex#14642](https://github.com/openai/codex/issues/14642)) — the
  harness now recovers those post-hoc from their rollout files, and a natively
  priced Codex `cost_usd` includes that spend too — each subagent thread priced at
  the model its rollout names (`raw["subagent_usage"][thread]["model"]`); a model
  `pricing` does not know makes `cost_usd` `None` — never a guess — with a
  `subagent_price_unknown` status event, tokens still counted in `usage`. The mid-turn `max_turns`/
  `max_budget_usd` checks still cannot see it while the turn runs, but the budget
  is re-checked at turn end, so subagent spend that crosses the cap yields
  `error_budget_exceeded` rather than `success`; a subagent still running at turn
  end is flagged by a `subagent_usage_partial` status event, and a wire-seen
  subagent thread with no rollout found by a `subagent_usage_missing` one (the
  recovery reads non-public rollout details; `make live-usage` re-checks them).
  **Update notes:** readers of Claude Code's verbatim dicts find them on the result
  event's `raw` — `raw["model_usage"]` (the CLI's complete per-model record) and
  `raw["cli_usage"]` (the old `usage` value). Codex readers: `cached_input_tokens`
  is now `cache_read_input_tokens`, `input_tokens` no longer includes the
  cached/cache-written share, `total_tokens` is gone (it is the three input buckets
  + `output_tokens`; `reasoning_output_tokens` is already inside `output_tokens`), and
  per-thread subagent detail is at `raw["subagent_usage"]`. The harness conformance
  suite now asserts the normalized shape. The harness code runs inside the engine,
  so a deployed sandbox engine keeps emitting the legacy `usage` shape until it is
  redeployed with this version (the client passes results through unchanged), and
  events already recorded stay in the legacy shape. `num_turns` is unchanged and now documented: the
  main agent's model calls on both harnesses, cumulative across re-invocation
  segments, subagent calls not counted — the scope `max_turns` caps. ([#36])

### Added

- `agent-run-gcp-setup` (a console script; also `python -m
  agent_run.runtime.sandbox.project_setup`): one-command GCP project setup
  for the `sandbox` backend. Audits a project against the README's "GCP setup & required
  permissions" section — required APIs, the staging/output buckets (+ handoff lifecycle
  rules), the operator SA with project roles and *bucket-scoped* storage grants, the
  impersonation grant, the runtime service account engines run as (`agent-run-runtime`: the
  `agentRunPredict` custom role with only `aiplatform.endpoints.predict`, its project
  roles, `objectViewer` on the staging bucket, the conditional `jobs/` + `events/agent-run-`
  bindings on the output bucket, and the operator's `serviceAccountUser` on it; the
  default Agent Runtime service agent is granted nothing, and grants it still holds from
  the earlier identity model are reported as a migration note), and live Claude-on-Vertex
  model checks (1-token probes; by default Haiku 4.5 / Sonnet 5 / Opus 5 required and
  Fable 5 optional — reported but non-blocking; `--model` / `--optional-model` to tune)
  — then asks for confirmation, applies what's missing, and re-audits.
  Additive-only and idempotent, so it is safe on existing non-empty projects.
  `--check` audits without changing anything; `--yes` applies without a prompt;
  `--verify` proves the end state with a throwaway warm-pool deploy as the runtime
  service account + one Haiku turn (and refuses to spend on the deploy while a check it
  depends on still fails — the model check itself is a 1-token live probe, reported
  before anything is applied).

- Talking to a running turn ([#44]). `Session.send()` on a RUNNING session no longer
  dispatches a second concurrent turn (two workers writing one session's stream,
  checkpoint and transcript); it delivers the message INTO the running turn and returns
  the same `Run`: `interrupt=False` steers (the model sees it at its next step — Claude
  Code's mid-turn `query()`, Codex's `turn/steer`), `interrupt=True` interrupts the model
  first and continues from the message in the same harness session and workspace. Each
  delivered message is a `user` event on the stream / mirror / `history()`, carrying the
  caller's `message_id=` — the acknowledgement that the model has it; the worker dedupes
  on the id, so a retried send never reaches the model twice. On `sandbox` the transport
  is a GCS inbox (`control/<sid>/` under the output bucket, `AGENT_CONTROL_GCS` in the
  engine env) the worker polls every ~1.5 s during the turn, cold and warm alike, from
  any process that holds the session id; the worker announces it with a `control_ready`
  status event. `run()` on a running session raises; `secrets` / `config` / `hooks` are
  rejected on a running `send()`; a running turn on an engine revision deployed before
  the inbox existed raises the new typed `ControlUnavailable` (exported at package
  level) instead of writing into the void. `local` implements the same semantics over an
  in-process queue, and `make live-interactive` checks the four transitions against real
  models on both harnesses and prints how long a steer, an interrupt and a stop take
  (TESTING.md has the numbers). `make chat` (`dev/chat.py`) is a local chat page for
  trying turn control by hand; a dev tool like the live probes.
- Both harnesses can run `openrouter/*` models with an `OPENROUTER_API_KEY`
  per-invocation secret. Kimi K3, GLM-5.3, and DeepSeek v4 Flash/Pro have known
  context sizes. Each OpenRouter response reports its selected upstream and exact
  charge as an `openrouter_request` event. Results and budget checks use that charge,
  and an OpenRouter model is never priced from the toolkit's own table: a turn
  OpenRouter reported no charge for gets `cost_usd=None` and a `cost_unknown` event
  (why an estimate cannot replace it: the `harness/pricing.py` docstring).
  `openrouter_provider` selects one OpenRouter provider per agent, session, or turn and
  disables fallbacks. Paid local and Gemini Agent Runtime checks cover both
  harnesses. ([#34])
- `openrouter_routing` carries OpenRouter's whole `provider` object instead of that one
  pinned slug — several providers, a deny list, fallbacks on, a price or throughput sort —
  and sends it verbatim. Same three scopes, and the two fields cannot be combined.
  `provider_matches_request` is only reported when the object names a closed set of
  providers (`only`, or `order` with `allow_fallbacks` false); an open set reports `null`
  instead of a verdict the request cannot support. Every `openrouter_request` event carries
  the object as `requested_routing`, including the one a pinned slug resolves to — the slug is
  turned into that object before the proxy starts. Because a config field is new, a client staging it
  against an older engine fails closed, as the same-revision rule requires. ([#34])
- Codex emits a `model_routing` status event from the app-server's thread record. ([#34])
- Every OpenRouter model call, on both harnesses, passes through a per-run localhost
  proxy, because the CLIs can neither send OpenRouter's provider field nor report what
  OpenRouter charged. It also enforces the selected model and the budget, and keeps the
  provider key in the parent process. See DESIGN.md.
  The proxy's lifecycle, the key lookup, the schema steer, the metadata drain and the
  `cost_unknown` event live once in `harness/_shared.py`; each binding keeps only how its
  own CLI is pointed at the proxy (environment variables for Claude Code, `--config`
  overrides for Codex). ([#34])

### Backwards-incompatible

- `RunResult.cost_usd` is now `float | None`, and defaults to `None`. It used to be
  `float` defaulting to `0.0`, so a run whose spend the toolkit could not determine
  reached the caller as `0.0` and read as a free run. The terminal event already
  carried `None` for that case, alongside a `cost_unknown` status event; the result
  object now carries it too. A run that died before reporting anything also reports
  `None` rather than `0.0`. **Update note:** code that does arithmetic or formatting
  on `result.cost_usd` needs a `None` check, e.g. `result.cost_usd or 0.0`. ([#34])

### Changed

- `Session.interrupt()` now means "interrupt and stop": the harness is asked to stop and
  the turn ends through its normal end-of-turn path — workspace checkpoint, transcript,
  a terminal result with the new `StopReason.INTERRUPTED` that keeps the turn's
  accounting — so the session is idle and a later `send()` resumes it, workspace changes
  of the interrupted turn included. Before, it cancelled the client-side tail (and the
  cold job): the run ended `is_error=True` / `StopReason.ERROR` with no checkpoint, and
  the next `send()` restored the previous turn's snapshot, so the interrupted turn's
  files were lost while its conversation survived. **Update note**: code that treated an
  interrupted session's `stop_reason == ERROR` as "stopped" should check for
  `INTERRUPTED`; the old cancel remains only as the fallback when the worker cannot be
  reached or does not stop in time, and the result's `warning` says so. On `sandbox` the
  new path needs an engine redeployed with this version (the worker polls the control
  inbox); against an older revision `interrupt()` falls back as before. ([#44])
- `AgentEvent.kind` has a new value, `"user"`: an operator message delivered into a
  running turn (see Added). Consumers that switch exhaustively on `kind` should add it.
- On Codex, a turn that Codex reports as `interrupted` now yields result subtype
  `interrupted` with `is_error=False` (it was `is_error=True`); a turn interrupted by the
  harness at `max_turns` / `max_budget_usd` still reports `error_max_turns` /
  `error_budget_exceeded`.
- Engines are deployed with `NUM_WORKERS=1`. The platform's serving harness (uvicorn, in
  the container base image) otherwise starts `os.cpu_count() + 1` worker processes —
  10–11 on the nodes seen live, sized to the host rather than the container's CPU limit
  — each importing the whole stack privately: ~300 MiB apiece, ~3 GiB of the default 4Gi
  worker consumed before the agent ran, so the `memory pressure` warning fired on nearly
  every turn and memory-heavy work OOM-killed easily. A query job serves one request per
  container, so one worker loses nothing: the idle baseline drops to ~450 MiB (cold and
  warm pools verified live) and job startup CPU from minutes to seconds. ([#42])
- `google-cloud-aiplatform` is now pinned `<2`: 2.0 was released in late August 2026 and
  is not validated with the toolkit yet, and the previous `>=1.163` floor would resolve
  to it on a fresh install (engine builds were already held at `1.165.1` by
  `constraints.txt`). Migration to 2.x is tracked separately. ([#38])
- The harness SDKs are now pinned exactly: `claude-agent-sdk==0.2.130` and
  `openai-codex==0.147.0`, matching what a deployed engine installs. Both were
  validated together on the local and Agent Runtime paths, and `openai-codex`
  ships the Codex CLI itself, whose behavior moves between versions (0.147
  dropped the `chat` wire API the OpenRouter route used to have a choice
  about). Installing this library therefore fixes those two versions ([#34]).

### Fixed

- Warm-pool workers cold-started under a previous engine revision no longer claim turns
  dispatched after a redeploy (they used to serve them with the old deploy-baked
  spec/skills for up to `pool_max_wait_s`, silently). Each `sandbox.deploy(warm_pool=True)`
  now mints a deploy-scoped dispatch topic/subscription pair and deletes the previous
  pair once the new pool is filled, so stale idle workers fail their next claim poll and
  exit within seconds. `get_engine(warm_pool=True)` reads the live subscription back from
  the deployed engine's env (the serving revision's when traffic is pinned) instead of
  deriving it from the engine name, and `wait_until_warm` tails the deploy-scoped
  readiness marker, so a previous pool's markers never count as this one being warm.
  With traffic pinned to an older revision, dispatch stays on the pinned (serving)
  revision's pair — resolved from that revision's own baked env, not the engine-level
  env, which under a pin names the latest, never-served revision's listener-less pair —
  and nothing is retired: each such deploy's fresh pair is kept (a later promotion
  needs it), and `delete(delete_pool_resources=True)` sweeps every revision's pair. A
  failed subscription read in `get_engine(warm_pool=True)` now raises (retryable)
  instead of silently falling back to legacy fixed names that fail only at the first
  publish. ([#38])
- Claude Code turns that finish without a final assistant message now return
  `error_no_final_text` for OpenRouter models. This has occurred intermittently with
  DeepSeek v4 and previously looked like a successful run with placeholder text. ([#34])
- Claude Code now passes `output_schema` to the Agent SDK as its native JSON-schema
  output format and preserves `ResultMessage.structured_output` in the terminal event.
  Earlier versions only parsed the final text on the client, so the model received no
  schema constraint and SDK-provided structured data was discarded. OpenRouter turns
  also receive the schema in their prompt because its selected model may be responsible
  for following it. ([#34])

[#34]: https://github.com/zytedata/remote-agent-toolkit/pull/34
[#36]: https://github.com/zytedata/remote-agent-toolkit/pull/36
[#38]: https://github.com/zytedata/remote-agent-toolkit/issues/38
[#42]: https://github.com/zytedata/remote-agent-toolkit/pull/42

## 0.2.0 — 2026-08-24

### Added

- Configuration now has three scopes, one type each: the `AgentSpec` baked at
  deploy, a `SessionConfig` bound once at `engine.start_session(config=…)`
  (the conversation's world: `repos`, `skills`, `mcp_servers`,
  `system_prompt`, `harness`, `checkpoint`, `extra_env`, plus session-wide
  knob defaults), and a `TurnConfig` passed per turn to `run()`/`send()`
  (`model`, `reasoning_effort`, budgets, `permission_mode`, tool lists,
  `output_schema`). Both configs are sparse overlays: a field left at
  `INHERIT` keeps the value from the layer below. Full parity between the
  `local` and `sandbox` backends; no more ~4 min engine redeploy to vary
  per-conversation or per-turn settings.
- The worker echoes the merged configuration it actually executed as an
  `effective_spec` event in the session's stream — the durable ground-truth
  record for debugging and replay.
- `AgentSpec(harnesses=("claude-code", "codex"))` bakes several harness CLIs
  into one image, so each session can pick its harness; selecting one the
  image doesn't carry fails the turn loudly.

(all [#16])

- `sandbox.deploy(..., pool_max_wait_s=...)` sets how long an idle warm-pool
  worker waits for an assignment before exiting. The worker side always read
  `AGENT_POOL_MAX_WAIT_S`, but nothing plumbed it into the engine env, so
  the knob was unreachable; passing it without `warm_pool=True` now fails
  loudly instead of being swallowed by `deploy()`'s kwargs catch-all, and
  invalid values (zero, negative, NaN, infinity) are rejected at the
  `deploy()` boundary — before the pub/sub ensure, so bad input never leaves
  an orphaned topic/subscription behind ([#28]).
- `AgentSpec(transcript=True)` persists the harness's own transcript — the full
  per-turn record: usage, tool statuses, subagent trees, permission denials —
  and `Session.transcripts()` reads it back on both runtimes, keyed by `"main"`
  plus each subagent subpath. Reading a transcript used to require
  `checkpoint=True`, which also archives the entire working directory to blobs
  at every turn: hundreds of MB per run for an agent that writes a lot, paid
  purely to get at a JSONL. `checkpoint=True` still implies `transcript`
  (resume needs the transcript), so existing specs are unaffected. On `sandbox`
  the flag is what opens the checkpoint bucket, so a transcript-only spec needs
  a redeploy with an `output_bucket`. `transcript` on its own is purely
  observational: `send()` still needs `checkpoint=True` to continue a
  conversation. `transcripts()` raises when the spec persists nothing and reads
  `{}` when persistence is on but nothing is written yet, so an empty result is
  never a misconfiguration in disguise; the `codex` harness persists its
  conversation under `checkpoint` alone, so it reads `{}` there and the run says
  so with a `spec_warning` event. Treat what `transcripts()` returns as
  sensitive: unlike events, it is the verbatim record of everything the agent
  saw ([#26]).
- `run(hooks=…)` / `send(hooks=…)` pass Claude Agent SDK hook callbacks for one
  turn, so a caller can observe or gate every individual tool call — a
  `PreToolUse` hook fires under every `permission_mode`, unlike the SDK's
  `can_use_tool`, which the default `bypassPermissions` shadows entirely. Hooks
  are live callables, so they ride the run plane next to `secrets` rather than a
  config overlay (configs are serialized data) and are `local`-only: `sandbox`
  runs the turn in a remote worker and rejects them, and the `codex` harness
  fails the turn rather than run it with the hooks never called ([#25]).
- `local.deploy(spec, workspace=…)` (and `local.run(…, workspace=…)`) runs every
  session of that engine in a directory you name instead of a per-session
  `<workdir>/jobs/<session-id>/workspace`. The cwd is part of the agent's system
  prompt, so a suite of short sessions pays prompt-cache creation on every one of
  them; one shared cwd keeps the prefix identical and the cache warm (measured 2x
  on a 111-attempt suite). The directory is yours: sessions are no longer isolated
  from each other there, `repos` is rejected (they would all clone to the same
  path), and `checkpoint=True` checkpoints the conversation only, so a resume
  continues in the directory as it stands instead of restoring a snapshot over it.
  `sandbox.deploy(workspace=…)` raises — a worker's cwd is its own `/tmp` ([#29]).
- `run_harness_conformance()` is the `Harness` port's conformance suite, the
  sibling of `run_session_store_conformance()`: hand it a callable that runs one
  turn and it asserts what callers read off `AgentEvent.raw` beyond the event's
  own fields — an `init` status event and the terminal `result` event's spend
  (a float, or `None` when the backend cannot price the run), usage and turn
  count. `raw` is documented as a pass-through, so a harness or SDK reshaping
  it broke downstream readers silently; run this against your own `Harness`
  implementation, or against a shipped one after a toolkit or SDK upgrade, to
  catch that at test time. `run_claude_code_harness_conformance()` layers
  Claude Code's stricter promise on top — the `init` event's backend payload
  under `raw["data"]`, with the session id in it — which Codex's `init` event
  does not carry ([#27]).

### Changed

- Claude Code runs now pass `--strict-mcp-config`: MCP servers come from
  `mcp_servers` in the spec/configs only. MCP config is loaded outside the
  setting sources, so the existing project-only isolation did not stop the CLI
  from picking up a `.mcp.json` sitting in the workspace — and the workspace
  is a cloned repo, which could ship a `stdio` server (an arbitrary command)
  for the agent to run. A repo whose `.mcp.json` servers *are* wanted must
  mirror them into `mcp_servers` ([#21]).
- The pinned engine `google-cloud-aiplatform` is now 1.165.1 (Google's release
  made fresh venvs drift past the old `==1.164.0` pin, failing
  `verify_deploy_env()` — and CI — by design). Deploy venvs must carry 1.165.1
  from now on (`uv sync` suffices); already-deployed engines are unaffected,
  the pin is baked into their image. Verified with a clean `make live-smoke`
  ([#24], [#33]).
- The default idle life of a warm-pool worker is now a day, up from 30
  minutes. An idle-expired worker exits **without replacement**, and a pool
  that drains to empty never self-recovers (the post-dispatch refill worker
  claims the pending dispatch itself) — so the short default silently turned
  any pool quiet for half an hour permanently cold. Note the cost implication
  on redeploy: idle workers now bill for up to a day; pass `pool_max_wait_s`
  to dial it back for pools with steady traffic ([#28]).

### Fixed

- A single oversized message on the Claude Code CLI's stdout no longer destroys the
  whole turn. The Agent SDK caps one NDJSON message at 1 MiB and raises from inside
  its read loop when a message exceeds it, so the harness generator died mid-run and
  the turn ended as `agent run failed: Failed to decode JSON: JSON message exceeded
  maximum buffer size of 1048576 bytes` with no result — everything the agent had
  done was discarded. Nothing plumbed the SDK's `max_buffer_size` option, so the
  1 MiB default was in force; an agent that `Read`s a screenshot (base64-encoded into
  one line) or gets a large tool result hit it routinely. The cap is now
  `AgentSpec.max_buffer_size`, default 32 MiB, and settable per session or per turn
  like the other invocation knobs. `claude-code` only: the Codex app-server SDK
  frames its own stream and has no equivalent ([#32]).
- Structured output is no longer lost when a background-task notification arrives
  after the agent has already delivered its answer: the model's reply to the stale
  notification became the turn's final message, and structured parsing — which reads
  the final message only — returned nothing, so a successful run read as having no
  deliverable. The final result event now carries the displaced segment-boundary
  results' texts (`raw["segment_summaries"]`, newest first) and result building
  falls back over them; `RunResult.structured_output_recovered` marks a recovered
  value ([#20]).
- An unset `system_prompt` now gives the Claude Code harness its own preset
  prompt, instead of an *empty* system prompt (the Agent SDK's `None` means
  `--system-prompt ""`). Previously a default spec produced a bare agent on
  non-interactive runs — no Claude Code identity, no `CLAUDE.md` — while
  interactive runs got the full preset; now `None` behaves like
  `SystemPrompt.inherit()` on both harnesses. Note the behavior change on
  redeploy: default-spec agents get the preset prompt and start honoring
  `CLAUDE.md` from cloned repos ([#22]).
- A second `local.deploy()` into the same `workdir` no longer crashes on
  provisioning the engine venv (uv ≥ 0.12 errors on `uv venv` over an existing
  venv), which broke cross-process re-attach — `get_session`, checkpoint
  resume — for specs with `packages`. The existing venv is now reused (declared
  packages are still re-applied onto it), so an agent's mid-run installs
  survive re-attach; the `UV_VENV_CLEAR=1` workaround discarded them. Reuse
  requires the venv's Python to match the engine contract's at major.minor
  (checked via `pyvenv.cfg`) — an out-of-contract or interpreter-less venv is
  re-provisioned from scratch. Note convergence is one-way: packages *removed*
  from the spec stay installed in a reused venv; delete the workdir's `venv/`
  to rebuild from the spec alone ([#23]).
- The synchronous `local.run(spec, message, …)` convenience now forwards
  `config` and `hooks` to the turn it runs; both were silently swallowed by the
  backend-symmetry `**_` catch-all, so a `TurnConfig` passed there had no
  effect ([#25]).

### Backwards-incompatible

- `get_engine(spec=…)` is removed, with no deprecation shim — passing `spec`
  raises a `TypeError` pointing at the replacement ([#16]). It was
  only ever a client-side parsing hint; runs always executed the
  deploy-baked spec. **To update:** `get_engine(…)` now only addresses an
  engine; bind a `SessionConfig` at `start_session` for anything the old
  spec was supposed to change — including `output_schema`/`checkpoint`,
  which also restore the client-side structured parsing and idle
  stop-reason the old hint provided. Per-turn knobs go to
  `run(config=TurnConfig(…))`.
- Engines deployed from an older toolkit revision keep working with new
  client code **as long as you pass no configs** (nothing config-related
  rides the invocation then). To *use* `SessionConfig`/`TurnConfig` against
  one, redeploy it first: an old worker predates the fail-closed contract,
  so it would silently run its baked spec instead (and on the cold path the
  unrecognized directive line would additionally leak into the prompt).

[#16]: https://github.com/zytedata/remote-agent-toolkit/pull/16
[#20]: https://github.com/zytedata/remote-agent-toolkit/pull/20
[#21]: https://github.com/zytedata/remote-agent-toolkit/pull/21
[#22]: https://github.com/zytedata/remote-agent-toolkit/pull/22
[#23]: https://github.com/zytedata/remote-agent-toolkit/pull/23
[#24]: https://github.com/zytedata/remote-agent-toolkit/pull/24
[#25]: https://github.com/zytedata/remote-agent-toolkit/pull/25
[#26]: https://github.com/zytedata/remote-agent-toolkit/pull/26
[#27]: https://github.com/zytedata/remote-agent-toolkit/pull/27
[#28]: https://github.com/zytedata/remote-agent-toolkit/pull/28
[#29]: https://github.com/zytedata/remote-agent-toolkit/pull/29
[#32]: https://github.com/zytedata/remote-agent-toolkit/pull/32
[#33]: https://github.com/zytedata/remote-agent-toolkit/pull/33

## 0.1.0 — 2026-08-07

Initial release. What's in the box:

### Agent definition

- `AgentSpec`: one declarative spec for an agent — system prompt, skills
  (local or packaged), MCP servers (local or remote), extra Python/apt
  packages, git repositories to provision (with host-aware auth), per-run
  secrets, structured `output_schema`, model override, `reasoning_effort`
  ([#3]), and container `resource_limits` (CPU/memory, [#6]).
- Harnesses: Claude Code (via the Claude Agent SDK) and Codex (OpenAI models,
  [#2]) behind the same `AgentSpec`/engine/session API.

### Run planes

- Local: run the same spec on your machine for fast iteration; a dev parity
  image mirrors the engine's install contract.
- Gemini Agent Runtime (Vertex AI Agent Engine): `deploy()` /
  `get_engine()` / sessions with multi-turn `run()`, real interrupt,
  checkpoint/resume (canonical-UUID session ids, [#1]), and agent versions —
  `deploy` creates revisions and `get_engine(version=…)` pins one ([#13]).
- Warm pool: pre-provisioned workers claimed per turn for lower latency,
  with readiness signaling, pre-warm tuning, and robust cross-process
  teardown.
- Live event streaming over a GCS mirror (quota-free); Cloud Logging is
  emit-only. Clock-free turn boundaries prevent stale-result replay on fast
  resume. Out-of-order live logs are reconciled on history recovery
  ([#10], [#15]).
- Worker CPU/RAM self-sampling with `Session.resource_samples()` for OOM
  forensics ([#7]).

### Deploy reliability

- The platform-critical engine layer (aiplatform, cloudpickle, pydantic,
  google-adk, telemetry stack, …) is pinned via an in-tree
  `constraints.txt`, merged into engine requirements client-side;
  `verify_deploy_env()` fails fast when the deploying venv drifts from the
  pickle-coupled pins ([#18]).
- The deploy target project/location is pinned into the pickled `AdkApp`
  (`build_adk_app`), so engine telemetry can no longer follow the deployer's
  stray local gcloud default into the wrong project (persistent span/metric
  export 403s) ([#17]).
- Retry-safe secrets handoff: per-invocation secrets are deleted at turn
  end, not on first read ([#5]).

[#1]: https://github.com/zytedata/remote-agent-toolkit/pull/1
[#2]: https://github.com/zytedata/remote-agent-toolkit/pull/2
[#3]: https://github.com/zytedata/remote-agent-toolkit/pull/3
[#5]: https://github.com/zytedata/remote-agent-toolkit/pull/5
[#6]: https://github.com/zytedata/remote-agent-toolkit/pull/6
[#7]: https://github.com/zytedata/remote-agent-toolkit/pull/7
[#10]: https://github.com/zytedata/remote-agent-toolkit/pull/10
[#13]: https://github.com/zytedata/remote-agent-toolkit/pull/13
[#15]: https://github.com/zytedata/remote-agent-toolkit/pull/15
[#17]: https://github.com/zytedata/remote-agent-toolkit/pull/17
[#18]: https://github.com/zytedata/remote-agent-toolkit/pull/18
[#44]: https://github.com/zytedata/remote-agent-toolkit/issues/44

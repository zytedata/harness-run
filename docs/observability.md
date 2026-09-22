# Observability

What a run leaves behind and how to read it back: the durable event record, harness transcripts,
cost and token accounting, and the sandbox's own resource samples.

## Past jobs: listing sessions & reading history

Every `sandbox` run leaves a durable record, and the library reads it back — from any process, long after
the run:

```python
engine = sandbox.get_engine("code-agent", project=..., location=...)

for info in engine.list_sessions():            # newest first: {"session_id", "sources", "last_file"}
    session = engine.get_session(info["session_id"])
    events = session.history()                 # the persisted AgentEvents, oldest first
    result = session.last_result               # reconstructed from the terminal event (None if unfinished)
    print(info["session_id"], result and result.text)
```

Where the record lives: the **event mirror** — `gs://<output_bucket>/events/<session_id>/*.jsonl`, small
batch files the worker writes with the run-scoped token as the turn runs, keeping **all turns** of a
multi-turn session. It is also the client's fallback stream when a sandbox stops answering (an OOM shows up
as the sandbox going away mid-turn: the run ends with an explained `sandbox_unreachable` error result unless
the mirror already holds the terminal result).

`session.last_result` on a re-attached session reads the record on first access and adopts a turn still
running there (["From another process"](runs-and-sessions.md#talking-to-a-running-turn-steer-interrupt-stop-exec) above); without an event loop it
re-reads the record on each access until a result exists — prefer `current_run` / `run.done` for
in-flight runs. Caveats: the mirror is bucket-wide, so engines
sharing an output bucket see each other's sessions in `list_sessions`; the `local` runtime keeps no durable
event log (`list_sessions` shows its workdir's session dirs; `history()` raises).

### Reading the harness transcript

`AgentEvent`s are summaries. When you need the harness's own record — per-turn usage, tool statuses,
subagent trees, permission denials — deploy with `transcript=True` and read it back on both runtimes:

```python
spec = AgentSpec(name="scorer", model="claude-sonnet-4-6", transcript=True)
...
transcripts = await session.transcripts()   # {"main": [...], "subagents/<id>": [...], ...}
```

`checkpoint=True` implies it (resume needs the transcript), but `transcript=True` alone adds **no
workspace snapshot** — a run with a multi-hundred-MB working directory pays for the transcript only. It is
purely observational: continuing a conversation across turns (`send`) still takes `checkpoint=True`. The
`codex` harness keeps its conversation under `checkpoint` alone, so `transcripts()` reads back `{}` there
(the run says so, as a `spec_warning` status event). Without either flag `transcripts()` raises; `{}` means
persistence is on and nothing has been written yet.

Unlike events, a transcript is verbatim: prompts, tool inputs and outputs, anything the agent saw or
echoed — including a secret a coerced agent printed. Treat the checkpoint prefix, and everything
`transcripts()` returns, as sensitive.

## What you get, what you don't

The sandbox runtime keeps **one record**: the event mirror (`session.history()`, and the live stream).
There is no Cloud Logging step log and no Cloud Trace span tree any more — the sandbox has no Google
identity to ship telemetry with. What replaces them:

- **Events are the record.** Every turn streams `turn_started` (with the sandbox id), `effective_spec`,
  `workspace_ready`, the harness events and the terminal `result`; `session.history()` reads the same
  objects back from any process. Large payloads survive (the mirror has no per-entry size cap; the live
  channel pages).
- **A sandbox that dies** (OOM, TTL, deleted elsewhere) ends the run with an explained
  `sandbox_unreachable` error result within ~30 s — never a hang, never a silent retry. Raise
  `resource_limits` for agents whose turns build dependencies or import big projects (up to 8 vCPU;
  16 GiB verified).
- **Cost** is reported per run as `result.cost_usd` as before.

**CPU / memory.** The platform exposes **no** resource metrics for sandboxes (no Cloud Monitoring metric
type, no usage fields on the sandbox resource — checked 2026-09-16), so the worker samples **itself**: gVisor
mounts cgroup v1 accounting and reports the template's memory limit as `MemTotal`. Three outputs:

- **Per-session samples** — every 20 s (`HARNESS_RUN_RESOURCE_SAMPLE_S` in the image env; `0` disables) to the
  session's **event mirror only**, not the live stream, so a watcher is not drowned and the record survives a
  mid-turn kill: after an OOM the last sample sits at most one interval before death. `session.history()`
  skips them (`history(include_samples=True)` keeps them); read them as rows with:

  ```python
  session = engine.get_session("<session-id>")      # or any session you already hold
  for row in session.resource_samples():            # oldest first
      print(row["time"], row.get("memory_current_bytes"), row.get("memory_limit_bytes"), row.get("cpu_usec"))
  ```
- **A visible warning** — the first time memory crosses 85 % of the limit, a `memory pressure: …` status
  event lands on the normal event stream, so a watcher sees trouble before the platform kills the sandbox
  at the limit (the fix: deploy with higher `resource_limits`).
- **Peak in every result** — `result.resources` (and the terminal result event's `raw`) carries
  `memory_peak_bytes` / `memory_limit_bytes` / `cpu_usec`, so completed *and failed* turns report their
  high-water mark for free.

The idle baseline with Claude Code is ~550 MiB (the CLI ~500 MiB, the worker ~110 MiB); what the samples
show above that is your agent's own work.

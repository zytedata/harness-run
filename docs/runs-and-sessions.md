# Sessions and runs

`session.run(message)` starts a turn and returns a `Run`; `session.send(message)` continues the
conversation, or talks to a turn that is still running. Both work the same on `local` and `sandbox`.

## Consuming a run: wait, stream, or poll

`session.run(msg)` (and `session.send(msg)` to resume) returns a `Run` handle, consumable three ways — the
same on `local` and `sandbox`:

```python
# (a) wait for completion (simplest — the headline default). Pass any credentials the run
#     needs as per-invocation secrets (name → value); see "Secrets & security" below.
result = await session.run("Fix the failing tests", secrets={"SERVICE_API_KEY": os.environ["SERVICE_API_KEY"]})

# (b) stream events as they happen (UI / live logs)
async for ev in session.run("Fix the failing tests"):
    print(ev.kind, ev.summary)               # thinking | tool_use | tool_result | message | status | result
result = session.last_result                 # populated once the run completes

# (c) poll — e.g. a web handler that can't hold the connection, or another process
run = session.run("Fix the failing tests")             # returns immediately
while not run.done:
    await asyncio.sleep(2)
    print(run.status)                        # pending | running | idle | terminated
result = run.result

# re-attach later / elsewhere by session id, then poll or continue
session = engine.get_session(session_id)
if (run := session.current_run) is not None:             # a turn still running (even under another process)
    async for ev in run:
        print(ev.kind, ev.summary)
if session.status == "idle" and session.stop_reason == "needs_input":
    await session.send("yes, that schema looks right")     # resumes the conversation (a fresh turn)
```

**Results are kept honest.** `RunResult` keeps its accounting on *error* results too — an
`error_max_turns` run reports its real `cost_usd`/`num_turns`/`usage` (it spent right up to its limit;
treating it as free corrupts budget bookkeeping). And the terminal result is authoritative: if the
underlying `claude` CLI exits non-zero *after* the final result was streamed (it happens transiently even
on finished runs), the run keeps its result and gets `result.warning` set instead of being voided into an
error. For post-mortems, the CLI's stderr is captured to `<workdir>/jobs/<session-id>/stderr.log` (beside,
not inside, the workspace); on a CLI-exit failure its tail is also surfaced as a `claude_stderr` status
event and embedded in the error text.

**What the numbers count.** `result.cost_usd` is the turn's total spend in dollars (or `None` when the
backend can't price it — never a guess). `result.usage` is the library-normalized token record, identical
in shape and meaning on every harness and both runtimes: a flat dict whose five keys are always present —
`input_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`, `output_tokens`,
`reasoning_output_tokens`. The three input buckets are disjoint (fresh / cache-read / cache-written), and
`reasoning_output_tokens` is the reasoning *share* of `output_tokens`, not a further bucket — so total
tokens = the three input buckets + `output_tokens`. A key is `None` when the backend doesn't report that number
(Anthropic has no reasoning split; OpenAI doesn't bill cache writes separately, the count is
informational). Both metrics cover the **whole turn**: subagent sessions and background-task
re-invocation segments included — on Claude Code aggregated from the CLI's complete per-model record, on
Codex recovered from the subagent rollout files (Codex reports no subagent usage on its wire). The
backends' own verbatim records stay on the result event's `raw`: `model_usage` and `cli_usage` on Claude
Code, `subagent_usage` (per-thread) on Codex. `result.num_turns` counts the **main agent's** model calls
(same unit on both harnesses): cumulative across re-invocation segments, subagent calls **not** included —
the same scope `max_turns` caps.

**Background tasks are honored.** If the agent starts a background job (Bash `run_in_background`) or arms
the Monitor tool and then ends its turn — the trained, efficient behavior for waiting on long processes
like a long test suite — the run does **not** end there: the harness holds the session open and the
model is re-invoked when the task completes, exactly as in interactive Claude Code. Interim turn ends
surface as `awaiting_tasks` status events; the run's single `result` comes when no work is pending.
`spec.background_task_timeout` (seconds, default 3600) bounds how long a turn waits on still-running
tasks — size it to the longest job the agent legitimately waits on. Two accounting notes: `num_turns`,
`cost_usd` and `usage` are cumulative across these re-invocations (see "What the numbers count"), and
`max_turns` caps each invocation segment rather than the
whole run (`max_budget_usd` remains a global cap). Event-driven waiting instead of poll-loops is exactly
what keeps long waits nearly free in turns.

On `sandbox`, live events and recovery records carry the submitted turn's ID. Reattaching a
session and immediately sending again cannot replay an earlier result. Per-worker warm
dispatch also matches the addressed worker: only its `turn_started` acknowledges pickup,
and late events from abandoned workers are ignored. Every storage poll is time-bounded.
Cold jobs and per-worker warm jobs have a watchdog that recovers a matching durable result
after job termination, or reports an error after its grace period. Legacy shared pools
have no per-worker job handle and rely on bounded polling.

When upgrading to turn IDs, redeploy engines before updating clients. Submission fails
before dispatch if the serving revision does not advertise the protocol. With pinned or
split traffic, every serving revision must support it. Older clients can drive upgraded
workers during the rollout; `Session.history()` continues to return all turns.

## Talking to a running turn: steer, interrupt, stop, exec

The session API is turn-based, and an interactive loop like Claude Code's needs two more things while
a turn is **running**: send a message into it, and cut it short without losing the session (a third,
looking into its workspace, is [below](#looking-into-a-running-turn-exec)). Both work the same on
`local` and `sandbox`, and from any process that holds the session id (`engine.get_session(session_id)`):

```python
run = session.run("Add a CLI entry point")

# The operator changes their mind while the agent works. Same Run, same worker, same workspace:
session.send("Actually skip the docs, only the code changes")            # steer: the model sees it at its next step
session.send("Stop that approach, reuse the existing helper instead", interrupt=True)  # interrupt, then continue from this message

# Or just stop. The turn ends through its normal end-of-turn path (checkpoint, transcript, accounting):
await session.interrupt()
assert session.stop_reason == "interrupted"     # idle and resumable
await session.send("OK, now do X")              # a normal turn, from the interrupted turn's checkpoint
```

- **`send()` on a running session** delivers the message into the running turn and returns the **same
  `Run`**: its events keep flowing and its single `result` covers everything. With `interrupt=False`
  the message is queued for the model's next step (both harnesses: Claude Code's mid-turn `query()`,
  Codex's `turn/steer`). With `interrupt=True` the model is interrupted first and continues from the
  message in the same harness session — no checkpoint round-trip, no second job. On an idle session
  `send()` is the usual resume (`interrupt` is ignored).
- **The `user` event is the acknowledgement.** Each delivered message appears on the stream, in the
  mirror and in `history()` as an event of kind `user` (the text as `summary`; `raw["message_id"]`,
  `raw["interrupt"]`), emitted when the harness hands it to the model. Pass your own
  `message_id=` to match it up; the worker dedupes on it, so a retried send after re-attaching never
  reaches the model twice. Until that event arrives the message is "waiting" — on `sandbox` well under a
  second to reach the worker, plus whatever tool call the model is in the middle of.
- **A message sent while the turn is still starting is queued, not refused.** Between `run()` and
  the worker's `control_ready` event (~1 s from the ready pool, 15–25 s when a sandbox has to be
  created) the session holds the message and posts it the moment the worker is ready — in order,
  acknowledged by the same `user` event, on the same `Run`. An `interrupt=True` message queued that
  early interrupts the harness right after it starts, so the model's first step is the message. If
  the turn ends before the queue drained (dispatch failed, the sandbox died) the messages are dropped
  and `result.warning` says how many. No retry loop needed on the caller's side.
- **`interrupt()` is "interrupt and stop"**: the harness stops what the model is doing and the turn
  ends with `StopReason.INTERRUPTED` — not an error; `cost_usd` / `usage` / `num_turns` are real, the
  checkpoint includes the interrupted turn's workspace changes, and the next `send()` resumes from it.
  Only when the worker cannot be reached (an engine deployed before this existed, or a cold job that
  has not started yet) or does not stop within the timeout does it fall back to cancelling the run:
  an error result whose `warning` says so, and no checkpoint.
- **Guards.** `run()` on a running session raises: a second concurrent turn under one session id would
  corrupt its checkpoint and transcript. `secrets` / `config` / `hooks` cannot change mid-turn and are
  rejected on a running `send()`. `send()` raises `ControlUnavailable` (a typed exception; nothing was
  delivered) only when a *ready* worker did not take the message: the platform proxy failed, or the
  turn ended between your `busy` check and the post. Mind that race on your side: `send()` on a session
  that has just gone idle is a resume, i.e. a new turn with no secrets — check `session.busy` right
  before, and catch `ControlUnavailable`.
- **From another process (`sandbox`).** `engine.get_session(id)` in the process that started the turn
  returns the live session. In a *different* process (a poller, or a worker adopting a job whose
  owner died mid-turn) the session holds no run, so its first `send()`, `interrupt()`, `exec()`,
  `run()`, `current_run`, `busy` or `last_result` reads the session's event record once — every
  event names its turn and sandbox — and, if a turn is still running there, **adopts** it: its
  events replay from the worker and then flow live on the session's `Run` — `session.current_run`
  hands you that `Run`, so an adopter consumes the events exactly as the owner would (`async for ev
  in run`, `await run`) instead of polling `last_result` — `send()` steers it (no second turn is
  started; `run()` raises as it would locally), `interrupt()` stops it, `exec()` probes it, and
  `last_result` lands at its end. The adopter also keeps the turn's tokens fresh and deletes the
  sandbox at the result (a few seconds late, in case the owner is still reading), so a job whose
  owner died is cleaned up — a poller that only asks `busy` or `last_result` gets the same, so a
  long turn does not lose model access for want of the right accessor. Adoption drives the turn on
  the running event loop; a plain sync poller with no loop keeps re-reading the record on each
  `last_result` until the result appears. An adopter that merely exits leaves the turn to its owner.
  With nothing running there `send()` is the usual resume. Same-process re-attach needs none of
  this: the engine hands back the live session.
- **Transport (`sandbox`).** The client POSTs the message to the worker's `/control` endpoint through
  the platform's proxy (sub-second); the worker feeds it to the harness and dedupes on `message_id`.
  The worker announces the channel with a `control_ready` status event at the start of the turn;
  until then the message waits in the client's queue (above). A message can only reach a running
  turn — nothing carries over to the session's next turn. On `local` the same channel is an
  in-process queue.

The interactive system-prompt suffix (`checkpoint=True`) still tells the model to end its turn for a
genuine decision; steering is additive — the way to talk to an agent that is *already* working.

### Looking into a running turn: `exec()`

`await session.exec(command)` runs a shell command in the running turn's workspace and returns its
output — a read-only probe for showing the agent's work *while* it works. The events are not enough
for that: a Claude Code edit arrives as old/new strings without context, a Codex change as paths only,
and a `git checkout` or a shell one-liner escapes both. One `git diff` in the workspace is exact and
harness-independent:

```python
run = session.run("Fix the failing tests")
while not run.done:
    r = await session.exec(
        "git add -A --intent-to-add . && git diff origin/main",   # what the agent has changed so far
        timeout=30,
    )
    if r.ok:
        render(r.stdout)                                          # e.g. refresh a change panel
    await asyncio.sleep(10)
```

- **What it returns.** `ExecResult(stdout, stderr, returncode, truncated, duration_ms)` (`r.ok` is
  `returncode == 0`). The command runs under `/bin/bash -c` with the agent's working directory as cwd
  (`cwd=` relative to it, or absolute). `timeout` in seconds (default 60; on `sandbox` at most 240,
  the platform proxy cuts a call at ~300 s) kills the command and reports `returncode` 124. Output is
  kept up to 500 000 characters per stream and `truncated` says when it was cut — a probe never
  fails for printing too much. The command's own failure is data, not an exception: read
  `returncode` and `stderr`.
- **Only a running turn has a workspace.** On `sandbox` the sandbox exists for the turn's duration
  and is deleted at the terminal event; `local` keeps the same rule so code written against it
  behaves the same remotely (between turns, read `session.workspace` directly there). Called right
  after `run()`, before the sandbox is known, `exec()` waits for the worker (~1 s from the ready
  pool, 15–25 s on a fresh sandbox) — so a poller can start at once. Called with no turn running,
  or after the turn ended, it raises `ControlUnavailable`; a poller treats that as "the turn is
  over" and reads `run.result`.
- **Read-only is your side of the contract.** Nothing polices the command; one that writes into
  the workspace races the agent.
- **From another process it works too.** A re-attached session's `exec()` first adopts the turn
  running there (see [From another process](#talking-to-a-running-turn-steer-interrupt-stop-exec))
  and probes its sandbox; with nothing running it raises `ControlUnavailable`, as usual.
- **Transport (`sandbox`).** The worker's `/exec` endpoint through the platform proxy (~0.3 s plus
  the command itself); the same endpoint `dev/live_smoke.py` uses to test the sandbox's isolation.

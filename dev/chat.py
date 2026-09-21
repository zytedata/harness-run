"""Local chat UI for trying turn control by hand (issue #44): steer, interrupt, stop.

Same status as the live probes in this directory: a dev tool, not part of the library.
One page, one local engine (``local.deploy``), one session at a time. Type while the agent
works and the message is steered into the running turn; "Interrupt & send" interrupts
first; "Stop" interrupts and leaves the session resumable. Events stream over SSE.

COSTS REAL MONEY (whatever the model you chat with charges) and needs ``ANTHROPIC_API_KEY``
/ ``OPENAI_API_KEY``. Needs ``fastapi`` and ``uvicorn`` in the venv (not in the dev group).
Run from an unsandboxed shell: the Claude CLI's Bash tool writes under ``~/.claude``.

    make chat                        # claude-code, Haiku
    HARNESS=codex make chat
    .venv/bin/python dev/chat.py --harness codex --model gpt-5.6-luna --port 8765

then open http://127.0.0.1:8765 .
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, is_dataclass

from agent_run import AgentSpec, local
from agent_run.events import RunStatus

try:
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
except ImportError as exc:
    sys.exit(f"dev/chat.py needs fastapi and uvicorn ({exc}); install them into the venv: "
             ".venv/bin/pip install fastapi uvicorn")

DEFAULT_MODELS = {"claude-code": "claude-haiku-4-5-20251001", "codex": "gpt-5.6-luna"}

parser = argparse.ArgumentParser()
parser.add_argument("--harness", default="claude-code", choices=["claude-code", "codex"])
parser.add_argument("--model", default=None)
parser.add_argument("--workdir", default=None, help="engine workdir (default: a temp dir)")
parser.add_argument("--port", type=int, default=8765)
parser.add_argument("--max-budget", type=float, default=2.0)
args = parser.parse_args()

MODEL = args.model or DEFAULT_MODELS[args.harness]
WORKDIR = args.workdir or tempfile.mkdtemp(prefix="agent-run-chat-")
SPEC = AgentSpec(
    name=f"chat-{args.harness}", model=MODEL, harness=args.harness, checkpoint=True,
    max_turns=60, max_budget_usd=args.max_budget,
)
ENGINE = local.deploy(SPEC, workdir=WORKDIR)
SECRETS = {k: os.environ[k] for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY") if k in os.environ}

app = FastAPI()


class Conversation:
    """One session plus the SSE broadcast of everything that happens in it."""

    def __init__(self) -> None:
        self.session = ENGINE.start_session()
        self.subscribers: list[asyncio.Queue] = []
        self.log: list[dict] = []  # everything published, for late subscribers
        self.total_cost = 0.0
        self.turns = 0
        self.pump: asyncio.Task | None = None

    # -- publish / subscribe --------------------------------------------------------
    def publish(self, item: dict) -> None:
        item.setdefault("t", time.time())
        self.log.append(item)
        for q in list(self.subscribers):
            q.put_nowait(item)

    def state(self) -> dict:
        s = self.session
        return {
            "type": "state",
            "t": time.time(),
            "session_id": s.session_id,
            "status": s.status.value,
            "stop_reason": s.stop_reason.value if s.stop_reason else None,
            "busy": s.busy,
            "total_cost": self.total_cost,
            "turns": self.turns,
            "workspace": str(s.workspace),
            "harness": args.harness,
            "model": MODEL,
        }

    # -- actions --------------------------------------------------------------------
    def send(self, message: str, *, interrupt: bool, message_id: str) -> str:
        s = self.session
        if s.busy:
            s.send(message, interrupt=interrupt, message_id=message_id)
            mode = "interrupt" if interrupt else "steer"
            self.publish({"type": "sent", "mode": mode, "message": message, "message_id": message_id})
            self.publish(self.state())
            return mode
        run = s.send(message, secrets=SECRETS) if s.last_result or s.status != RunStatus.PENDING \
            else s.run(message, secrets=SECRETS)
        self.turns += 1
        self.publish({"type": "sent", "mode": "turn", "message": message, "message_id": message_id})
        self.publish(self.state())
        self.pump = asyncio.ensure_future(self._pump(run))
        return "turn"

    async def _pump(self, run) -> None:
        try:
            async for ev in run:
                self.publish({"type": "event", "kind": ev.kind, "summary": ev.summary,
                              "raw": _jsonable(ev.raw), "cost_usd": ev.cost_usd, "usage": ev.usage})
        except Exception as exc:  # noqa: BLE001
            self.publish({"type": "event", "kind": "status", "summary": f"stream error: {exc}", "raw": {}})
        result = run.result
        if result is not None:
            if result.cost_usd:
                self.total_cost += result.cost_usd
            self.publish({"type": "result", "text": result.text, "is_error": result.is_error,
                          "cost_usd": result.cost_usd, "num_turns": result.num_turns,
                          "warning": result.warning,
                          "stop_reason": self.session.stop_reason.value if self.session.stop_reason else None})
        self.publish(self.state())

    async def interrupt(self) -> None:
        self.publish({"type": "sent", "mode": "stop", "message": "", "message_id": str(uuid.uuid4())})
        t0 = time.monotonic()
        await self.session.interrupt()
        self.publish({"type": "stopped", "seconds": round(time.monotonic() - t0, 2)})
        self.publish(self.state())


def _jsonable(x):
    if is_dataclass(x):
        x = asdict(x)
    try:
        json.dumps(x)
        return x
    except (TypeError, ValueError):
        return json.loads(json.dumps(x, default=str))


CONVOS: dict[str, Conversation] = {}


def _get(sid: str) -> Conversation:
    if sid not in CONVOS:
        raise KeyError(sid)
    return CONVOS[sid]


@app.post("/api/session")
async def new_session():
    c = Conversation()
    CONVOS[c.session.session_id] = c
    return c.state()


@app.get("/api/state")
async def state(sid: str):
    return _get(sid).state()


@app.post("/api/send")
async def send(req: Request):
    body = await req.json()
    c = _get(body["sid"])
    message_id = body.get("message_id") or str(uuid.uuid4())
    try:
        mode = c.send(body["message"], interrupt=bool(body.get("interrupt")), message_id=message_id)
    except Exception as exc:  # noqa: BLE001 - show ControlUnavailable and friends in the UI
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=409)
    return {"mode": mode, "message_id": message_id}


@app.post("/api/interrupt")
async def interrupt(req: Request):
    body = await req.json()
    c = _get(body["sid"])
    asyncio.ensure_future(c.interrupt())
    return {"ok": True}


@app.get("/api/events")
async def events(sid: str):
    c = _get(sid)
    q: asyncio.Queue = asyncio.Queue()
    for item in c.log:  # replay for a reconnecting tab
        q.put_nowait(item)
    q.put_nowait(c.state())
    c.subscribers.append(q)

    async def gen():
        try:
            while True:
                try:
                    item = await asyncio.wait_for(q.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                yield f"data: {json.dumps(item)}\n\n"
        finally:
            c.subscribers.remove(q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML


HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>agent-run chat</title>
<style>
  :root { --bg:#0f1115; --panel:#171a21; --line:#2a2f3a; --fg:#e6e6e6; --muted:#8b93a7;
          --user:#2b4c7e; --steer:#5b3f8c; --assist:#1f2a37; --ok:#2f9e44; --warn:#e8a33d; --err:#d9480f; }
  body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.45 system-ui, sans-serif; display:flex; flex-direction:column; height:100vh; }
  header { display:flex; gap:14px; align-items:center; padding:10px 16px; border-bottom:1px solid var(--line); background:var(--panel); flex-wrap:wrap; }
  header .pill { padding:2px 10px; border-radius:999px; font-weight:600; font-size:12px; background:#333; }
  .pill.running { background:#1c5f9a; } .pill.idle { background:#3b3f4a; } .pill.needs_input { background:var(--ok); }
  .pill.interrupted { background:var(--warn); color:#111; } .pill.error { background:var(--err); } .pill.pending { background:#3b3f4a; }
  header .muted { color:var(--muted); font-size:12px; }
  main { flex:1; overflow:auto; padding:16px; display:flex; flex-direction:column; gap:6px; }
  .row { max-width: 78%; padding:8px 12px; border-radius:10px; white-space:pre-wrap; word-break:break-word; }
  .row.user { align-self:flex-end; background:var(--user); }
  .row.steer { align-self:flex-end; background:var(--steer); border:1px dashed #b39ddb; }
  .row.steer .ack { display:block; font-size:11px; color:#d8ccf0; margin-top:4px; }
  .row.assistant { align-self:flex-start; background:var(--assist); }
  .row.thinking { align-self:flex-start; color:var(--muted); font-style:italic; font-size:12px; background:transparent; padding:2px 12px; }
  .row.tool { align-self:flex-start; font-family:ui-monospace, monospace; font-size:12px; color:#b9c2d0; background:#141821; border-left:3px solid #3a4a63; }
  .row.status { align-self:center; color:var(--muted); font-size:11px; background:transparent; padding:0 12px; }
  .row.status.important { color:var(--warn); font-weight:600; }
  .row.result { align-self:center; font-size:12px; color:#cfd8e3; border:1px solid var(--line); background:var(--panel); }
  .row.result.err { border-color:var(--err); }
  footer { border-top:1px solid var(--line); background:var(--panel); padding:10px 16px; display:flex; gap:8px; align-items:flex-end; }
  textarea { flex:1; min-height:44px; max-height:160px; resize:vertical; background:#0f1115; color:var(--fg); border:1px solid var(--line); border-radius:8px; padding:8px; font:inherit; }
  button { background:#2b4c7e; color:#fff; border:0; border-radius:8px; padding:10px 14px; font-weight:600; cursor:pointer; }
  button.alt { background:var(--steer); } button.stop { background:var(--err); } button.ghost { background:#333; }
  button:disabled { opacity:.4; cursor:not-allowed; }
  .hint { color:var(--muted); font-size:11px; padding:0 16px 8px; }
</style></head>
<body>
<header>
  <strong>agent-run chat</strong>
  <span id="pill" class="pill pending">no session</span>
  <span class="muted" id="meta"></span>
  <span class="muted" id="cost"></span>
  <span style="flex:1"></span>
  <button class="ghost" id="newsess">New session</button>
</header>
<main id="log"></main>
<footer>
  <textarea id="box" placeholder="Type a message… (Enter to send, Shift+Enter for a newline)"></textarea>
  <button id="send" title="Idle: start a turn. Running: the model sees it at its next step (same run).">Send</button>
  <button id="isend" class="alt" title="Interrupt the model now, then it continues from this message (same run).">Interrupt &amp; send</button>
  <button id="stop" class="stop" title="Interrupt and end the turn; the session stays resumable.">Stop</button>
</footer>
<div class="hint" id="hint"></div>
<script>
let sid = null, es = null, state = null;
const pending = new Map();   // message_id -> steer row waiting for its ack
const $ = id => document.getElementById(id);
const log = $('log');

function row(cls, text) {
  const d = document.createElement('div'); d.className = 'row ' + cls; d.textContent = text;
  log.appendChild(d); log.scrollTop = log.scrollHeight; return d;
}
function setState(s) {
  state = s;
  const pill = $('pill');
  let label = s.status;
  if (s.status === 'idle' && s.stop_reason) label = s.stop_reason;
  pill.textContent = label; pill.className = 'pill ' + label;
  $('meta').textContent = `${s.harness} · ${s.model} · session ${s.session_id.slice(0,8)} · ${s.workspace}`;
  $('cost').textContent = `turns ${s.turns} · $${s.total_cost.toFixed(4)}`;
  const busy = s.busy;
  $('send').textContent = busy ? 'Send (steer)' : 'Send';
  $('isend').disabled = !busy; $('stop').disabled = !busy;
  $('hint').textContent = busy
    ? 'The agent is working. Send = the model reads it at its next step. Interrupt & send = stop the current step first. Stop = end the turn, keep the session.'
    : (s.stop_reason === 'interrupted' ? 'Stopped. The session is resumable: your next message continues from the checkpoint.'
       : 'Idle. Your next message starts a turn.');
}
function onItem(it) {
  if (it.type === 'state') return setState(it);
  if (it.type === 'sent') {
    if (it.mode === 'turn') row('user', it.message);
    else if (it.mode === 'stop') row('status important', '⏹ stop requested');
    else { const r = row('steer', (it.mode === 'interrupt' ? '⚡ ' : '↪ ') + it.message);
           const a = document.createElement('span'); a.className='ack'; a.textContent = 'waiting for the model to take it…'; r.appendChild(a); pending.set(it.message_id, r); }
    return;
  }
  if (it.type === 'stopped') return row('status important', `⏹ stopped in ${it.seconds}s`);
  if (it.type === 'result') {
    const r = row('result' + (it.is_error ? ' err' : ''), `■ turn ended: ${it.stop_reason}` + (it.cost_usd != null ? ` · $${it.cost_usd.toFixed(4)}` : ' · cost unknown') + ` · ${it.num_turns} model calls` + (it.warning ? `\n⚠ ${it.warning}` : ''));
    return;
  }
  if (it.type !== 'event') return;
  const raw = it.raw || {}; const tag = raw.event || raw.subtype || '';
  switch (it.kind) {
    case 'message': row('assistant', it.summary); break;
    case 'thinking': row('thinking', '💭 ' + it.summary.slice(0, 300)); break;
    case 'tool_use': row('tool', '▶ ' + it.summary); break;
    case 'tool_result': { const c = raw.content; const s = typeof c === 'string' ? c : JSON.stringify(c ?? ''); row('tool', '◀ ' + it.summary + (s ? ': ' + s.slice(0, 200) : '')); break; }
    case 'user': {
      const r = pending.get(raw.message_id);
      if (r) { r.querySelector('.ack').textContent = '✓ delivered to the model'; pending.delete(raw.message_id); }
      else row('steer', '↪ ' + it.summary + ' (delivered)');
      break; }
    case 'result': break; // the 'result' item carries the summary line
    case 'status': {
      const important = ['interrupt_requested','interrupted_segment','awaiting_steer','checkpoint_saved','limit_interrupt','task_wait_timeout'].includes(tag);
      row('status' + (important ? ' important' : ''), it.summary); break; }
  }
}
async function api(path, body) {
  const r = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body || {})});
  const j = await r.json(); if (!r.ok) { row('status important', '✗ ' + (j.error || r.statusText)); throw new Error(j.error); } return j;
}
async function newSession() {
  const s = await api('/api/session'); sid = s.session_id; log.innerHTML = ''; pending.clear();
  if (es) es.close();
  es = new EventSource('/api/events?sid=' + sid); es.onmessage = e => onItem(JSON.parse(e.data));
  setState(s);
}
async function send(interrupt) {
  const text = $('box').value.trim(); if (!text || !sid) return;
  $('box').value = '';
  await api('/api/send', {sid, message: text, interrupt, message_id: crypto.randomUUID()});
}
$('send').onclick = () => send(false);
$('isend').onclick = () => send(true);
$('stop').onclick = () => api('/api/interrupt', {sid});
$('newsess').onclick = newSession;
$('box').addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(false); } });
newSession();
</script>
</body></html>
"""

if __name__ == "__main__":
    print(f"agent-run chat: harness={args.harness} model={MODEL} workdir={WORKDIR}", flush=True)
    print(f"open http://127.0.0.1:{args.port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")

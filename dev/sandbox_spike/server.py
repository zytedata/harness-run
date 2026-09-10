"""Sandbox-spike worker: the toolkit harness behind a tiny HTTP server (stdlib only).

This is the "runner" a custom-container Agent Sandbox needs: the platform's ``execute``
call forwards an HTTP POST (URI + port + JSON body) to the container, so anything that
answers JSON on the template's declared port works. Endpoints (all POST, JSON in/out):

* ``/health``  -> uptime, user, cwd facts (the client polls this until the sandbox answers)
* ``/exec``    -> ``{command, cwd?, timeout?}`` -> ``{stdout, stderr, returncode,
  duration_ms}`` — the same contract as Google's shell image, so the SDK's ``execute_bash``
  helper works against this container too
* ``/turn``    -> ``{turn_id, spec, prompt, env?, secrets?}`` starts ONE harness turn in a
  background thread via ``local.deploy``; 409 while a turn is running
* ``/events``  -> ``{turn_id, since}`` -> ``{events[since:], done, result, error}``

The sandbox has no ambient Google credentials on purpose; the client hands the model
token in through ``env`` (``CLAUDE_CODE_SKIP_VERTEX_AUTH=1`` + ``ANTHROPIC_AUTH_TOKEN``).
Nothing here logs env or secret values.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

BOOT = time.time()
WORKSPACE = os.environ.get("SPIKE_WORKSPACE", "/workspace")
_LOCK = threading.Lock()
_TURNS: dict[str, dict] = {}


def _health() -> dict:
    return {
        "ok": True,
        "uptime_s": round(time.time() - BOOT, 3),
        "pid": os.getpid(),
        "uid": os.getuid(),
        "cwd_writable": os.access(WORKSPACE, os.W_OK),
        "home_writable": os.access(os.path.expanduser("~"), os.W_OK),
        "python": sys.version.split()[0],
    }


def _exec(body: dict) -> dict:
    cmd = body.get("command")
    if not isinstance(cmd, str):
        raise ValueError("command (str) is required")
    cwd = body.get("cwd") or WORKSPACE
    timeout = body.get("timeout")
    t0 = time.time()
    try:
        p = subprocess.run(
            ["/bin/bash", "-c", cmd], cwd=cwd, capture_output=True, text=True,
            timeout=float(timeout) if timeout else None,
        )
        out, err, rc = p.stdout, p.stderr, p.returncode
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = f"command exceeded {timeout}s and was killed"
        rc = 124
    return {"stdout": out, "stderr": err, "returncode": rc, "duration_ms": int((time.time() - t0) * 1000)}


def _run_turn(turn_id: str, body: dict) -> None:
    rec = _TURNS[turn_id]
    try:
        from remote_agent_toolkit import AgentSpec
        from remote_agent_toolkit.runtime import local

        env = body.get("env") or {}
        os.environ.update({str(k): str(v) for k, v in env.items()})
        spec = AgentSpec(**body["spec"])
        workdir = body.get("workdir") or os.path.join(WORKSPACE, "ratk", turn_id)

        async def go() -> dict | None:
            engine = local.deploy(spec, workdir=workdir)
            session = engine.start_session()
            run = session.run(body["prompt"], secrets=body.get("secrets"))
            async for ev in run:
                with _LOCK:
                    rec["events"].append({
                        "t": time.time(), "kind": ev.kind,
                        "summary": " ".join((ev.summary or "").split())[:200],
                        "cost_usd": ev.cost_usd,
                    })
            r = run.result
            if r is None:
                return None
            return {"text": r.text, "is_error": r.is_error, "num_turns": r.num_turns,
                    "cost_usd": r.cost_usd, "session_id": r.session_id}

        rec["result"] = asyncio.run(go())
    except Exception:  # noqa: BLE001 — surfaced to the client as `error`
        rec["error"] = traceback.format_exc()[-3000:]
    finally:
        rec["done"] = True
        rec["finished_at"] = time.time()


def _start_turn(body: dict) -> tuple[int, dict]:
    turn_id = str(body.get("turn_id") or f"t{int(time.time() * 1000)}")
    with _LOCK:
        running = [t for t, r in _TURNS.items() if not r["done"]]
        if running:
            return 409, {"error": "a turn is running", "running": running}
        _TURNS[turn_id] = {"events": [], "done": False, "result": None, "error": None,
                           "started_at": time.time()}
    threading.Thread(target=_run_turn, args=(turn_id, body), daemon=True).start()
    return 202, {"turn_id": turn_id, "accepted_at": _TURNS[turn_id]["started_at"]}


def _events(body: dict) -> tuple[int, dict]:
    rec = _TURNS.get(str(body.get("turn_id")))
    if rec is None:
        return 404, {"error": "unknown turn_id"}
    since = int(body.get("since") or 0)
    with _LOCK:
        events = list(rec["events"][since:])
    return 200, {"events": events, "next": since + len(events), "done": rec["done"],
                 "result": rec["result"], "error": rec["error"],
                 "started_at": rec["started_at"], "finished_at": rec.get("finished_at")}


class Handler(BaseHTTPRequestHandler):
    server_version = "ratk-sandbox-spike/0.1"

    def _reply(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        if not raw.strip():
            return {}
        return json.loads(raw)

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path in ("/", "/health"):
            self._reply(200, _health())
        else:
            self._reply(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        try:
            body = self._body()
            if path == "/health":
                self._reply(200, _health())
            elif path == "/exec":
                self._reply(200, _exec(body))
            elif path == "/turn":
                self._reply(*_start_turn(body))
            elif path == "/events":
                self._reply(*_events(body))
            else:
                self._reply(404, {"error": "not found"})
        except Exception as exc:  # noqa: BLE001
            self._reply(400, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, fmt: str, *args) -> None:  # quiet: no request bodies in logs
        sys.stderr.write(f"{time.strftime('%H:%M:%S')} {self.command} {urlsplit(self.path).path}\n")


def main() -> None:
    port = int(os.environ.get("PORT", "8080"))
    os.makedirs(WORKSPACE, exist_ok=True)
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    sys.stderr.write(f"ratk sandbox spike server on :{port} (uid {os.getuid()})\n")
    srv.serve_forever()


if __name__ == "__main__":
    main()

"""The sandbox worker: the toolkit harness behind a small HTTP server (DESIGN.md §13.1).

This is the process a sandbox container runs (``CMD`` of the image ``_image.py`` builds).
The platform's ``execute`` call forwards an HTTP POST (URI + port + JSON) into the
container, so the whole worker contract is a handful of JSON-over-POST endpoints:

* ``/health``  — liveness + facts (uptime, uid, writable dirs, the running turn if any).
  The client polls it after creating a sandbox until the proxy routes.
* ``/turn``    — start ONE turn (``{turn_id, session_id, prompt, resume, session_config,
  turn_config, secrets, gcs, model_env, sandbox}``); ``{ok: false}`` while one runs.
  The worker overlays the configs on the image's **baked** spec exactly as the query-job
  worker did, echoes the merged ``effective_spec``, prepares the workspace (restore or
  skills + repos) and drives the harness. Every event goes to the in-memory turn record
  (served by ``/events``) AND to the GCS event mirror written with the run-scoped token
  (the durable record and the client's fallback stream).
* ``/events``  — ``{turn_id, since, wait}`` long-poll: held until events past ``since``
  exist or ``wait`` seconds pass; answers are paged under a byte budget (the proxy caps a
  response at ~2 MB, measured 2026-09-11).
* ``/control`` — ``{turn_id, op, message?, message_id?}``: steer / interrupt / stop into
  the running turn (the ``ControlChannel`` the harness reads; replaces the GCS inbox).
* ``/token``   — ``{turn_id, model_token}``: replaces the turn's model-token file (Claude Code
  does not re-read its token, so this only serves tooling that does; the client bounds a
  turn's model access by the token's lifetime instead, see ``model_token.py``).
* ``/exec``    — ``{command, cwd?, timeout?}`` → ``{stdout, stderr, returncode}``: the
  same contract as Google's shell image, for probes and debugging.

Application errors are answered with HTTP 200 and ``{"ok": false, "error": ...}`` — the
proxy's treatment of non-2xx worker answers is not documented, and the client only needs
the JSON. Nothing here logs env, secret or token values.

The container has no Google identity on purpose: GCS access runs on the run-scoped token
the client sends (``scoped_gcs.py``), model access on the model token in ``model_env``.
Stdlib HTTP server; the toolkit and the harness SDKs are imported inside the turn.
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
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ...control import ControlMessage
from ...events import AgentEvent

BOOT = time.time()
WORKSPACE_ROOT = os.environ.get("RATK_WORKSPACE_ROOT", "/workspace")
BAKED_SPEC_PATH = os.environ.get("RATK_BAKED_SPEC", "/opt/toolkit/spec.json")

# /events paging: the proxy rejects answers past ~2 MB ("Response size too large", measured
# at 2,002,042 bytes), so a page stops once its serialized events pass this budget. A single
# event whose summary alone exceeds it is truncated to fit.
EVENTS_PAGE_BYTES = 1_000_000
EVENT_SUMMARY_CAP = 200_000
# Longest a /events call is held open. The proxy allowed 120 s calls (measured); the client
# asks for less, this only bounds a misbehaving caller.
MAX_WAIT_S = 90.0



class _TurnControl:
    """Worker-side ``ControlChannel``: an asyncio queue on the turn's loop, fed from HTTP threads."""

    def __init__(self, loop: asyncio.AbstractEventLoop, lock: threading.Lock) -> None:
        self._loop = loop
        self._lock = lock
        self._queue: asyncio.Queue[ControlMessage] = asyncio.Queue()
        self._seen: set[str] = set()

    def send_threadsafe(self, msg: ControlMessage) -> bool:
        """Queue ``msg`` for the harness; ``False`` for a message id already delivered."""
        with self._lock:
            if msg.message_id in self._seen:
                return False
            self._seen.add(msg.message_id)
        self._loop.call_soon_threadsafe(self._queue.put_nowait, msg)
        return True

    async def receive(self) -> ControlMessage:
        return await self._queue.get()


class TurnRecord:
    """Everything the endpoints know about one turn (guarded by the worker's lock / ``cond``)."""

    def __init__(self, turn_id: str, lock: threading.Lock) -> None:
        self.turn_id = turn_id
        self.events: list[dict] = []  # mirror-line dicts, in emission order
        self.done = False
        self.error: str | None = None
        self.started_at = time.time()
        self.finished_at: float | None = None
        self.cond = threading.Condition(lock)
        self.control: _TurnControl | None = None
        self.model_token_file: Path | None = None

    def append(self, line: dict) -> None:
        with self.cond:
            self.events.append(line)
            self.cond.notify_all()

    def finish(self, error: str | None = None) -> None:
        with self.cond:
            self.done = True
            self.error = error
            self.finished_at = time.time()
            self.cond.notify_all()


def _exec(body: dict, workspace_root: str) -> dict:
    cmd = body.get("command")
    if not isinstance(cmd, str):
        return {"ok": False, "error": "command (str) is required"}
    cwd = body.get("cwd") or workspace_root
    if not os.path.isdir(cwd):
        cwd = None
    timeout = body.get("timeout")
    t0 = time.time()
    try:
        p = subprocess.run(
            ["/bin/bash", "-c", cmd], cwd=cwd, capture_output=True, text=True,
            timeout=float(timeout) if timeout else None,
        )
        out, err, rc = p.stdout, p.stderr, p.returncode
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = f"command exceeded {timeout}s and was killed"
        rc = 124
    return {"ok": True, "stdout": out, "stderr": err, "returncode": rc,
            "duration_ms": int((time.time() - t0) * 1000)}


# -- the turn ------------------------------------------------------------------------


def load_baked_spec(path: str = BAKED_SPEC_PATH) -> Any:
    """The spec baked into this image at deploy (``RATK_BAKED_SPEC``)."""
    from ...spec import AgentSpec

    with open(path, encoding="utf-8") as fh:
        return AgentSpec.from_dict(json.load(fh))


def find_baked_skills(spec_path: str = BAKED_SPEC_PATH) -> Path | None:
    """The ``skills/`` dir baked next to the spec (and the toolkit package) at deploy, else ``None``."""
    import remote_agent_toolkit

    pkg = Path(remote_agent_toolkit.__file__).resolve().parent
    for cand in (Path(spec_path).parent / "skills", pkg.parent / "skills"):
        if cand.is_dir():
            return cand
    return None


def _prepare_workspace(rc: Any, prefer_baked_skills: bool = True, baked_dir: Path | None = None) -> dict:
    """Restore a prior workspace (resume) or stage skills + clone repos into a fresh cwd. Sync.

    ``prefer_baked_skills=False`` resolves ``rc.spec.skills`` from their sources at run time
    instead of ``baked_dir`` (the deploy-baked dir) — when the effective spec's skills differ
    from the baked ones (including declaring none).
    """
    from ...checkpoint.workspace import RestoreResult, try_restore, workspace_ready_event
    from ...skills import provision, skills_subdir
    from ...spec import SkillSource

    restored, restore_error = RestoreResult(found=False), None
    if rc.resume_sid and rc.blobs is not None:
        restored, restore_error = try_restore(rc.blobs, rc.resume_sid, str(rc.workspace))
    names: list[str] = []
    repos: list[str] = []
    if restored:
        if rc.spec.repos:
            from ...integrations.git import reauth_repos

            repos = reauth_repos(rc.workspace, rc.spec.repos, rc.secrets)
    else:
        rc.workspace.mkdir(parents=True, exist_ok=True)
        baked = baked_dir if prefer_baked_skills else None
        sources = (SkillSource.local(str(baked)),) if baked else rc.spec.skills
        if sources:
            names = provision(sources, str(rc.workspace), skills_subdir(rc.spec.harness))
        if rc.spec.repos:
            from ...integrations.git import provision_repos

            repos = provision_repos(rc.workspace, rc.spec.repos, rc.secrets)
    return workspace_ready_event(restored, restore_error, names, repos)


def _line(event: AgentEvent) -> dict:
    from .history import mirror_line

    line = mirror_line(event)
    if isinstance(line.get("summary"), str) and len(line["summary"]) > EVENT_SUMMARY_CAP:
        line["summary"] = line["summary"][:EVENT_SUMMARY_CAP] + " …[truncated]"
    return line


def _write_private(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(value)


class Worker:
    """The endpoints and the turn runner of one sandbox process.

    ``baked`` overrides the spec read from ``baked_spec_path`` (tests); ``blob_store_factory``
    builds the checkpoint ``BlobStore`` for ``(bucket, prefix)`` (default ``GcsBlobStore``).
    """

    def __init__(
        self,
        *,
        workspace_root: str = WORKSPACE_ROOT,
        baked_spec_path: str = BAKED_SPEC_PATH,
        baked: Any | None = None,
        baked_skills: Path | None | str = "auto",
        blob_store_factory: Any | None = None,
        mirror_factory: Any | None = None,
    ) -> None:
        self.workspace_root = workspace_root
        self.baked_spec_path = baked_spec_path
        self._baked = baked
        self._baked_skills = baked_skills
        self._blob_store_factory = blob_store_factory
        self._mirror_factory = mirror_factory
        self._lock = threading.Lock()
        self._turns: dict[str, TurnRecord] = {}

    # -- plumbing ---------------------------------------------------------------------

    def baked_spec(self) -> Any:
        if self._baked is None:
            self._baked = load_baked_spec(self.baked_spec_path)
        return self._baked

    def baked_skills(self) -> Path | None:
        if self._baked_skills == "auto":
            self._baked_skills = find_baked_skills(self.baked_spec_path)
        return self._baked_skills  # type: ignore[return-value]

    def _blobs(self, bucket: str, prefix: str) -> Any:
        if self._blob_store_factory is not None:
            return self._blob_store_factory(bucket, prefix)
        from ...ports.blobstore import GcsBlobStore

        return GcsBlobStore(bucket, prefix)

    def _mirror(self, events_uri: str, session_id: str, turn_id: str) -> Any:
        if self._mirror_factory is not None:
            return self._mirror_factory(events_uri, session_id, turn_id)
        from .stream import MirrorStream

        return MirrorStream(events_uri, session_id, turn_id=turn_id)

    def handle(self, path: str, body: dict) -> dict:
        """Route one request: what the HTTP handler and an in-process caller (tests) share."""
        if path == "/health":
            return self.health()
        if path == "/turn":
            return self.start_turn(body)
        if path == "/events":
            return self.events(body)
        if path == "/control":
            return self.control(body)
        if path == "/token":
            return self.token(body)
        if path == "/exec":
            return _exec(body, self.workspace_root)
        return {"ok": False, "error": "not found"}

    # -- the turn -----------------------------------------------------------------------

    async def run_turn(self, record: TurnRecord, body: dict) -> None:
        """Process one turn: the sandbox equivalent of the query-job worker's ``_run_turn``."""
        baked = self.baked_spec()
        from ...checkpoint.session_store import _claude_session_id
        from ...config import (
            SessionConfig,
            TurnConfig,
            apply_session_config,
            apply_turn_config,
            validate_harness_choice,
        )
        from ...harness import resolve_harness
        from ...harness.context import RunContext
        from ...ports.blobstore import parse_gcs_uri, set_default_gcs_credentials
        from .history import mirror_line, write_turn_mirror
        from .scoped_gcs import _parse_expiry, worker_credentials

        turn_id = record.turn_id
        session_id = str(body["session_id"])
        prompt = str(body.get("prompt") or "")
        resume = bool(body.get("resume"))
        sandbox = body.get("sandbox")
        gcs = body.get("gcs") or {}
        output_bucket = gcs.get("output_bucket")
        events_uri = f"{output_bucket}/events" if output_bucket else None
        claude_sid = _claude_session_id(session_id)

        # Every GCS access of this turn runs on the run-scoped token (never an ambient identity —
        # the sandbox has none that could open the bucket anyway).
        token = gcs.get("token")
        if token:
            expiry = _parse_expiry(gcs.get("expiry"))
            set_default_gcs_credentials(
                worker_credentials(token, output_bucket, session_id, expiry=expiry)
            )
        else:
            set_default_gcs_credentials(None)

        def identify(event: AgentEvent) -> AgentEvent:
            event.raw = {**(event.raw or {}), "turn_id": turn_id, "worker": sandbox}
            return event

        def terminal_error(summary: str) -> None:
            ev = identify(AgentEvent(
                kind="result", summary=summary,
                raw={"event": "config_error", "is_error": True, "subtype": "error",
                     "session_id": session_id},
            ))
            record.append(_line(ev))
            if events_uri:
                write_turn_mirror(events_uri, session_id, [mirror_line(ev)],
                                  now_ms=int(time.time() * 1000), turn_id=turn_id)

        # Effective spec: baked ← session config ← turn config (the worker's merge is the
        # ground truth; the client keeps its own view only for result parsing).
        spec = baked
        prefer_baked_skills = True
        session_cfg_dict = body.get("session_config")
        turn_cfg_dict = body.get("turn_config")
        try:
            session_cfg = SessionConfig.from_dict(session_cfg_dict) if session_cfg_dict else None
            turn_cfg = TurnConfig.from_dict(turn_cfg_dict) if turn_cfg_dict else None
            spec = apply_turn_config(apply_session_config(baked, session_cfg), turn_cfg)
            validate_harness_choice(baked, spec)
        except Exception as exc:  # noqa: BLE001 — wrong-config runs must fail loudly
            terminal_error(f"{str(exc)[:300]}; failing the turn")
            return
        if session_cfg_dict or turn_cfg_dict:
            prefer_baked_skills = spec.to_dict().get("skills") == baked.to_dict().get("skills")

        secrets = {str(k): str(v) for k, v in (body.get("secrets") or {}).items()}
        model_env = {str(k): str(v) for k, v in (body.get("model_env") or {}).items()}

        blobs = session_store = None
        if (spec.checkpoint or spec.transcript) and output_bucket:
            from ...checkpoint.session_store import BlobSessionStore

            bucket, prefix = parse_gcs_uri(f"{output_bucket}/checkpoints")
            blobs = self._blobs(bucket, (prefix + "/") if prefix else "")
            session_store = BlobSessionStore(blobs)

        control = _TurnControl(asyncio.get_running_loop(), self._lock)
        with self._lock:
            record.control = control
        job_dir = Path(self.workspace_root) / "jobs" / claude_sid
        job_dir.mkdir(parents=True, exist_ok=True)
        if model_env.get("ANTHROPIC_AUTH_TOKEN"):
            # Kept for a helper-based refresh (see /token); the env token is what the CLI reads.
            record.model_token_file = job_dir / "model-token"
            _write_private(record.model_token_file, model_env["ANTHROPIC_AUTH_TOKEN"])
        rc = RunContext(
            spec=spec,
            prompt=prompt,
            job_dir=job_dir,
            session_id=claude_sid,
            secrets=secrets,
            env=model_env or None,
            resume_sid=(
                _claude_session_id(session_id)
                if (resume and spec.checkpoint and session_store is not None) else None
            ),
            session_store=session_store,
            blobs=blobs,
            interactive=spec.checkpoint if spec.interactive is None else spec.interactive,
            control=control,
        )

        stream = self._mirror(events_uri, session_id, turn_id) if events_uri else None

        def surface(event: AgentEvent) -> None:
            identify(event)
            record.append(_line(event))
            if stream is not None:
                stream.append(event)

        saw_result = False
        try:
            surface(AgentEvent(
                kind="status",
                summary=f"turn started on sandbox {sandbox}" if sandbox else "turn started",
                raw={"event": "turn_started", "session_id": session_id, "worker": sandbox,
                     "warm": bool(body.get("warm"))},
            ))
            if session_cfg_dict or turn_cfg_dict:
                surface(AgentEvent(
                    kind="status",
                    summary=(
                        "effective spec resolved "
                        f"(session config: {bool(session_cfg_dict)}, "
                        f"turn config: {bool(turn_cfg_dict)})"
                    ),
                    raw={"event": "effective_spec", "spec": spec.to_dict(),
                         "session_config_gcs": body.get("session_config_gcs"),
                         "turn_config_gcs": body.get("turn_config_gcs")},
                ))
            # The worker reads /control for the whole turn: send() may steer or interrupt,
            # interrupt() waits for the worker instead of deleting the sandbox.
            surface(AgentEvent(
                kind="status",
                summary="control channel ready (steer / interrupt accepted)",
                raw={"event": "control_ready", "channel": "http"},
            ))
            prep = await asyncio.to_thread(
                _prepare_workspace, rc, prefer_baked_skills, self.baked_skills()
            )
            prep["scoped_gcs"] = bool(token)
            surface(AgentEvent(kind="status", summary=prep["summary"], raw=prep))

            async for event in resolve_harness(spec).run(spec, rc):
                saw_result = saw_result or event.kind == "result"
                surface(event)
        except Exception as exc:  # noqa: BLE001 — a silent death is undebuggable
            if saw_result:
                surface(AgentEvent(
                    kind="status",
                    summary=(
                        "harness exited abnormally after the terminal result "
                        f"(result kept): {str(exc)[:300]}"
                    ),
                    raw={"event": "late_harness_error", "session_id": session_id},
                ))
            else:
                surface(AgentEvent(
                    kind="result",
                    summary=f"agent run failed: {str(exc)[:300]}",
                    raw={"event": "harness_error", "is_error": True, "subtype": "error",
                         "session_id": session_id},
                ))
        finally:
            if stream is not None:
                stream.close()
            set_default_gcs_credentials(None)


    def _turn_thread(self, record: TurnRecord, body: dict) -> None:
        error = None
        try:
            asyncio.run(self.run_turn(record, body))
        except Exception:  # noqa: BLE001 — surfaced to the client as `error`
            error = traceback.format_exc()[-3000:]
        finally:
            record.finish(error)

    # -- endpoints ---------------------------------------------------------------------

    def health(self) -> dict:
        with self._lock:
            running = [t for t, r in self._turns.items() if not r.done]
            turns = len(self._turns)
        home = os.path.expanduser("~")
        return {
            "ok": True,
            "uptime_s": round(time.time() - BOOT, 3),
            "pid": os.getpid(),
            "uid": os.getuid(),
            "workspace_writable": os.access(self.workspace_root, os.W_OK),
            "home_writable": os.access(home, os.W_OK),
            "python": sys.version.split()[0],
            "baked_spec": self._baked is not None or os.path.exists(self.baked_spec_path),
            "running_turn": running[0] if running else None,
            "turns": turns,
        }

    def start_turn(self, body: dict) -> dict:
        turn_id = str(body.get("turn_id") or "")
        if not turn_id or not body.get("session_id"):
            return {"ok": False, "error": "turn_id and session_id are required"}
        with self._lock:
            running = [t for t, r in self._turns.items() if not r.done]
            if running:
                return {"ok": False, "error": "a turn is running", "running": running}
            if turn_id in self._turns:
                return {"ok": False, "error": "turn_id already used on this sandbox"}
            record = TurnRecord(turn_id, self._lock)
            self._turns[turn_id] = record
        threading.Thread(target=self._turn_thread, args=(record, body),
                         name=f"turn-{turn_id[:8]}", daemon=True).start()
        return {"ok": True, "turn_id": turn_id, "accepted_at": record.started_at}

    def events(self, body: dict) -> dict:
        record = self._turns.get(str(body.get("turn_id")))
        if record is None:
            return {"ok": False, "error": "unknown turn_id"}
        since = max(0, int(body.get("since") or 0))
        wait = min(float(body.get("wait") or 0.0), MAX_WAIT_S)
        deadline = time.monotonic() + wait
        with record.cond:
            while len(record.events) <= since and not record.done:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                record.cond.wait(remaining)
            page: list[dict] = []
            size = 0
            for line in record.events[since:]:
                encoded = len(json.dumps(line))
                if page and size + encoded > EVENTS_PAGE_BYTES:
                    break
                page.append(line)
                size += encoded
            nxt = since + len(page)
            return {
                "ok": True, "events": page, "next": nxt,
                # done only once the caller has seen every event
                "done": record.done and nxt >= len(record.events),
                "error": record.error, "started_at": record.started_at,
                "finished_at": record.finished_at,
            }

    def control(self, body: dict) -> dict:
        record = self._turns.get(str(body.get("turn_id")))
        if record is None:
            return {"ok": False, "error": "unknown turn_id"}
        try:
            msg = ControlMessage.from_dict(body)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        with self._lock:
            control, done = record.control, record.done
        if done:
            return {"ok": False, "error": "turn already finished"}
        if control is None:
            return {"ok": False, "error": "turn not ready for control yet"}
        delivered = control.send_threadsafe(msg)
        return {"ok": True, "delivered": delivered, "message_id": msg.message_id}

    def token(self, body: dict) -> dict:
        record = self._turns.get(str(body.get("turn_id")))
        if record is None:
            return {"ok": False, "error": "unknown turn_id"}
        token = body.get("model_token")
        if not isinstance(token, str) or not token:
            return {"ok": False, "error": "model_token is required"}
        if record.model_token_file is None:
            return {"ok": False, "error": "this turn runs without a model token"}
        _write_private(record.model_token_file, token)
        return {"ok": True}


WORKER: Worker | None = None


class Handler(BaseHTTPRequestHandler):
    server_version = "ratk-sandbox-worker/1"

    def _reply(self, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        if not raw.strip():
            return {}
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}

    def do_GET(self) -> None:  # noqa: N802
        assert WORKER is not None
        if urlsplit(self.path).path in ("/", "/health"):
            self._reply(WORKER.health())
        else:
            self._reply({"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        assert WORKER is not None
        path = urlsplit(self.path).path
        try:
            self._reply(WORKER.handle(path, self._body()))
        except Exception as exc:  # noqa: BLE001
            self._reply({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet: never request bodies
        sys.stderr.write(f"{time.strftime('%H:%M:%S')} {self.command} {urlsplit(self.path).path}\n")


def main() -> None:
    global WORKER
    port = int(os.environ.get("PORT", "8080"))
    os.makedirs(WORKSPACE_ROOT, exist_ok=True)
    WORKER = Worker()
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    sys.stderr.write(f"ratk sandbox worker on :{port} (uid {os.getuid()})\n")
    srv.serve_forever()


if __name__ == "__main__":
    main()

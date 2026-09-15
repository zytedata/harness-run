"""Offline stand-ins for the sandbox platform: a ``SandboxProvider`` over in-memory state.

``FakeSandboxProvider`` keeps templates and sandboxes in dicts and routes every ``call``
to the sandbox's *worker* — either the real :class:`~remote_agent_toolkit.runtime.gemini.worker.Worker`
(driven in-process, its harness faked by the test) or a :class:`ScriptedWorker` a test
feeds events to. Failure injection: ``fail_next[path]`` makes the next calls to ``path``
raise, ``vanish(sandbox)`` makes a sandbox answer :class:`SandboxGone`.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Callable

from remote_agent_toolkit.events import AgentEvent
from remote_agent_toolkit.runtime.gemini import backend
from remote_agent_toolkit.runtime.gemini.history import mirror_line
from remote_agent_toolkit.runtime.gemini.provider import SandboxError, SandboxGone, SandboxHandle
from remote_agent_toolkit.runtime.gemini.roster import InMemoryRosterStore
from remote_agent_toolkit.spec import AgentSpec

INSTANCE = "projects/p/locations/l/reasoningEngines/host"


def result_event(text="done", subtype="success", is_error=False, num_turns=2, cost=0.3):
    return AgentEvent(
        kind="result", summary=text, cost_usd=cost, usage={"input_tokens": 5},
        raw={"subtype": subtype, "is_error": is_error, "num_turns": num_turns, "session_id": "claude-sid"},
    )


class ScriptedWorker:
    """A worker whose events a test emits by hand (``emit`` / ``finish``).

    ``/turn`` accepts once and records the body; ``/events`` answers like the real worker
    (blocks up to ``wait`` for news, pages everything past ``since``); ``/control`` records
    the message and calls ``on_control`` (a test hook that may emit the reaction).
    """

    def __init__(self, *, on_control: Callable[[ScriptedWorker, dict], None] | None = None,
                 auto: list[AgentEvent] | None = None) -> None:
        self.turns: list[dict] = []
        self.controls: list[dict] = []
        self.tokens: list[dict] = []
        self.lines: list[dict] = []
        self.done = False
        self.error: str | None = None
        self.on_control = on_control
        self.auto = auto  # events emitted the moment a turn starts (then finish)
        self._cond = threading.Condition()

    def emit(self, *events: AgentEvent) -> None:
        with self._cond:
            for ev in events:
                turn_id = self.turns[-1]["turn_id"] if self.turns else None
                ev.raw = {**(ev.raw or {}), "turn_id": (ev.raw or {}).get("turn_id", turn_id)}
                self.lines.append(mirror_line(ev))
            self._cond.notify_all()

    def finish(self, error: str | None = None) -> None:
        with self._cond:
            self.done, self.error = True, error
            self._cond.notify_all()

    def handle(self, path: str, body: dict) -> dict:
        if path == "/health":
            return {"ok": True, "uptime_s": 1.0}
        if path == "/turn":
            if self.turns and not self.done:
                return {"ok": False, "error": "a turn is running"}
            self.turns.append(dict(body))
            if self.auto is not None:
                self.emit(*self.auto)
                self.finish()
            return {"ok": True, "turn_id": body["turn_id"]}
        if path == "/events":
            since = int(body.get("since") or 0)
            deadline = time.monotonic() + float(body.get("wait") or 0)
            with self._cond:
                while len(self.lines) <= since and not self.done:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._cond.wait(remaining)
                page = list(self.lines[since:])
                return {"ok": True, "events": page, "next": since + len(page),
                        "done": self.done and since + len(page) >= len(self.lines),
                        "error": self.error}
        if path == "/control":
            self.controls.append(dict(body))
            if self.on_control is not None:
                self.on_control(self, body)
            return {"ok": True, "delivered": True, "message_id": body.get("message_id")}
        if path == "/token":
            self.tokens.append(dict(body))
            return {"ok": True}
        return {"ok": False, "error": "not found"}


class FakeSandboxProvider:
    """In-memory ``SandboxProvider``; see the module docstring."""

    def __init__(self, worker_factory: Callable[[str], Any] | None = None) -> None:
        self.worker_factory = worker_factory or (lambda name: ScriptedWorker(auto=[result_event()]))
        self.templates: dict[str, dict] = {}
        self.sandboxes: dict[str, dict] = {}
        self.workers: dict[str, Any] = {}
        self.deleted: list[str] = []
        self.deleted_templates: list[str] = []
        self.calls: list[tuple[str, str, dict]] = []
        self.fail_next: dict[str, list[Exception]] = {}  # path -> exceptions to raise, in order
        self.gone: set[str] = set()
        self._n = 0
        self.lock = threading.Lock()

    # -- helpers for tests ------------------------------------------------------------

    def add_template(self, display_name: str, image: str = "img:1", cpu="4", memory="8Gi",
                     create_time: float | None = None) -> str:
        with self.lock:
            self._n += 1
            name = f"{INSTANCE}/sandboxEnvironmentTemplates/t{self._n}"
            self.templates[name] = {
                "name": name, "display_name": display_name, "image_uri": image, "cpu": cpu,
                "memory": memory, "create_time": create_time or time.time() + self._n, "state": "ACTIVE",
            }
        return name

    def vanish(self, sandbox: str) -> None:
        """The sandbox is gone (TTL / OOM / deleted elsewhere): every call answers SandboxGone."""
        self.gone.add(sandbox)
        self.sandboxes.pop(sandbox, None)

    def worker(self, sandbox: str) -> Any:
        return self.workers[sandbox]

    def live(self) -> list[str]:
        return sorted(self.sandboxes)

    # -- SandboxProvider ----------------------------------------------------------------

    def create(self, template: str, *, ttl_s: float, display_name: str) -> SandboxHandle:
        if template not in self.templates:
            raise SandboxError(f"unknown template {template}")
        with self.lock:
            self._n += 1
            name = f"{INSTANCE}/sandboxEnvironments/s{self._n}"
        now = time.time()
        self.sandboxes[name] = {
            "name": name, "display_name": display_name, "state": "STATE_RUNNING", "template": template,
            "create_time": now, "expire_time": now + ttl_s, "ttl_s": ttl_s,
        }
        self.workers[name] = self.worker_factory(name)
        return SandboxHandle(name=name, created_at=now, expires_at=now + ttl_s)

    def call(self, sandbox: str, path: str, body: dict | None = None, *, timeout_s: float | None = None) -> dict:
        body = body or {}
        self.calls.append((sandbox, path, body))
        queued = self.fail_next.get(path)
        if queued:
            raise queued.pop(0)
        if sandbox in self.gone or sandbox not in self.sandboxes:
            raise SandboxGone(f"{sandbox} is gone")
        return self.workers[sandbox].handle(path, body)

    def delete(self, sandbox: str) -> None:
        self.deleted.append(sandbox)
        self.sandboxes.pop(sandbox, None)

    def list(self, *, display_prefix: str | None = None) -> list[dict]:
        return [dict(sb) for sb in self.sandboxes.values()
                if not display_prefix or sb["display_name"].startswith(display_prefix)]

    def create_template(self, *, display_name: str, image_uri: str, cpu: str, memory: str,
                        internet_access: bool = True, log=None) -> str:
        if log is not None:
            log(f"template {display_name}: PROVISIONING for 0s")
        return self.add_template(display_name, image_uri, cpu, memory)

    def list_templates(self, *, display_name: str | None = None) -> list[dict]:
        rows = [dict(t) for t in self.templates.values()
                if display_name is None or t["display_name"] == display_name]
        rows.sort(key=lambda r: r["create_time"], reverse=True)
        return rows

    def delete_template(self, name: str) -> None:
        if any(sb["template"] == name for sb in self.sandboxes.values()):
            raise SandboxError("template still has sandboxes")
        self.deleted_templates.append(name)
        self.templates.pop(name, None)


def make_engine(provider: FakeSandboxProvider, *, spec: AgentSpec | None = None, warm: bool = False,
                template: str | None = None, output_bucket: str | None = "gs://out", **kw) -> backend.GeminiEngine:
    """A ``GeminiEngine`` over ``provider`` with an in-memory roster and no GCP client."""
    spec = spec or AgentSpec(name="g", model="m")
    template = template or provider.add_template(spec.name)
    kw.setdefault("model_service_account", "ratk-model@p.iam.gserviceaccount.com")
    kw.setdefault("roster_store", InMemoryRosterStore())
    return backend.GeminiEngine(
        template=template, spec=spec, project="p", location="l", output_bucket=output_bucket,
        provider=provider, warm=warm, **kw,
    )


def fresh_id() -> str:
    return str(uuid.uuid4())

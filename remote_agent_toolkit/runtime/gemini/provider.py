"""``SandboxProvider`` — the one platform seam of the sandbox runtime (DESIGN.md §13.6).

The runtime asks the platform for four things only: create a container from a template,
reach an HTTP port on it through an authenticated proxy, delete it, and enforce a TTL —
plus template create/list/delete at deploy. Everything else (dispatch, events, checkpoints,
control, secrets) is ours, over HTTP into the container and an object store. Keeping the
provider-specific part behind this protocol keeps the worker image and the control plane
provider-agnostic; :class:`AgentSandboxProvider` (Gemini Enterprise Agent Platform's Agent
Sandbox, ``google-cloud-agentplatform`` 2.x) is the first and only implementation, and
tests drive the backend through an in-memory fake.

Stdlib-only at import; the Google SDK is imported lazily inside the adapter's methods.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

# The display name of the parent "instance" every template and sandbox hangs off. Agent
# Sandbox resources are children of a ``reasoningEngine``; the toolkit keeps exactly one
# such (empty) resource per project/location and never serves anything from it.
HOST_DISPLAY_NAME = "ratk-sandbox-host"

# The HTTP port the worker listens on inside the container (the template declares it; the
# proxy forwards only declared ports).
WORKER_PORT = 8080


class SandboxError(RuntimeError):
    """A provider call failed (the proxy, the platform API, or the worker's HTTP answer)."""


class SandboxGone(SandboxError):
    """The sandbox no longer exists (deleted, TTL-expired, or never created)."""


@dataclass(frozen=True)
class SandboxHandle:
    """One created sandbox: its resource name and the lifetime the platform enforces."""

    name: str  # full resource name (.../sandboxEnvironments/<id>)
    created_at: float  # epoch seconds
    expires_at: float  # epoch seconds: created_at + TTL (the platform deletes it then)

    @property
    def id(self) -> str:
        return self.name.rsplit("/", 1)[-1]


class SandboxProvider(Protocol):
    """What the runtime needs from a hosted-sandbox platform (all calls are blocking)."""

    def create(self, template: str, *, ttl_s: float, display_name: str) -> SandboxHandle:
        """Create a sandbox from ``template`` that the platform deletes after ``ttl_s``."""
        ...

    def call(
        self, sandbox: str, path: str, body: dict | None = None, *, timeout_s: float | None = None
    ) -> dict:
        """POST ``body`` (JSON) to the worker's ``path`` inside ``sandbox``; return its JSON.

        Raises :class:`SandboxGone` when the sandbox does not exist any more and
        :class:`SandboxError` for any other failure (proxy, platform, malformed answer).
        """
        ...

    def delete(self, sandbox: str) -> None:
        """Delete a sandbox (idempotent: a missing sandbox is not an error)."""
        ...

    def list(self, *, display_prefix: str | None = None) -> list[dict]:
        """Sandboxes under the host instance: ``{name, display_name, state, template,
        create_time, expire_time}``; optionally only those whose display name starts with
        ``display_prefix``."""
        ...

    def create_template(
        self, *, display_name: str, image_uri: str, cpu: str, memory: str,
        internet_access: bool = True, log: Callable[[str], None] | None = None,
    ) -> str:
        """Create an immutable custom-container template; return its resource name once it
        is ACTIVE. ``log`` gets progress lines while the platform provisions it. A template
        that ends FAILED raises ``SandboxError`` with the platform's reason."""
        ...

    def list_templates(self, *, display_name: str | None = None) -> list[dict]:
        """Live templates under the host instance, newest first: ``{name, display_name,
        image_uri, cpu, memory, create_time, state}`` (deleted ones are not listed)."""
        ...

    def delete_template(self, name: str) -> None:
        """Delete a template (fails while sandboxes created from it still exist)."""
        ...


TEMPLATE_POLL_S = 5.0
TEMPLATE_REPORT_S = 30.0
TEMPLATE_TIMEOUT_S = 30 * 60.0


def _state_name(obj: Any) -> str:
    state = getattr(obj, "state", None)
    return getattr(state, "name", str(state) if state else "") or ""


def _error_text(error: Any) -> str:
    """A long-running operation's ``error`` (a Status-like object or dict) as one line."""
    if isinstance(error, dict):
        return str(error.get("message") or error)[:500]
    message = getattr(error, "message", None)
    return str(message or error)[:500]


def template_id(name: str) -> str:
    """The bare id of a template resource name (id in → id out)."""
    return name.rsplit("/", 1)[-1]


def _ts(value: Any) -> float | None:
    """A datetime-ish SDK field as epoch seconds (``None`` when absent)."""
    if value is None:
        return None
    if hasattr(value, "timestamp"):
        return float(value.timestamp())
    try:
        import datetime as _dt

        return _dt.datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


class AgentSandboxProvider:
    """Agent Sandbox (Gemini Enterprise Agent Platform) adapter over ``agentplatform`` 2.x.

    Resolves the host instance lazily (created on first use when missing). Every call is
    one platform API request; the execute proxy adds ~0.2–0.3 s per call (measured).
    """

    def __init__(
        self,
        project: str,
        location: str,
        credentials: Any | None = None,
        *,
        instance: str | None = None,
    ) -> None:
        self._project = project
        self._location = location
        self._credentials = credentials
        self._instance = instance
        self._client_cached: Any = None

    # -- SDK plumbing -----------------------------------------------------------

    def _client(self) -> Any:
        if self._client_cached is None:
            import agentplatform  # lazy: google-cloud-agentplatform 2.x

            self._client_cached = agentplatform.Client(
                project=self._project, location=self._location, credentials=self._credentials
            )
        return self._client_cached

    def instance(self) -> str:
        """The host instance's resource name (found by display name, or created)."""
        if self._instance is None:
            client = self._client()
            for runtime in client.runtimes.list():
                api = getattr(runtime, "api_resource", runtime)
                if getattr(api, "display_name", None) == HOST_DISPLAY_NAME:
                    self._instance = api.name
                    break
            else:
                created = client.runtimes.create(
                    config={
                        "display_name": HOST_DISPLAY_NAME,
                        "description": (
                            "remote-agent-toolkit: parent of the sandbox templates and "
                            "sandboxes; serves nothing itself"
                        ),
                    }
                )
                self._instance = created.api_resource.name
        return self._instance

    @staticmethod
    def _translate(exc: BaseException) -> SandboxError:
        text = str(exc)
        code = getattr(exc, "code", None)
        if code == 404 or "NOT_FOUND" in text:
            return SandboxGone(text[:300])
        if "Precondition check failed" in text and "Execution Failed" not in text:
            # What an expired (TTL) sandbox answers to execute/get (measured 2026-09-10).
            return SandboxGone(text[:300])
        return SandboxError(text[:500])

    # -- sandboxes ------------------------------------------------------------------

    def create(self, template: str, *, ttl_s: float, display_name: str) -> SandboxHandle:
        ttl = int(max(60, ttl_s))
        created_at = time.time()
        try:
            op = self._client().sandboxes.create(
                name=self.instance(),
                config={
                    "display_name": display_name,
                    "sandbox_environment_template": template,
                    "wait_for_completion": True,
                    "ttl": f"{ttl}s",
                },
            )
        except Exception as exc:  # noqa: BLE001
            raise self._translate(exc) from exc
        resp = op.response
        name = getattr(resp, "name", None)
        if not name:
            raise SandboxError(f"sandbox create returned no resource name: {op}")
        expires = _ts(getattr(resp, "expire_time", None)) or (created_at + ttl)
        return SandboxHandle(name=name, created_at=created_at, expires_at=expires)

    def call(
        self, sandbox: str, path: str, body: dict | None = None, *, timeout_s: float | None = None
    ) -> dict:
        from agentplatform._genai import types  # lazy

        inputs = [
            types.Chunk(mime_type="application/x.sandbox-request-uri", data=path.encode()),
            types.Chunk(mime_type="application/x.sandbox-request-port", data=str(WORKER_PORT).encode()),
            types.Chunk(mime_type="application/json", data=json.dumps(body or {}).encode()),
        ]
        config = {"http_options": {"timeout": int(timeout_s * 1000)}} if timeout_s else None
        try:
            resp = self._client().sandboxes._execute_code(name=sandbox, inputs=inputs, config=config)
        except Exception as exc:  # noqa: BLE001
            raise self._translate(exc) from exc
        for out in getattr(resp, "outputs", None) or []:
            data = getattr(out, "data", None)
            if data:
                try:
                    parsed = json.loads(data.decode("utf-8"))
                except (ValueError, UnicodeDecodeError) as exc:
                    raise SandboxError(f"non-JSON answer from the worker at {path}") from exc
                if not isinstance(parsed, dict):
                    raise SandboxError(f"unexpected answer shape from the worker at {path}")
                return parsed
        raise SandboxError(f"empty answer from the sandbox at {path}")

    def delete(self, sandbox: str) -> None:
        try:
            self._client().sandboxes.delete(name=sandbox)
        except Exception as exc:  # noqa: BLE001
            if isinstance(self._translate(exc), SandboxGone):
                return
            raise self._translate(exc) from exc

    def list(self, *, display_prefix: str | None = None) -> list[dict]:
        rows = []
        try:
            sandboxes = list(self._client().sandboxes.list(name=self.instance()))
        except Exception as exc:  # noqa: BLE001
            raise self._translate(exc) from exc
        for sb in sandboxes:
            display = getattr(sb, "display_name", None) or ""
            if display_prefix and not display.startswith(display_prefix):
                continue
            state = getattr(sb, "state", None)
            rows.append({
                "name": sb.name,
                "display_name": display,
                "state": getattr(state, "name", str(state) if state else None),
                "template": getattr(sb, "sandbox_environment_template", None),
                "create_time": _ts(getattr(sb, "create_time", None)),
                "expire_time": _ts(getattr(sb, "expire_time", None)),
            })
        return rows

    # -- templates ------------------------------------------------------------------

    def create_template(
        self, *, display_name: str, image_uri: str, cpu: str, memory: str,
        internet_access: bool = True, log: Callable[[str], None] | None = None,
    ) -> str:
        """Create the template and return as soon as it lists as ACTIVE.

        The create is a long-running operation whose completion can lag the template's
        ACTIVE state by minutes (nine, measured 2026-09-15), so this does not block on the
        operation: it polls the template listing (the new entry under ``display_name``)
        every ``TEMPLATE_POLL_S`` and returns at ACTIVE, reporting progress through ``log``
        every ``TEMPLATE_REPORT_S``. The operation is consulted for the failure reason when
        the template ends FAILED, or when it finishes with an error first.
        """
        templates = self._client().sandboxes.templates
        say = log or (lambda msg: None)
        try:
            known = {t.name for t in templates.list(name=self.instance())
                     if (getattr(t, "display_name", None) or "") == display_name}
            op = templates.create(
                name=self.instance(),
                display_name=display_name,
                config={
                    "custom_container_environment": {
                        "custom_container_spec": {"image_uri": image_uri},
                        "ports": [{"port": WORKER_PORT, "protocol": "TCP"}],
                        "resources": {
                            "requests": {"cpu": cpu, "memory": memory},
                            "limits": {"cpu": cpu, "memory": memory},
                        },
                    },
                    "egress_control_config": {"internet_access": internet_access},
                    "wait_for_completion": False,
                },
            )
        except Exception as exc:  # noqa: BLE001
            raise self._translate(exc) from exc
        started = time.monotonic()
        next_report = started + TEMPLATE_REPORT_S
        target: str | None = getattr(op.response, "name", None)
        last_state = "PROVISIONING"
        while True:
            try:
                rows = [t for t in templates.list(name=self.instance())
                        if (getattr(t, "display_name", None) or "") == display_name
                        and t.name not in known and "DELETED" not in _state_name(t)]
            except Exception as exc:  # noqa: BLE001 — a listing blip; the op check below still runs
                rows = []
                say(f"template listing failed ({type(exc).__name__}); retrying")
            if target is not None:
                rows = [t for t in rows if t.name == target] or rows
            if rows:
                rows.sort(key=lambda t: _ts(getattr(t, "create_time", None)) or 0.0, reverse=True)
                target = rows[0].name
                last_state = _state_name(rows[0]) or last_state
                if "ACTIVE" in last_state:
                    return target
                if "FAILED" in last_state:
                    raise SandboxError(
                        f"template {template_id(target)} ({display_name}) ended FAILED: "
                        f"{self._operation_error(templates, op.name) or 'the platform gave no reason'}"
                    )
            error = None
            try:
                current = templates.get_sandbox_environment_template_operation(operation_name=op.name)
                if getattr(current, "done", False):
                    error = getattr(current, "error", None)
                    if error:
                        raise SandboxError(f"template create for {display_name} failed: {_error_text(error)}")
                    name = getattr(getattr(current, "response", None), "name", None)
                    if name and target is None:
                        target = name
            except SandboxError:
                raise
            except Exception as exc:  # noqa: BLE001 — the listing is the source of truth
                say(f"operation check failed ({type(exc).__name__}); relying on the template listing")
            elapsed = time.monotonic() - started
            if elapsed > TEMPLATE_TIMEOUT_S:
                raise SandboxError(
                    f"template {display_name} still {last_state} after {elapsed:.0f}s "
                    f"(operation {op.name}); it may still come up — re-run the deploy later, or "
                    "delete it via engine.delete_version()"
                )
            if time.monotonic() >= next_report:
                say(f"template {display_name}: {last_state} for {elapsed:.0f}s")
                next_report = time.monotonic() + TEMPLATE_REPORT_S
            time.sleep(TEMPLATE_POLL_S)

    @staticmethod
    def _operation_error(templates: Any, operation_name: str) -> str | None:
        try:
            op = templates.get_sandbox_environment_template_operation(operation_name=operation_name)
        except Exception:  # noqa: BLE001
            return None
        error = getattr(op, "error", None)
        return _error_text(error) if error else None

    def list_templates(self, *, display_name: str | None = None) -> list[dict]:
        rows = []
        try:
            templates = list(self._client().sandboxes.templates.list(name=self.instance()))
        except Exception as exc:  # noqa: BLE001
            raise self._translate(exc) from exc
        for tpl in templates:
            display = getattr(tpl, "display_name", None) or ""
            if display_name is not None and display != display_name:
                continue
            state = getattr(tpl, "state", None)
            state_name = getattr(state, "name", str(state) if state else None)
            if state_name and ("DELETED" in state_name or "DEPROVISIONING" in state_name):
                # The platform keeps deleted templates in the listing (observed 2026-09-11),
                # and shows DEPROVISIONING while a delete runs; neither is a version anyone
                # can dispatch to. FAILED and PROVISIONING ones are listed (with their state).
                continue
            env = getattr(tpl, "custom_container_environment", None)
            spec = getattr(env, "custom_container_spec", None)
            limits = getattr(getattr(env, "resources", None), "limits", None) or {}
            rows.append({
                "name": tpl.name,
                "display_name": display,
                "image_uri": getattr(spec, "image_uri", None),
                "cpu": limits.get("cpu") if isinstance(limits, dict) else None,
                "memory": limits.get("memory") if isinstance(limits, dict) else None,
                "create_time": _ts(getattr(tpl, "create_time", None)),
                "state": state_name,
            })
        rows.sort(key=lambda r: r["create_time"] or 0.0, reverse=True)
        return rows

    def delete_template(self, name: str) -> None:
        try:
            self._client().sandboxes.templates.delete(name=name)
        except Exception as exc:  # noqa: BLE001
            if isinstance(self._translate(exc), SandboxGone):
                return
            raise self._translate(exc) from exc

"""``AgentSandboxProvider.create_template`` over a stubbed SDK client: it returns at ACTIVE
without waiting for the long-running operation, reports progress, and surfaces failures."""

from __future__ import annotations

import types

import pytest

from remote_agent_toolkit.runtime.gemini import provider as prov
from remote_agent_toolkit.runtime.gemini.provider import AgentSandboxProvider, SandboxError

INSTANCE = "projects/p/locations/l/reasoningEngines/1"


class _State:
    def __init__(self, name):
        self.name = name


class _Template:
    def __init__(self, name, display_name, state, create_time=1.0):
        self.name, self.display_name, self.state, self.create_time = name, display_name, _State(state), create_time


class _Op:
    def __init__(self, name, done=False, error=None, response=None):
        self.name, self.done, self.error, self.response = name, done, error, response


class _Templates:
    """The SDK's ``client.sandboxes.templates`` with a scripted timeline."""

    def __init__(self, timeline):
        # timeline: list of (template state or None, op) per poll, consumed in order
        self.timeline = list(timeline)
        self.created = []
        self.polls = 0

    def list(self, *, name):
        rows = [_Template(f"{INSTANCE}/sandboxEnvironmentTemplates/old", "g", "ACTIVE", 0.5)]
        if self.created:
            state, _ = self.timeline[min(self.polls, len(self.timeline) - 1)]
            if state is not None:
                rows.append(_Template(f"{INSTANCE}/sandboxEnvironmentTemplates/new", "g", state, 2.0))
        return iter(rows)

    def create(self, *, name, display_name, config):
        assert config["wait_for_completion"] is False
        self.created.append(config)
        return _Op("projects/p/locations/l/operations/op1")

    def get_sandbox_environment_template_operation(self, *, operation_name):
        _, op = self.timeline[min(self.polls, len(self.timeline) - 1)]
        self.polls += 1
        return op


def _provider(monkeypatch, timeline):
    templates = _Templates(timeline)
    client = types.SimpleNamespace(sandboxes=types.SimpleNamespace(templates=templates))
    p = AgentSandboxProvider(project="p", location="l")
    p._client_cached = client
    monkeypatch.setattr(p, "instance", lambda: INSTANCE)
    monkeypatch.setattr(prov, "TEMPLATE_POLL_S", 0.0)
    monkeypatch.setattr(prov, "TEMPLATE_REPORT_S", 0.0)
    return p, templates


def test_create_template_returns_at_active_without_waiting_for_the_operation(monkeypatch):
    pending = _Op("op1")
    p, templates = _provider(monkeypatch, [
        (None, pending),              # not listed yet
        ("PROVISIONING", pending),
        ("ACTIVE", pending),          # ACTIVE while the operation is still running (the 9-minute lag)
    ])
    lines = []
    name = p.create_template(display_name="g", image_uri="img", cpu="1", memory="1Gi", log=lines.append)
    assert name.endswith("/new")
    assert templates.created[0]["custom_container_environment"]["custom_container_spec"]["image_uri"] == "img"
    assert any("PROVISIONING for" in line for line in lines)


def test_create_template_failed_state_raises_with_the_operation_error(monkeypatch):
    failed = _Op("op1", done=True, error={"message": "image pull failed: manifest unknown"})
    p, _ = _provider(monkeypatch, [("PROVISIONING", _Op("op1")), ("FAILED", failed)])
    with pytest.raises(SandboxError, match="ended FAILED: image pull failed"):
        p.create_template(display_name="g", image_uri="img", cpu="1", memory="1Gi")


def test_create_template_operation_error_before_the_template_lists(monkeypatch):
    p, _ = _provider(monkeypatch, [(None, _Op("op1", done=True, error={"message": "quota exceeded"}))])
    with pytest.raises(SandboxError, match="failed: quota exceeded"):
        p.create_template(display_name="g", image_uri="img", cpu="1", memory="1Gi")


def test_create_template_times_out_with_the_operation_name(monkeypatch):
    p, _ = _provider(monkeypatch, [("PROVISIONING", _Op("op1"))])
    monkeypatch.setattr(prov, "TEMPLATE_TIMEOUT_S", -1.0)
    with pytest.raises(SandboxError, match="still PROVISIONING.*operations/op1"):
        p.create_template(display_name="g", image_uri="img", cpu="1", memory="1Gi")

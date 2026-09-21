"""``AgentSandboxProvider.create_template`` over a stubbed SDK client: it returns at ACTIVE
without waiting for the long-running operation, reports progress, surfaces failures, and clears
FAILED / stuck templates, which hold capacity from the next create."""

from __future__ import annotations

import types

import pytest

from remote_agent_toolkit.runtime.gemini import provider as prov
from remote_agent_toolkit.runtime.gemini.provider import AgentSandboxProvider, SandboxError

INSTANCE = "projects/p/locations/l/reasoningEngines/1"


def _tpl(tid: str) -> str:
    return f"{INSTANCE}/sandboxEnvironmentTemplates/{tid}"


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

    def __init__(self, timeline, preexisting=None, refuse_delete=()):
        # timeline: list of (template state or None, op) per poll, consumed in order
        self.timeline = list(timeline)
        self.preexisting = list(preexisting or [_Template(_tpl("old"), "g", "ACTIVE", 0.5)])
        self.refuse_delete = set(refuse_delete)
        self.created = []
        self.deleted = []
        self.polls = 0

    def list(self, *, name):
        rows = [t for t in self.preexisting if t.name not in self.deleted]
        if self.created:
            state, _ = self.timeline[min(self.polls, len(self.timeline) - 1)]
            if state is not None and _tpl("new") not in self.deleted:
                rows.append(_Template(_tpl("new"), "g", state, 2.0))
        return iter(rows)

    def create(self, *, name, display_name, config):
        assert config["wait_for_completion"] is False
        self.created.append(config)
        return _Op("projects/p/locations/l/operations/op1")

    def delete(self, *, name):
        if name in self.refuse_delete:
            raise RuntimeError("409 template is in use")
        self.deleted.append(name)

    def get_sandbox_environment_template_operation(self, *, operation_name):
        _, op = self.timeline[min(self.polls, len(self.timeline) - 1)]
        self.polls += 1
        return op


def _provider(monkeypatch, timeline, **kw):
    templates = _Templates(timeline, **kw)
    client = types.SimpleNamespace(sandboxes=types.SimpleNamespace(templates=templates))
    p = AgentSandboxProvider(project="p", location="l")
    p._client_cached = client
    monkeypatch.setattr(p, "instance", lambda: INSTANCE)
    monkeypatch.setattr(prov, "TEMPLATE_POLL_S", 0.0)
    monkeypatch.setattr(prov, "TEMPLATE_REPORT_S", 0.0)
    return p, templates


def _create(p, **kw):
    return p.create_template(display_name="g", image_uri="img", cpu="1", memory="1Gi", **kw)


def test_create_template_returns_at_active_without_waiting_for_the_operation(monkeypatch):
    pending = _Op("op1")
    p, templates = _provider(monkeypatch, [
        (None, pending),              # not listed yet
        ("PROVISIONING", pending),
        ("ACTIVE", pending),          # ACTIVE while the operation is still running (the 9-minute lag)
    ])
    lines = []
    name = _create(p, log=lines.append)
    assert name.endswith("/new")
    assert templates.created[0]["custom_container_environment"]["custom_container_spec"]["image_uri"] == "img"
    assert any("PROVISIONING for" in line for line in lines)
    assert templates.deleted == []  # nothing FAILED around: nothing reaped


def test_create_template_sweeps_failed_templates_of_any_engine_first(monkeypatch):
    p, templates = _provider(monkeypatch, [("ACTIVE", _Op("op1"))], preexisting=[
        _Template(_tpl("mine-active"), "g", "ACTIVE", 0.5),
        _Template(_tpl("mine-failed"), "g", "FAILED", 0.7),
        _Template(_tpl("theirs-failed"), "other-engine", "FAILED", 0.8),
        _Template(_tpl("theirs-active"), "other-engine", "ACTIVE", 0.9),
        _Template(_tpl("gone"), "g", "DELETED", 0.3),
    ])
    lines = []
    name = _create(p, log=lines.append)
    assert name.endswith("/new")
    # the FAILED ones went (whoever's they were), before the create; ACTIVE and DELETED ones stayed
    assert templates.deleted == [_tpl("mine-failed"), _tpl("theirs-failed")]
    assert len(templates.created) == 1
    reaped = [line for line in lines if line.startswith("deleted template")]
    assert len(reaped) == 2 and "other-engine" in reaped[1] and "holds warm-pool capacity" in reaped[0]


def test_create_template_failed_state_raises_with_the_operation_error_and_reaps_it(monkeypatch):
    failed = _Op("op1", done=True, error={"message": "image pull failed: manifest unknown"})
    p, templates = _provider(monkeypatch, [("PROVISIONING", _Op("op1")), ("FAILED", failed)])
    lines = []
    with pytest.raises(SandboxError, match=r"ended FAILED: image pull failed.*; deleted it"):
        _create(p, log=lines.append)
    assert templates.deleted == [_tpl("new")]
    assert any(line.startswith("deleted template new (g)") for line in lines)


def test_create_template_names_a_failed_template_the_platform_would_not_delete(monkeypatch):
    failed = _Op("op1", done=True, error=None)
    p, templates = _provider(monkeypatch, [("FAILED", failed)], refuse_delete={_tpl("new")})
    with pytest.raises(SandboxError, match=r"the platform gave no reason; could not delete it \(RuntimeError\)"
                                            r".*engine.delete_version\(\)"):
        _create(p)
    assert templates.deleted == []


def test_create_template_operation_error_before_the_template_lists(monkeypatch):
    p, templates = _provider(monkeypatch, [(None, _Op("op1", done=True, error={"message": "quota exceeded"}))])
    with pytest.raises(SandboxError, match="failed: quota exceeded$"):
        _create(p)
    assert templates.deleted == []  # nothing listed yet, nothing to reap


def test_create_template_operation_error_after_the_template_lists_reaps_it(monkeypatch):
    p, templates = _provider(monkeypatch, [("PROVISIONING", _Op("op1", done=True, error={"message": "INTERNAL"}))])
    with pytest.raises(SandboxError, match=r"failed: INTERNAL; deleted it"):
        _create(p)
    assert templates.deleted == [_tpl("new")]


def test_create_template_times_out_past_the_platform_deadline_and_reaps_the_stuck_one(monkeypatch):
    assert prov.TEMPLATE_TIMEOUT_S > 30 * 60 + 31  # the platform's own verdict comes at 30 min 31 s
    p, templates = _provider(monkeypatch, [("PROVISIONING", _Op("op1"))])
    monkeypatch.setattr(prov, "TEMPLATE_TIMEOUT_S", -1.0)
    lines = []
    with pytest.raises(SandboxError, match=r"still PROVISIONING.*operations/op1.*deleted it.*re-run the deploy"):
        _create(p, log=lines.append)
    assert templates.deleted == [_tpl("new")]
    assert any("stuck template holds capacity" in line for line in lines)


def test_create_template_times_out_before_the_template_ever_lists(monkeypatch):
    p, templates = _provider(monkeypatch, [(None, _Op("op1"))])
    monkeypatch.setattr(prov, "TEMPLATE_TIMEOUT_S", -1.0)
    with pytest.raises(SandboxError, match="nothing listed under that name to delete"):
        _create(p)
    assert templates.deleted == []

"""The remote probe's engine teardown — offline, with a stub engine.

A `dev/` script does not normally get tests. This one does, because the mechanism it
guards costs real money when it fails: the probe runs for many minutes, so it is routinely
wrapped in a `timeout` or interrupted, and SIGTERM/SIGINT kill the process WITHOUT running
`finally`. That happened while the probe was being written and left an engine
billing until it was deleted by hand. These tests pin the behaviors that prevent a repeat,
including an interrupt during the deploy, when there is no engine object yet and the engine
has to be deleted by name. They touch no network and no cloud.
"""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dev"))

probe = pytest.importorskip("live_openrouter_remote_probe")


class _StubEngine:
    """Stands in for a deployed engine: records how many times it was deleted."""

    def __init__(self) -> None:
        self.deletes = 0

    def delete(self) -> None:
        self.deletes += 1


@pytest.fixture(autouse=True)
def _reset_teardown_flag(monkeypatch):
    monkeypatch.setattr(probe, "_TORN_DOWN", False)
    monkeypatch.setattr(probe, "_ENGINE", None)
    monkeypatch.delenv("KEEP", raising=False)


def _deployed(monkeypatch, engine):
    """Put the probe in the state it reaches once sandbox.deploy has returned."""
    monkeypatch.setattr(probe, "_ENGINE", engine)
    return engine


def test_teardown_is_idempotent(monkeypatch):
    """The normal path and a signal can both reach it; the engine dies once."""
    engine = _deployed(monkeypatch, _StubEngine())
    probe._teardown()
    probe._teardown()
    assert engine.deletes == 1


def test_teardown_respects_keep(monkeypatch):
    """KEEP=1 hands the operator the cleanup instead of deleting."""
    monkeypatch.setenv("KEEP", "1")
    engine = _deployed(monkeypatch, _StubEngine())
    probe._teardown()
    assert engine.deletes == 0


def test_teardown_survives_a_failed_delete(monkeypatch, capsys):
    """A delete that raises must say so loudly, not propagate and hide the verdicts."""

    class Broken(_StubEngine):
        def delete(self) -> None:
            raise RuntimeError("platform said no")

    _deployed(monkeypatch, Broken())
    probe._teardown()  # must not raise
    assert "TEARDOWN FAILED" in capsys.readouterr().out


def test_sigterm_tears_down_and_exits(monkeypatch):
    """The case that leaked an engine: killed mid-run, `finally` never reached."""
    engine = _deployed(monkeypatch, _StubEngine())
    probe._install_signal_teardown()
    try:
        with pytest.raises(SystemExit) as exc:
            os.kill(os.getpid(), signal.SIGTERM)
        assert exc.value.code == 130
        assert engine.deletes == 1
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)


def test_sigterm_during_the_deploy_deletes_by_name(monkeypatch):
    """The window this fixes: interrupted before sandbox.deploy returned.

    There is no engine object yet, but the platform may already have created the engine
    and it bills while it exists. The name is fixed, so the handler looks it up and
    deletes it.
    """
    engine = _StubEngine()
    looked_up = {}

    def fake_get_engine(name, project, location, credentials=None):
        looked_up.update(name=name, project=project, location=location)
        return engine

    monkeypatch.setattr(probe.sandbox, "get_engine", fake_get_engine)
    probe._install_signal_teardown()  # no engine: deploy has not returned
    try:
        with pytest.raises(SystemExit) as exc:
            os.kill(os.getpid(), signal.SIGTERM)
        assert exc.value.code == 130
        assert engine.deletes == 1
        assert looked_up == {"name": probe.NAME, "project": probe.PROJECT,
                             "location": probe.LOCATION}
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)


def test_a_failed_delete_by_name_prints_the_manual_steps(monkeypatch, capsys):
    """If the lookup or the delete fails, say the name so it can be removed by hand."""

    def broken_get_engine(*_a, **_kw):
        raise RuntimeError("lookup failed")

    monkeypatch.setattr(probe.sandbox, "get_engine", broken_get_engine)
    probe._teardown()  # must not raise
    out = capsys.readouterr().out
    assert "COULD NOT DELETE" in out and probe.NAME in out
    assert "lookup failed" in out  # the reason rides on the same line, not a traceback


def test_keep_wins_before_the_deploy_returns(monkeypatch):
    """KEEP=1 must not trigger a delete-by-name either."""
    monkeypatch.setenv("KEEP", "1")
    called = []
    monkeypatch.setattr(probe.sandbox, "get_engine", lambda *a, **k: called.append(1))
    probe._teardown()
    assert called == []

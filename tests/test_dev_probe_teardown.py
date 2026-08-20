"""The remote probe's engine teardown — offline, with a stub engine.

A `dev/` script does not normally get tests. This one does, because the mechanism it
guards costs real money when it fails: the probe runs for ~35-40 min, so it is routinely
wrapped in a `timeout` or interrupted, and SIGTERM/SIGINT kill the process WITHOUT running
`finally`. That happened while the probe was being written and left an Agent Engine
billing until it was deleted by hand. These tests pin the three behaviors that prevent a
repeat; they touch no network and no cloud.
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
    monkeypatch.delenv("KEEP", raising=False)


def test_teardown_is_idempotent():
    """The normal path and a signal can both reach it; the engine dies once."""
    engine = _StubEngine()
    probe._teardown(engine)
    probe._teardown(engine)
    assert engine.deletes == 1


def test_teardown_respects_keep(monkeypatch):
    """KEEP=1 hands the operator the cleanup instead of deleting."""
    monkeypatch.setenv("KEEP", "1")
    engine = _StubEngine()
    probe._teardown(engine)
    assert engine.deletes == 0


def test_teardown_survives_a_failed_delete(capsys):
    """A delete that raises must say so loudly, not propagate and hide the verdicts."""

    class Broken(_StubEngine):
        def delete(self) -> None:
            raise RuntimeError("platform said no")

    probe._teardown(Broken())  # must not raise
    assert "TEARDOWN FAILED" in capsys.readouterr().out


def test_sigterm_tears_down_and_exits():
    """The case that leaked an engine: killed mid-run, `finally` never reached."""
    engine = _StubEngine()
    probe._install_signal_teardown(engine)
    try:
        with pytest.raises(SystemExit) as exc:
            os.kill(os.getpid(), signal.SIGTERM)
        assert exc.value.code == 130
        assert engine.deletes == 1
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)

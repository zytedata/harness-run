"""The sandbox worker closes its memory to same-user processes (the agent's shell)."""

from __future__ import annotations

import subprocess
import sys

import pytest

from harness_run.runtime.sandbox import worker

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="prctl is Linux-only")

# Runs in a child interpreter: the flag cannot be taken back, and the test process should
# stay debuggable. The grandchild is what the agent's shell is to the worker.
_SCRIPT = """
import subprocess, sys
from harness_run.runtime.sandbox import worker
probe = '''
import os
for name in ("mem", "environ", "maps"):
    try:
        open(f"/proc/{os.getppid()}/{name}", "rb").close()
        print(name, "open")
    except PermissionError:
        print(name, "denied")
'''
print("dumpable", worker._dumpable())
worker._refuse_inspection()
print("dumpable", worker._dumpable())
print(subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True).stdout)
"""


def test_a_child_process_cannot_open_the_workers_memory_or_environment():
    out = subprocess.run(
        [sys.executable, "-c", _SCRIPT], capture_output=True, text=True, timeout=30, check=True
    ).stdout.split("\n")
    assert out[:2] == ["dumpable True", "dumpable False"]
    assert out[2:5] == ["mem denied", "environ denied", "maps denied"]


def test_the_test_process_itself_stays_inspectable():
    assert worker._dumpable() is True

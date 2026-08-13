"""Deploy-time agent venv for the local runtime (``spec.packages`` parity with ``gemini``).

``spec.packages`` means "the agent's Python packages" on BOTH backends: ``gemini`` bakes them
into the engine image; ``local`` resolves them here into a **per-engine venv** at
``local.deploy()`` time. A fresh venv per engine (rather than a shared content-hash cache)
keeps runs hermetic — an agent that installs extra packages mid-run mutates only its own
engine's venv — while staying fast: ``uv`` hardlinks from its global package cache, so a warm
provision takes seconds. The interpreter is pinned to the engine contract's Python (3.12,
DESIGN.md §6); ``uv`` provisions it as a managed interpreter if the host lacks it.

This also gives the agent an interpreter + site-packages *separate from the harness venv*
(the env-isolation concern): the venv's ``bin`` leads the agent's PATH.

Stdlib-only at import; ``uv`` is located lazily via its wheel (a direct toolkit dependency).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

# The engine image's interpreter (DESIGN.md §6 deploy contracts) — local parity pins the same.
ENGINE_PYTHON = "3.12"


def _uv_bin() -> str:
    from uv import find_uv_bin  # lazy: the wheel knows the bundled binary's real location

    return find_uv_bin()


def _run(cmd: list[str], what: str) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        # Loud on purpose: declared packages are explicit caller intent, so a resolution
        # failure is a deploy error, not something to paper over at runtime.
        raise RuntimeError(f"{what} failed: {proc.stderr.strip()[-1000:]}")


def provision_venv(root: Path, packages: tuple[str, ...], python: str = ENGINE_PYTHON) -> Path:
    """Create — or reuse — ``root/venv`` with ``packages`` installed; return the venv path.

    Called by ``local.deploy()`` when the spec declares packages. Uses ``uv`` end to end:
    ``uv venv --python 3.12`` (downloads a managed interpreter if needed) then
    ``uv pip install`` (hardlinked from uv's global cache — warm installs take seconds).

    Idempotent: a venv already provisioned in ``root`` is reused, not replaced. Re-attaching
    to a session from another process goes through ``local.deploy()`` into the same workdir,
    so replacing would discard an agent's mid-run installs (supported, see module docstring) —
    and ``uv`` >= 0.12 errors on ``uv venv`` over an existing venv anyway. The declared
    ``packages`` are (re)applied either way; a leftover directory that lacks an interpreter
    is removed and provisioned from scratch.
    """
    venv = Path(root) / "venv"
    uv = _uv_bin()
    if not (venv / "bin" / "python").exists():
        shutil.rmtree(venv, ignore_errors=True)
        _run([uv, "venv", str(venv), "--python", python], "uv venv")
    _run([uv, "pip", "install", "--python", str(venv / "bin" / "python"), *packages], "uv pip install")
    return venv


def venv_agent_env(venv: Path, base_path: str | None) -> dict[str, str]:
    """The agent-subprocess env entries that activate ``venv``.

    ``PATH`` composes ``venv/bin`` onto ``base_path`` (the caller's ``spec.env`` PATH when
    set, else the ambient one) so activation never *discards* an existing PATH — the harness
    then prepends the uv dir on top of whatever we return.
    """
    import os

    base = base_path or os.environ.get("PATH", "")
    bin_dir = str(venv / "bin")
    return {
        "VIRTUAL_ENV": str(venv),
        "PATH": f"{bin_dir}:{base}" if base else bin_dir,
    }

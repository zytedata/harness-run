"""spec.packages on the local runtime: deploy-time venv provisioning + agent-env activation.

The uv subprocess calls are mocked (no network / interpreter downloads in unit tests); the
composition and wiring are what these lock in. The real `uv venv` + install path is exercised
manually / by consumers (the eval harness runs it in anger).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_run import AgentSpec, local
from agent_run.runtime import venv as venv_mod


def test_provision_venv_uses_engine_python_and_surfaces_failure(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, what):
        calls.append(cmd)

    monkeypatch.setattr(venv_mod, "_run", fake_run)
    monkeypatch.setattr(venv_mod, "_uv_bin", lambda: "/opt/uv")

    got = venv_mod.provision_venv(tmp_path, ("scrapy>=2.11", "scrapy-zyte-api"))
    assert got == tmp_path / "venv"
    assert calls[0] == ["/opt/uv", "venv", str(got), "--python", "3.12"]  # engine-contract Python
    assert calls[1][:4] == ["/opt/uv", "pip", "install", "--python"]
    assert calls[1][-2:] == ["scrapy>=2.11", "scrapy-zyte-api"]

    # A failing uv command raises loudly (declared packages are explicit intent).
    def boom(cmd, what):
        raise RuntimeError(f"{what} failed: no such package")

    monkeypatch.setattr(venv_mod, "_run", boom)
    with pytest.raises(RuntimeError, match="failed"):
        venv_mod.provision_venv(tmp_path, ("nope-does-not-exist",))


def _plant_venv(root, version="3.12.3"):
    """A fake provisioned venv: interpreter, pyvenv.cfg (uv's layout), and a marker file
    standing in for an agent's mid-run install."""
    marker = root / "venv" / "lib" / "mid-run-install"
    py = root / "venv" / "bin" / "python"
    for f in (marker, py):
        f.parent.mkdir(parents=True, exist_ok=True)
        f.touch()
    (root / "venv" / "pyvenv.cfg").write_text(
        f"home = /usr/bin\nimplementation = CPython\nversion_info = {version}\n"
    )
    return marker


def test_provision_venv_reuses_existing_venv(monkeypatch, tmp_path):
    """Re-deploy into the same workdir must not replace the venv (uv >= 0.12 errors on
    an existing venv; replacing would also discard an agent's mid-run installs)."""
    calls = []
    monkeypatch.setattr(venv_mod, "_run", lambda cmd, what: calls.append(cmd))
    monkeypatch.setattr(venv_mod, "_uv_bin", lambda: "/opt/uv")
    marker = _plant_venv(tmp_path)

    got = venv_mod.provision_venv(tmp_path, ("scrapy",))
    assert got == tmp_path / "venv"
    assert [c[1] for c in calls] == ["pip"]  # no `uv venv` — the existing venv is reused
    assert marker.exists()  # the venv's contents survived
    # ...but the declared packages are still (re)applied on the reused venv.
    assert calls[0][-1] == "scrapy"


def test_provision_venv_replaces_wrong_python_version(monkeypatch, tmp_path):
    """A venv whose interpreter doesn't match the engine contract's Python is re-provisioned:
    a persistent workdir must not silently keep the old interpreter across an
    ENGINE_PYTHON bump (local/sandbox parity is the whole point of the venv)."""
    calls = []
    monkeypatch.setattr(venv_mod, "_run", lambda cmd, what: calls.append(cmd))
    monkeypatch.setattr(venv_mod, "_uv_bin", lambda: "/opt/uv")
    marker = _plant_venv(tmp_path, version="3.11.9")

    venv_mod.provision_venv(tmp_path, ("scrapy",))
    assert [c[1] for c in calls] == ["venv", "pip"]  # provisioned from scratch
    assert not marker.exists()  # the out-of-contract venv was cleared
    assert calls[0][3:] == ["--python", "3.12"]


def test_provision_venv_replaces_venv_without_pyvenv_cfg(monkeypatch, tmp_path):
    """An interpreter with no readable pyvenv.cfg version is treated as out of contract."""
    calls = []
    monkeypatch.setattr(venv_mod, "_run", lambda cmd, what: calls.append(cmd))
    monkeypatch.setattr(venv_mod, "_uv_bin", lambda: "/opt/uv")
    _plant_venv(tmp_path)
    (tmp_path / "venv" / "pyvenv.cfg").unlink()

    venv_mod.provision_venv(tmp_path, ("scrapy",))
    assert [c[1] for c in calls] == ["venv", "pip"]


def test_provision_venv_replaces_dir_without_interpreter(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(venv_mod, "_run", lambda cmd, what: calls.append(cmd))
    monkeypatch.setattr(venv_mod, "_uv_bin", lambda: "/opt/uv")
    leftover = tmp_path / "venv" / "pyvenv.cfg"  # a venv dir with no bin/python: unusable
    leftover.parent.mkdir(parents=True)
    leftover.touch()

    venv_mod.provision_venv(tmp_path, ("scrapy",))
    assert [c[1] for c in calls] == ["venv", "pip"]  # provisioned from scratch
    assert not leftover.exists()  # the unusable leftover was cleared first


def test_venv_agent_env_composes_path():
    env = venv_mod.venv_agent_env(Path("/w/venv"), base_path="/custom:/usr/bin")
    assert env["VIRTUAL_ENV"] == "/w/venv"
    assert env["PATH"] == "/w/venv/bin:/custom:/usr/bin"  # caller base kept, venv leads
    # Without a caller base, the ambient PATH is the base (never empty in practice).
    env2 = venv_mod.venv_agent_env(Path("/w/venv"), base_path=None)
    assert env2["PATH"].startswith("/w/venv/bin:")


def test_local_deploy_provisions_and_activates_packages_venv(monkeypatch, tmp_path):
    provisioned = {}

    def fake_provision(root, packages, python="3.12"):
        provisioned.update(root=Path(root), packages=packages)
        return Path(root) / "venv"

    monkeypatch.setattr(venv_mod, "provision_venv", fake_provision)

    spec = AgentSpec(name="a", model="m", packages=["scrapy"], env={"PATH": "/spec/bin"})
    engine = local.deploy(spec, workdir=str(tmp_path))

    assert provisioned["packages"] == ("scrapy",)
    assert engine._agent_env["VIRTUAL_ENV"] == str(tmp_path / "venv")
    # venv bin leads; the spec.env PATH survives as the base (issue-2 composition).
    assert engine._agent_env["PATH"] == f"{tmp_path}/venv/bin:/spec/bin"

    # No packages -> no venv, no runtime env injected (current fast-path behavior unchanged).
    plain = local.deploy(AgentSpec(name="b", model="m"), workdir=str(tmp_path / "b"))
    assert plain._agent_env is None
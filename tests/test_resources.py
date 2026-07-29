"""Worker CPU/RAM self-sampling (resources.py): cgroup readers, sampler, turn wiring.

Why it exists: job containers run in a Google tenant project, so platform metrics and OOM
kills are invisible from the user project — the worker samples itself and ships the record
to Cloud Logging, where it survives a mid-turn kill.
"""

from __future__ import annotations

import asyncio
import time

from remote_agent_toolkit import AgentSpec
from remote_agent_toolkit.events import AgentEvent
from remote_agent_toolkit.runtime.gemini import adk_agent, resources


def _write(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def _cgroup_v2(root, current="1073741824", peak="2147483648", limit="4294967296"):
    _write(root, "memory.current", current)
    _write(root, "memory.peak", peak)
    _write(root, "memory.max", limit)
    _write(root, "cpu.stat", "usage_usec 5000000\nuser_usec 4000000\n")


def test_read_usage_cgroup_v2(tmp_path):
    _cgroup_v2(tmp_path)
    assert resources.read_usage(tmp_path) == {
        "memory_current_bytes": 1 << 30,
        "memory_peak_bytes": 2 << 30,
        "memory_limit_bytes": 4 << 30,
        "cpu_usec": 5_000_000,
    }
    # An unlimited cgroup reports "max": the limit key is simply absent (no pressure math).
    _write(tmp_path, "memory.max", "max")
    assert "memory_limit_bytes" not in resources.read_usage(tmp_path)


def test_read_usage_cgroup_v1_and_empty(tmp_path):
    _write(tmp_path, "memory/memory.usage_in_bytes", str(1 << 30))
    _write(tmp_path, "memory/memory.max_usage_in_bytes", str(2 << 30))
    # v1 encodes "unlimited" as a huge sentinel — treated as no limit.
    _write(tmp_path, "memory/memory.limit_in_bytes", str(1 << 62))
    _write(tmp_path, "cpuacct/cpuacct.usage", "5000000000")  # ns -> usec
    usage = resources.read_usage(tmp_path)
    assert usage["memory_current_bytes"] == 1 << 30
    assert usage["memory_peak_bytes"] == 2 << 30
    assert "memory_limit_bytes" not in usage
    assert usage["cpu_usec"] == 5_000_000
    assert resources.read_usage(tmp_path / "nothing-here") == {}  # no cgroup -> no data


def test_sampler_samples_pressure_once_and_enriches(tmp_path):
    _cgroup_v2(tmp_path, current=str(4 << 30), peak=str(4 << 30), limit=str(4 << 30))
    side, main = [], []
    sampler = resources.ResourceSampler(
        "sid-1", side_emit=side.append, on_event=main.append, sample_s=0.01, root=tmp_path
    ).start()
    for _ in range(200):  # a few sampling periods
        if len(side) >= 3:
            break
        time.sleep(0.01)
    sampler.stop()

    assert side and side[0].raw["event"] == "resource_sample"
    assert side[0].raw["memory_current_bytes"] == 4 << 30
    # 100% of the limit crossed the 85% threshold — exactly ONE visible pressure event.
    assert [e.raw["event"] for e in main] == ["memory_pressure"]
    assert "resource_limits" in main[0].summary  # points at the deploy-time fix

    raw: dict = {}
    sampler.enrich_result(raw)
    assert raw == {
        "memory_peak_bytes": 4 << 30, "memory_limit_bytes": 4 << 30, "cpu_usec": 5_000_000,
    }


def test_sampler_no_pressure_below_threshold(tmp_path):
    _cgroup_v2(tmp_path, current=str(1 << 30), peak=str(1 << 30), limit=str(4 << 30))
    side, main = [], []
    sampler = resources.ResourceSampler(
        "sid-2", side_emit=side.append, on_event=main.append, sample_s=0.01, root=tmp_path
    ).start()
    for _ in range(100):
        if side:
            break
        time.sleep(0.01)
    sampler.stop()
    assert side and not main  # samples flow; 25% of the limit stays quiet


def test_start_sampler_disabled_or_no_cgroup(tmp_path, monkeypatch):
    _cgroup_v2(tmp_path)
    monkeypatch.setenv("AGENT_RESOURCE_SAMPLE_S", "0")
    assert resources.start_sampler("sid", on_event=lambda e: None, root=tmp_path) is None
    monkeypatch.setenv("AGENT_RESOURCE_SAMPLE_S", "not-a-number")
    assert resources.start_sampler("sid", on_event=lambda e: None, root=tmp_path) is None
    monkeypatch.delenv("AGENT_RESOURCE_SAMPLE_S")
    # Readable cgroup + default cadence -> a live sampler (stopped right away).
    sampler = resources.start_sampler(
        "sid", on_event=lambda e: None, side_sink=type("S", (), {"emit": lambda self, e: None})(),
        root=tmp_path,
    )
    assert sampler is not None
    sampler.stop()
    # No cgroup at all -> disabled (e.g. local/dev containers without the mount).
    assert resources.start_sampler(
        "sid", on_event=lambda e: None, root=tmp_path / "missing"
    ) is None


def test_run_turn_stamps_peak_into_result(tmp_path, monkeypatch):
    """End-to-end: the sampler enriches the terminal result raw with the memory peak."""
    from remote_agent_toolkit.ports.eventsink import InMemorySink

    monkeypatch.setenv("AGENT_JOBS_ROOT", str(tmp_path / "jobs"))
    monkeypatch.chdir(tmp_path)  # no baked skills dir on the lookup paths
    cg = tmp_path / "cg"
    _cgroup_v2(cg, current=str(2 << 30), peak=str(3 << 30), limit=str(4 << 30))
    monkeypatch.setenv("AGENT_RESOURCE_SAMPLE_S", "0.01")
    # Route the sampler at the fake cgroup + a capturing side sink.
    real_start = resources.start_sampler
    side_events = []

    def patched_start(session_id, on_event, side_sink=None, root=None):
        return real_start(
            session_id, on_event,
            side_sink=type("S", (), {"emit": lambda self, e: side_events.append(e)})(),
            root=cg,
        )

    monkeypatch.setattr(resources, "start_sampler", patched_start)

    class SlowDoneHarness:
        async def run(self, spec, rc):
            await asyncio.sleep(0.15)  # a few sampling periods
            yield AgentEvent(kind="result", summary="done",
                             raw={"subtype": "success", "is_error": False})

    import remote_agent_toolkit.harness.claude_code as harness_mod
    import remote_agent_toolkit.ports.eventsink as eventsink_mod
    monkeypatch.setattr(harness_mod, "ClaudeCodeHarness", SlowDoneHarness)
    monkeypatch.setattr(eventsink_mod, "CloudLoggingSink",
                        lambda **kw: InMemorySink(session_id="77"))

    spec = AgentSpec(name="w", model="m")
    agent = adk_agent.build_agent(spec)

    async def drive():
        return [ev async for ev in agent._run_turn(spec, "77", "go", None)]

    events = asyncio.run(drive())
    result = events[-1]
    assert result.custom_metadata["kind"] == "result"
    raw = result.custom_metadata["raw"]
    assert raw["memory_peak_bytes"] == 3 << 30
    assert raw["memory_limit_bytes"] == 4 << 30
    assert side_events and side_events[0].raw["event"] == "resource_sample"

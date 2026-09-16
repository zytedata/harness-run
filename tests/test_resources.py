"""In-sandbox CPU/RAM sampling (runtime/gemini/resources.py): reads, the sampler, the rows."""
import datetime as dt
import time
from pathlib import Path

from remote_agent_toolkit.events import AgentEvent
from remote_agent_toolkit.runtime.gemini import resources


def _v1_tree(root: Path, *, usage=575_750_144, limit=None, cpu_ns=3_420_000_000, meminfo_kb=1_048_576) -> Path:
    (root / "memory").mkdir(parents=True)
    (root / "cpuacct").mkdir()
    (root / "memory" / "memory.usage_in_bytes").write_text(f"{usage}\n")
    # gVisor: the v1 limit file reads as "unlimited"; the template's limit shows as MemTotal.
    (root / "memory" / "memory.limit_in_bytes").write_text(f"{limit if limit is not None else 9223372036854775807}\n")
    (root / "cpuacct" / "cpuacct.usage").write_text(f"{cpu_ns}\n")
    meminfo = root / "meminfo"
    meminfo.write_text(f"MemTotal:        {meminfo_kb} kB\nMemFree:          485748 kB\n")
    return meminfo


def test_read_usage_v1_as_gvisor_mounts_it_takes_the_limit_from_meminfo(tmp_path):
    meminfo = _v1_tree(tmp_path)
    assert resources.read_usage(tmp_path, meminfo) == {
        "memory_current_bytes": 575_750_144, "memory_limit_bytes": 1 << 30, "cpu_usec": 3_420_000}


def test_read_usage_v1_prefers_a_real_cgroup_limit_and_peak(tmp_path):
    meminfo = _v1_tree(tmp_path, limit=2 << 30)
    (tmp_path / "memory" / "memory.max_usage_in_bytes").write_text("700000000\n")
    usage = resources.read_usage(tmp_path, meminfo)
    assert usage["memory_limit_bytes"] == 2 << 30 and usage["memory_peak_bytes"] == 700_000_000


def test_read_usage_v2(tmp_path):
    (tmp_path / "memory.current").write_text("123\n")
    (tmp_path / "memory.peak").write_text("456\n")
    (tmp_path / "memory.max").write_text("max\n")
    (tmp_path / "cpu.stat").write_text("usage_usec 9000\nuser_usec 8000\n")
    assert resources.read_usage(tmp_path, tmp_path / "absent-meminfo") == {
        "memory_current_bytes": 123, "memory_peak_bytes": 456, "cpu_usec": 9000}


def test_read_usage_with_nothing_readable_is_empty(tmp_path):
    assert resources.read_usage(tmp_path / "nope", tmp_path / "nope2") == {}


def _wait(pred, timeout=3.0):
    deadline = time.time() + timeout
    while not pred() and time.time() < deadline:
        time.sleep(0.005)
    return pred()


def test_sampler_mirrors_samples_warns_once_on_pressure_and_enriches_the_result(tmp_path):
    meminfo = _v1_tree(tmp_path, usage=100 << 20, meminfo_kb=1_048_576)  # 100 MiB of 1 GiB
    mirrored, streamed = [], []
    sampler = resources.start_sampler("s" * 32, on_event=streamed.append, sample_emit=mirrored.append,
                                      sample_s=0.01, root=tmp_path, meminfo=meminfo)
    assert sampler is not None
    try:
        assert _wait(lambda: len(mirrored) >= 2)
        first = mirrored[0]
        assert first.kind == "status" and first.raw["event"] == "resource_sample"
        assert first.raw["memory_current_bytes"] == 100 << 20 and first.raw["memory_limit_bytes"] == 1 << 30
        assert dt.datetime.fromisoformat(first.raw["at"]).tzinfo is not None
        assert "mem 100MiB/1024MiB" in first.summary
        assert streamed == []  # nothing on the live stream below the threshold

        (tmp_path / "memory" / "memory.usage_in_bytes").write_text(f"{900 << 20}\n")  # 88 %
        assert _wait(lambda: len(streamed) == 1)
        (tmp_path / "memory" / "memory.usage_in_bytes").write_text(f"{950 << 20}\n")
        assert _wait(lambda: any(e.raw.get("memory_current_bytes") == 950 << 20 for e in mirrored))
        assert len(streamed) == 1  # warned once
        pressure = streamed[0]
        assert pressure.raw["event"] == "memory_pressure" and "memory pressure: 900MiB of 1024MiB (88%)" in pressure.summary
    finally:
        sampler.stop()
    # The result takes one last sample: a spike after the final periodic one still counts.
    (tmp_path / "memory" / "memory.usage_in_bytes").write_text(f"{990 << 20}\n")
    (tmp_path / "cpuacct" / "cpuacct.usage").write_text("5000000000\n")
    before = len(mirrored)
    raw = {"subtype": "success"}
    sampler.enrich_result(raw)
    assert raw["memory_peak_bytes"] == 990 << 20 and raw["memory_limit_bytes"] == 1 << 30 and raw["cpu_usec"] == 5_000_000
    assert raw["subtype"] == "success" and len(mirrored) == before + 1


def test_sampler_is_off_when_disabled_or_nothing_is_readable(tmp_path, monkeypatch):
    meminfo = _v1_tree(tmp_path)
    assert resources.start_sampler("s", on_event=lambda e: None, sample_emit=None, sample_s=0, root=tmp_path, meminfo=meminfo) is None
    assert resources.start_sampler("s", on_event=lambda e: None, sample_emit=None, sample_s=1, root=tmp_path / "x", meminfo=tmp_path / "y") is None
    monkeypatch.setenv(resources.SAMPLE_S_ENV, "0")
    assert resources.start_sampler("s", on_event=lambda e: None, sample_emit=None, root=tmp_path, meminfo=meminfo) is None
    monkeypatch.setenv(resources.SAMPLE_S_ENV, "not-a-number")
    assert resources.start_sampler("s", on_event=lambda e: None, sample_emit=None, root=tmp_path, meminfo=meminfo) is None
    monkeypatch.setenv(resources.SAMPLE_S_ENV, "0.01")
    got = []
    sampler = resources.start_sampler("s", on_event=lambda e: None, sample_emit=got.append, root=tmp_path, meminfo=meminfo)
    assert sampler is not None and _wait(lambda: len(got) >= 1)
    sampler.stop()


def test_sample_rows_keeps_only_samples_and_parses_the_time():
    events = [
        AgentEvent(kind="status", summary="turn started", raw={"event": "turn_started"}),
        AgentEvent(kind="status", summary="resource sample", raw={
            "event": "resource_sample", "at": "2026-09-16T17:00:00+00:00", "turn_id": "t",
            "memory_current_bytes": 5, "memory_limit_bytes": 10, "cpu_usec": 7}),
        AgentEvent(kind="status", summary="memory pressure", raw={"event": "memory_pressure", "memory_current_bytes": 9}),
        AgentEvent(kind="status", summary="resource sample", raw={"event": "resource_sample", "at": "garbage", "cpu_usec": 8}),
        AgentEvent(kind="result", summary="42", raw={"memory_peak_bytes": 9}),
    ]
    rows = resources.sample_rows(events)
    assert rows == [
        {"time": dt.datetime(2026, 9, 16, 17, tzinfo=dt.UTC), "memory_current_bytes": 5, "memory_limit_bytes": 10, "cpu_usec": 7},
        {"time": None, "cpu_usec": 8},
    ]
    assert [resources.is_sample(e) for e in events] == [False, True, False, True, False]


def test_run_result_carries_the_sampled_high_water_marks():
    from remote_agent_toolkit.runtime._run import build_result
    from remote_agent_toolkit.spec import AgentSpec

    spec = AgentSpec(name="a", model="m")
    ev = AgentEvent(kind="result", summary="42", raw={"subtype": "success", "num_turns": 1,
                                                     "memory_peak_bytes": 600, "memory_limit_bytes": 1000, "cpu_usec": 5})
    result, _ = build_result(ev, "sid", spec)
    assert result.resources == {"memory_peak_bytes": 600, "memory_limit_bytes": 1000, "cpu_usec": 5}
    bare, _ = build_result(AgentEvent(kind="result", summary="42", raw={"subtype": "success", "memory_peak_bytes": True}), "sid", spec)
    assert bare.resources is None

"""Worker self-monitoring: CPU/RAM sampled from inside the sandbox (OOM forensics).

Why self-sampling: the platform exposes **nothing** about a sandbox's resources from outside
(checked 2026-09-16: no Cloud Monitoring metric type mentions sandboxes, the
``reasoning_engine/*/allocation_time`` series report only for Agent Runtime engines, the
sandbox resource carries no usage fields, Cloud Logging holds only the audit entries). The
worker itself is the only vantage point — and gVisor does mount cgroup **v1** accounting
(``cpuacct.usage``, ``memory/memory.usage_in_bytes``; probed live on a running turn) and
reports the template's memory limit as ``/proc/meminfo`` ``MemTotal``.

Three outputs, three destinations, none of which needs a Google identity:

1. **Periodic samples** → the turn's GCS **event mirror only** (``event: resource_sample``;
   default every 20 s). Not the live ``/events`` stream a ``Run`` consumes — a sample every
   20 s would drown the turn for a watcher — so the record is durable and survives a
   mid-turn kill (the last sample lands at most one interval before death, which is what an
   OOM post-mortem needs). ``Session.resource_samples()`` reads them back; ``history()``
   skips them unless asked.
2. **One visible "memory pressure" status event** on the main stream the first time usage
   crosses :data:`PRESSURE_FRACTION` of the limit — someone watching the run sees the
   warning before the kill, and it lands in the durable history next to the death.
3. **Peak / limit / CPU stamped into the terminal result** (``memory_peak_bytes``,
   ``memory_limit_bytes``, ``cpu_usec`` on the result event's ``raw``) — every finished
   turn, failed ones included, reports its high-water mark for free.

Reads cgroup v2 first (``memory.current`` / ``memory.peak`` / ``memory.max``, ``cpu.stat``),
then v1 (``memory/memory.usage_in_bytes`` …, ``cpuacct/cpuacct.usage``); a missing or
"unlimited" cgroup memory limit falls back to ``MemTotal`` (what gVisor sets it to). Anything
unreadable just drops that key — never an exception.

Sampling is a daemon THREAD (not an asyncio task): emits are sync/best-effort, and the
thread keeps sampling while the event loop is busy driving the harness.
"""

from __future__ import annotations

import datetime as _dt
import os
import threading
from pathlib import Path
from typing import Any, Callable

from ...events import AgentEvent

#: First crossing of this fraction of the memory limit emits the visible pressure event.
PRESSURE_FRACTION = 0.85

#: Sampling cadence (seconds); ``RATK_RESOURCE_SAMPLE_S`` in the worker's env overrides, ``0`` disables.
DEFAULT_SAMPLE_S = 20.0
SAMPLE_S_ENV = "RATK_RESOURCE_SAMPLE_S"

SAMPLE_EVENT = "resource_sample"
PRESSURE_EVENT = "memory_pressure"

# cgroup v1 reports "unlimited" as a huge page-rounded number; treat anything this large
# (>= 1 EiB) as no limit at all.
_V1_UNLIMITED = 1 << 60

DEFAULT_CGROUP_ROOT = Path("/sys/fs/cgroup")
DEFAULT_MEMINFO = Path("/proc/meminfo")


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _read_bytes_value(path: Path) -> int | None:
    """An integer byte count, or ``None`` for missing/unlimited (``max`` / v1 sentinel)."""
    text = _read_text(path)
    if text is None or text == "max" or not text.isdigit():
        return None
    value = int(text)
    return None if value >= _V1_UNLIMITED else value


def _meminfo_total(meminfo: Path) -> int | None:
    text = _read_text(meminfo) or ""
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) * 1024  # kB
    return None


def read_usage(root: Path = DEFAULT_CGROUP_ROOT, meminfo: Path = DEFAULT_MEMINFO) -> dict[str, int]:
    """This container's memory/CPU usage; missing data → missing keys.

    Keys (all optional): ``memory_current_bytes``, ``memory_peak_bytes`` (the kernel's
    high-water mark where available), ``memory_limit_bytes``, ``cpu_usec`` (cumulative).
    """
    usage: dict[str, int] = {}
    if (root / "memory.current").exists():  # cgroup v2 (unified hierarchy)
        pairs = {
            "memory_current_bytes": root / "memory.current",
            "memory_peak_bytes": root / "memory.peak",  # kernel >= 5.19
            "memory_limit_bytes": root / "memory.max",
        }
        for line in (_read_text(root / "cpu.stat") or "").splitlines():
            if line.startswith("usage_usec "):
                usage["cpu_usec"] = int(line.split()[1])
                break
    else:  # cgroup v1 (split hierarchies; what gVisor mounts)
        pairs = {
            "memory_current_bytes": root / "memory" / "memory.usage_in_bytes",
            "memory_peak_bytes": root / "memory" / "memory.max_usage_in_bytes",
            "memory_limit_bytes": root / "memory" / "memory.limit_in_bytes",
        }
        nanos = _read_bytes_value(root / "cpuacct" / "cpuacct.usage")
        if nanos is not None:
            usage["cpu_usec"] = nanos // 1000
    for key, path in pairs.items():
        value = _read_bytes_value(path)
        if value is not None:
            usage[key] = value
    if "memory_limit_bytes" not in usage:
        total = _meminfo_total(meminfo)
        if total is not None:
            usage["memory_limit_bytes"] = total
    return usage


def _mib(n: int) -> str:
    return f"{n / (1 << 20):.0f}MiB"


class ResourceSampler:
    """Daemon thread sampling :func:`read_usage` every ``sample_s`` for one turn.

    ``sample_emit`` receives the periodic sample events (→ the mirror); ``on_event`` the
    single memory-pressure event (→ the main stream). Both are called from the sampler
    THREAD and must be thread-safe — the mirror stream's ``append`` and the turn record's
    ``append`` both are. Everything is best-effort: a sampling or emit crash kills only the
    sampler, never the turn.
    """

    def __init__(
        self,
        session_id: str,
        sample_emit: Callable[[AgentEvent], None],
        on_event: Callable[[AgentEvent], None],
        sample_s: float = DEFAULT_SAMPLE_S,
        root: Path = DEFAULT_CGROUP_ROOT,
        meminfo: Path = DEFAULT_MEMINFO,
    ) -> None:
        self._session_id = session_id
        self._sample_emit = sample_emit
        self._on_event = on_event
        self._sample_s = sample_s
        self._root = root
        self._meminfo = meminfo
        self._stop = threading.Event()
        self._pressure_fired = False
        self._peak_bytes: int | None = None
        self._limit_bytes: int | None = None
        self._cpu_usec: int | None = None
        self.samples = 0
        self._thread = threading.Thread(
            target=self._run, name=f"ratk-resources-{session_id[:8]}", daemon=True
        )

    def start(self) -> ResourceSampler:
        # The first sample is taken synchronously: a turn that dies at once still reports
        # its baseline, and the result's peak never depends on the thread's timing.
        try:
            self._sample()
        except Exception:  # noqa: BLE001 — monitoring must never take the turn down
            pass
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self._sample_s):
            try:
                self._sample()
            except Exception:  # noqa: BLE001 — monitoring must never take the turn down
                pass

    def _sample(self) -> None:
        usage = read_usage(self._root, self._meminfo)
        if not usage:
            return
        current = usage.get("memory_current_bytes")
        limit = usage.get("memory_limit_bytes")
        self._limit_bytes = limit if limit is not None else self._limit_bytes
        self._cpu_usec = usage.get("cpu_usec", self._cpu_usec)
        observed = max(
            (v for v in (usage.get("memory_peak_bytes"), current) if v is not None),
            default=None,
        )
        if observed is not None:
            self._peak_bytes = max(self._peak_bytes or 0, observed)
        summary = "resource sample"
        if current is not None:
            summary += f": mem {_mib(current)}" + (f"/{_mib(limit)}" if limit else "")
        self.samples += 1
        self._sample_emit(AgentEvent(
            kind="status", summary=summary,
            raw={"event": SAMPLE_EVENT, "at": _dt.datetime.now(_dt.UTC).isoformat(), **usage},
        ))
        if (
            not self._pressure_fired
            and current is not None and limit
            and current / limit >= PRESSURE_FRACTION
        ):
            self._pressure_fired = True
            self._on_event(AgentEvent(
                kind="status",
                summary=(
                    f"memory pressure: {_mib(current)} of {_mib(limit)} "
                    f"({current / limit:.0%}) — the platform kills the sandbox at the "
                    "limit; consider deploying with higher resource_limits"
                ),
                raw={"event": PRESSURE_EVENT, **usage},
            ))

    def enrich_result(self, raw: dict) -> None:
        """Stamp the observed peak/limit/CPU into a terminal result event's ``raw``.

        Samples once more first: a turn shorter than the cadence would otherwise report
        only its start-of-turn baseline (the worker before the harness started), and the
        CPU total is cumulative, so the last reading is the one that counts.
        """
        try:
            self._sample()
        except Exception:  # noqa: BLE001 — best-effort, like every sample
            pass
        if self._peak_bytes is not None:
            raw.setdefault("memory_peak_bytes", self._peak_bytes)
        if self._limit_bytes is not None:
            raw.setdefault("memory_limit_bytes", self._limit_bytes)
        if self._cpu_usec is not None:
            raw.setdefault("cpu_usec", self._cpu_usec)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


def start_sampler(
    session_id: str,
    on_event: Callable[[AgentEvent], None],
    sample_emit: Callable[[AgentEvent], None] | None,
    *,
    sample_s: float | None = None,
    root: Path = DEFAULT_CGROUP_ROOT,
    meminfo: Path = DEFAULT_MEMINFO,
) -> ResourceSampler | None:
    """Start the per-turn sampler (``None`` when disabled or nothing is readable).

    ``sample_s`` ``None`` reads :data:`SAMPLE_S_ENV` (default 20; ``0``/negative or a
    non-number disables). ``sample_emit`` ``None`` (no mirror) keeps the pressure event and
    the result's peak but drops the periodic samples. A first read that yields nothing (no
    cgroup, no meminfo) disables sampling for the turn.
    """
    if sample_s is None:
        try:
            sample_s = float(os.environ.get(SAMPLE_S_ENV, DEFAULT_SAMPLE_S))
        except ValueError:
            return None
    if sample_s <= 0 or not read_usage(root, meminfo):
        return None
    return ResourceSampler(
        session_id, sample_emit=sample_emit or (lambda ev: None), on_event=on_event,
        sample_s=sample_s, root=root, meminfo=meminfo,
    ).start()


def is_sample(event: AgentEvent) -> bool:
    return event.kind == "status" and (event.raw or {}).get("event") == SAMPLE_EVENT


def sample_rows(events: list[AgentEvent]) -> list[dict[str, Any]]:
    """The ``resource_sample`` events among ``events`` as rows: ``time`` (aware datetime,
    the worker's clock) + the usage keys; input order kept."""
    rows: list[dict[str, Any]] = []
    for ev in events:
        if not is_sample(ev):
            continue
        raw = dict(ev.raw or {})
        at = raw.pop("at", None)
        try:
            when = _dt.datetime.fromisoformat(at) if isinstance(at, str) else None
        except ValueError:
            when = None
        row: dict[str, Any] = {"time": when}
        row.update({k: v for k, v in raw.items() if k.startswith(("memory_", "cpu_"))})
        rows.append(row)
    return rows

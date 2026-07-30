"""Worker self-monitoring: CPU/RAM sampled from inside the job container (OOM forensics).

Why self-sampling: query jobs execute as Cloud Run jobs in a Google TENANT project, so their
container metrics (``run.googleapis.com/container/*``) and kill events are invisible from the
user project (verified live 2026-07-30: zero time series there). The worker itself is the
only vantage point with access to its own usage — via its cgroup. Samples are shipped to
Cloud Logging in the user project, so the record SURVIVES a mid-turn kill: the last sample
lands at most one interval before death, which is exactly what an OOM post-mortem needs.

Three outputs, three destinations:

1. **Periodic samples** → a side log (:data:`RESOURCES_LOG`, labelled ``session_id``) —
   deliberately NOT the event stream clients tail; a ~20s cadence would drown the turn
   history. Query it per session, or chart it with a log-based metric.
2. **One visible "memory pressure" status event** on the main event stream the first time
   usage crosses :data:`PRESSURE_FRACTION` of the limit — someone watching the run sees the
   warning before a kill, and it lands in the durable history next to the death.
3. **Peak/limit stamped into the terminal result** (``memory_peak_bytes`` etc. in the result
   event's ``raw``, via :meth:`ResourceSampler.enrich_result`) — every completed turn
   reports its high-water mark for free.

Reads cgroup v2 first (``/sys/fs/cgroup/memory.current`` / ``memory.peak`` / ``memory.max``,
``cpu.stat``), falling back to cgroup v1; anything missing just drops that key — never an
exception, and on an unlimited cgroup the limit key is absent (no pressure events).

Sampling is a daemon THREAD (not an asyncio task): emits are sync/best-effort, and the
thread keeps sampling while the event loop is busy driving the harness.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Callable

from ...events import AgentEvent

#: Side log for periodic samples (same Cloud Logging plumbing as the steps log).
RESOURCES_LOG = "remote_agent_toolkit_resources"

#: First crossing of this fraction of the memory limit emits the visible pressure event.
PRESSURE_FRACTION = 0.85

#: Sampling cadence (seconds); the ``AGENT_RESOURCE_SAMPLE_S`` env overrides, ``0`` disables.
DEFAULT_SAMPLE_S = 20.0

# cgroup v1 reports "unlimited" as a huge page-rounded number; treat anything this large
# (>= 1 EiB) as no limit at all.
_V1_UNLIMITED = 1 << 60


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


def read_usage(root: Path = Path("/sys/fs/cgroup")) -> dict[str, int]:
    """This container's memory/CPU usage from its cgroup; missing data → missing keys.

    Keys (all optional): ``memory_current_bytes``, ``memory_peak_bytes`` (kernel
    high-water mark where available), ``memory_limit_bytes``, ``cpu_usec`` (cumulative).
    """
    usage: dict[str, int] = {}
    if (root / "memory.current").exists():  # cgroup v2 (unified hierarchy)
        pairs = {
            "memory_current_bytes": root / "memory.current",
            "memory_peak_bytes": root / "memory.peak",  # kernel >= 5.19
            "memory_limit_bytes": root / "memory.max",
        }
        cpu_stat = _read_text(root / "cpu.stat") or ""
        for line in cpu_stat.splitlines():
            if line.startswith("usage_usec "):
                usage["cpu_usec"] = int(line.split()[1])
                break
    else:  # cgroup v1 (split hierarchies)
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
    return usage


def _mib(n: int) -> str:
    return f"{n / (1 << 20):.0f}MiB"


class ResourceSampler:
    """Daemon thread sampling :func:`read_usage` every ``sample_s`` for one turn.

    ``side_emit`` receives the periodic sample events (→ the :data:`RESOURCES_LOG` side
    log); ``on_event`` receives the single memory-pressure event (→ the main stream: sink +
    mirror). Both are called from the sampler THREAD and must be thread-safe — the Cloud
    Logging emit and a list append both are. Everything is best-effort: a sampling or emit
    crash kills only the sampler, never the turn.
    """

    def __init__(
        self,
        session_id: str,
        side_emit: Callable[[AgentEvent], None],
        on_event: Callable[[AgentEvent], None],
        sample_s: float = DEFAULT_SAMPLE_S,
        root: Path = Path("/sys/fs/cgroup"),
    ) -> None:
        self._session_id = session_id
        self._side_emit = side_emit
        self._on_event = on_event
        self._sample_s = sample_s
        self._root = root
        self._stop = threading.Event()
        self._pressure_fired = False
        self._peak_bytes: int | None = None
        self._limit_bytes: int | None = None
        self._cpu_usec: int | None = None
        self._thread = threading.Thread(
            target=self._run, name=f"ratk-resources-{session_id[:8]}", daemon=True
        )

    def start(self) -> ResourceSampler:
        self._thread.start()
        return self

    def _run(self) -> None:
        while True:
            try:
                self._sample()
            except Exception:  # noqa: BLE001 — monitoring must never take the turn down
                pass
            if self._stop.wait(self._sample_s):
                return

    def _sample(self) -> None:
        usage = read_usage(self._root)
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
        self._side_emit(AgentEvent(
            kind="status", summary=summary, raw={"event": "resource_sample", **usage},
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
                    f"({current / limit:.0%}) — the platform OOM-kills the worker at the "
                    "limit; consider deploying with higher resource_limits"
                ),
                raw={"event": "memory_pressure", **usage},
            ))

    def enrich_result(self, raw: dict) -> None:
        """Stamp the observed peak/limit/CPU into a terminal result event's ``raw``."""
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
    side_sink: Any | None = None,
    root: Path = Path("/sys/fs/cgroup"),
) -> ResourceSampler | None:
    """Start the per-turn sampler (``None`` when disabled or nothing is readable).

    Cadence comes from ``AGENT_RESOURCE_SAMPLE_S`` (default ``20``; ``0``/negative or a
    non-number disables). ``side_sink`` overrides the sample destination for tests; by
    default samples go to :data:`RESOURCES_LOG` via a dedicated ``CloudLoggingSink``.
    A first read that yields nothing (no cgroup mounted) disables sampling for the turn.
    """
    try:
        sample_s = float(os.environ.get("AGENT_RESOURCE_SAMPLE_S", DEFAULT_SAMPLE_S))
    except ValueError:
        return None
    if sample_s <= 0 or not read_usage(root):
        return None
    if side_sink is None:
        from ...ports.eventsink import CloudLoggingSink

        side_sink = CloudLoggingSink(session_id=session_id, log_name=RESOURCES_LOG)
    return ResourceSampler(
        session_id, side_emit=side_sink.emit, on_event=on_event, sample_s=sample_s, root=root
    ).start()


def _sample_rows(entries: Any) -> list[dict]:
    """Map raw Cloud Logging entries to sample rows (pure; separated for offline tests).

    Each row: ``{"time": <datetime>, **usage keys}`` — e.g. ``memory_current_bytes``,
    ``memory_limit_bytes``, ``cpu_usec``. Malformed entries are skipped.
    """
    rows: list[dict] = []
    for entry in entries:
        payload = entry.payload if isinstance(entry.payload, dict) else {}
        raw = payload.get("raw") or {}
        if raw.get("event") != "resource_sample":
            continue
        rows.append({"time": entry.timestamp,
                     **{k: v for k, v in raw.items() if k != "event"}})
    return rows


def read_samples(
    session_id: str,
    project: str | None = None,
    credentials: Any | None = None,
) -> list[dict]:
    """All persisted resource samples for ``session_id``, oldest first.

    Reads the :data:`RESOURCES_LOG` side log (bounded by log retention, ~30 days default),
    so it works long after the run — including for a worker the platform killed mid-turn,
    whose last sample landed at most one interval before death. Each row carries ``time``
    (an aware datetime) plus the usage keys of :func:`read_usage`.
    """
    import google.cloud.logging  # lazy: client-side helper, engine never calls this

    client = google.cloud.logging.Client(project=project, credentials=credentials)
    filter_str = (
        f'logName="projects/{client.project}/logs/{RESOURCES_LOG}" '
        f'AND labels.session_id="{session_id}"'
    )
    entries = client.list_entries(
        filter_=filter_str, order_by=google.cloud.logging.ASCENDING, page_size=1000
    )
    return _sample_rows(entries)

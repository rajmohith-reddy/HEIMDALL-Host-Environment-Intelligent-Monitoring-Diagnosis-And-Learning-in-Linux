"""
System Snapshot Orchestrator
==============================
Calls all sub-collectors, merges their outputs into a single `SystemSnapshot`
dataclass, timestamps it (ISO-8601), and serialises it to JSON.

Also maintains a thread-safe circular buffer of the last N snapshots so that
`feature_engineer.py` can compute temporal delta/rolling features.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Deque

import psutil as _psutil

from heimdall.collector import cpu, disk, memory, network, process, system
from heimdall.config import settings


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class SystemSnapshot:
    """All telemetry from one collection cycle, with a UTC timestamp."""
    timestamp:  str                    # ISO-8601 UTC
    label:      str                    # class label (filled during data collection)
    cpu:        dict[str, Any]         = field(default_factory=dict)
    memory:     dict[str, Any]         = field(default_factory=dict)
    disk:       dict[str, Any]         = field(default_factory=dict)
    network:    dict[str, Any]         = field(default_factory=dict)
    process:    dict[str, Any]         = field(default_factory=dict)
    system:     dict[str, Any]         = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SystemSnapshot":
        return cls(**d)


# ── Snapshot buffer ───────────────────────────────────────────────────────────

_buffer:    Deque[SystemSnapshot] = deque(maxlen=settings.SNAPSHOT_BUFFER_SIZE)
_last_mono: float                 = 0.0
_lock:      Lock                  = Lock()


def get_buffer() -> list[SystemSnapshot]:
    """Return a shallow copy of the current snapshot ring buffer."""
    with _lock:
        return list(_buffer)


def push_snapshot(snap: SystemSnapshot) -> None:
    """Push an externally created or simulated snapshot into the ring buffer."""
    global _last_mono
    with _lock:
        _buffer.append(snap)
        _last_mono = time.monotonic()


# ── Collection logic ──────────────────────────────────────────────────────────

# Prime psutil CPU counter on import
_psutil.cpu_percent(interval=None, percpu=True)


def collect(label: str = "unknown") -> SystemSnapshot:
    """
    Run all sub-collectors, merge results, push to the ring buffer, and
    return the new `SystemSnapshot`.
    """
    global _last_mono
    ts = datetime.now(timezone.utc).isoformat()

    snap = SystemSnapshot(
        timestamp=ts,
        label=label,
        cpu=cpu.collect(),
        memory=memory.collect(),
        disk=disk.collect(),
        network=network.collect(),
        process=process.collect(),
        system=system.collect(),
    )

    with _lock:
        _buffer.append(snap)
        _last_mono = time.monotonic()

    return snap


def collect_primed(
    label: str = "unknown",
    warmup_sec: float = settings.COLD_START_WARMUP_SEC,
) -> SystemSnapshot:
    """
    Two-point primed collection.
    If the ring buffer is empty OR the previous snapshot is stale (> 6 seconds old),
    primes all rate counters (`cpu`, `memory`, `disk`, `network`) and `psutil.cpu_percent`,
    sleeps `warmup_sec`, and then captures a fresh instantaneous snapshot.
    """
    with _lock:
        buf_len = len(_buffer)
        elapsed = time.monotonic() - _last_mono

    if buf_len == 0 or elapsed > 6.0:
        _psutil.cpu_percent(interval=None, percpu=True)
        collect(label=label)
        time.sleep(max(warmup_sec, 0.15))

    return collect(label=label)


# ── Persistence helpers ───────────────────────────────────────────────────────

def save_snapshot(snap: SystemSnapshot, output_dir: Path | str) -> Path:
    """
    Serialise *snap* as JSON to *output_dir*/<label>/<timestamp>.json.
    Directories are created automatically.
    """
    output_dir = Path(output_dir)
    label_dir  = output_dir / snap.label
    label_dir.mkdir(parents=True, exist_ok=True)

    safe_ts  = snap.timestamp.replace(":", "-").replace("+", "p")
    out_path = label_dir / f"{safe_ts}.json"
    out_path.write_text(snap.to_json(), encoding="utf-8")
    return out_path


def load_snapshot(path: Path | str) -> SystemSnapshot:
    """Load a previously saved snapshot JSON file."""
    path = Path(path)
    return SystemSnapshot.from_dict(json.loads(path.read_text(encoding="utf-8")))

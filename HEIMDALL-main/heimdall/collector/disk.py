"""
Disk Telemetry Collector
=========================
Collects per-device I/O counters (read/write throughput, IOPS) and
per-mountpoint disk usage. Rates are derived from counter deltas.
"""

from __future__ import annotations

import time
from typing import Any

import psutil


# ── State for rate computation ────────────────────────────────────────────────
_prev_io:   dict[str, Any] | None = None
_prev_time: float | None = None


# ── Public API ────────────────────────────────────────────────────────────────

def collect() -> dict[str, Any]:
    """
    Return a dict of disk metrics.

    Keys
    ----
    disk_read_mbps      : float  — aggregate read throughput across all devices (MB/s)
    disk_write_mbps     : float  — aggregate write throughput (MB/s)
    disk_read_iops      : float  — aggregate read IOPS
    disk_write_iops     : float  — aggregate write IOPS
    disk_read_ms        : float  — aggregate read time (ms/s, i.e., ms spent reading per second)
    disk_write_ms       : float  — aggregate write time (ms/s)
    disk_used_pct       : float  — percentage used on the root ( / ) mountpoint
    disk_free_gb        : float  — free space on root mountpoint (GB)
    disk_used_gb        : float  — used space on root mountpoint (GB)
    mountpoints         : list   — list of {mountpoint, used_pct, free_gb} for major mounts
    """
    global _prev_io, _prev_time

    now = time.monotonic()
    io  = psutil.disk_io_counters(perdisk=False)

    if io is not None and _prev_io is not None and _prev_time is not None:
        dt = max(now - _prev_time, 1e-9)
        mb = 1024 ** 2
        read_mbps   = max((io.read_bytes  - _prev_io.read_bytes)  / mb / dt, 0.0)
        write_mbps  = max((io.write_bytes - _prev_io.write_bytes) / mb / dt, 0.0)
        read_iops   = max((io.read_count  - _prev_io.read_count)  / dt, 0.0)
        write_iops  = max((io.write_count - _prev_io.write_count) / dt, 0.0)
        read_ms_s   = max((io.read_time   - _prev_io.read_time)   / dt, 0.0)
        write_ms_s  = max((io.write_time  - _prev_io.write_time)  / dt, 0.0)
    else:
        read_mbps = write_mbps = read_iops = write_iops = read_ms_s = write_ms_s = 0.0

    _prev_io   = io
    _prev_time = now

    # Root mountpoint usage
    try:
        import sys, os
        root_path = (os.environ.get("SystemDrive", "C:") + "\\") if sys.platform == "win32" else "/"
        root_usage = psutil.disk_usage(root_path)
        disk_used_pct = round(root_usage.percent, 2)
        disk_free_gb  = round(root_usage.free  / (1024 ** 3), 2)
        disk_used_gb  = round(root_usage.used  / (1024 ** 3), 2)
    except Exception:
        disk_used_pct = disk_free_gb = disk_used_gb = 0.0

    # Per-mountpoint summary (major filesystems only)
    mountpoints: list[dict[str, Any]] = []
    try:
        for part in psutil.disk_partitions(all=False):
            if not part.mountpoint:
                continue
            opts = (part.opts or "").lower()
            if "cdrom" in opts or "removable" in opts:
                continue
            try:
                u = psutil.disk_usage(part.mountpoint)
                mountpoints.append({
                    "mountpoint": part.mountpoint,
                    "used_pct":   round(u.percent, 1),
                    "free_gb":    round(u.free / (1024 ** 3), 2),
                })
            except (PermissionError, OSError, Exception):
                pass
    except Exception:
        pass

    return {
        "disk_read_mbps":  round(read_mbps, 3),
        "disk_write_mbps": round(write_mbps, 3),
        "disk_read_iops":  round(read_iops, 1),
        "disk_write_iops": round(write_iops, 1),
        "disk_read_ms":    round(read_ms_s, 2),
        "disk_write_ms":   round(write_ms_s, 2),
        "disk_used_pct":   disk_used_pct,
        "disk_free_gb":    disk_free_gb,
        "disk_used_gb":    disk_used_gb,
        "mountpoints":     mountpoints,
    }

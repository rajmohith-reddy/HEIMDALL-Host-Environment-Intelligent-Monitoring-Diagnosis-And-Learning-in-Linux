"""
Process Telemetry Collector
============================
Collects system-wide process counts, thread counts, zombie counts,
open file handles, and the top-N CPU / memory consuming processes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import psutil


_TOP_N = 5   # number of top processes to record


def _get_open_files_count() -> int:
    """Fast kernel read of allocated file handles on Linux (/proc/sys/fs/file-nr)."""
    try:
        file_nr = Path("/proc/sys/fs/file-nr")
        if file_nr.exists():
            parts = file_nr.read_text(encoding="utf-8").split()
            if parts:
                return int(parts[0])
    except Exception:
        pass
    return 0


def collect() -> dict[str, Any]:
    """
    Return a dict of process-level system metrics.

    Keys
    ----
    num_processes       : int   — total number of running processes
    num_threads         : int   — total threads across all processes
    num_zombies         : int   — zombie process count
    open_files          : int   — total open file descriptors (from /proc/sys/fs/file-nr)
    top_cpu_procs       : list  — top-N processes by CPU%, each {pid, name, cpu_pct, mem_pct}
    top_mem_procs       : list  — top-N processes by memory%, same shape
    """
    procs_info: list[dict[str, Any]] = []
    num_threads  = 0
    num_zombies  = 0
    open_files   = _get_open_files_count()

    for proc in psutil.process_iter(
        attrs=["pid", "name", "status", "cpu_percent", "memory_percent", "num_threads"],
        ad_value=None,
    ):
        try:
            info = proc.info  # type: ignore[attr-defined]
            if not info:
                continue

            status = info.get("status") or ""
            if status == psutil.STATUS_ZOMBIE:
                num_zombies += 1

            threads = info.get("num_threads") or 0
            num_threads += threads

            procs_info.append({
                "pid":      info.get("pid", -1),
                "name":     info.get("name", "?") or "?",
                "cpu_pct":  round(info.get("cpu_percent") or 0.0, 2),
                "mem_pct":  round(info.get("memory_percent") or 0.0, 3),
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, Exception):
            continue

    num_processes = len(procs_info)

    top_cpu = sorted(procs_info, key=lambda p: p["cpu_pct"], reverse=True)[:_TOP_N]
    top_mem = sorted(procs_info, key=lambda p: p["mem_pct"], reverse=True)[:_TOP_N]

    return {
        "num_processes": num_processes,
        "num_threads":   num_threads,
        "num_zombies":   num_zombies,
        "open_files":    open_files,
        "top_cpu_procs": top_cpu,
        "top_mem_procs": top_mem,
    }

"""
Memory Telemetry Collector
===========================
Collects virtual memory, swap usage, and paging statistics from psutil
and /proc/vmstat (Linux). Rates (page-in/out per second, swap in/out per
second) are computed from deltas between successive calls.
"""

from __future__ import annotations

import time
from typing import Any

import psutil


# ── State for rate computation ────────────────────────────────────────────────
_prev_vmstat: dict[str, int] | None = None
_prev_swap:   dict[str, int] | None = None
_prev_time:   float | None = None


def _read_vmstat_fields(*fields: str) -> dict[str, int]:
    """Parse selected fields from /proc/vmstat. Returns 0 for missing fields."""
    result: dict[str, int] = {f: 0 for f in fields}
    try:
        with open("/proc/vmstat", "r") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 2 and parts[0] in result:
                    result[parts[0]] = int(parts[1])
    except FileNotFoundError:
        pass  # Non-Linux; all values stay 0
    return result


# ── Public API ────────────────────────────────────────────────────────────────

def collect() -> dict[str, Any]:
    """
    Return a dict of memory metrics.

    Keys
    ----
    mem_total_mb        : float  — total physical RAM (MB)
    mem_used_mb         : float  — used RAM (MB)
    mem_available_mb    : float  — available RAM (MB)
    mem_used_pct        : float  — used RAM percentage
    mem_cached_mb       : float  — kernel page cache (MB)
    swap_total_mb       : float  — total swap space (MB)
    swap_used_mb        : float  — used swap (MB)
    swap_used_pct       : float  — swap utilisation (%)
    swap_in_per_sec     : float  — swap-in pages per second
    swap_out_per_sec    : float  — swap-out pages per second
    page_faults_per_sec : float  — major page faults per second (/proc/vmstat pgmajfault)
    """
    global _prev_vmstat, _prev_swap, _prev_time

    vm    = psutil.virtual_memory()
    swap  = psutil.swap_memory()
    now   = time.monotonic()

    vm_fields = _read_vmstat_fields("pgmajfault", "pswpin", "pswpout")

    # Compute rates
    if _prev_vmstat is not None and _prev_time is not None:
        dt = max(now - _prev_time, 1e-9)
        swap_in_rate    = max((vm_fields["pswpin"]     - _prev_vmstat["pswpin"])    / dt, 0.0)
        swap_out_rate   = max((vm_fields["pswpout"]    - _prev_vmstat["pswpout"])   / dt, 0.0)
        page_fault_rate = max((vm_fields["pgmajfault"] - _prev_vmstat["pgmajfault"]) / dt, 0.0)
    else:
        swap_in_rate = swap_out_rate = page_fault_rate = 0.0

    _prev_vmstat = vm_fields
    _prev_time   = now

    mb = 1024 ** 2
    return {
        "mem_total_mb":        round(vm.total / mb, 1),
        "mem_used_mb":         round(vm.used  / mb, 1),
        "mem_available_mb":    round(vm.available / mb, 1),
        "mem_used_pct":        round(vm.percent, 2),
        "mem_cached_mb":       round(getattr(vm, "cached", 0) / mb, 1),
        "swap_total_mb":       round(swap.total / mb, 1),
        "swap_used_mb":        round(swap.used  / mb, 1),
        "swap_used_pct":       round(swap.percent, 2),
        "swap_in_per_sec":     round(swap_in_rate, 2),
        "swap_out_per_sec":    round(swap_out_rate, 2),
        "page_faults_per_sec": round(page_fault_rate, 2),
    }

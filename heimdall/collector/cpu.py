"""
CPU Telemetry Collector
========================
Collects CPU utilization, frequency, load averages, context switches,
and hardware interrupts from psutil and /proc/loadavg.

All functions return plain Python dicts so they can be JSON-serialised
directly by snapshot.py without any extra conversion step.
"""

from __future__ import annotations

import os
import time
from typing import Any

import psutil


# ── Internal helpers ──────────────────────────────────────────────────────────

_prev_cpu_stats: dict[str, float] | None = None
_prev_cpu_time: float | None = None


def _read_loadavg() -> tuple[float, float, float]:
    """Read /proc/loadavg (Linux only). Falls back to psutil on other OS."""
    try:
        with open("/proc/loadavg", "r") as fh:
            parts = fh.read().split()
        return float(parts[0]), float(parts[1]), float(parts[2])
    except FileNotFoundError:
        # Non-Linux fallback (e.g., macOS, Windows in dev)
        avg = os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)
        return avg[0], avg[1], avg[2]


# ── Public API ────────────────────────────────────────────────────────────────

def collect() -> dict[str, Any]:
    """
    Return a dict of CPU metrics for the current moment.

    Keys
    ----
    cpu_percent_per_core : list[float]   — per-core utilisation (%)
    cpu_percent_mean     : float         — mean utilisation across all cores
    cpu_percent_max      : float         — highest single-core utilisation
    cpu_percent_std      : float         — std-dev of per-core utilisation
    cpu_freq_mhz         : float         — current CPU frequency in MHz (0 if unavailable)
    load_avg_1           : float         — 1-minute load average
    load_avg_5           : float         — 5-minute load average
    load_avg_15          : float         — 15-minute load average
    cpu_ctx_switches_per_sec : float     — context switches per second since last call
    cpu_interrupts_per_sec   : float     — hardware interrupts per second since last call
    num_logical_cpus     : int           — total logical CPU count
    """
    global _prev_cpu_stats, _prev_cpu_time

    # Per-core utilisation (non-blocking interval=None uses last cached value;
    # we call with interval=0.1 on first call to prime the counter, then None)
    per_core: list[float] = psutil.cpu_percent(interval=None, percpu=True)  # type: ignore[assignment]
    n = len(per_core)
    mean_pct = sum(per_core) / n if n else 0.0
    max_pct  = max(per_core) if per_core else 0.0
    # population std-dev
    std_pct  = (sum((x - mean_pct) ** 2 for x in per_core) / n) ** 0.5 if n > 1 else 0.0

    # CPU frequency
    freq = psutil.cpu_freq()
    freq_mhz = freq.current if freq else 0.0

    # Load averages
    load1, load5, load15 = _read_loadavg()

    # Context switches and interrupts (rate per second)
    now       = time.monotonic()
    cpu_stats = psutil.cpu_stats()
    raw_ctx   = cpu_stats.ctx_switches
    raw_irq   = cpu_stats.interrupts

    if _prev_cpu_stats is not None and _prev_cpu_time is not None:
        elapsed = now - _prev_cpu_time
        elapsed = elapsed if elapsed > 0 else 1e-9
        ctx_per_sec = (raw_ctx - _prev_cpu_stats["ctx"]) / elapsed
        irq_per_sec = (raw_irq - _prev_cpu_stats["irq"]) / elapsed
    else:
        ctx_per_sec = 0.0
        irq_per_sec = 0.0

    _prev_cpu_stats = {"ctx": raw_ctx, "irq": raw_irq}
    _prev_cpu_time  = now

    return {
        "cpu_percent_per_core":      per_core,
        "cpu_percent_mean":          round(mean_pct, 2),
        "cpu_percent_max":           round(max_pct, 2),
        "cpu_percent_std":           round(std_pct, 2),
        "cpu_freq_mhz":              round(freq_mhz, 1),
        "load_avg_1":                round(load1, 2),
        "load_avg_5":                round(load5, 2),
        "load_avg_15":               round(load15, 2),
        "cpu_ctx_switches_per_sec":  round(max(ctx_per_sec, 0.0), 1),
        "cpu_interrupts_per_sec":    round(max(irq_per_sec, 0.0), 1),
        "num_logical_cpus":          n,
    }

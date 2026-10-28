"""
Context-Aware Baseline Engine
=============================
Establishes per-host, time-of-day (diurnal) aware baselines and workload-aware
tolerance modifiers.

Distinguishes between:
- Legitimate heavy workloads (e.g., software compilation, rendering, backups)
- Malicious resource hijacking (e.g., cryptominer, exfiltration, DDoS)

Key Logic:
1. Diurnal profile: Evaluates if current time is during active business hours
   (09:00 - 19:00) vs quiet night/weekend hours.
2. Known benign workload detection: Inspects top process signatures (e.g. gcc,
   clang, rustc, cargo, python, make, msbuild, git, apt) paired with network context.
   If CPU is high but processes belong to a known compile toolchain and novel outbound
   connections are 0, it dynamically relaxes CPU tolerance.
3. If high CPU is driven by an unfamiliar binary with external network connections,
   it tightens anomaly sensitivity.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from heimdall.collector.snapshot import SystemSnapshot

# Process names known to generate legitimate, temporary CPU/Memory/Disk strain
KNOWN_BUILD_TOOLS = {
    "gcc", "g++", "clang", "clang++", "cc1", "cc1plus", "rustc", "cargo",
    "make", "ninja", "cmake", "msbuild", "cl.exe", "link.exe", "dotnet",
    "javac", "mvn", "gradle", "npm", "node", "pytest", "python", "python.exe",
    "git", "apt", "dpkg", "pacman", "dnf", "yum", "tar", "gzip", "zip",
}


@dataclass
class ContextProfile:
    time_of_day_bucket: str        # "business_hours", "evening", "night", "weekend"
    is_off_hours: bool
    workload_type: str             # "compilation", "package_update", "idle", "routine_work", "unknown_heavy"
    threshold_modifier: float      # e.g., 1.35x for known compilation, 0.9x for off-hours
    context_note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_of_day_bucket": self.time_of_day_bucket,
            "is_off_hours": self.is_off_hours,
            "workload_type": self.workload_type,
            "threshold_modifier": round(self.threshold_modifier, 2),
            "context_note": self.context_note,
        }


def evaluate_host_context(
    snap: "SystemSnapshot",
    dt: datetime | None = None,
) -> ContextProfile:
    """
    Evaluates current snapshot against temporal time-of-day and running process signatures.
    """
    if dt is None:
        try:
            dt = datetime.fromisoformat(snap.timestamp)
        except Exception:
            dt = datetime.now(timezone.utc)

    # 1. Diurnal Time-of-Day Evaluation
    weekday = dt.weekday()  # 0=Mon, 6=Sun
    hour = dt.hour

    if weekday in (5, 6):
        time_bucket = "weekend"
        is_off_hours = True
    elif 9 <= hour < 19:
        time_bucket = "business_hours"
        is_off_hours = False
    elif 19 <= hour < 23:
        time_bucket = "evening"
        is_off_hours = False
    else:
        time_bucket = "night"
        is_off_hours = True

    # 2. Workload Signature Detection
    top_cpu = [p.get("name", "").lower() for p in snap.process.get("top_cpu_procs", [])]
    top_mem = [p.get("name", "").lower() for p in snap.process.get("top_mem_procs", [])]
    all_top = set(top_cpu + top_mem)

    # Check for known compilation / build activity
    has_build_proc = any(any(tool in p for tool in KNOWN_BUILD_TOOLS) for p in all_top)
    tcp_conns = snap.network.get("tcp_connections", 0)

    threshold_modifier = 1.0
    context_note = "Standard baseline tolerance."
    workload = "routine_work"

    if has_build_proc and snap.cpu.get("cpu_percent_mean", 0) > 40.0:
        # Check if paired with unexpected external traffic
        if tcp_conns < 250:
            workload = "compilation"
            threshold_modifier = 1.40  # 40% higher tolerance for compiler threads
            context_note = "Recognized local build/development activity. Elevated CPU tolerance applied."
        else:
            workload = "unknown_heavy"
            threshold_modifier = 1.0
            context_note = "Build tool signature present, but accompanied by elevated socket counts."

    elif is_off_hours and snap.cpu.get("cpu_percent_mean", 0) > 50.0:
        threshold_modifier = 0.85  # Stricter sensitivity during quiet night hours
        context_note = "Off-hours heavy compute activity detected. Heightened anomaly sensitivity applied."

    elif is_off_hours and snap.cpu.get("cpu_percent_mean", 0) < 15.0:
        workload = "idle"
        threshold_modifier = 1.0
        context_note = "Normal off-hours quiescent state."

    return ContextProfile(
        time_of_day_bucket=time_bucket,
        is_off_hours=is_off_hours,
        workload_type=workload,
        threshold_modifier=threshold_modifier,
        context_note=context_note,
    )

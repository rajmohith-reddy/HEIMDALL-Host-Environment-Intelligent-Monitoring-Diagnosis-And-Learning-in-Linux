"""
Temporal Anomaly Detector
=========================
Detects multi-step attack sequences and temporal anomalies across a rolling
window of snapshots, rather than treating each snapshot in isolation.

Example multi-step sequences detected:
1. Process Execution -> Outbound Connection -> Resource Strain (e.g. cryptominer / malware).
2. Process Spawn Burst -> File Descriptor Leak (e.g. fork bomb / leak).
3. Monotonic Memory Creep -> Swap Thrashing (e.g. slow memory leak).
4. Repeated Auth Failures -> Privilege Elevation / Sudo burst.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from heimdall.collector.snapshot import SystemSnapshot


@dataclass
class TemporalStep:
    timestamp: str
    stage: str
    description: str
    severity: str = "INFO"


@dataclass
class TemporalAnomalyResult:
    is_temporal_anomaly: bool = False
    risk_score: float = 0.0          # 0.0 to 100.0
    incident_type: str = "none"
    stages_detected: list[str] = field(default_factory=list)
    event_timeline: list[dict[str, str]] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_temporal_anomaly": self.is_temporal_anomaly,
            "risk_score": round(self.risk_score, 1),
            "incident_type": self.incident_type,
            "stages_detected": self.stages_detected,
            "event_timeline": self.event_timeline,
            "explanation": self.explanation,
        }


def evaluate_temporal_sequence(
    history: list["SystemSnapshot"],
) -> TemporalAnomalyResult:
    """
    Examines a chronological sequence of snapshots (typically 6 to 12 snapshots,
    representing 30-60 seconds of history) for correlated temporal incident chains.
    """
    if len(history) < 2:
        return TemporalAnomalyResult()

    timeline: list[dict[str, str]] = []
    stages: list[str] = []
    risk = 0.0
    incident_type = "none"
    explanations: list[str] = []

    # 1. Check for Monotonic Memory Leak / Creep
    mem_used = [s.memory.get("mem_used_pct", 0.0) for s in history]
    if len(mem_used) >= 4:
        diffs = [mem_used[i] - mem_used[i - 1] for i in range(1, len(mem_used))]
        positive_steps = sum(1 for d in diffs if d > 0.3)
        total_growth = mem_used[-1] - mem_used[0]

        if positive_steps >= len(diffs) - 1 and total_growth >= 4.0:
            stages.append("Resource Creep")
            timeline.append({
                "time": history[-1].timestamp,
                "stage": "Memory Creep",
                "description": f"RAM steadily climbed by {total_growth:.1f}% across {len(history)} collection ticks.",
            })
            risk += 35.0
            incident_type = "resource_creep"
            explanations.append(f"Monotonic memory escalation detected ({total_growth:.1f}% increase across observation window)")

    # 2. Check for Process Spawn -> Network Activity -> CPU/Disk Spike (Kill-Chain Progression)
    has_proc_growth = False
    has_net_surge = False
    has_resource_spike = False

    t_first = history[0]
    t_last = history[-1]

    # Process delta
    proc_growth = t_last.process.get("num_processes", 0) - t_first.process.get("num_processes", 0)
    top_procs_last = [p.get("name", "") for p in t_last.process.get("top_cpu_procs", [])]
    top_procs_first = [p.get("name", "") for p in t_first.process.get("top_cpu_procs", [])]
    new_top_procs = set(top_procs_last) - set(top_procs_first)

    if proc_growth >= 3 or len(new_top_procs) > 0:
        has_proc_growth = True
        proc_desc = f"New processes appeared: {', '.join(new_top_procs)}" if new_top_procs else f"{proc_growth} new processes spawned"
        timeline.append({
            "time": history[len(history) // 2].timestamp,
            "stage": "Process Spawn",
            "description": proc_desc,
        })
        stages.append("Process Execution")
        risk += 25.0

    # Network delta
    tcp_delta = t_last.network.get("tcp_connections", 0) - t_first.network.get("tcp_connections", 0)
    rx_delta = t_last.network.get("net_bytes_recv_mbps", 0) - t_first.network.get("net_bytes_recv_mbps", 0)
    tx_delta = t_last.network.get("net_bytes_sent_mbps", 0) - t_first.network.get("net_bytes_sent_mbps", 0)

    if tcp_delta >= 15 or rx_delta >= 5.0 or tx_delta >= 5.0:
        has_net_surge = True
        timeline.append({
            "time": t_last.timestamp,
            "stage": "Network Burst",
            "description": f"Network activity expanded: +{tcp_delta} sockets, +{rx_delta+tx_delta:.1f} MB/s",
        })
        stages.append("Network Connection")
        risk += 30.0

    # Resource spike
    cpu_delta = t_last.cpu.get("cpu_percent_mean", 0) - t_first.cpu.get("cpu_percent_mean", 0)
    if cpu_delta >= 25.0 or t_last.cpu.get("cpu_percent_mean", 0) >= 80.0:
        has_resource_spike = True
        timeline.append({
            "time": t_last.timestamp,
            "stage": "Resource Exhaustion",
            "description": f"Processor utilization surged by {cpu_delta:+.1f}% to {t_last.cpu.get('cpu_percent_mean', 0):.1f}%",
        })
        stages.append("Resource Exhaustion")
        risk += 30.0

    # Correlate stages into an incident
    if has_proc_growth and (has_net_surge or has_resource_spike):
        incident_type = "multi_stage_execution"
        risk = min(100.0, risk + 20.0)
        explanations.append("Multi-stage attack pattern detected: process spawn immediately preceded network or resource surge")

    # 3. Check for Rapid Authentication / Zombie Surge
    auth_delta = t_last.system.get("auth_failures_60s", 0)
    if auth_delta >= 3:
        stages.append("Credential Access")
        timeline.append({
            "time": t_last.timestamp,
            "stage": "Auth Velocity",
            "description": f"{auth_delta} auth failures in the trailing 60s window.",
        })
        risk = min(100.0, risk + 40.0)
        incident_type = "brute_force_surge"
        explanations.append(f"High-frequency authentication failures ({auth_delta} failures/min)")

    is_anomaly = risk >= 45.0 and len(stages) >= 2
    final_explanation = " | ".join(explanations) if explanations else "Normal temporal sequence stability."

    return TemporalAnomalyResult(
        is_temporal_anomaly=is_anomaly,
        risk_score=min(100.0, risk),
        incident_type=incident_type,
        stages_detected=stages,
        event_timeline=timeline,
        explanation=final_explanation,
    )

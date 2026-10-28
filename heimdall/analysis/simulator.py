"""
HEIMDALL — Live Workload & Anomaly Simulator
=============================================
Generates realistic, hardware-anchored telemetry snapshots for testing and
demonstrating the MLP Classifier, Production Autoencoder, and XAI Root-Cause
Engine without requiring root privileges or external stress tools.
"""

from __future__ import annotations

import copy
import random
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from heimdall.collector import snapshot as snap_mod

if TYPE_CHECKING:
    from heimdall.collector.snapshot import SystemSnapshot


VALID_MODES = (
    "live",
    "normal",
    "cpu_stress",
    "memory_pressure",
    "disk_pressure",
    "network_flood",
    "system_anomaly",
)


def create_simulated_snapshot(
    mode: str = "live",
    intensity: float = 0.85,
) -> tuple["SystemSnapshot", "SystemSnapshot"]:
    """
    Capture a live primed snapshot and apply a realistic, physically consistent
    workload overlay for *mode*.
    """
    buf = snap_mod.get_buffer()
    if mode == "live" or not buf:
        base = snap_mod.collect_primed(label=mode if mode != "live" else "unknown")
        buf  = snap_mod.get_buffer()
    else:
        base = buf[-1]
    prev = buf[-2] if len(buf) >= 2 else copy.deepcopy(base)

    if mode == "live":
        return base, prev

    rng = random.Random(42)
    num_cpus  = max(int(base.cpu.get("num_logical_cpus", 8) or 8), 1)
    total_mb  = max(float(base.memory.get("mem_total_mb", 16000.0) or 16000.0), 2048.0)
    base_freq = max(float(base.cpu.get("cpu_freq_mhz", 2200.0) or 2200.0), 1400.0)
    base_ctx  = max(float(base.cpu.get("cpu_ctx_switches_per_sec", 6500.0) or 6500.0), 2500.0)
    base_irq  = max(float(base.cpu.get("cpu_interrupts_per_sec", 3200.0) or 3200.0), 1200.0)
    base_mem  = float(base.memory.get("mem_used_pct", 30.0) or 30.0)
    base_disk = float(base.disk.get("disk_used_pct", 45.0) or 45.0)
    base_proc = max(int(base.process.get("num_processes", 220) or 220), 80)
    base_thrd = max(int(base.process.get("num_threads", 1400) or 1400), 250)
    base_file = max(int(base.process.get("open_files", 4800) or 4800), 640)
    base_user = max(int(base.system.get("logged_in_users", 1) or 1), 1)
    base_svcs = int(base.system.get("failed_services", 0) or 0)
    base_tcp  = max(int(base.network.get("tcp_connections", 18) or 18), 2)
    base_udp  = max(int(base.network.get("udp_connections", 10) or 10), 2)

    # First construct a clean, host-calibrated normal snapshot (so non-target subsystems are calm)
    snap = copy.deepcopy(base)
    snap.timestamp = datetime.now(timezone.utc).isoformat()
    snap.label = mode

    cores = [round(rng.uniform(4.0, 16.0), 1) for _ in range(num_cpus)]
    mean_c = sum(cores) / num_cpus
    std_c = (sum((x - mean_c) ** 2 for x in cores) / num_cpus) ** 0.5
    base_l1 = max(float(base.cpu.get("load_avg_1", num_cpus * 0.22) or (num_cpus * 0.22)), num_cpus * 0.20)
    base_l5 = max(float(base.cpu.get("load_avg_5", num_cpus * 0.20) or (num_cpus * 0.20)), num_cpus * 0.18)
    base_l15 = max(float(base.cpu.get("load_avg_15", num_cpus * 0.18) or (num_cpus * 0.18)), num_cpus * 0.16)
    snap.cpu.update({
        "cpu_percent_per_core": cores,
        "cpu_percent_mean": round(mean_c, 2),
        "cpu_percent_max": round(max(cores), 2),
        "cpu_percent_std": round(std_c, 2),
        "cpu_freq_mhz": round(base_freq, 1),
        "load_avg_1": round(base_l1, 2),
        "load_avg_5": round(base_l5, 2),
        "load_avg_15": round(base_l15, 2),
        "cpu_ctx_switches_per_sec": round(base_ctx + mean_c * 150.0, 1),
        "cpu_interrupts_per_sec": round(base_irq + mean_c * 80.0, 1),
    })
    used_pct = round(max(18.0, min(65.0, base_mem)), 2)
    snap.memory.update({
        "mem_used_pct": used_pct,
        "mem_available_mb": round(total_mb * (1.0 - used_pct / 100.0), 1),
        "swap_used_pct": round(float(base.memory.get("swap_used_pct", 2.0)), 2),
        "swap_in_per_sec": 0.0,
        "swap_out_per_sec": 0.0,
    })
    d_read  = 0.65
    d_write = 0.95
    snap.disk.update({
        "disk_read_mbps": round(d_read, 3),
        "disk_write_mbps": round(d_write, 3),
        "disk_read_iops": round(d_read * 20.0 + 3.0, 1),
        "disk_write_iops": round(d_write * 22.0 + 4.0, 1),
        "disk_used_pct": round(base_disk, 2),
    })
    n_sent = 1.20
    n_recv = 1.80
    snap.network.update({
        "net_bytes_sent_mbps": round(n_sent, 4),
        "net_bytes_recv_mbps": round(n_recv, 4),
        "net_packets_sent_per_sec": round(n_sent * 750.0 + 20.0, 1),
        "net_packets_recv_per_sec": round(n_recv * 820.0 + 25.0, 1),
        "net_errin_per_sec": 0.0,
        "net_errout_per_sec": 0.0,
        "net_dropin_per_sec": 0.0,
        "net_dropout_per_sec": 0.0,
        "tcp_connections": base_tcp,
        "udp_connections": base_udp,
    })
    snap.process.update({
        "num_processes": base_proc,
        "num_threads": base_thrd,
        "num_zombies": int(base.process.get("num_zombies", 0) or 0),
        "open_files": base_file,
    })
    snap.system.update({
        "logged_in_users": base_user,
        "failed_services": base_svcs,
        "auth_failures_60s": 0,
    })

    prev_sim = copy.deepcopy(snap)
    lvl = max(0.55, min(intensity, 1.0))

    if mode == "cpu_stress":
        cores = [round(min(100.0, rng.uniform(82.0, 99.5) * lvl), 1) for _ in range(num_cpus)]
        mean_c = sum(cores) / num_cpus
        std_c = (sum((x - mean_c) ** 2 for x in cores) / num_cpus) ** 0.5
        snap.cpu.update({
            "cpu_percent_per_core": cores,
            "cpu_percent_mean": round(mean_c, 2),
            "cpu_percent_max": round(max(cores), 2),
            "cpu_percent_std": round(std_c, 2),
            "cpu_freq_mhz": round(max(base_freq * 1.25, 3400.0), 1),
            "load_avg_1": round(num_cpus * rng.uniform(1.45, 2.25) * lvl, 2),
            "load_avg_5": round(num_cpus * rng.uniform(1.10, 1.75) * lvl, 2),
            "load_avg_15": round(num_cpus * rng.uniform(0.85, 1.35) * lvl, 2),
            "cpu_ctx_switches_per_sec": round(max(base_ctx * 4.5, 58000.0) + 45000.0 * lvl, 1),
            "cpu_interrupts_per_sec": round(max(base_irq * 4.5, 28000.0) + 22000.0 * lvl, 1),
        })
        if snap.process.get("top_cpu_procs"):
            snap.process["top_cpu_procs"][0]["name"] = "stress-ng-cpu"
            snap.process["top_cpu_procs"][0]["cpu_pct"] = round(mean_c, 1)

    elif mode == "memory_pressure":
        used_pct = round(rng.uniform(89.0, 97.5) * lvl, 2)
        avail_mb = round(max(64.0, total_mb * (1.0 - used_pct / 100.0)), 1)
        swap_pct = round(rng.uniform(62.0, 92.0) * lvl, 2)
        snap.memory.update({
            "mem_used_pct": used_pct,
            "mem_available_mb": avail_mb,
            "swap_used_pct": swap_pct,
            "swap_in_per_sec": round(rng.uniform(220.0, 580.0) * lvl, 2),
            "swap_out_per_sec": round(rng.uniform(380.0, 980.0) * lvl, 2),
        })
        if snap.process.get("top_mem_procs"):
            snap.process["top_mem_procs"][0]["name"] = "python3-memleak"
            snap.process["top_mem_procs"][0]["mem_pct"] = round(used_pct * 0.68, 2)

    elif mode == "disk_pressure":
        r_mbps = round(rng.uniform(120.0, 270.0) * lvl, 3)
        w_mbps = round(rng.uniform(95.0, 230.0) * lvl, 3)
        snap.disk.update({
            "disk_read_mbps": r_mbps,
            "disk_write_mbps": w_mbps,
            "disk_read_iops": round(rng.uniform(1800.0, 4400.0) * lvl, 1),
            "disk_write_iops": round(rng.uniform(1500.0, 3900.0) * lvl, 1),
            "disk_used_pct": round(max(base_disk + 18.0, rng.uniform(86.0, 94.0)), 2),
        })

    elif mode == "network_flood":
        rx_mbps = round(rng.uniform(55.0, 130.0) * lvl, 4)
        tx_mbps = round(rng.uniform(35.0, 85.0) * lvl, 4)
        snap.network.update({
            "net_bytes_recv_mbps": rx_mbps,
            "net_bytes_sent_mbps": tx_mbps,
            "net_packets_recv_per_sec": round(rng.uniform(38000.0, 88000.0) * lvl, 1),
            "net_packets_sent_per_sec": round(rng.uniform(26000.0, 62000.0) * lvl, 1),
            "net_errin_per_sec": round(rng.uniform(8.0, 28.0) * lvl, 2),
            "net_errout_per_sec": round(rng.uniform(2.0, 8.0) * lvl, 2),
            "net_dropin_per_sec": round(rng.uniform(25.0, 95.0) * lvl, 2),
            "net_dropout_per_sec": round(rng.uniform(5.0, 20.0) * lvl, 2),
            "tcp_connections": int(rng.uniform(340, 780) * lvl),
            "udp_connections": int(rng.uniform(120, 290) * lvl),
        })

    elif mode == "system_anomaly":
        snap.process.update({
            "num_processes": int(base_proc + rng.randint(200, 420)),
            "num_threads": int(base_thrd + rng.randint(1400, 3200)),
            "num_zombies": int(rng.randint(8, 22) * lvl),
            "open_files": int(base_file + rng.randint(18000, 42000)),
        })
        snap.system.update({
            "failed_services": base_svcs + rng.randint(4, 7),
            "auth_failures_60s": rng.randint(12, 28),
        })

    return snap, prev_sim

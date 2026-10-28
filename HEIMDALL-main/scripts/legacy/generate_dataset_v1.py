"""
HEIMDALL Dataset Generator
==========================
Generates a realistic, calibrated synthetic telemetry dataset anchored on the
current host system's hardware baseline. Produces labelled snapshots for all
6 HEIMDALL operating states in data/raw/:
  - normal
  - cpu_stress
  - memory_pressure
  - disk_pressure
  - network_flood
  - system_anomaly

Usage:
    python scripts/generate_dataset.py [--count 150]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import json
import random
from pathlib import Path

from heimdall.collector.snapshot import SystemSnapshot, collect, save_snapshot
from heimdall.config import settings


def generate_scenario_snapshots(
    base_snap: SystemSnapshot,
    label: str,
    count: int = 150,
) -> list[SystemSnapshot]:
    """
    Generate realistic sequential snapshots for a given scenario anchored on
    the host system's actual baseline.
    """
    snapshots: list[SystemSnapshot] = []
    base_time = datetime.now(timezone.utc) - timedelta(seconds=count * 5)
    num_cpus = base_snap.cpu.get("num_logical_cpus", 12)
    mem_total = base_snap.memory.get("mem_total_mb", 16000.0)
    swap_total = base_snap.memory.get("swap_total_mb", 8000.0)
    disk_total = base_snap.disk.get("disk_used_gb", 70.0) + base_snap.disk.get("disk_free_gb", 50.0)

    for i in range(count):
        ts = (base_time + timedelta(seconds=i * 5)).isoformat()
        
        # Deep copies of sub-dictionaries
        cpu_data = dict(base_snap.cpu)
        mem_data = dict(base_snap.memory)
        disk_data = dict(base_snap.disk)
        net_data = dict(base_snap.network)
        proc_data = dict(base_snap.process)
        sys_data = dict(base_snap.system)

        # Baseline jitter
        noise = random.gauss(0, 0.05)

        if label == "normal":
            cpu_mean = max(1.0, min(35.0, 10.0 + random.uniform(-4.0, 8.0)))
            cpu_max = min(100.0, cpu_mean + random.uniform(5.0, 20.0))
            cpu_data["cpu_percent_mean"] = round(cpu_mean, 2)
            cpu_data["cpu_percent_max"] = round(cpu_max, 2)
            cpu_data["cpu_percent_std"] = round(random.uniform(1.0, 6.0), 2)
            cpu_data["load_avg_1"] = round(random.uniform(0.1, 0.9), 2)
            cpu_data["load_avg_5"] = round(random.uniform(0.2, 0.8), 2)
            cpu_data["load_avg_15"] = round(random.uniform(0.2, 0.7), 2)
            cpu_data["cpu_ctx_switches_per_sec"] = round(random.uniform(800, 3500), 1)
            cpu_data["cpu_interrupts_per_sec"] = round(random.uniform(500, 2000), 1)

            mem_pct = max(10.0, min(50.0, 20.0 + random.uniform(-3.0, 8.0)))
            mem_data["mem_used_pct"] = round(mem_pct, 2)
            mem_data["mem_used_mb"] = round(mem_total * (mem_pct / 100.0), 1)
            mem_data["mem_available_mb"] = round(mem_total - mem_data["mem_used_mb"], 1)
            mem_data["swap_used_pct"] = round(random.uniform(0.0, 4.0), 2)
            mem_data["swap_in_per_sec"] = 0.0
            mem_data["swap_out_per_sec"] = 0.0

            disk_data["disk_read_mbps"] = round(random.uniform(0.0, 1.5), 3)
            disk_data["disk_write_mbps"] = round(random.uniform(0.0, 3.0), 3)
            disk_data["disk_read_iops"] = round(random.uniform(0, 30), 1)
            disk_data["disk_write_iops"] = round(random.uniform(0, 60), 1)

            net_data["net_bytes_recv_mbps"] = round(random.uniform(0.001, 0.2), 4)
            net_data["net_bytes_sent_mbps"] = round(random.uniform(0.001, 0.1), 4)
            net_data["net_packets_recv_per_sec"] = round(random.uniform(10, 80), 1)
            net_data["net_packets_sent_per_sec"] = round(random.uniform(10, 60), 1)
            net_data["tcp_connections"] = random.randint(15, 45)
            net_data["udp_connections"] = random.randint(1, 8)

            proc_data["num_zombies"] = 0
            sys_data["auth_failures_60s"] = 0
            sys_data["failed_services"] = random.choice([0, 1])

        elif label == "cpu_stress":
            cpu_mean = max(75.0, min(100.0, 92.0 + random.uniform(-6.0, 7.0)))
            cpu_max = max(95.0, min(100.0, cpu_mean + random.uniform(1.0, 5.0)))
            cpu_data["cpu_percent_mean"] = round(cpu_mean, 2)
            cpu_data["cpu_percent_max"] = round(cpu_max, 2)
            cpu_data["cpu_percent_std"] = round(random.uniform(1.0, 4.0), 2)
            cpu_data["load_avg_1"] = round(random.uniform(num_cpus * 0.9, num_cpus * 2.2), 2)
            cpu_data["load_avg_5"] = round(random.uniform(num_cpus * 0.7, num_cpus * 1.8), 2)
            cpu_data["load_avg_15"] = round(random.uniform(num_cpus * 0.5, num_cpus * 1.5), 2)
            cpu_data["cpu_ctx_switches_per_sec"] = round(random.uniform(15000, 65000), 1)
            cpu_data["cpu_interrupts_per_sec"] = round(random.uniform(10000, 35000), 1)

            proc_data["top_cpu_procs"] = [
                {"name": "stress-ng-cpu", "pid": 9101, "cpu_pct": round(cpu_mean * 0.85, 1), "mem_pct": 0.5},
                {"name": "stress-ng-cpu", "pid": 9102, "cpu_pct": round(cpu_mean * 0.12, 1), "mem_pct": 0.5},
            ]

        elif label == "memory_pressure":
            mem_pct = max(82.0, min(98.5, 91.0 + random.uniform(-4.0, 6.0)))
            mem_data["mem_used_pct"] = round(mem_pct, 2)
            mem_data["mem_used_mb"] = round(mem_total * (mem_pct / 100.0), 1)
            mem_data["mem_available_mb"] = round(mem_total - mem_data["mem_used_mb"], 1)
            swap_pct = max(45.0, min(95.0, 72.0 + random.uniform(-10.0, 15.0)))
            mem_data["swap_used_pct"] = round(swap_pct, 2)
            mem_data["swap_used_mb"] = round(swap_total * (swap_pct / 100.0), 1)
            mem_data["swap_in_per_sec"] = round(random.uniform(50.0, 450.0), 1)
            mem_data["swap_out_per_sec"] = round(random.uniform(80.0, 650.0), 1)
            mem_data["page_faults_per_sec"] = round(random.uniform(8000, 45000), 1)

            proc_data["top_mem_procs"] = [
                {"name": "stress-ng-vm", "pid": 9201, "cpu_pct": 12.0, "mem_pct": 52.0},
                {"name": "stress-ng-vm", "pid": 9202, "cpu_pct": 10.0, "mem_pct": 32.0},
            ]

        elif label == "disk_pressure":
            disk_data["disk_read_mbps"] = round(random.uniform(35.0, 140.0), 2)
            disk_data["disk_write_mbps"] = round(random.uniform(50.0, 210.0), 2)
            disk_data["disk_read_iops"] = round(random.uniform(800, 4500), 1)
            disk_data["disk_write_iops"] = round(random.uniform(1200, 6500), 1)
            disk_data["disk_read_ms"] = round(random.uniform(15.0, 85.0), 1)
            disk_data["disk_write_ms"] = round(random.uniform(25.0, 110.0), 1)
            disk_data["disk_used_pct"] = round(min(95.0, base_snap.disk.get("disk_used_pct", 50.0) + random.uniform(5.0, 18.0)), 1)
            cpu_data["cpu_ctx_switches_per_sec"] = round(random.uniform(8000, 25000), 1)

        elif label == "network_flood":
            net_data["net_bytes_recv_mbps"] = round(random.uniform(60.0, 320.0), 2)
            net_data["net_bytes_sent_mbps"] = round(random.uniform(50.0, 280.0), 2)
            net_data["net_packets_recv_per_sec"] = round(random.uniform(25000, 95000), 1)
            net_data["net_packets_sent_per_sec"] = round(random.uniform(20000, 85000), 1)
            net_data["net_dropin_per_sec"] = round(random.uniform(10.0, 250.0), 1)
            net_data["net_errin_per_sec"] = round(random.uniform(0.0, 25.0), 1)
            net_data["tcp_connections"] = random.randint(250, 1800)
            net_data["udp_connections"] = random.randint(50, 400)
            cpu_data["cpu_interrupts_per_sec"] = round(random.uniform(15000, 48000), 1)

        elif label == "system_anomaly":
            proc_data["num_zombies"] = random.randint(5, 35)
            sys_data["failed_services"] = random.randint(4, 16)
            sys_data["auth_failures_60s"] = random.randint(6, 45)
            proc_data["num_processes"] = random.randint(700, 2200)
            proc_data["num_threads"] = random.randint(3500, 8500)
            cpu_data["cpu_percent_mean"] = round(random.uniform(30.0, 75.0), 2)
            mem_data["mem_used_pct"] = round(random.uniform(55.0, 85.0), 2)

        snap = SystemSnapshot(
            timestamp=ts,
            label=label,
            cpu=cpu_data,
            memory=mem_data,
            disk=disk_data,
            network=net_data,
            process=proc_data,
            system=sys_data,
        )
        snapshots.append(snap)

    return snapshots


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic benchmark dataset for HEIMDALL")
    parser.add_argument("--count", type=int, default=150, help="Number of snapshots per class (default: 150)")
    args = parser.parse_args()

    print("══════════════════════════════════════════════════════")
    print("  HEIMDALL Synthetic Telemetry Dataset Generator")
    print(f"  Target: {args.count} snapshots per class × {len(settings.CLASS_NAMES)} classes")
    print("══════════════════════════════════════════════════════\n")

    base_snap = collect(label="baseline")
    total_saved = 0

    for label in settings.CLASS_NAMES:
        print(f"  Generating class '{label}' ({args.count} samples) …")
        snaps = generate_scenario_snapshots(base_snap, label, count=args.count)
        for snap in snaps:
            save_snapshot(snap, settings.DATA_RAW_DIR)
            total_saved += 1

    print(f"\n[SUCCESS] Successfully generated and saved {total_saved} raw snapshots to {settings.DATA_RAW_DIR}")
    print("Next steps:")
    print("  1. python -m heimdall build-dataset")
    print("  2. python -m heimdall train")
    print("  3. python -m heimdall train-ae")
    print("  4. python -m heimdall evaluate")
    print("  5. python -m heimdall diagnose")


if __name__ == "__main__":
    main()

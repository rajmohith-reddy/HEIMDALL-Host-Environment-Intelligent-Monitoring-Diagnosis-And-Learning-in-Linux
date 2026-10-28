"""
HEIMDALL v2: Context Signals Collector
======================================
Captures system context beyond raw resource counters:
1. Process Tree Context: Top process CPU share, new process names, spawn rate.
2. Network Context: Listening ports, port diffs, distinct outbound IPs, SYN_RECV count.
3. Auth & Health: Auth failures from journald/auth.log, sudo invocations, heartbeat.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import psutil


@dataclass
class ContextSignals:
    # Process context
    proc_top_cpu_share: float = 0.0
    proc_new_names_count: float = 0.0
    proc_spawn_rate: float = 0.0

    # Network context
    net_listening_ports: float = 0.0
    net_listening_port_changes: float = 0.0
    net_distinct_outbound_ips: float = 0.0
    net_new_dest_rate: float = 0.0
    net_syn_recv_count: float = 0.0

    # Auth & Host health
    auth_failures_rate: float = 0.0
    auth_sudo_events: float = 0.0
    collector_heartbeat: float = 1.0

    def to_dict(self) -> dict[str, float]:
        return {
            "proc_top_cpu_share": self.proc_top_cpu_share,
            "proc_new_names_count": self.proc_new_names_count,
            "proc_spawn_rate": self.proc_spawn_rate,
            "net_listening_ports": self.net_listening_ports,
            "net_listening_port_changes": self.net_listening_port_changes,
            "net_distinct_outbound_ips": self.net_distinct_outbound_ips,
            "net_new_dest_rate": self.net_new_dest_rate,
            "net_syn_recv_count": self.net_syn_recv_count,
            "auth_failures_rate": self.auth_failures_rate,
            "auth_sudo_events": self.auth_sudo_events,
            "collector_heartbeat": self.collector_heartbeat,
        }


class ContextCollector:
    """Stateful collector to track deltas, rates, and novel process/IP encounters."""

    def __init__(self, history_window: int = 100) -> None:
        self.known_process_names: set[str] = set()
        self.previous_listening_ports: set[int] = set()
        self.known_outbound_ips: set[str] = set()
        self.last_timestamp: float = time.time()
        self.last_total_procs: int = 0
        self.history_window = history_window
        self.recent_new_names: deque[str] = deque(maxlen=200)

    def sample(self) -> ContextSignals:
        now = time.time()
        dt = max(now - self.last_timestamp, 0.1)

        # ── 1. Process context ───────────────────────────────────────────────
        top_cpu_share = 0.0
        current_proc_names: set[str] = set()
        total_procs = 0

        try:
            for p in psutil.process_iter(["name", "cpu_percent"]):
                total_procs += 1
                name = p.info.get("name")
                if name:
                    current_proc_names.add(name)
                cpu_p = p.info.get("cpu_percent") or 0.0
                if cpu_p > top_cpu_share:
                    top_cpu_share = cpu_p
        except Exception:
            pass

        # Identify newly spawned binary names not in known set
        if self.known_process_names:
            new_names = current_proc_names - self.known_process_names
            new_names_count = float(len(new_names))
            self.known_process_names.update(new_names)
        else:
            self.known_process_names = set(current_proc_names)
            new_names_count = 0.0

        spawn_rate = max((total_procs - self.last_total_procs) / dt, 0.0) if self.last_total_procs > 0 else 0.0
        self.last_total_procs = total_procs

        # ── 2. Network context ───────────────────────────────────────────────
        listening_ports: set[int] = set()
        outbound_ips: set[str] = set()
        syn_recv_count = 0

        try:
            for conn in psutil.net_connections(kind="inet"):
                if conn.status == "LISTEN" and conn.laddr:
                    listening_ports.add(conn.laddr.port)
                elif conn.status == "SYN_RECV":
                    syn_recv_count += 1
                elif conn.status in ("ESTABLISHED", "SYN_SENT") and conn.raddr:
                    ip = conn.raddr.ip
                    # Filter loopback
                    if not ip.startswith("127.") and ip != "::1":
                        outbound_ips.add(ip)
        except (psutil.AccessDenied, PermissionError):
            # Non-root fallback: inspect /proc/net/tcp or sockets available to user
            pass

        port_changes = float(len(listening_ports.symmetric_difference(self.previous_listening_ports))) if self.previous_listening_ports else 0.0
        self.previous_listening_ports = listening_ports

        new_dests = outbound_ips - self.known_outbound_ips
        new_dest_rate = float(len(new_dests)) / (dt / 60.0) if dt > 0 else 0.0
        self.known_outbound_ips.update(outbound_ips)

        # ── 3. Auth & Health context ─────────────────────────────────────────
        auth_fails = 0.0
        sudo_events = 0.0
        # Check journald for recent 60s auth failures/sudo if available
        try:
            # Query last 1 min of auth/sudo from journalctl quickly (<50ms)
            cmd = ["journalctl", "-q", "-S", "-60s", "_COMM=sudo", "_COMM=sshd", "-o", "cat"]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=0.5)
            if res.returncode == 0 and res.stdout:
                for line in res.stdout.splitlines():
                    lower = line.lower()
                    if "failed" in lower or "failure" in lower or "invalid user" in lower:
                        auth_fails += 1.0
                    if "sudo" in lower and ("command=" in lower or "session opened" in lower):
                        sudo_events += 1.0
        except Exception:
            pass

        self.last_timestamp = now

        return ContextSignals(
            proc_top_cpu_share=float(top_cpu_share),
            proc_new_names_count=float(new_names_count),
            proc_spawn_rate=float(spawn_rate),
            net_listening_ports=float(len(listening_ports)),
            net_listening_port_changes=float(port_changes),
            net_distinct_outbound_ips=float(len(outbound_ips)),
            net_new_dest_rate=float(new_dest_rate),
            net_syn_recv_count=float(syn_recv_count),
            auth_failures_rate=float(auth_fails),
            auth_sudo_events=float(sudo_events),
            collector_heartbeat=1.0,
        )

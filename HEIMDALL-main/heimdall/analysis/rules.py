"""
Rules Engine
============
Deterministic threshold checks that run regardless of model output.
Each rule function returns a RuleResult(triggered, message, severity).

These rules provide a fast, interpretable safety layer that catches obvious
threshold violations even if the ML models are not yet loaded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from heimdall.analysis.state import SeverityLevel, max_severity
from heimdall.config import settings

if TYPE_CHECKING:
    from heimdall.collector.snapshot import SystemSnapshot


# ── Rule result type ──────────────────────────────────────────────────────────

class RuleResult:
    __slots__ = ("triggered", "message", "severity")

    def __init__(self, triggered: bool, message: str, severity: SeverityLevel = "OK") -> None:
        self.triggered = triggered
        self.message   = message
        self.severity  = severity


# ── Individual rule functions ─────────────────────────────────────────────────

def rule_cpu_warning(snap: "SystemSnapshot") -> RuleResult:
    pct = snap.cpu.get("cpu_percent_mean", 0.0)
    if pct >= settings.RULE_CPU_CRITICAL_PCT:
        return RuleResult(True, f"CPU usage critically high: {pct:.1f}% (threshold: {settings.RULE_CPU_CRITICAL_PCT}%)", "CRITICAL")
    if pct >= settings.RULE_CPU_WARNING_PCT:
        return RuleResult(True, f"CPU usage elevated: {pct:.1f}% (threshold: {settings.RULE_CPU_WARNING_PCT}%)", "WARNING")
    return RuleResult(False, "")


def rule_memory_free(snap: "SystemSnapshot") -> RuleResult:
    avail_mb = snap.memory.get("mem_available_mb", 9999.0)
    used_pct = snap.memory.get("mem_used_pct", 0.0)
    if avail_mb <= settings.RULE_MEM_FREE_CRITICAL_MB or used_pct >= settings.RULE_MEM_USED_CRITICAL_PCT:
        return RuleResult(True, f"Memory critically stressed: {used_pct:.1f}% used, {avail_mb:.0f} MB available", "CRITICAL")
    if avail_mb <= settings.RULE_MEM_FREE_WARNING_MB or used_pct >= settings.RULE_MEM_USED_WARNING_PCT:
        return RuleResult(True, f"Memory pressure elevated: {used_pct:.1f}% used, {avail_mb:.0f} MB available", "WARNING")
    return RuleResult(False, "")


def rule_swap_usage(snap: "SystemSnapshot") -> RuleResult:
    swap_pct = snap.memory.get("swap_used_pct", 0.0)
    if swap_pct >= settings.RULE_SWAP_WARNING_PCT:
        return RuleResult(True, f"Swap utilisation high: {swap_pct:.1f}% (threshold: {settings.RULE_SWAP_WARNING_PCT}%)", "WARNING")
    return RuleResult(False, "")


def rule_disk_utilisation(snap: "SystemSnapshot") -> RuleResult:
    disk_pct = snap.disk.get("disk_used_pct", 0.0)
    if disk_pct >= settings.RULE_DISK_USED_WARNING_PCT:
        return RuleResult(True, f"Root filesystem almost full: {disk_pct:.1f}% used (threshold: {settings.RULE_DISK_USED_WARNING_PCT}%)", "WARNING")
    return RuleResult(False, "")


def rule_disk_io_throughput(snap: "SystemSnapshot") -> RuleResult:
    read_mbps  = snap.disk.get("disk_read_mbps", 0.0)
    write_mbps = snap.disk.get("disk_write_mbps", 0.0)
    total_mbps = read_mbps + write_mbps
    if total_mbps >= settings.RULE_DISK_IO_CRITICAL_MBPS:
        return RuleResult(True, f"Heavy disk I/O saturation: {total_mbps:.1f} MB/s (R={read_mbps:.1f}, W={write_mbps:.1f})", "CRITICAL")
    if total_mbps >= settings.RULE_DISK_IO_WARNING_MBPS:
        return RuleResult(True, f"Elevated disk I/O throughput: {total_mbps:.1f} MB/s (R={read_mbps:.1f}, W={write_mbps:.1f})", "WARNING")
    return RuleResult(False, "")


def rule_network_drops(snap: "SystemSnapshot") -> RuleResult:
    drops = (
        snap.network.get("net_dropin_per_sec", 0.0)
        + snap.network.get("net_dropout_per_sec", 0.0)
        + snap.network.get("net_errin_per_sec", 0.0)
        + snap.network.get("net_errout_per_sec", 0.0)
    )
    if drops >= settings.RULE_NET_DROP_WARNING_PS:
        return RuleResult(True, f"Network packet drops/errors elevated: {drops:.1f}/s", "WARNING")
    return RuleResult(False, "")


def rule_zombie_processes(snap: "SystemSnapshot") -> RuleResult:
    zombies = snap.process.get("num_zombies", 0)
    if zombies >= settings.RULE_ZOMBIE_WARNING:
        return RuleResult(True, f"Zombie process count elevated: {zombies} (threshold: {settings.RULE_ZOMBIE_WARNING})", "WARNING")
    return RuleResult(False, "")


def rule_failed_services(snap: "SystemSnapshot") -> RuleResult:
    failed = snap.system.get("failed_services", 0)
    if failed >= settings.RULE_FAILED_SERVICES_WARN:
        return RuleResult(True, f"{failed} systemd service(s) in failed state", "WARNING")
    return RuleResult(False, "")


def rule_auth_failures(snap: "SystemSnapshot") -> RuleResult:
    fails = snap.system.get("auth_failures_60s", 0)
    if fails >= settings.RULE_AUTH_FAIL_CRITICAL:
        return RuleResult(True, f"Security alert: {fails} authentication failures in last 60 s (potential brute-force)", "CRITICAL")
    return RuleResult(False, "")


# ── All rules registry ────────────────────────────────────────────────────────

ALL_RULES = [
    rule_cpu_warning,
    rule_memory_free,
    rule_swap_usage,
    rule_disk_utilisation,
    rule_disk_io_throughput,
    rule_network_drops,
    rule_zombie_processes,
    rule_failed_services,
    rule_auth_failures,
]


# ── Aggregate runner ──────────────────────────────────────────────────────────

def evaluate_rules(snap: "SystemSnapshot") -> tuple[list[str], SeverityLevel]:
    """
    Run all rules against *snap* and return:
        (list of triggered rule messages, overall severity)
    """
    triggered_messages: list[str] = []
    overall_severity: SeverityLevel = "OK"

    for rule_fn in ALL_RULES:
        result = rule_fn(snap)
        if result.triggered:
            triggered_messages.append(result.message)
            overall_severity = max_severity(overall_severity, result.severity)

    return triggered_messages, overall_severity

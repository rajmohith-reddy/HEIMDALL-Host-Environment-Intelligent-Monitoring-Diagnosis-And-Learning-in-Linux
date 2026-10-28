"""
Feature Engineer
================
Converts a `SystemSnapshot` (optionally with a preceding snapshot for delta
features) into a fixed-length float32 feature vector whose order is defined
by `settings.FEATURE_NAMES`.

Usage
-----
    from heimdall.preprocessor.feature_engineer import extract_features
    vec = extract_features(current_snap, prev_snap)   # list[float], len == NUM_FEATURES
    vec = extract_features(current_snap)               # deltas will be 0.0
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from heimdall.config import settings

if TYPE_CHECKING:
    from heimdall.collector.snapshot import SystemSnapshot


def _safe(value: object, default: float = 0.0) -> float:
    """Return float(value) or *default* if None / not numeric."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def extract_features(
    snap: "SystemSnapshot",
    prev_snap: "SystemSnapshot | None" = None,
) -> list[float]:
    """
    Build the feature vector for *snap*.

    The vector follows `settings.FEATURE_NAMES` exactly, so positional
    order must be preserved here if FEATURE_NAMES is changed.

    Parameters
    ----------
    snap      : current SystemSnapshot
    prev_snap : immediately preceding snapshot (for delta features).
                If None, delta features are set to 0.0.

    Returns
    -------
    list[float] — length == settings.NUM_FEATURES
    """
    c  = snap.cpu
    m  = snap.memory
    d  = snap.disk
    n  = snap.network
    p  = snap.process
    s  = snap.system

    # ── CPU ──────────────────────────────────────────────────────────────────
    cpu_mean = _safe(c.get("cpu_percent_mean"))
    cpu_max  = _safe(c.get("cpu_percent_max"))
    cpu_std  = _safe(c.get("cpu_percent_std"))
    cpu_freq = _safe(c.get("cpu_freq_mhz"))
    load1    = _safe(c.get("load_avg_1"))
    load5    = _safe(c.get("load_avg_5"))
    load15   = _safe(c.get("load_avg_15"))
    ctx_sw   = _safe(c.get("cpu_ctx_switches_per_sec"))
    irq      = _safe(c.get("cpu_interrupts_per_sec"))

    # ── Memory ───────────────────────────────────────────────────────────────
    mem_used_pct   = _safe(m.get("mem_used_pct"))
    mem_avail_mb   = _safe(m.get("mem_available_mb"))
    swap_used_pct  = _safe(m.get("swap_used_pct"))
    swap_in        = _safe(m.get("swap_in_per_sec"))
    swap_out       = _safe(m.get("swap_out_per_sec"))

    # ── Disk ─────────────────────────────────────────────────────────────────
    disk_read_mbps  = _safe(d.get("disk_read_mbps"))
    disk_write_mbps = _safe(d.get("disk_write_mbps"))
    disk_read_iops  = _safe(d.get("disk_read_iops"))
    disk_write_iops = _safe(d.get("disk_write_iops"))
    disk_used_pct   = _safe(d.get("disk_used_pct"))

    # ── Network ──────────────────────────────────────────────────────────────
    net_sent_mbps    = _safe(n.get("net_bytes_sent_mbps"))
    net_recv_mbps    = _safe(n.get("net_bytes_recv_mbps"))
    pkt_sent_ps      = _safe(n.get("net_packets_sent_per_sec"))
    pkt_recv_ps      = _safe(n.get("net_packets_recv_per_sec"))
    errin            = _safe(n.get("net_errin_per_sec"))
    errout           = _safe(n.get("net_errout_per_sec"))
    dropin           = _safe(n.get("net_dropin_per_sec"))
    dropout          = _safe(n.get("net_dropout_per_sec"))
    tcp_conns        = _safe(n.get("tcp_connections"))
    udp_conns        = _safe(n.get("udp_connections"))

    # ── Process / System ─────────────────────────────────────────────────────
    num_procs    = _safe(p.get("num_processes"))
    num_threads  = _safe(p.get("num_threads"))
    num_zombies  = _safe(p.get("num_zombies"))
    open_files   = _safe(p.get("open_files"))
    users        = _safe(s.get("logged_in_users"))
    failed_svc   = _safe(s.get("failed_services"))
    auth_fails   = _safe(s.get("auth_failures_60s"))

    # ── Temporal delta features ───────────────────────────────────────────────
    if prev_snap is not None:
        pc = prev_snap.cpu
        pm = prev_snap.memory
        pd = prev_snap.disk
        pn = prev_snap.network

        delta_cpu   = cpu_mean - _safe(pc.get("cpu_percent_mean"))
        delta_mem   = mem_used_pct - _safe(pm.get("mem_used_pct"))
        delta_disk  = disk_read_mbps - _safe(pd.get("disk_read_mbps"))
        delta_net   = net_recv_mbps - _safe(pn.get("net_bytes_recv_mbps"))
    else:
        delta_cpu = delta_mem = delta_disk = delta_net = 0.0

    # ── Assemble vector (must match FEATURE_NAMES order) ─────────────────────
    vec: list[float] = [
        # CPU (9)
        cpu_mean, cpu_max, cpu_std, cpu_freq,
        load1, load5, load15,
        ctx_sw, irq,
        # Memory (5)
        mem_used_pct, mem_avail_mb, swap_used_pct, swap_in, swap_out,
        # Disk (5)
        disk_read_mbps, disk_write_mbps, disk_read_iops, disk_write_iops, disk_used_pct,
        # Network (10)
        net_sent_mbps, net_recv_mbps,
        pkt_sent_ps, pkt_recv_ps,
        errin, errout, dropin, dropout,
        tcp_conns, udp_conns,
        # Process/System (7)
        num_procs, num_threads, num_zombies, open_files,
        users, failed_svc, auth_fails,
        # Temporal deltas (4)
        delta_cpu, delta_mem, delta_disk, delta_net,
    ]

    assert len(vec) == settings.NUM_FEATURES, (
        f"Feature vector length mismatch: got {len(vec)}, "
        f"expected {settings.NUM_FEATURES}. "
        "Update settings.FEATURE_NAMES to match extract_features()."
    )

    return vec

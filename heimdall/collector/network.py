"""
Network Telemetry Collector
============================
Collects aggregate network I/O rates (bytes, packets, errors, drops) and
connection counts by protocol (TCP/UDP) from psutil.
"""

from __future__ import annotations

import time
from typing import Any

import psutil


# ── State for rate computation ────────────────────────────────────────────────
_prev_net:  Any | None = None
_prev_time: float | None = None


def _count_connections(timeout: float = 0.3) -> tuple[int, int]:
    """Safely count TCP/UDP connections with a hard timeout to prevent Windows hangs."""
    import concurrent.futures

    def _query() -> tuple[int, int]:
        tcp = udp = 0
        try:
            for c in psutil.net_connections(kind="inet"):
                if c.type == 1:   # SOCK_STREAM — TCP
                    tcp += 1
                elif c.type == 2: # SOCK_DGRAM  — UDP
                    udp += 1
        except Exception:
            pass
        return tcp, udp

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_query)
            return future.result(timeout=timeout)
    except Exception:
        return 0, 0


# ── Public API ────────────────────────────────────────────────────────────────

def collect() -> dict[str, Any]:
    """
    Return a dict of network metrics.

    Keys
    ----
    net_bytes_sent_mbps       : float  — aggregate TX throughput (MB/s)
    net_bytes_recv_mbps       : float  — aggregate RX throughput (MB/s)
    net_packets_sent_per_sec  : float  — TX packets per second
    net_packets_recv_per_sec  : float  — RX packets per second
    net_errin_per_sec         : float  — inbound errors per second
    net_errout_per_sec        : float  — outbound errors per second
    net_dropin_per_sec        : float  — inbound drops per second
    net_dropout_per_sec       : float  — outbound drops per second
    tcp_connections           : int    — number of ESTABLISHED TCP connections
    udp_connections           : int    — number of UDP sockets
    """
    global _prev_net, _prev_time

    now = time.monotonic()
    net = psutil.net_io_counters(pernic=False)
    mb  = 1024 ** 2

    if net is not None and _prev_net is not None and _prev_time is not None:
        dt = max(now - _prev_time, 1e-9)
        sent_mbps   = max((net.bytes_sent    - _prev_net.bytes_sent)    / mb / dt, 0.0)
        recv_mbps   = max((net.bytes_recv    - _prev_net.bytes_recv)    / mb / dt, 0.0)
        pkt_sent    = max((net.packets_sent  - _prev_net.packets_sent)  / dt, 0.0)
        pkt_recv    = max((net.packets_recv  - _prev_net.packets_recv)  / dt, 0.0)
        errin       = max((net.errin         - _prev_net.errin)         / dt, 0.0)
        errout      = max((net.errout        - _prev_net.errout)        / dt, 0.0)
        dropin      = max((net.dropin        - _prev_net.dropin)        / dt, 0.0)
        dropout     = max((net.dropout       - _prev_net.dropout)       / dt, 0.0)
    else:
        sent_mbps = recv_mbps = pkt_sent = pkt_recv = 0.0
        errin = errout = dropin = dropout = 0.0

    _prev_net  = net
    _prev_time = now

    # Connection counts by protocol (safe & non-blocking)
    tcp_count, udp_count = _count_connections(timeout=0.3)

    return {
        "net_bytes_sent_mbps":      round(sent_mbps, 4),
        "net_bytes_recv_mbps":      round(recv_mbps, 4),
        "net_packets_sent_per_sec": round(pkt_sent, 1),
        "net_packets_recv_per_sec": round(pkt_recv, 1),
        "net_errin_per_sec":        round(errin, 2),
        "net_errout_per_sec":       round(errout, 2),
        "net_dropin_per_sec":       round(dropin, 2),
        "net_dropout_per_sec":      round(dropout, 2),
        "tcp_connections":          tcp_count,
        "udp_connections":          udp_count,
    }

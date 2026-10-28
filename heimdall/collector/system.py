"""
System Telemetry Collector
===========================
Collects system-level state: uptime, logged-in users, failed systemd
services, and recent authentication failure count from auth.log / journald.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_uptime_seconds() -> float:
    """Return system uptime in seconds."""
    try:
        with open("/proc/uptime", "r") as fh:
            return float(fh.read().split()[0])
    except FileNotFoundError:
        return time.monotonic()  # rough fallback


def _count_failed_services() -> int:
    """Count failed systemd units via `systemctl`. Returns 0 if unavailable."""
    if sys.platform == "win32" or not shutil.which("systemctl"):
        return 0
    try:
        result = subprocess.run(
            ["systemctl", "list-units", "--state=failed", "--no-legend", "--no-pager"],
            capture_output=True, text=True, timeout=2,
        )
        lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
        return len(lines)
    except Exception:
        return 0


def _parse_log_line_timestamp(line: str, now_dt: datetime) -> float | None:
    """
    Parse either ISO-8601 ('2026-10-13T10:15:30...') or traditional syslog
    ('Oct 13 10:15:30 ...') timestamp at the start of an auth.log line.
    Returns epoch seconds or None if unparseable.
    """
    parts = line.split()
    if not parts:
        return None

    # 1. Try ISO-8601 prefix (Ubuntu 24.04+ rsyslog default)
    first = parts[0]
    if len(first) >= 19 and first[4:5] == "-" and "T" in first:
        try:
            dt = datetime.fromisoformat(first.replace("Z", "+00:00"))
            return dt.timestamp()
        except Exception:
            pass

    # 2. Try traditional syslog: 'Oct 13 10:15:30'
    if len(parts) >= 3:
        ts_str = f"{now_dt.year} {parts[0]} {parts[1]} {parts[2]}"
        try:
            dt = datetime.strptime(ts_str, "%Y %b %d %H:%M:%S")
            return dt.timestamp()
        except Exception:
            pass

    return None


def _count_auth_failures(window_seconds: int = 60) -> int:
    """
    Count SSH/PAM authentication failures in the last *window_seconds* seconds.
    Checks timestamped tail of /var/log/auth.log first, then falls back to journalctl.
    Returns 0 if neither source is accessible.
    """
    if sys.platform == "win32":
        return 0

    pattern = re.compile(r"(Failed password|authentication failure|Invalid user)", re.IGNORECASE)
    count = 0

    # --- /var/log/auth.log (Debian/Ubuntu) ---
    auth_log = Path("/var/log/auth.log")
    if auth_log.exists():
        try:
            cutoff = time.time() - window_seconds
            now_dt = datetime.now()
            # Read only the last ~64 KB for speed and check line timestamps
            with auth_log.open("rb") as fh:
                fh.seek(0, 2)
                size = fh.tell()
                fh.seek(max(0, size - 65536), 0)
                tail_text = fh.read().decode("utf-8", errors="replace")

            for line in tail_text.splitlines():
                if pattern.search(line):
                    ts = _parse_log_line_timestamp(line, now_dt)
                    if ts is not None and ts >= cutoff:
                        count += 1
            return count
        except (PermissionError, Exception):
            pass

    # --- journalctl fallback ---
    if not shutil.which("journalctl"):
        return 0

    try:
        result = subprocess.run(
            ["journalctl", "_SYSTEMD_UNIT=sshd.service",
             "--since", f"-{window_seconds}s",
             "--no-pager", "--output=short"],
            capture_output=True, text=True, timeout=2,
        )
        for line in result.stdout.splitlines():
            if pattern.search(line):
                count += 1
    except Exception:
        pass

    return count


# ── Public API ────────────────────────────────────────────────────────────────

def collect() -> dict[str, Any]:
    """
    Return a dict of system-level metrics.

    Keys
    ----
    uptime_seconds       : float  — system uptime in seconds
    logged_in_users      : int    — number of currently logged-in users
    failed_services      : int    — count of failed systemd units
    auth_failures_60s    : int    — authentication failures in the last 60 s
    """
    uptime   = _get_uptime_seconds()
    users    = len(psutil.users())
    failed   = _count_failed_services()
    auth_f   = _count_auth_failures(window_seconds=60)

    return {
        "uptime_seconds":    round(uptime, 1),
        "logged_in_users":   users,
        "failed_services":   failed,
        "auth_failures_60s": auth_f,
    }

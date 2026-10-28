"""
Diagnosis Engine (Production Hybrid + XAI Root-Cause Fusion)
=============================================================
Maps (mlp_class, triggered_rules, ae_result, snapshot) to human-readable
diagnosis sentences, XAI root-cause feature attributions, and concrete
recommended actions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from heimdall.analysis.rules import evaluate_rules
from heimdall.analysis.state import SeverityLevel, SystemState, max_severity
from heimdall.config import settings

if TYPE_CHECKING:
    from heimdall.collector.snapshot import SystemSnapshot
    from heimdall.models.autoencoder.inference import AEResult
    from heimdall.models.mlp.inference import MLPResult


# ── Diagnosis helpers ─────────────────────────────────────────────────────────

def _top_proc_name(snap: "SystemSnapshot", key: str = "top_cpu_procs") -> str:
    """Return the name of the top process by CPU or memory."""
    procs = snap.process.get(key, [])
    if procs:
        p = procs[0]
        return f"'{p.get('name', '?')}' (PID {p.get('pid', '?')})"
    return "an unknown process"


def _format_xai_drivers(top_features: list[dict[str, Any]], limit: int = 3) -> str:
    """Format the top anomalous features into a concise root-cause string."""
    active = [f for f in top_features if f.get("z_error", 0.0) > 0.25][:limit]
    if not active:
        active = top_features[:limit]
    if not active:
        return ""
    parts = [
        f"{f['feature']} ({f['z_error']:.1f}σ, {f['contribution_pct']:.0f}%)"
        for f in active
    ]
    return "Primary telemetry drivers: " + ", ".join(parts) + "."


def _diagnose_cpu_stress(snap: "SystemSnapshot") -> tuple[list[str], list[str]]:
    top = _top_proc_name(snap, "top_cpu_procs")
    load1 = snap.cpu.get("load_avg_1", 0)
    cpu_pct = snap.cpu.get("cpu_percent_mean", 0)
    num_cpus = snap.cpu.get("num_logical_cpus", 1) or 1
    diag = [
        f"CPU stress detected (utilisation: {cpu_pct:.1f}%, 1m load avg: {load1:.2f})."
    ]
    if load1 > num_cpus * 1.2:
        diag.append(f"Run-queue saturation: load avg ({load1:.2f}) exceeds logical CPU count ({num_cpus}).")
    diag.append(f"Top CPU consumer: {top}.")
    recs = [
        f"Inspect {top} with `top -c -p <PID>` or `pidstat -p <PID> 1`.",
        "Throttle non-critical compute with `renice +10 <PID>` or `cpulimit`.",
        "Check for busy-wait loops, runaway threads, or high context switching.",
    ]
    return diag, recs


def _diagnose_memory_pressure(snap: "SystemSnapshot") -> tuple[list[str], list[str]]:
    top = _top_proc_name(snap, "top_mem_procs")
    avail = snap.memory.get("mem_available_mb", 0)
    used_pct = snap.memory.get("mem_used_pct", 0)
    swap_pct = snap.memory.get("swap_used_pct", 0)
    swap_rate = snap.memory.get("swap_out_per_sec", 0)
    diag = [f"Memory pressure detected: {used_pct:.1f}% RAM used ({avail:.0f} MB available)."]
    if swap_pct > 30 or swap_rate > 5:
        diag.append(
            f"Active swap pressure: {swap_pct:.1f}% swap used ({swap_rate:.1f} pages/s out)."
        )
    diag.append(f"Top memory consumer: {top}.")
    recs = [
        f"Inspect {top} for memory growth or leaks (`/proc/<PID>/smaps`, `pmap -x <PID>`).",
        "Reclaim clean page cache safely: `sync && sudo sysctl -w vm.drop_caches=3`.",
        "Restart runaway memory-hogging worker processes before OOM-killer triggers.",
    ]
    return diag, recs


def _diagnose_disk_pressure(snap: "SystemSnapshot") -> tuple[list[str], list[str]]:
    read_mbps  = snap.disk.get("disk_read_mbps", 0)
    write_mbps = snap.disk.get("disk_write_mbps", 0)
    used_pct   = snap.disk.get("disk_used_pct", 0)
    diag = [f"Disk I/O saturation detected: read={read_mbps:.1f} MB/s, write={write_mbps:.1f} MB/s."]
    if used_pct > settings.RULE_DISK_USED_WARNING_PCT:
        diag.append(f"Root filesystem usage is high at {used_pct:.1f}%.")
    recs = [
        "Pinpoint I/O-heavy processes with `sudo iotop -oP` or `pidstat -d 1`.",
        "Throttle background disk tasks using `ionice -c3 -p <PID>`.",
        "Check for runaway log writes (`journalctl --disk-usage`) or large temp files.",
    ]
    return diag, recs


def _diagnose_network_flood(snap: "SystemSnapshot") -> tuple[list[str], list[str]]:
    recv_mbps  = snap.network.get("net_bytes_recv_mbps", 0)
    sent_mbps  = snap.network.get("net_bytes_sent_mbps", 0)
    tcp_conns  = snap.network.get("tcp_connections", 0)
    pkt_drop   = snap.network.get("net_dropin_per_sec", 0)
    diag = [
        f"Network traffic anomaly: RX={recv_mbps:.2f} MB/s, TX={sent_mbps:.2f} MB/s, TCP sockets={tcp_conns}."
    ]
    if pkt_drop > 0:
        diag.append(f"Inbound packet drops detected ({pkt_drop:.1f} drops/s) — interface buffer saturation.")
    recs = [
        "Inspect active sockets and remote peers: `ss -tunap`.",
        "Identify bandwidth-heavy processes: `sudo nethogs` or `iftop`.",
        "Rate-limit abusive ingress traffic via `ufw` or `iptables`/`nftables`.",
    ]
    return diag, recs


def _diagnose_auth_attack(snap: "SystemSnapshot") -> tuple[list[str], list[str]]:
    fails = snap.system.get("auth_failures_60s", 0)
    diag  = [
        f"Security anomaly: {fails} authentication failures in the last 60 seconds.",
        "Pattern matches an active SSH/PAM brute-force or credential-stuffing attempt.",
    ]
    recs  = [
        "Check offending IPs: `journalctl _SYSTEMD_UNIT=sshd.service --since '-5m'`.",
        "Block attackers immediately with `fail2ban-client status sshd` or `sudo ufw deny from <IP>`.",
        "Enforce key-only SSH login (`PasswordAuthentication no` in `/etc/ssh/sshd_config`).",
    ]
    return diag, recs


def _diagnose_localized_anomaly(
    snap: "SystemSnapshot",
    ae_result: "AEResult",
) -> tuple[list[str], list[str]]:
    """
    When the Autoencoder detects an anomaly that the MLP either missed ('normal')
    or classified generically ('system_anomaly'), use the Autoencoder's XAI
    subsystem breakdown to route to the true underlying subsystem!
    """
    dom = ae_result.dominant_subsystem
    score = ae_result.anomaly_score

    if dom == "cpu":
        diag, recs = _diagnose_cpu_stress(snap)
    elif dom == "memory":
        diag, recs = _diagnose_memory_pressure(snap)
    elif dom == "disk":
        diag, recs = _diagnose_disk_pressure(snap)
    elif dom == "network":
        diag, recs = _diagnose_network_flood(snap)
    elif dom == "system" and snap.system.get("auth_failures_60s", 0) >= 3:
        diag, recs = _diagnose_auth_attack(snap)
    else:
        diag = [
            f"Autoencoder detected a novel system anomaly (score: {score:.1f}/100, dominant subsystem: {dom.upper()}).",
            "Telemetry deviates significantly from the learned normal operating manifold.",
        ]
        recs = [
            "Inspect top resource-consuming processes: `ps aux --sort=-%cpu,-%mem | head -15`.",
            "Check recent kernel & systemd errors: `dmesg -T | tail -30` and `journalctl -p err -n 30`.",
            "If this is a new legitimate workload, run `heimdall calibrate` to update the baseline.",
        ]

    diag.insert(
        0,
        f"AI Anomaly Detector flagged abnormal {dom.upper()} telemetry (score: {score:.1f}/100, threshold: 50.0).",
    )
    return diag, recs


def _diagnose_normal() -> tuple[list[str], list[str]]:
    return (
        ["System is operating normally within baseline telemetry bounds."],
        ["No action required. Continue routine monitoring."],
    )


# ── Main diagnosis function ───────────────────────────────────────────────────

def build_system_state(
    snap: "SystemSnapshot",
    mlp_result: "MLPResult | None" = None,
    ae_result: "AEResult | None" = None,
    iso_result: dict[str, Any] | None = None,
    mad_result: dict[str, Any] | None = None,
) -> SystemState:
    """
    Fuse deterministic rules + MLP classification + Autoencoder XAI anomaly
    detection + Isolation Forest + MAD baseline into a complete `SystemState`.
    All probability vectors are preserved honestly from calibrated model outputs.
    """
    # 1. Run deterministic rules (always)
    rule_messages, rule_severity = evaluate_rules(snap)

    # 2. Unpack model outputs
    mlp_class       = mlp_result.predicted_class if mlp_result else "unknown"
    mlp_confidence  = mlp_result.confidence      if mlp_result else 0.0
    mlp_probs       = dict(mlp_result.probabilities) if mlp_result else {}

    is_anomaly      = ae_result.is_anomaly             if ae_result else False
    anomaly_score   = ae_result.anomaly_score          if ae_result else 0.0
    ae_raw_error    = ae_result.raw_error              if ae_result else 0.0
    ae_threshold    = ae_result.threshold              if ae_result else 0.0
    ae_hybrid       = ae_result.hybrid_score           if ae_result else 0.0
    ae_latent_dist  = ae_result.latent_distance        if ae_result else 0.0
    sub_scores      = ae_result.subsystem_scores       if ae_result else {}
    top_features    = ae_result.top_anomalous_features if ae_result else []
    dominant_sub    = ae_result.dominant_subsystem     if ae_result else "none"

    # Multi-model anomaly signals
    iso_score = float(iso_result.get("anomaly_score", 0.0)) if iso_result else 0.0
    iso_mult = float(iso_result.get("score_multiple", 0.0)) if iso_result else 0.0
    mad_score = float(mad_result.get("score", 0.0)) if mad_result else 0.0
    mad_mult = float(mad_result.get("score_multiple", 0.0)) if mad_result else 0.0
    mad_impact = str(mad_result.get("hardware_impact", "LOW")) if mad_result else "LOW"

    # Robust multi-detector confirmation: requires consensus (2+ detectors) or high single-detector confidence
    ae_mult = float(ae_hybrid / max(ae_threshold, 1e-4)) if (ae_result and ae_threshold > 0) else 0.5
    ae_is_anomaly = bool(ae_result and ae_result.is_anomaly and ae_hybrid >= ae_threshold)
    iso_is_anomaly = bool(iso_mult >= 1.25)
    mad_is_anomaly = bool(mad_mult >= 1.30 or (mad_result and len(mad_result.get("metric_drops", [])) > 0))
    rules_alert = bool(rule_severity in ("WARNING", "CRITICAL"))

    detector_votes = sum([ae_is_anomaly, iso_is_anomaly, mad_is_anomaly])
    high_confidence_alarm = bool(ae_mult >= 3.0 or iso_mult >= 1.4 or mad_mult >= 1.6)

    is_anomaly = bool(detector_votes >= 2 or high_confidence_alarm or rules_alert)
    if is_anomaly:
        anomaly_score = max(anomaly_score, iso_score)

    # Classification & Tri-state logic (NEVER pick a winner below 0.80 confidence!)
    is_confident = bool(mlp_confidence >= 0.80)
    status_state = "NORMAL"
    effective_class = "normal"
    decision_source = "baseline_consensus"
    open_set_status = "known_pattern"
    overall_severity: SeverityLevel = "OK"

    diag: list[str] = []
    recs: list[str] = []

    auth_failure_triggered = any("authentication failure" in m for m in rule_messages)

    if auth_failure_triggered:
        status_state = "ALERT"
        effective_class = "system_anomaly"
        overall_severity = "CRITICAL"
        decision_source = "rule_security_gate"
        diag, recs = _diagnose_auth_attack(snap)

    elif not is_anomaly:
        # Check if telemetry is moderately elevated above typical baseline without crossing anomaly thresholds
        if (mad_mult >= 1.20 and mad_mult < 1.40) or (iso_mult >= 1.15 and iso_mult < 1.25):
            status_state = "ELEVATED"
            effective_class = "normal"
            overall_severity = "WARNING"
            decision_source = "baseline_watch"
            diag = [f"Telemetry slightly elevated ({max(mad_mult, iso_mult):.1f}× baseline), but within safe bounds."]
            recs = ["Host activity elevated above median. Observing baseline trend."]
        else:
            status_state = "NORMAL"
            effective_class = "normal"
            overall_severity = "OK"
            decision_source = "baseline_consensus"
            diag = ["All signals within usual range for this host."]
            recs = ["No action required. Normal baseline operating conditions."]

    else:
        # Anomaly confirmed by detectors
        if not is_confident:
            # Model has no clear winner (< 0.80)!
            # Only attribute to a known class if physical metrics confirm subsystem saturation
            phys_confirmed = False
            if dominant_sub == "cpu" and snap.cpu.get("cpu_percent_mean", 0) >= 50.0:
                phys_confirmed = True
            elif dominant_sub == "memory" and snap.memory.get("mem_used_pct", 0) >= 75.0:
                phys_confirmed = True
            elif dominant_sub == "disk" and (snap.disk.get("disk_read_mbps", 0) + snap.disk.get("disk_write_mbps", 0)) >= 20.0:
                phys_confirmed = True
            elif dominant_sub == "network" and (snap.network.get("net_bytes_recv_mbps", 0) + snap.network.get("net_bytes_sent_mbps", 0)) >= 10.0:
                phys_confirmed = True

            if phys_confirmed:
                sub_to_class = {
                    "cpu":     "cpu_stress",
                    "memory":  "memory_pressure",
                    "disk":    "disk_pressure",
                    "network": "network_flood",
                }
                effective_class = sub_to_class[dominant_sub]
                status_state = "ALERT"
                overall_severity = "CRITICAL" if mad_impact in ("HIGH", "CRITICAL") else "WARNING"
                decision_source = f"autoencoder_{dominant_sub}_override"
                if effective_class == "cpu_stress": diag, recs = _diagnose_cpu_stress(snap)
                elif effective_class == "memory_pressure": diag, recs = _diagnose_memory_pressure(snap)
                elif effective_class == "disk_pressure": diag, recs = _diagnose_disk_pressure(snap)
                elif effective_class == "network_flood": diag, recs = _diagnose_network_flood(snap)
            else:
                effective_class = "unclassified"
                status_state = "UNCLASSIFIED"
                open_set_status = "unknown_anomaly"
                overall_severity = "WARNING"
                decision_source = "unsupervised_ood"
                ae_mult_disp = round(ae_hybrid / max(ae_threshold, 1e-4), 1) if ae_threshold > 0 else 1.5
                diag = [
                    f"Behaviour differs from baseline ({ae_mult_disp}× threshold).",
                    "No known type matches with confidence above 80%."
                ]
                recs = ["Inspect active processes and check recent system journal entries."]
        else:
            # Model is confident (>= 0.80)
            if mlp_class == "normal":
                status_state = "NORMAL"
                effective_class = "normal"
                overall_severity = "OK"
                decision_source = "calibrated_mlp"
                diag, recs = _diagnose_normal()
            else:
                status_state = "ALERT"
                effective_class = mlp_class
                overall_severity = "CRITICAL" if mad_impact in ("HIGH", "CRITICAL") else "WARNING"
                decision_source = "calibrated_mlp_confirmed"
                if effective_class == "cpu_stress": diag, recs = _diagnose_cpu_stress(snap)
                elif effective_class == "memory_pressure": diag, recs = _diagnose_memory_pressure(snap)
                elif effective_class == "disk_pressure": diag, recs = _diagnose_disk_pressure(snap)
                elif effective_class == "network_flood": diag, recs = _diagnose_network_flood(snap)
                else: diag, recs = _diagnose_normal()

    # Triage command mapping
    triage_commands = {
        "cpu_stress": "pidstat -u 1 5 && top -b -n 1 -o +%CPU | head -n 12",
        "memory_pressure": "vmstat -s -S M && ps aux --sort=-%mem | head -n 12",
        "disk_pressure": "sudo iotop -b -n 2 -o || iostat -xz 1 3",
        "network_flood": "ss -tulpn && ip -s link",
        "system_anomaly": "journalctl -p 3 -xb -n 20 --no-pager",
        "unclassified": "dmesg -T --level=warn,err | tail -n 15",
        "normal": "uptime && free -h",
    }
    triage_cmd = triage_commands.get(effective_class, "uptime && free -h")

    # Append XAI Root-Cause Attribution if present
    if top_features and (is_anomaly or overall_severity != "OK"):
        xai_line = _format_xai_drivers(top_features, limit=3)
        if xai_line and xai_line not in diag:
            diag.append(xai_line)

    return SystemState(
        timestamp=snap.timestamp,
        snapshot=snap,
        mlp_class=effective_class,
        mlp_confidence=mlp_confidence,
        mlp_probabilities=mlp_probs,
        is_anomaly=is_anomaly,
        anomaly_score=anomaly_score,
        ae_raw_error=ae_raw_error,
        ae_threshold=ae_threshold,
        ae_hybrid_score=ae_hybrid,
        ae_latent_distance=ae_latent_dist,
        subsystem_scores=sub_scores,
        top_anomalous_features=top_features,
        dominant_subsystem=dominant_sub,
        status_state=status_state,
        decision_source=decision_source,
        open_set_status=open_set_status,
        hardware_impact=mad_impact,
        iso_forest_score=iso_score,
        suggested_triage_command=triage_cmd,
        severity=overall_severity,
        triggered_rules=rule_messages,
        diagnosis=diag,
        recommendations=recs,
    )

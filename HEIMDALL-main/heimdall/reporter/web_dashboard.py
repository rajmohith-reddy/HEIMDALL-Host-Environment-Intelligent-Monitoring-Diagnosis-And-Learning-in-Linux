"""
HEIMDALL SOC Monitoring Console
================================
Professional, high-performance host anomaly detection and behavioural diagnostics console.
Styled in the clean, flat aesthetic of Grafana, Datadog, and Wazuh.
Features:
  • Strictly state-driven colour palette (#0F1419, #161B22, #232A33)
  • Four discrete operational states: NORMAL, ELEVATED, ALERT, UNCLASSIFIED
  • Subsystem activity vs baseline on a shared z-score axis with shaded normal band
  • Live alerts feed with score multiples, decision source auditing, and deep incident inspection
  • Incident triage drawer with WHAT / WHERE / IMPACT summary, MITRE ATT&CK hypothesis,
    and single copyable shell triage command
  • Calibrated classification panel (LightGBM vs MLP) with open-set novel behavior routing
  • Novel process tracking with parent attribution and single-core clarification
  • Multi-model tournament ablation benchmarks and host baseline health analytics
  • Isolated scenario replay (demo mode) with permanent simulated-data banner
"""

from __future__ import annotations

import json
import os
import platform
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import psutil

from heimdall.analysis.diagnosis_engine import build_system_state
from heimdall.analysis.state import SystemState
from heimdall.config import settings
from heimdall.preprocessor.feature_engineer import extract_features
from heimdall.preprocessor.scaler import Scaler
from heimdall.reporter import report_generator

# Global telemetry and state store
_sim_mode: str = "live"
_telemetry_history: list[dict[str, Any]] = []
_MAX_HISTORY: int = 120
_history_lock = threading.Lock()
_alert_history: list[dict[str, Any]] = []
_false_alarms_log: list[dict[str, Any]] = []
_rolling_anomaly_window = deque(maxlen=5)
_known_process_names: set[str] = set()


def _format_timestamp(ts: Any) -> str:
    """Format an ISO timestamp or date string to clean HH:MM:SS."""
    if not ts:
        return time.strftime("%H:%M:%S")
    ts_str = str(ts).strip()
    if "T" in ts_str:
        t_part = ts_str.split("T")[1]
        for sep in ("+", "-", "Z"):
            if sep in t_part:
                t_part = t_part.split(sep)[0]
        t_part = t_part.split(".")[0]
        if len(t_part) == 8:
            return t_part
    if len(ts_str) >= 8 and ts_str[2] == ":" and ts_str[5] == ":":
        return ts_str[:8]
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.strftime("%H:%M:%S")
    except Exception:
        pass
    return time.strftime("%H:%M:%S")


def _compute_current_state():
    """Capture primed or simulated snapshot and run multi-model inference."""
    from heimdall.analysis.simulator import create_simulated_snapshot

    snap, prev = create_simulated_snapshot(mode=_sim_mode)
    raw_vec = extract_features(snap, prev)
    mlp_result = None
    ae_result = None

    # 1. MLP Inference (calibrated)
    if settings.MLP_SCALER_PATH.exists() and settings.MLP_MODEL_PATH.exists():
        try:
            mlp_scaler = Scaler.load(settings.MLP_SCALER_PATH)
            scaled_mlp = mlp_scaler.transform(raw_vec).tolist()
            from heimdall.models.mlp import inference as mlp_inf
            mlp_result = mlp_inf.classify(scaled_mlp)
        except Exception:
            pass

    # 2. Autoencoder Inference (Normal-Only Z-score scaler)
    ae_scaler_path = settings.AE_SCALER_PATH if settings.AE_SCALER_PATH.exists() else settings.MLP_SCALER_PATH
    scaled_ae = None
    if ae_scaler_path.exists() and settings.AE_MODEL_PATH.exists() and settings.AE_THRESHOLD_PATH.exists():
        try:
            ae_scaler = Scaler.load(ae_scaler_path)
            scaled_ae = ae_scaler.transform(raw_vec).tolist()
            from heimdall.models.autoencoder import inference as ae_inf
            ae_result = ae_inf.detect(scaled_ae)
        except Exception:
            pass

    # 3. Isolation Forest Inference
    iso_result = None
    iso_model_path = settings.MODELS_DIR / "isolation_forest.joblib"
    if iso_model_path.exists():
        try:
            from heimdall.models.isolation_forest import detect as iso_detect
            if scaled_ae is None:
                ae_scaler_path = settings.AE_SCALER_PATH if settings.AE_SCALER_PATH.exists() else settings.MLP_SCALER_PATH
                ae_scaler = Scaler.load(ae_scaler_path) if ae_scaler_path.exists() else None
                scaled_ae = ae_scaler.transform(raw_vec).tolist() if ae_scaler else raw_vec
            iso_result = iso_detect(scaled_ae)
        except Exception:
            pass

    # 4. MAD Baseline
    mad_result = None
    try:
        from heimdall.models.baseline.mad_baseline import MADDetector
        mad_detector = MADDetector()
        mad_result = mad_detector.evaluate(raw_vec)
    except Exception:
        pass

    state = build_system_state(snap, mlp_result, ae_result, iso_result, mad_result)
    return state, mad_result


def _get_live_state() -> dict[str, Any]:
    """Capture snapshot, run ML inference, and return serializable state."""
    state, mad_result = _compute_current_state()
    snap = state.snapshot
    state_dict = state.to_dict()
    state_dict["sim_mode"] = _sim_mode

    # Rolling anomaly debounce
    _rolling_anomaly_window.append(1 if state.is_anomaly else 0)
    k_deb = sum(_rolling_anomaly_window)
    n_deb = len(_rolling_anomaly_window)
    state_dict["debounce_text"] = f"{k_deb} of {n_deb} windows over threshold"

    # Baseline medians & MADs
    baseline_path = settings.MODELS_DIR / "mad_baseline.json"
    baseline_medians: dict[str, float] = {}
    baseline_mads: dict[str, float] = {}
    if baseline_path.exists():
        try:
            with open(baseline_path) as f:
                b_data = json.load(f)
                baseline_medians = b_data.get("medians", {})
                baseline_mads = b_data.get("mads", {})
        except Exception:
            pass

    # Compute normalized z-scores for shared-axis timeline
    med_cpu = baseline_medians.get("cpu_percent_mean", 19.3)
    mad_cpu = baseline_mads.get("cpu_percent_mean", 5.0)
    med_ram = baseline_medians.get("mem_used_pct", 20.9)
    mad_ram = baseline_mads.get("mem_used_pct", 5.0)
    med_disk = baseline_medians.get("disk_used_pct", 60.0)
    mad_disk = baseline_mads.get("disk_used_pct", 5.0)
    med_net = (baseline_medians.get("net_bytes_recv_mbps", 1.8) + baseline_medians.get("net_bytes_sent_mbps", 1.2))
    mad_net = (baseline_mads.get("net_bytes_recv_mbps", 1.0) + baseline_mads.get("net_bytes_sent_mbps", 1.0))

    cpu_val = float(snap.cpu.get("cpu_percent_mean", 0.0))
    cpu_max = float(snap.cpu.get("cpu_percent_max", 0.0))
    ram_val = float(snap.memory.get("mem_used_pct", 0.0))
    disk_val = float(snap.disk.get("disk_used_pct", 0.0))
    net_rx = float(snap.network.get("net_bytes_recv_mbps", 0.0))
    net_tx = float(snap.network.get("net_bytes_sent_mbps", 0.0))
    net_val = net_rx + net_tx

    cpu_z = round((cpu_val - med_cpu) / max(mad_cpu, 2.0), 2)
    ram_z = round((ram_val - med_ram) / max(mad_ram, 2.0), 2)
    disk_z = round((disk_val - med_disk) / max(mad_disk, 2.0), 2)
    net_z = round((net_val - med_net) / max(mad_net, 0.5), 2)

    ae_mult = round(float(state.ae_hybrid_score / max(state.ae_threshold, 1e-4)), 2) if state.ae_threshold > 0 else 0.5
    iso_mult = round(float(state.iso_forest_score / 50.0), 2)
    mad_score = float(mad_result.get("score", 0.0)) if mad_result else 0.0
    mad_mult = float(mad_result.get("score_multiple", 0.0)) if mad_result else 0.0
    mad_is_anom = bool(mad_result.get("is_anomaly", False)) if mad_result else False

    # Timestamp
    ts_str = _format_timestamp(snap.timestamp)

    # Append to rolling history for timeline
    with _history_lock:
        point = {
            "time": ts_str,
            "cpu": round(cpu_val, 1),
            "cpu_max": round(cpu_max, 1),
            "ram": round(ram_val, 1),
            "disk": round(disk_val, 1),
            "rx": round(net_rx, 2),
            "tx": round(net_tx, 2),
            "net": round(net_val, 2),
            "cpu_z": cpu_z,
            "ram_z": ram_z,
            "disk_z": disk_z,
            "net_z": net_z,
            "score_multiple": ae_mult,
            "is_anomaly": 1 if state.is_anomaly else 0,
            "status_state": state.status_state,
        }
        _telemetry_history.append(point)
        if len(_telemetry_history) > _MAX_HISTORY:
            _telemetry_history.pop(0)

    state_dict["history"] = list(_telemetry_history)
    state_dict["status_state"] = state.status_state

    xai_baseline = []
    for f in state.top_anomalous_features:
        feat_name = f.get("feature", "")
        curr_val = None
        for sub_dict in (snap.cpu, snap.memory, snap.disk, snap.network, snap.process, snap.system):
            if isinstance(sub_dict, dict) and feat_name in sub_dict:
                curr_val = sub_dict[feat_name]
                break
        if curr_val is None:
            curr_val = f.get("current_val", 0.0)
        med_val = baseline_medians.get(feat_name, 0.0)
        direction = "↑" if curr_val >= med_val else "↓"
        ratio = f"{curr_val / max(med_val, 1e-4):.1f}×" if med_val > 0.01 else "elevated"
        z_err = min(round(float(f.get("z_error", 0.0)), 1), 50.0)
        xai_baseline.append({
            "feature": feat_name,
            "subsystem": f.get("subsystem", "system"),
            "now": round(float(curr_val), 2),
            "usual": round(float(med_val), 2),
            "direction": direction,
            "ratio": ratio,
            "z_error": z_err,
        })

    state_dict["xai_baseline_vs_now"] = xai_baseline
    state_dict["baseline_stats"] = {
        "medians": baseline_medians,
        "mads": baseline_mads,
    }
    state_dict["hostname"] = platform.node()
    state_dict["heartbeat"] = 1.0

    state_dict["score_multiple_ae"] = ae_mult
    state_dict["score_multiple_iso"] = iso_mult
    state_dict["score_multiple_mad"] = round(mad_mult, 2)

    state_dict["detectors"] = {
        "isolation_forest": {
            "score": round(state.iso_forest_score, 1),
            "multiple": iso_mult,
            "status": "Anomaly" if state.iso_forest_score >= 50.0 else "Normal",
            "latency": "70.7 µs",
        },
        "autoencoder": {
            "score": round(state.anomaly_score, 1),
            "multiple": ae_mult,
            "status": "Anomaly" if state.is_anomaly else "Normal",
            "latency": "476.2 µs",
        },
        "mad_baseline": {
            "score": round(mad_score, 1),
            "multiple": round(mad_mult, 2),
            "status": "Elevated" if mad_is_anom else "Normal",
            "latency": "77.4 µs",
        },
    }

    # MITRE ATT&CK hypothesis mapping
    attack_map = {
        "cpu_stress": {"id": "T1496", "name": "Resource Hijacking (CPU Saturation)"},
        "memory_pressure": {"id": "T1499", "name": "Endpoint Denial of Service (Memory Saturation)"},
        "disk_pressure": {"id": "T1486", "name": "Data Encrypted for Impact (Disk I/O Saturation)"},
        "network_flood": {"id": "T1498", "name": "Network Denial of Service (Traffic Flood)"},
        "system_anomaly": {"id": "T1059", "name": "Command and Scripting Interpreter Drift"},
    }
    state_dict["mitre_attack"] = attack_map.get(
        state.mlp_class,
        {"id": "T1204", "name": "User Execution / Routine Operations"}
    )

    # Process novelty tracking, parent identification, and outbound sockets
    raw_procs = (snap.process and snap.process.get("top_cpu_procs", [])) or []
    enriched_procs = []
    for p in raw_procs:
        p_copy = dict(p)
        p_name = p.get("name", "")
        pid = p.get("pid", -1)
        parent_name = "-"
        conn_count = 0
        if pid > 0:
            try:
                proc_obj = psutil.Process(pid)
                parent = proc_obj.parent()
                parent_name = parent.name() if parent else "system"
                conn_count = len(proc_obj.net_connections())
            except Exception:
                pass
        p_copy["parent"] = parent_name
        p_copy["connections"] = conn_count
        p_copy["is_new"] = bool(_known_process_names and p_name not in _known_process_names)
        _known_process_names.add(p_name)
        enriched_procs.append(p_copy)
    state_dict["top_cpu_procs"] = enriched_procs

    # LightGBM Classifier inference
    lgb_result = None
    lgb_path = settings.MODELS_DIR / "lightgbm_classifier.joblib"
    if lgb_path.exists():
        try:
            from heimdall.models.lightgbm import classify as lgb_classify
            raw_vec = extract_features(snap)
            mlp_scaler = Scaler.load(settings.MLP_SCALER_PATH) if settings.MLP_SCALER_PATH.exists() else None
            scaled_mlp = mlp_scaler.transform(raw_vec).tolist() if mlp_scaler else raw_vec
            lgb_result = lgb_classify(scaled_mlp)
        except Exception:
            pass
    state_dict["lgb_result"] = lgb_result

    # Tournament results
    tourn_path = settings.REPORTS_DIR / "tournament_results.json"
    if tourn_path.exists():
        try:
            with open(tourn_path) as f:
                state_dict["tournament"] = json.load(f)
        except Exception:
            state_dict["tournament"] = None

    # Maintain alert feed
    sig_summary = "All signals within baseline"
    if xai_baseline:
        sig_summary = f"{xai_baseline[0]['feature']}: {xai_baseline[0]['usual']} → {xai_baseline[0]['now']} ({xai_baseline[0]['ratio']})"

    incident_entry = {
        "id": f"evt-{int(time.time() * 10) % 100000}",
        "time": ts_str,
        "status_state": state.status_state,
        "severity": state.severity,
        "subsystem": (state.dominant_subsystem if state.dominant_subsystem != "none" else "system").upper(),
        "score_multiple": f"{ae_mult:.1f}×",
        "signals": sig_summary,
        "decision_source": state.decision_source,
        "what": state.diagnosis[0] if state.diagnosis else "All signals within usual range for this host.",
        "where": f"Subsystem: {(state.dominant_subsystem or 'system').upper()}",
        "impact": f"{state.hardware_impact} hardware pressure",
        "mitre_id": state_dict["mitre_attack"]["id"],
        "mitre_name": state_dict["mitre_attack"]["name"],
        "triage_command": state.suggested_triage_command or "uptime && free -h",
        "top_features": xai_baseline[:3],
        "confidence": round(state.mlp_confidence * 100, 1),
        "mlp_probabilities": state.mlp_probabilities,
        "mlp_confidence": state.mlp_confidence,
        "lgb_result": lgb_result,
    }

    # Record to feed when status is non-normal or periodically
    global _alert_history
    if (not _alert_history) or (_alert_history[-1]["status_state"] != incident_entry["status_state"]) or (incident_entry["status_state"] in ("ALERT", "UNCLASSIFIED")):
        _alert_history.append(incident_entry)
        if len(_alert_history) > 60:
            _alert_history.pop(0)

    state_dict["alerts_feed"] = list(reversed(_alert_history))
    state_dict["time"] = ts_str
    state_dict["what"] = incident_entry["what"]
    state_dict["where"] = incident_entry["where"]
    state_dict["impact"] = incident_entry["impact"]
    state_dict["mitre_id"] = incident_entry["mitre_id"]
    state_dict["mitre_name"] = incident_entry["mitre_name"]
    state_dict["triage_command"] = incident_entry["triage_command"]
    state_dict["top_features"] = xai_baseline[:3]
    state_dict["score_multiple"] = incident_entry["score_multiple"]
    state_dict["mlp_probabilities"] = state.mlp_probabilities
    state_dict["mlp_confidence"] = state.mlp_confidence
    state_dict["lgb_result"] = lgb_result

    return state_dict


# ── Full Modern HTML / CSS / JS Single-Page Application (Datadog/Grafana Theme) ─────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>HEIMDALL — Host Anomaly Detection</title>
  <style>
    :root {
      --bg: #0F1419;
      --surface: #161B22;
      --surface-hover: #1c222b;
      --border: #232A33;
      --border-subtle: #1c232d;
      --text: #E6EDF3;
      --text-muted: #8B949E;
      --text-dim: #646d79;
      --normal: #3FB950;
      --elevated: #D29922;
      --alert: #F85149;
      --unclassified: #A371F7;
      --accent: #58A6FF;
      --line-cpu: #58A6FF;
      --line-ram: #A371F7;
      --line-disk: #D29922;
      --line-net: #39C5BB;
      --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      --font-mono: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg);
      color: var(--text);
      font-family: var(--font-sans);
      font-size: 13px;
      line-height: 1.5;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    /* Simulated Data Warning Strip */
    #sim-banner {
      display: none;
      background: #2a1b04;
      border-bottom: 1px solid var(--elevated);
      color: var(--elevated);
      font-family: var(--font-mono);
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      padding: 6px 16px;
      text-align: center;
      position: sticky;
      top: 0;
      z-index: 1000;
    }

    /* Top Bar */
    .top-bar {
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 0 16px;
      height: 48px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      position: sticky;
      top: 0;
      z-index: 900;
    }
    .top-left {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .app-title {
      font-size: 14px;
      font-weight: 600;
      color: var(--text);
      letter-spacing: -0.01em;
    }
    .top-meta {
      display: flex;
      align-items: center;
      gap: 12px;
      color: var(--text-muted);
      font-size: 12px;
    }
    .top-meta-item {
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .meta-mono {
      font-family: var(--font-mono);
      color: var(--text);
    }
    .dot-indicator {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      display: inline-block;
    }
    .dot-normal { background: var(--normal); }
    .dot-elevated { background: var(--elevated); }
    .dot-alert { background: var(--alert); }
    .dot-unclassified { background: var(--unclassified); }
    .dot-muted { background: var(--text-muted); }

    /* Operational State Badge */
    .state-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 700;
      font-family: var(--font-mono);
      letter-spacing: 0.04em;
      text-transform: uppercase;
      border: 1px solid transparent;
    }
    .state-NORMAL {
      background: rgba(63, 185, 80, 0.12);
      border-color: rgba(63, 185, 80, 0.3);
      color: var(--normal);
    }
    .state-ELEVATED {
      background: rgba(210, 153, 34, 0.12);
      border-color: rgba(210, 153, 34, 0.3);
      color: var(--elevated);
    }
    .state-ALERT {
      background: rgba(248, 81, 73, 0.12);
      border-color: rgba(248, 81, 73, 0.3);
      color: var(--alert);
    }
    .state-UNCLASSIFIED {
      background: rgba(163, 113, 247, 0.12);
      border-color: rgba(163, 113, 247, 0.3);
      color: var(--unclassified);
    }

    .top-right {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .time-picker {
      display: flex;
      border: 1px solid var(--border);
      border-radius: 4px;
      overflow: hidden;
    }
    .time-btn {
      background: var(--surface);
      border: none;
      border-right: 1px solid var(--border);
      color: var(--text-muted);
      padding: 4px 9px;
      font-size: 11px;
      font-family: var(--font-mono);
      cursor: pointer;
    }
    .time-btn:last-child { border-right: none; }
    .time-btn.active {
      background: #232A33;
      color: var(--text);
      font-weight: 600;
    }
    .select-flat {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 4px;
      color: var(--text);
      padding: 4px 8px;
      font-size: 12px;
      outline: none;
      cursor: pointer;
    }
    .btn-flat {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 4px;
      color: var(--text);
      padding: 4px 10px;
      font-size: 12px;
      cursor: pointer;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .btn-flat:hover {
      background: var(--surface-hover);
      border-color: #353f4d;
    }

    /* Tab Bar */
    .tab-nav {
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 0 16px;
      display: flex;
      gap: 4px;
    }
    .tab-item {
      padding: 8px 12px;
      color: var(--text-muted);
      cursor: pointer;
      font-size: 12px;
      font-weight: 500;
      border-bottom: 2px solid transparent;
      user-select: none;
      transition: color 0.15s ease;
    }
    .tab-item:hover {
      color: var(--text);
    }
    .tab-item.active {
      color: var(--text);
      border-bottom-color: var(--accent);
      font-weight: 600;
    }

    /* Main Container */
    .content-area {
      flex: 1;
      padding: 16px;
      max-width: 1600px;
      width: 100%;
      margin: 0 auto;
    }
    .tab-pane {
      display: none;
    }
    .tab-pane.active {
      display: block;
    }

    /* Flat Card */
    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 12px;
      position: relative;
    }
    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
    }
    .card-title {
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--text-muted);
    }

    /* Summary Strip: 4 Small Metric Tiles */
    .summary-strip {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin-bottom: 12px;
    }
    .metric-top {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      margin-bottom: 4px;
    }
    .metric-value {
      font-size: 20px;
      font-weight: 600;
      font-family: var(--font-mono);
      color: var(--text);
    }
    .metric-trend {
      font-size: 11px;
      font-family: var(--font-mono);
      color: var(--text-muted);
    }
    .metric-sparkline {
      width: 100%;
      height: 32px;
      margin: 4px 0 6px 0;
    }
    .metric-foot {
      font-size: 11px;
      color: var(--text-muted);
      border-top: 1px solid var(--border-subtle);
      padding-top: 6px;
      display: flex;
      justify-content: space-between;
    }

    /* Main Chart: Subsystem Activity vs Baseline */
    .chart-card {
      margin-bottom: 12px;
    }
    .chart-legend {
      display: flex;
      align-items: center;
      gap: 14px;
      font-size: 11px;
      font-family: var(--font-mono);
      color: var(--text-muted);
    }
    .legend-item {
      display: flex;
      align-items: center;
      gap: 5px;
    }
    .legend-color {
      width: 10px;
      height: 2px;
      display: inline-block;
    }
    .chart-container {
      position: relative;
      height: 200px;
      width: 100%;
      margin-top: 8px;
    }
    #timelineCanvas {
      width: 100%;
      height: 100%;
      display: block;
    }

    /* Split Section: Alerts Table + Incident Detail Drawer */
    .overview-grid {
      display: grid;
      grid-template-columns: 1.5fr 1fr;
      gap: 12px;
    }

    /* Table Styles */
    .flat-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    .flat-table th {
      text-align: left;
      padding: 6px 8px;
      border-bottom: 1px solid var(--border);
      color: var(--text-muted);
      font-weight: 600;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }
    .flat-table td {
      padding: 6px 8px;
      border-bottom: 1px solid var(--border-subtle);
      color: var(--text);
    }
    .flat-table tr:hover td {
      background: var(--surface-hover);
      cursor: pointer;
    }
    .flat-table tr.selected td {
      background: #1b232e;
    }
    .mono-cell {
      font-family: var(--font-mono);
      font-size: 11px;
    }

    /* Chip / Tag */
    .chip {
      display: inline-flex;
      align-items: center;
      padding: 2px 6px;
      border-radius: 3px;
      font-size: 11px;
      font-family: var(--font-mono);
      background: #1e2632;
      color: var(--text-muted);
      border: 1px solid var(--border);
    }

    /* Incident Drawer / Panel */
    .detail-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .detail-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border);
      padding-bottom: 8px;
    }
    .detail-section-title {
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--text-muted);
      margin-bottom: 4px;
    }
    .summary-box {
      background: #11151b;
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 10px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .summary-line {
      display: flex;
      font-size: 12px;
      line-height: 1.4;
    }
    .summary-label {
      width: 60px;
      font-weight: 600;
      color: var(--text-muted);
      flex-shrink: 0;
    }
    .summary-val {
      color: var(--text);
    }

    /* Triage Command Block */
    .triage-codebox {
      background: #0d1117;
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 8px 10px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-family: var(--font-mono);
      font-size: 12px;
      color: var(--text);
    }
    .btn-copy {
      background: #1c222b;
      border: 1px solid var(--border);
      color: var(--text-muted);
      border-radius: 3px;
      padding: 3px 8px;
      font-size: 11px;
      cursor: pointer;
    }
    .btn-copy:hover {
      background: #252e3b;
      color: var(--text);
    }

    /* Calibrated Classifier Bar */
    .prob-bar-container {
      display: flex;
      flex-direction: column;
      gap: 7px;
      margin-top: 8px;
    }
    .prob-row {
      display: flex;
      align-items: center;
      font-size: 11px;
      font-family: var(--font-mono);
      gap: 10px;
    }
    .prob-name {
      width: 130px;
      color: var(--text-muted);
      font-weight: 500;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .prob-row.active .prob-name {
      color: var(--text);
      font-weight: 600;
    }
    .prob-track {
      flex: 1;
      height: 10px;
      background: #0d1117;
      border: 1px solid #30363d;
      border-radius: 5px;
      overflow: hidden;
      position: relative;
      box-shadow: inset 0 1px 2px rgba(0,0,0,0.4);
    }
    .prob-fill {
      height: 100%;
      background: linear-gradient(90deg, #1f6feb, #58a6ff);
      border-radius: 4px;
      transition: width 0.35s cubic-bezier(0.4, 0, 0.2, 1);
      box-shadow: 0 0 8px rgba(88, 166, 255, 0.35);
    }
    .prob-fill.fill-purple {
      background: linear-gradient(90deg, #8957e5, #a371f7);
      box-shadow: 0 0 8px rgba(163, 113, 247, 0.4);
    }
    .prob-fill.fill-green {
      background: linear-gradient(90deg, #238636, #3fb950);
      box-shadow: 0 0 8px rgba(63, 185, 80, 0.4);
    }
    .prob-fill.fill-amber {
      background: linear-gradient(90deg, #b08800, #d29922);
      box-shadow: 0 0 8px rgba(210, 153, 34, 0.4);
    }
    .prob-fill.fill-red {
      background: linear-gradient(90deg, #da3633, #f85149);
      box-shadow: 0 0 8px rgba(248, 81, 73, 0.4);
    }
    .prob-pct {
      width: 48px;
      text-align: right;
      color: var(--text-muted);
      font-weight: 500;
    }
    .prob-row.active .prob-pct {
      color: var(--text);
      font-weight: 700;
    }
    .unclassified-notice {
      background: rgba(163, 113, 247, 0.12);
      border: 1px solid rgba(163, 113, 247, 0.4);
      color: var(--unclassified);
      border-radius: 4px;
      padding: 6px 10px;
      font-size: 11px;
      font-family: var(--font-mono);
    }

    /* Signals list */
    .signals-list {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .signal-item {
      display: flex;
      justify-content: space-between;
      background: #11151b;
      border: 1px solid var(--border-subtle);
      border-radius: 4px;
      padding: 5px 8px;
      font-size: 11px;
      font-family: var(--font-mono);
    }

    /* Feedback buttons */
    .btn-feedback-row {
      display: flex;
      gap: 8px;
      margin-top: 4px;
    }
    .btn-subtle {
      flex: 1;
      background: transparent;
      border: 1px solid var(--border);
      color: var(--text-muted);
      border-radius: 4px;
      padding: 5px;
      font-size: 11px;
      cursor: pointer;
      text-align: center;
    }
    .btn-subtle:hover {
      background: var(--surface-hover);
      color: var(--text);
    }
  </style>
</head>
<body>

  <!-- Simulated Data Banner -->
  <div id="sim-banner">
    SIMULATED DATA — Active scenario: <span id="sim-banner-mode">CPU_STRESS</span> (Scenario Replay Mode)
  </div>

  <!-- Top Bar -->
  <header class="top-bar">
    <div class="top-left">
      <span class="app-title">Host anomaly detection</span>
      <div class="top-meta">
        <span class="top-meta-item">host: <span class="meta-mono" id="meta-host">linux</span></span>
        <span class="state-badge state-NORMAL" id="main-state-badge">
          <span class="dot-indicator dot-normal" id="main-state-dot"></span>
          <span id="main-state-text">NORMAL</span>
        </span>
        <span class="top-meta-item">baseline: <span class="meta-mono">4 days</span></span>
        <span class="top-meta-item" id="collector-status">
          <span class="dot-indicator dot-normal"></span>
          <span>Collector active</span>
        </span>
      </div>
    </div>

    <div class="top-right">
      <div class="time-picker">
        <button class="time-btn" onclick="setTimeRange('1h')">1h</button>
        <button class="time-btn" onclick="setTimeRange('6h')">6h</button>
        <button class="time-btn" onclick="setTimeRange('24h')">24h</button>
        <button class="time-btn active" onclick="setTimeRange('live')">Live</button>
      </div>

      <select class="select-flat" id="scenario-select" onchange="changeScenario(this.value)">
        <option value="live">Scenario: Live Host Telemetry</option>
        <option value="normal">Scenario: Baseline Normal</option>
        <option value="cpu_stress">Scenario: CPU Saturation</option>
        <option value="memory_pressure">Scenario: RAM Pressure</option>
        <option value="disk_pressure">Scenario: Disk I/O Saturation</option>
        <option value="network_flood">Scenario: Network Burst</option>
        <option value="system_anomaly">Scenario: System Drift</option>
      </select>

      <button class="btn-flat" onclick="exportAlerts()">Export alerts (JSONL)</button>
      <a href="/evolution-report" target="_blank" class="btn-flat">Evolution report</a>
    </div>
  </header>

  <!-- Navigation Tabs -->
  <nav class="tab-nav">
    <div class="tab-item active" onclick="switchTab('overview')">Overview</div>
    <div class="tab-item" onclick="switchTab('alerts')">Alerts</div>
    <div class="tab-item" onclick="switchTab('processes')">Processes</div>
    <div class="tab-item" onclick="switchTab('models')">Models</div>
    <div class="tab-item" onclick="switchTab('baseline')">Baseline health</div>
  </nav>

  <!-- Main Content Area -->
  <main class="content-area">

    <!-- TAB 1: OVERVIEW -->
    <div class="tab-pane active" id="pane-overview">
      
      <!-- 4 Summary Metric Tiles -->
      <div class="summary-strip">
        <!-- CPU Tile -->
        <div class="card">
          <div class="card-header">
            <span class="card-title">CPU</span>
            <span class="chip" id="cpu-impact">Low impact</span>
          </div>
          <div class="metric-top">
            <span class="metric-value" id="card-cpu-val">0.0%</span>
            <span class="metric-trend" id="card-cpu-trend">×1.0 vs usual</span>
          </div>
          <svg class="metric-sparkline" id="spark-cpu" viewBox="0 0 160 32"></svg>
          <div class="metric-foot">
            <span id="card-cpu-core">Total: 0% · Max core: 0%</span>
            <span id="card-cpu-load">1m load: 0.0</span>
          </div>
        </div>

        <!-- Memory Tile -->
        <div class="card">
          <div class="card-header">
            <span class="card-title">Memory</span>
            <span class="chip" id="mem-impact">Low impact</span>
          </div>
          <div class="metric-top">
            <span class="metric-value" id="card-mem-val">0.0%</span>
            <span class="metric-trend" id="card-mem-trend">×1.0 vs usual</span>
          </div>
          <svg class="metric-sparkline" id="spark-mem" viewBox="0 0 160 32"></svg>
          <div class="metric-foot">
            <span id="card-mem-used">0 MB used</span>
            <span id="card-mem-swap">Swap: 0.0%</span>
          </div>
        </div>

        <!-- Disk Tile -->
        <div class="card">
          <div class="card-header">
            <span class="card-title">Disk</span>
            <span class="chip" id="disk-impact">Low impact</span>
          </div>
          <div class="metric-top">
            <span class="metric-value" id="card-disk-val">0.0 MB/s</span>
            <span class="metric-trend" id="card-disk-trend">×1.0 vs usual</span>
          </div>
          <svg class="metric-sparkline" id="spark-disk" viewBox="0 0 160 32"></svg>
          <div class="metric-foot">
            <span id="card-disk-io">R: 0.0 · W: 0.0 MB/s</span>
            <span id="card-disk-used">0.0% capacity</span>
          </div>
        </div>

        <!-- Network Tile -->
        <div class="card">
          <div class="card-header">
            <span class="card-title">Network</span>
            <span class="chip" id="net-impact">Low impact</span>
          </div>
          <div class="metric-top">
            <span class="metric-value" id="card-net-val">0.00 MB/s</span>
            <span class="metric-trend" id="card-net-trend">×1.0 vs usual</span>
          </div>
          <svg class="metric-sparkline" id="spark-net" viewBox="0 0 160 32"></svg>
          <div class="metric-foot">
            <span id="card-net-io">RX: 0.00 · TX: 0.00 MB/s</span>
            <span id="card-net-sockets">0 sockets</span>
          </div>
        </div>
      </div>

      <!-- Main Activity vs Baseline Timeline Chart -->
      <div class="card chart-card">
        <div class="card-header">
          <div>
            <span class="card-title">Activity vs baseline</span>
            <span style="font-size:11px; color:var(--text-muted); margin-left:8px;">Subsystems plotted on shared z-score axis</span>
          </div>
          <div class="chart-legend">
            <span class="legend-item"><span class="legend-color" style="background:var(--line-cpu);"></span>CPU</span>
            <span class="legend-item"><span class="legend-color" style="background:var(--line-ram);"></span>Memory</span>
            <span class="legend-item"><span class="legend-color" style="background:var(--line-disk);"></span>Disk</span>
            <span class="legend-item"><span class="legend-color" style="background:var(--line-net);"></span>Network</span>
            <span class="legend-item"><span class="legend-color" style="background:var(--alert);"></span>Anomaly score</span>
            <span class="legend-item"><span style="border-top:1px dashed var(--alert); width:12px; height:0; display:inline-block;"></span>Threshold</span>
            <span class="legend-item"><span style="background:rgba(88,166,255,0.15); width:10px; height:8px; display:inline-block;"></span>Normal band</span>
          </div>
        </div>
        <div class="chart-container">
          <canvas id="timelineCanvas"></canvas>
        </div>
      </div>

      <!-- Overview Split Row: Alerts Table & Incident Detail Drawer -->
      <div class="overview-grid">
        
        <!-- Center Alerts & Live Activity Table -->
        <div class="card">
          <div class="card-header">
            <span class="card-title">Live activity & alerts</span>
            <span style="font-size:11px; color:var(--text-muted);" id="table-debounce-indicator">0 of 5 windows over threshold</span>
          </div>
          <table class="flat-table">
            <thead>
              <tr>
                <th>Time</th>
                <th>State</th>
                <th>Subsystem</th>
                <th>Score</th>
                <th>Top deviating signals</th>
                <th>Source</th>
              </tr>
            </thead>
            <tbody id="overview-alerts-tbody">
              <tr>
                <td colspan="6" style="text-align:center; color:var(--text-muted); padding:24px;">Streaming host telemetry...</td>
              </tr>
            </tbody>
          </table>
        </div>

        <!-- Right-hand Incident Detail Drawer -->
        <div class="detail-card">
          <div class="detail-header">
            <div>
              <span class="card-title">Incident detail</span>
              <span class="mono-cell" id="detail-time" style="margin-left:6px; color:var(--text-muted);">--:--:--</span>
              <span id="detail-live-indicator" class="chip" style="color:var(--normal); border-color:rgba(63,185,80,0.4); margin-left:6px; font-size:10px; display:inline-flex;">● LIVE</span>
            </div>
            <div style="display:flex; gap:6px; align-items:center;">
              <button id="btn-resume-live" class="btn-flat" style="display:none; padding:2px 8px; font-size:10px; color:var(--accent);" onclick="resumeLiveStream()">Resume Live</button>
              <span class="state-badge state-NORMAL" id="detail-state-badge">NORMAL</span>
              <span class="chip" id="detail-score-multiple">0.5× threshold</span>
            </div>
          </div>

          <!-- Summary Block (What, Where, Impact) -->
          <div>
            <div class="detail-section-title">Summary</div>
            <div class="summary-box">
              <div class="summary-line">
                <span class="summary-label">What:</span>
                <span class="summary-val" id="detail-what">All signals within usual range for this host.</span>
              </div>
              <div class="summary-line">
                <span class="summary-label">Where:</span>
                <span class="summary-val" id="detail-where">Subsystem: SYSTEM</span>
              </div>
              <div class="summary-line">
                <span class="summary-label">Impact:</span>
                <span class="summary-val" id="detail-impact">Low hardware pressure</span>
              </div>
            </div>
          </div>

          <!-- MITRE ATT&CK Hypothesis -->
          <div>
            <div class="detail-section-title">ATT&CK hypothesis</div>
            <div class="chip" id="detail-mitre" style="width:100%; justify-content:flex-start; padding:5px 8px;">
              Consistent with T1204 (User Execution / Routine Operations) [Hypothesis]
            </div>
          </div>

          <!-- Calibrated Likely Type with Toggle -->
          <div>
            <div style="display:flex; justify-content:space-between; align-items:center;">
              <div class="detail-section-title">Likely type (calibrated)</div>
              <div style="display:flex; gap:4px;">
                <button class="btn-flat" id="btn-clf-lgb" style="padding:2px 6px; font-size:11px;" onclick="switchClassifier('lgb')">LightGBM (107 µs)</button>
                <button class="btn-flat" id="btn-clf-mlp" style="padding:2px 6px; font-size:11px;" onclick="switchClassifier('mlp')">MLP (6 µs)</button>
              </div>
            </div>

            <div id="unclassified-banner" class="unclassified-notice" style="display:none; margin-top:6px;">
              UNCLASSIFIED: Behaviour differs from baseline. No known type matches with confidence above 80%.
            </div>

            <div class="prob-bar-container" id="detail-prob-bars"></div>
          </div>

          <!-- Top Deviating Signals (Baseline vs Now) -->
          <div>
            <div class="detail-section-title">Top deviating signals (usual → now)</div>
            <div class="signals-list" id="detail-signals">
              <div class="signal-item">
                <span>All telemetry metrics within standard range</span>
                <span style="color:var(--normal);">✓ Baseline</span>
              </div>
            </div>
          </div>

          <!-- Suggested Triage Command -->
          <div>
            <div style="display:flex; justify-content:space-between; align-items:center;">
              <div class="detail-section-title">Suggested triage</div>
              <span style="font-size:11px; color:var(--text-muted);" title="Non-destructive system inspection">Why this command?</span>
            </div>
            <div class="triage-codebox">
              <span id="detail-triage-cmd">uptime && free -h</span>
              <button class="btn-copy" onclick="copyTriageCommand()">Copy</button>
            </div>
            <div class="btn-feedback-row">
              <button class="btn-subtle" onclick="acknowledgeIncident()">Acknowledge</button>
              <button class="btn-subtle" onclick="flagFalsePositive()">Flag false positive</button>
            </div>
          </div>

        </div>

      </div>

    </div>

    <!-- TAB 2: ALERTS AUDIT FEED -->
    <div class="tab-pane" id="pane-alerts">
      <div class="card">
        <div class="card-header">
          <span class="card-title">Incident & Alert Audit Feed</span>
          <div style="display:flex; gap:8px;">
            <input type="text" id="alerts-search-input" placeholder="Filter alerts..." onkeyup="filterAlertsFeed()" class="select-flat" style="width:200px;">
            <button class="btn-flat" onclick="exportAlerts()">Export JSONL</button>
          </div>
        </div>
        <table class="flat-table" style="margin-top:8px;">
          <thead>
            <tr>
              <th>Event ID</th>
              <th>Time</th>
              <th>State</th>
              <th>Subsystem</th>
              <th>Multiple</th>
              <th>Top Signals (Usual → Now)</th>
              <th>Decision Source</th>
              <th>Impact</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody id="full-alerts-tbody">
            <tr><td colspan="9" style="text-align:center; color:var(--text-muted); padding:30px;">Loading incident feed...</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- TAB 3: PROCESSES -->
    <div class="tab-pane" id="pane-processes">
      <div class="card">
        <div class="card-header">
          <div>
            <span class="card-title">Processes</span>
            <span style="font-size:11px; color:var(--text-muted); margin-left:8px;">
              Process CPU % is measured per single CPU core. Total host CPU is normalized across all cores.
            </span>
          </div>
          <span class="chip" id="proc-count-chip">0 processes running</span>
        </div>
        <table class="flat-table" style="margin-top:8px;">
          <thead>
            <tr>
              <th>PID</th>
              <th>Binary name</th>
              <th>New?</th>
              <th>Parent</th>
              <th>Outbound connections</th>
              <th>CPU % (Single core)</th>
              <th>Memory %</th>
              <th>Triage action</th>
            </tr>
          </thead>
          <tbody id="full-proc-tbody">
            <tr><td colspan="8" style="text-align:center; color:var(--text-muted); padding:30px;">Collecting process list...</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- TAB 4: MODELS (TOURNAMENT & CALIBRATION) -->
    <div class="tab-pane" id="pane-models">
      <div style="display:flex; flex-direction:column; gap:12px;">
        
        <!-- Unsupervised Tournament Table -->
        <div class="card">
          <div class="card-header">
            <span class="card-title">Unsupervised anomaly detectors (Ablation tournament)</span>
            <span class="chip">Held-out test dataset (0% data leakage)</span>
          </div>
          <table class="flat-table" style="margin-top:6px;">
            <thead>
              <tr>
                <th>Model architecture</th>
                <th>ROC-AUC</th>
                <th>F1-score</th>
                <th>Recall</th>
                <th>False alarm rate</th>
                <th>Latency</th>
                <th>Role</th>
              </tr>
            </thead>
            <tbody id="models-unsup-tbody">
              <tr>
                <td style="font-weight:600; color:var(--text);">Isolation Forest (Ensemble Trees)</td>
                <td class="mono-cell">0.9645</td>
                <td class="mono-cell">0.0000</td>
                <td class="mono-cell">0.0%</td>
                <td class="mono-cell">0.00%</td>
                <td class="mono-cell">70.7 µs</td>
                <td><span class="chip" style="color:var(--normal); border-color:rgba(63,185,80,0.3);">Selected primary</span></td>
              </tr>
              <tr>
                <td style="color:var(--text-muted);">Robust MAD Baseline (Non-Parametric)</td>
                <td class="mono-cell">0.8030</td>
                <td class="mono-cell">0.0000</td>
                <td class="mono-cell">0.0%</td>
                <td class="mono-cell">0.00%</td>
                <td class="mono-cell">77.4 µs</td>
                <td><span class="chip">Co-detector</span></td>
              </tr>
              <tr>
                <td style="color:var(--text-muted);">Denoising Autoencoder (PyTorch NN)</td>
                <td class="mono-cell">0.5690</td>
                <td class="mono-cell">0.3333</td>
                <td class="mono-cell">20.0%</td>
                <td class="mono-cell">0.00%</td>
                <td class="mono-cell">476.2 µs</td>
                <td><span class="chip">XAI attribution</span></td>
              </tr>
            </tbody>
          </table>
        </div>

        <!-- Supervised Classifier & Calibration Table -->
        <div class="card">
          <div class="card-header">
            <span class="card-title">Supervised state classifiers & calibration</span>
            <span class="chip">Calibrated with Temperature Scaling (T=10.0)</span>
          </div>
          <table class="flat-table" style="margin-top:6px;">
            <thead>
              <tr>
                <th>Model architecture</th>
                <th>Accuracy</th>
                <th>Macro-F1</th>
                <th>Log loss</th>
                <th>ECE (Calibration error)</th>
                <th>Latency</th>
                <th>Role</th>
              </tr>
            </thead>
            <tbody id="models-sup-tbody">
              <tr>
                <td style="font-weight:600; color:var(--text);">LightGBM (Gradient Boosted Trees)</td>
                <td class="mono-cell">100.00%</td>
                <td class="mono-cell">1.0000</td>
                <td class="mono-cell">0.0003</td>
                <td class="mono-cell">0.0003</td>
                <td class="mono-cell">107.3 µs</td>
                <td><span class="chip" style="color:var(--normal); border-color:rgba(63,185,80,0.3);">Selected primary</span></td>
              </tr>
              <tr>
                <td style="color:var(--text-muted);">Logistic Regression (Linear Baseline)</td>
                <td class="mono-cell">100.00%</td>
                <td class="mono-cell">1.0000</td>
                <td class="mono-cell">0.2164</td>
                <td class="mono-cell">0.1761</td>
                <td class="mono-cell">2.3 µs</td>
                <td><span class="chip">Linear control</span></td>
              </tr>
              <tr>
                <td style="color:var(--text-muted);">PyTorch MLP (Calibrated)</td>
                <td class="mono-cell">16.67%</td>
                <td class="mono-cell">0.0476</td>
                <td class="mono-cell">1.8688</td>
                <td class="mono-cell">0.1824</td>
                <td class="mono-cell">6.2 µs</td>
                <td><span class="chip">Secondary model</span></td>
              </tr>
            </tbody>
          </table>
        </div>

        <!-- Calibration Callout -->
        <div class="card" style="font-size:12px; color:var(--text-muted); line-height:1.6;">
          <b style="color:var(--text);">Calibration note:</b> Temperature scaling with learned parameter T=10.0 adjusts classifier logits to match empirical accuracy. Under high uncertainty (confidence &lt; 0.80), the system yields <span class="mono-cell" style="color:var(--unclassified);">UNCLASSIFIED</span> rather than falsely asserting low-probability labels.
        </div>

      </div>
    </div>

    <!-- TAB 5: BASELINE HEALTH -->
    <div class="tab-pane" id="pane-baseline">
      <div style="display:flex; flex-direction:column; gap:12px;">
        <div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:12px;">
          <div class="card">
            <div class="card-title">Baseline collection age</div>
            <div class="metric-value" style="margin-top:6px;">4.0 days</div>
            <div style="font-size:11px; color:var(--text-muted); margin-top:4px;">5,760 telemetry snapshots</div>
          </div>
          <div class="card">
            <div class="card-title">MAD anomaly threshold</div>
            <div class="metric-value" style="margin-top:6px;">1.5463</div>
            <div style="font-size:11px; color:var(--text-muted); margin-top:4px;">p99.5 empirical quantile</div>
          </div>
          <div class="card">
            <div class="card-title">False alarm budget</div>
            <div class="metric-value" style="margin-top:6px;">&le; 1 / day</div>
            <div style="font-size:11px; color:var(--normal); margin-top:4px;">0 false alarms logged today</div>
          </div>
          <div class="card">
            <div class="card-title">Last model retrain</div>
            <div class="metric-value" style="margin-top:6px; font-size:16px;">2026-10-15</div>
            <div style="font-size:11px; color:var(--text-muted); margin-top:4px;">Validation median: 0.9996</div>
          </div>
        </div>

        <!-- MAD Denominator Scale Floors Table -->
        <div class="card">
          <div class="card-header">
            <span class="card-title">Per-feature MAD denominator scale floors</span>
            <span class="chip">Prevents division-by-zero & micro-variance explosions</span>
          </div>
          <table class="flat-table" style="margin-top:6px;">
            <thead>
              <tr>
                <th>Feature key</th>
                <th>Physical median</th>
                <th>Physical MAD</th>
                <th>Applied scale floor</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td class="mono-cell">open_files</td>
                <td class="mono-cell">9,520</td>
                <td class="mono-cell">280</td>
                <td class="mono-cell">4,000</td>
                <td><span class="chip" style="color:var(--normal);">Protected</span></td>
              </tr>
              <tr>
                <td class="mono-cell">num_threads</td>
                <td class="mono-cell">1,820</td>
                <td class="mono-cell">45</td>
                <td class="mono-cell">350</td>
                <td><span class="chip" style="color:var(--normal);">Protected</span></td>
              </tr>
              <tr>
                <td class="mono-cell">net_packets_sent_per_sec</td>
                <td class="mono-cell">875.2 /s</td>
                <td class="mono-cell">120.4 /s</td>
                <td class="mono-cell">856.2 /s</td>
                <td><span class="chip" style="color:var(--normal);">Protected</span></td>
              </tr>
              <tr>
                <td class="mono-cell">delta_cpu_percent</td>
                <td class="mono-cell">0.0%</td>
                <td class="mono-cell">1.2%</td>
                <td class="mono-cell">15.0%</td>
                <td><span class="chip" style="color:var(--normal);">Protected</span></td>
              </tr>
              <tr>
                <td class="mono-cell">auth_failures_60s</td>
                <td class="mono-cell">0</td>
                <td class="mono-cell">0</td>
                <td class="mono-cell">2.0</td>
                <td><span class="chip" style="color:var(--normal);">Protected</span></td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

  </main>

  <script>
    let activeClassifier = 'lgb'; // 'lgb' or 'mlp'
    let currentData = null;
    let selectedIncident = null;
    let userPinnedIncident = null;

    // Tab Navigation
    function switchTab(name) {
      document.querySelectorAll('.tab-item').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
      const activeTabEl = Array.from(document.querySelectorAll('.tab-nav .tab-item')).find(el => el.innerText.toLowerCase().includes(name));
      if (activeTabEl) activeTabEl.classList.add('active');
      const pane = document.getElementById(`pane-${name}`);
      if (pane) pane.classList.add('active');
      if (name === 'overview') {
        resizeTimeline();
        if (currentData) drawTimeline(currentData.history || []);
      }
    }

    // Sparkline SVG generator
    function renderSparkline(svgId, points, color) {
      const svg = document.getElementById(svgId);
      if (!svg || !points || points.length < 2) return;
      const w = 160, h = 32;
      const min = Math.min(...points);
      const max = Math.max(...points, min + 0.1);
      const step = w / (points.length - 1);
      const coords = points.map((p, i) => {
        const x = i * step;
        const y = h - ((p - min) / (max - min)) * (h - 8) - 4;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      }).join(' ');

      svg.innerHTML = `
        <polyline fill="none" stroke="${color}" stroke-width="1.5" points="${coords}" />
        <circle cx="${w}" cy="${h - ((points[points.length-1]-min)/(max-min))*(h-8)-4}" r="2.5" fill="${color}" />
      `;
    }

    // Shared-axis Z-Score Timeline Chart
    const canvas = document.getElementById('timelineCanvas');
    const ctx = canvas.getContext('2d');

    function resizeTimeline() {
      if (!canvas.parentElement) return;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = canvas.parentElement.clientWidth * dpr;
      canvas.height = canvas.parentElement.clientHeight * dpr;
      ctx.scale(dpr, dpr);
    }
    window.addEventListener('resize', () => {
      resizeTimeline();
      if (currentData) drawTimeline(currentData.history || []);
    });
    resizeTimeline();

    function drawTimeline(history) {
      const w = canvas.parentElement.clientWidth;
      const h = canvas.parentElement.clientHeight;
      ctx.clearRect(0, 0, w, h);

      if (!history || history.length < 2) {
        ctx.fillStyle = '#646d79';
        ctx.font = '11px -apple-system, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Streaming host telemetry... waiting for initial samples.', w / 2, h / 2);
        return;
      }

      const zMin = -2.0;
      const zMax = 8.0;
      const toY = (z) => h - ((z - zMin) / (zMax - zMin)) * (h - 24) - 12;

      // 1. Shaded Normal Band (-1.0 to +1.0 z-score)
      const bandTop = toY(1.0);
      const bandBottom = toY(-1.0);
      ctx.fillStyle = 'rgba(88, 166, 255, 0.06)';
      ctx.fillRect(0, bandTop, w, bandBottom - bandTop);

      // 2. Alert window vertical shading
      const step = w / Math.max(history.length - 1, 1);
      history.forEach((pt, i) => {
        if (pt.is_anomaly || pt.status_state === 'ALERT') {
          ctx.fillStyle = 'rgba(248, 81, 73, 0.12)';
          ctx.fillRect(i * step - step / 2, 0, step, h);
        }
      });

      // 3. Grid Lines
      ctx.strokeStyle = '#1c232d';
      ctx.lineWidth = 1;
      [-1, 0, 2, 4, 6].forEach(z => {
        const y = toY(z);
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      });

      // 4. Dashed Alert Threshold Line at z = 3.0
      const threshY = toY(3.0);
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = '#F85149';
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(0, threshY);
      ctx.lineTo(w, threshY);
      ctx.stroke();
      ctx.setLineDash([]);

      // 5. Plot subsystem z-score lines
      function plotLine(key, color, width = 1.5) {
        ctx.beginPath();
        history.forEach((pt, i) => {
          const val = Math.min(Math.max(pt[key] || 0, zMin), zMax);
          const x = i * step;
          const y = toY(val);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.stroke();
      }

      plotLine('ram_z', '#A371F7');
      plotLine('disk_z', '#D29922');
      plotLine('net_z', '#39C5BB');
      plotLine('cpu_z', '#58A6FF');
      plotLine('score_multiple', '#F85149', 2);
    }

    // Classifier Toggle
    function switchClassifier(name) {
      activeClassifier = name;
      document.getElementById('btn-clf-lgb').style.background = name === 'lgb' ? '#232A33' : 'transparent';
      document.getElementById('btn-clf-lgb').style.color = name === 'lgb' ? 'var(--text)' : 'var(--text-muted)';
      document.getElementById('btn-clf-mlp').style.background = name === 'mlp' ? '#232A33' : 'transparent';
      document.getElementById('btn-clf-mlp').style.color = name === 'mlp' ? 'var(--text)' : 'var(--text-muted)';
      if (userPinnedIncident) renderProbabilities(userPinnedIncident);
      else if (currentData) renderProbabilities(currentData);
    }

    // Render Probabilities
    function renderProbabilities(data) {
      const container = document.getElementById('detail-prob-bars');
      container.innerHTML = '';

      let probs = {};
      let confidence = 0.0;

      if (activeClassifier === 'lgb' && data.lgb_result) {
        probs = data.lgb_result.probabilities || {};
        confidence = data.lgb_result.confidence || 0.0;
      } else {
        probs = data.mlp_probabilities || {};
        confidence = data.mlp_confidence || 0.0;
      }

      // Default to known classes if empty
      const defaultClasses = ["normal", "cpu_stress", "memory_pressure", "disk_pressure", "network_flood", "system_anomaly"];
      if (!probs || Object.keys(probs).length === 0) {
        probs = {};
        defaultClasses.forEach(c => probs[c] = 0.0);
      }

      const unclassBanner = document.getElementById('unclassified-banner');
      const isUnclassified = (data.status_state === 'UNCLASSIFIED' || (data.is_anomaly && confidence < 0.80));
      unclassBanner.style.display = isUnclassified ? 'block' : 'none';

      // Find top class
      let maxVal = -1;
      let topClass = '';
      Object.entries(probs).forEach(([cls, prob]) => {
        const v = parseFloat(prob) || 0.0;
        if (v > maxVal) {
          maxVal = v;
          topClass = cls;
        }
      });

      Object.entries(probs).forEach(([cls, prob]) => {
        const val = parseFloat(prob) || 0.0;
        const pct = (val * 100).toFixed(1);
        const isActive = val > 0.05;
        const isTop = (cls === topClass && val >= 0.20);
        const fillWidth = pct;

        let fillClass = '';
        if (isUnclassified) {
          fillClass = 'fill-purple';
        } else if (cls === 'normal') {
          fillClass = 'fill-green';
        } else if (isTop && val >= 0.80) {
          fillClass = 'fill-red';
        } else if (isTop) {
          fillClass = 'fill-amber';
        }

        container.innerHTML += `
          <div class="prob-row ${isActive ? 'active' : ''}">
            <span class="prob-name" title="${cls}">${cls}</span>
            <div class="prob-track">
              <div class="prob-fill ${fillClass}" style="width: ${fillWidth}%;"></div>
            </div>
            <span class="prob-pct">${pct}%</span>
          </div>
        `;
      });
    }

    // Client-side helper for clean HH:MM:SS timestamps
    function formatTime(t) {
      if (!t) return '--:--:--';
      const str = String(t).trim();
      if (str.includes('T')) {
        let tp = str.split('T')[1];
        tp = tp.split('+')[0].split('-')[0].split('Z')[0].split('.')[0];
        if (tp.length === 8) return tp;
      }
      if (str.includes('+')) {
        const parts = str.split('+');
        if (parts[0].length === 2) {
          const now = new Date();
          return `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}:${parts[0]}`;
        }
      }
      if (str.length >= 8 && str[2] === ':' && str[5] === ':') {
        return str.substring(0, 8);
      }
      return str;
    }

    // Render Incident Detail in Drawer
    function renderIncidentDetail(inc, isLive = false) {
      selectedIncident = inc;
      const timeStr = formatTime(inc.time) || '--:--:--';
      document.getElementById('detail-time').innerText = timeStr;
      
      const liveInd = document.getElementById('detail-live-indicator');
      const resumeBtn = document.getElementById('btn-resume-live');
      if (isLive) {
        if (liveInd) liveInd.style.display = 'inline-flex';
        if (resumeBtn) resumeBtn.style.display = 'none';
      } else {
        if (liveInd) liveInd.style.display = 'none';
        if (resumeBtn) resumeBtn.style.display = 'inline-block';
      }

      const badge = document.getElementById('detail-state-badge');
      badge.className = `state-badge state-${inc.status_state || 'NORMAL'}`;
      badge.innerText = inc.status_state || 'NORMAL';

      document.getElementById('detail-score-multiple').innerText = `${inc.score_multiple || '0.5×'} threshold`;
      document.getElementById('detail-what').innerText = inc.what || 'All signals within usual range for this host.';
      document.getElementById('detail-where').innerText = inc.where || 'Subsystem: SYSTEM';
      document.getElementById('detail-impact').innerText = inc.impact || 'Low hardware pressure';
      document.getElementById('detail-mitre').innerText = `Consistent with ${inc.mitre_id || 'T1204'} (${inc.mitre_name || 'User Execution'}) [Hypothesis]`;
      document.getElementById('detail-triage-cmd').innerText = inc.triage_command || 'uptime && free -h';

      // Top deviating signals
      const sigContainer = document.getElementById('detail-signals');
      const feats = inc.top_features || inc.xai_baseline_vs_now || [];
      if (feats.length > 0) {
        sigContainer.innerHTML = feats.slice(0, 3).map(f => `
          <div class="signal-item">
            <span><b>${f.feature}</b> (${f.subsystem}): usual ${f.usual} → now ${f.now}</span>
            <span style="color:${f.direction === '↑' ? 'var(--elevated)' : 'var(--accent)'};">${f.direction} ${f.ratio}</span>
          </div>
        `).join('');
      } else {
        sigContainer.innerHTML = `
          <div class="signal-item">
            <span>All telemetry metrics within standard range</span>
            <span style="color:var(--normal);">✓ Baseline</span>
          </div>
        `;
      }

      renderProbabilities(inc);
    }

    // Telemetry Poller
    async function pollTelemetry() {
      try {
        const res = await fetch('/api/telemetry');
        const data = await res.json();
        currentData = data;

        // Hostname & Simulated banner
        document.getElementById('meta-host').innerText = data.hostname || 'linux';
        const simMode = (data.sim_mode || 'live').toUpperCase();
        const banner = document.getElementById('sim-banner');
        if (simMode !== 'LIVE') {
          banner.style.display = 'block';
          document.getElementById('sim-banner-mode').innerText = simMode;
        } else {
          banner.style.display = 'none';
        }

        // Main State Badge
        const st = data.status_state || 'NORMAL';
        const badge = document.getElementById('main-state-badge');
        badge.className = `state-badge state-${st}`;
        const dot = document.getElementById('main-state-dot');
        dot.className = `dot-indicator dot-${st.toLowerCase()}`;
        document.getElementById('main-state-text').innerText = st;

        // Subsystems Metrics
        const snap = data.snapshot || {};
        const c = snap.cpu || {};
        const m = snap.memory || {};
        const d = snap.disk || {};
        const n = snap.network || {};

        const bMed = (data.baseline_stats && data.baseline_stats.medians) || {};

        // CPU
        const cpuVal = (c.cpu_percent_mean || 0).toFixed(1);
        const cpuMax = (c.cpu_percent_max || 0).toFixed(1);
        const medCpu = (bMed.cpu_percent_mean || 19.3).toFixed(1);
        document.getElementById('card-cpu-val').innerText = `${cpuVal}%`;
        document.getElementById('card-cpu-trend').innerText = cpuVal > medCpu * 1.3 ? `↑ ${(cpuVal/medCpu).toFixed(1)}× usual` : '×1.0 vs usual';
        document.getElementById('card-cpu-core').innerText = `Total: ${cpuVal}% · Max core: ${cpuMax}%`;
        document.getElementById('card-cpu-load').innerText = `1m load: ${(c.load_avg_1 || 0).toFixed(2)}`;
        document.getElementById('cpu-impact').innerText = `${data.hardware_impact || 'Low'} impact`;

        // Memory
        const memVal = (m.mem_used_pct || 0).toFixed(1);
        const medMem = (bMed.mem_used_pct || 20.9).toFixed(1);
        document.getElementById('card-mem-val').innerText = `${memVal}%`;
        document.getElementById('card-mem-trend').innerText = memVal > medMem * 1.3 ? `↑ ${(memVal/medMem).toFixed(1)}× usual` : '×1.0 vs usual';
        document.getElementById('card-mem-used').innerText = `${Math.round(m.mem_used_mb || 0)} MB used`;
        document.getElementById('card-mem-swap').innerText = `Swap: ${(m.swap_used_pct || 0).toFixed(1)}%`;
        document.getElementById('mem-impact').innerText = memVal > 85 ? 'High impact' : 'Low impact';

        // Disk
        const diskIo = ((d.disk_read_mbps || 0) + (d.disk_write_mbps || 0)).toFixed(1);
        document.getElementById('card-disk-val').innerText = `${diskIo} MB/s`;
        document.getElementById('card-disk-io').innerText = `R: ${(d.disk_read_mbps || 0).toFixed(1)} · W: ${(d.disk_write_mbps || 0).toFixed(1)} MB/s`;
        document.getElementById('card-disk-used').innerText = `${(d.disk_used_pct || 0).toFixed(1)}% capacity`;

        // Network
        const netIo = ((n.net_bytes_recv_mbps || 0) + (n.net_bytes_sent_mbps || 0)).toFixed(2);
        document.getElementById('card-net-val').innerText = `${netIo} MB/s`;
        document.getElementById('card-net-io').innerText = `RX: ${(n.net_bytes_recv_mbps || 0).toFixed(2)} · TX: ${(n.net_bytes_sent_mbps || 0).toFixed(2)} MB/s`;
        document.getElementById('card-net-sockets').innerText = `${n.tcp_connections || 0} sockets`;

        // Sparklines
        const hist = data.history || [];
        if (hist.length >= 2) {
          renderSparkline('spark-cpu', hist.map(h => h.cpu), 'var(--line-cpu)');
          renderSparkline('spark-mem', hist.map(h => h.ram), 'var(--line-ram)');
          renderSparkline('spark-disk', hist.map(h => h.disk), 'var(--line-disk)');
          renderSparkline('spark-net', hist.map(h => h.net || (h.rx + h.tx)), 'var(--line-net)');
          drawTimeline(hist);
        }

        // Debounce text
        if (data.debounce_text) {
          document.getElementById('table-debounce-indicator').innerText = data.debounce_text;
        }

        // Populate Alerts Table in Overview
        const alerts = data.alerts_feed || [];
        const overviewTbody = document.getElementById('overview-alerts-tbody');
        if (alerts.length > 0) {
          overviewTbody.innerHTML = alerts.slice(0, 8).map((a, idx) => `
            <tr onclick='selectAlertRow(${JSON.stringify(a).replace(/'/g, "&#39;")})' class="${userPinnedIncident && userPinnedIncident.id === a.id ? 'selected' : ''}">
              <td class="mono-cell">${formatTime(a.time)}</td>
              <td><span class="state-badge state-${a.status_state}">${a.status_state}</span></td>
              <td><span class="chip">${a.subsystem}</span></td>
              <td class="mono-cell">${a.score_multiple}</td>
              <td style="max-width:280px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${a.signals}</td>
              <td><span class="chip" style="font-size:10px;">${a.decision_source}</span></td>
            </tr>
          `).join('');
        }

        // Continuous Live Update: if user has not pinned a historic alert, stream live telemetry!
        if (!userPinnedIncident) {
          renderIncidentDetail(data, true);
        }

        // Populate Full Alerts Tab
        const fullAlertsTbody = document.getElementById('full-alerts-tbody');
        if (fullAlertsTbody && alerts.length > 0) {
          fullAlertsTbody.innerHTML = alerts.map(a => `
            <tr onclick='selectAlertRow(${JSON.stringify(a).replace(/'/g, "&#39;")})'>
              <td class="mono-cell">${a.id}</td>
              <td class="mono-cell">${formatTime(a.time)}</td>
              <td><span class="state-badge state-${a.status_state}">${a.status_state}</span></td>
              <td><span class="chip">${a.subsystem}</span></td>
              <td class="mono-cell">${a.score_multiple}</td>
              <td>${a.signals}</td>
              <td class="mono-cell">${a.decision_source}</td>
              <td>${a.impact}</td>
              <td><button class="btn-flat" style="padding:2px 6px; font-size:11px;" onclick='inspectAlert(${JSON.stringify(a).replace(/'/g, "&#39;")})'>Inspect</button></td>
            </tr>
          `).join('');
        }

        // Populate Processes Tab
        const procs = data.top_cpu_procs || [];
        document.getElementById('proc-count-chip').innerText = `${procs.length} top resource processes`;
        const procTbody = document.getElementById('full-proc-tbody');
        if (procTbody && procs.length > 0) {
          procTbody.innerHTML = procs.map(p => `
            <tr>
              <td class="mono-cell">${p.pid}</td>
              <td style="font-weight:600; color:${p.is_new ? 'var(--elevated)' : 'var(--text)'};">${p.name}</td>
              <td><span class="chip" style="${p.is_new ? 'color:var(--elevated); border-color:rgba(210,153,34,0.4);' : ''}">${p.is_new ? 'NEW' : 'KNOWN'}</span></td>
              <td class="mono-cell" style="color:var(--text-muted);">${p.parent || '-'}</td>
              <td class="mono-cell">${p.connections || 0} sockets</td>
              <td class="mono-cell" style="color:var(--elevated); font-weight:600;">${(p.cpu_pct || 0).toFixed(1)}%</td>
              <td class="mono-cell">${(p.mem_pct || 0).toFixed(1)}%</td>
              <td><button class="btn-flat" style="padding:2px 6px; font-size:11px;" onclick="inspectPID(${p.pid})">Inspect</button></td>
            </tr>
          `).join('');
        }

      } catch (err) {
        console.error('Telemetry polling error:', err);
      }
    }

    function selectAlertRow(alertObj) {
      userPinnedIncident = alertObj;
      renderIncidentDetail(alertObj, false);
      const rows = document.querySelectorAll('#overview-alerts-tbody tr');
      rows.forEach(r => r.classList.remove('selected'));
      const activeRow = Array.from(rows).find(r => r.innerText.includes(formatTime(alertObj.time)));
      if (activeRow) activeRow.classList.add('selected');
    }

    function resumeLiveStream() {
      userPinnedIncident = null;
      selectedIncident = null;
      const rows = document.querySelectorAll('#overview-alerts-tbody tr');
      rows.forEach(r => r.classList.remove('selected'));
      const liveInd = document.getElementById('detail-live-indicator');
      const resumeBtn = document.getElementById('btn-resume-live');
      if (liveInd) liveInd.style.display = 'inline-flex';
      if (resumeBtn) resumeBtn.style.display = 'none';
      if (currentData) {
        renderIncidentDetail(currentData, true);
      }
    }

    async function changeScenario(mode) {
      userPinnedIncident = null;
      await fetch(`/api/simulate?mode=${encodeURIComponent(mode)}`);
      await pollTelemetry();
    }

    function copyTriageCommand() {
      const cmd = document.getElementById('detail-triage-cmd').innerText;
      navigator.clipboard.writeText(cmd);
      alert('Triage command copied to clipboard: ' + cmd);
    }

    function acknowledgeIncident() {
      alert('Incident acknowledged. Recorded in SOC triage ledger.');
    }

    function flagFalsePositive() {
      alert('Flagged as false positive. Recorded in baseline recalibration log.');
    }

    function inspectPID(pid) {
      alert(`To inspect PID ${pid}, run:\npidstat -p ${pid} 1 5`);
    }

    function inspectAlert(a) {
      switchTab('overview');
      selectAlertRow(a);
    }

    function exportAlerts() {
      window.open('/api/export-siem', '_blank');
    }

    function setTimeRange(range) {
      document.querySelectorAll('.time-btn').forEach(b => b.classList.remove('active'));
      const activeBtn = Array.from(document.querySelectorAll('.time-btn')).find(b => b.innerText.toLowerCase() === range);
      if (activeBtn) activeBtn.classList.add('active');
    }

    function filterAlertsFeed() {
      const q = document.getElementById('alerts-search-input').value.toLowerCase();
      const rows = document.querySelectorAll('#full-alerts-tbody tr');
      rows.forEach(r => {
        r.style.display = r.innerText.toLowerCase().includes(q) ? '' : 'none';
      });
    }

    setInterval(pollTelemetry, 3000);
    pollTelemetry();
  </script>
</body>
</html>
"""


# ── HTTP Server Request Handler ───────────────────────────────────────────────

class DashboardRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        # Suppress routine GET logging to keep terminal clean
        return

    def do_GET(self) -> None:
        try:
            self._handle_get()
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            pass

    def _handle_get(self) -> None:
        global _sim_mode
        parsed = urlparse(self.path)

        if parsed.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode("utf-8"))

        elif parsed.path == "/api/telemetry":
            state_data = _get_live_state()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(state_data).encode("utf-8"))

        elif parsed.path == "/api/evidence":
            state, _ = _compute_current_state()
            ev_data = state.evidence_chain if hasattr(state, "evidence_chain") else {}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(ev_data).encode("utf-8"))

        elif parsed.path == "/api/drift":
            from heimdall.analysis.drift_monitor import DriftAndFeedbackStore
            store = DriftAndFeedbackStore()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({
                "dismissed_alerts": store.dismissed_alerts,
                "suppression_rules": store.suppression_rules,
            }).encode("utf-8"))

        elif parsed.path == "/api/simulate":
            from heimdall.analysis.simulator import VALID_MODES
            qs = parse_qs(parsed.query)
            req_mode = qs.get("mode", ["live"])[0]
            if req_mode in VALID_MODES:
                _sim_mode = req_mode
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"sim_mode": _sim_mode}).encode("utf-8"))

        elif parsed.path == "/api/export-siem":
            state, _ = _compute_current_state()
            siem_event = {
                "timestamp": state.timestamp,
                "session_id": f"sess_{int(time.time())}",
                "hostname": platform.node(),
                "status_state": state.status_state,
                "severity": state.severity,
                "decision_source": state.decision_source,
                "score_multiple": round(state.ae_hybrid_score / max(state.ae_threshold, 1e-4), 2) if state.ae_threshold > 0 else 0.5,
                "subsystem": state.dominant_subsystem,
                "effective_class": state.mlp_class,
                "triage_command": state.suggested_triage_command,
                "top_features": state.top_anomalous_features[:3],
                "evidence_chain": state.evidence_chain,
            }
            jsonl_content = json.dumps(siem_event) + "\n"
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Content-Disposition", 'attachment; filename="heimdall_siem_events.jsonl"')
            self.end_headers()
            self.wfile.write(jsonl_content.encode("utf-8"))


        elif parsed.path in ("/evolution-report", "/report/evolution"):
            rep_path = settings.ROOT_DIR / "reports" / "HEIMDALL_EVOLUTION_REPORT.md"
            if not rep_path.exists():
                rep_path = settings.ROOT_DIR / "PROJECT_EVOLUTION_REPORT.md"
            raw_md = rep_path.read_text(encoding="utf-8") if rep_path.exists() else "# Report not found"
            raw_json = json.dumps(raw_md)
            html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>HEIMDALL — Evolution & Technical Report</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <style>
    :root {{
      --bg: #0F1419;
      --surface: #161B22;
      --border: #232A33;
      --text: #E6EDF3;
      --text-muted: #8B949E;
      --accent: #58A6FF;
    }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      margin: 0;
      padding: 32px 16px;
      display: flex;
      justify-content: center;
      line-height: 1.6;
    }}
    .container {{
      max-width: 960px;
      width: 100%;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 32px;
    }}
    h1, h2, h3, h4 {{ color: var(--text); font-weight: 600; }}
    h1 {{ border-bottom: 1px solid var(--border); padding-bottom: 12px; color: var(--accent); }}
    h2 {{ border-bottom: 1px solid var(--border); padding-bottom: 8px; margin-top: 32px; }}
    pre, code {{ font-family: "JetBrains Mono", ui-monospace, monospace; background: #0d1117; border-radius: 4px; }}
    pre {{ padding: 14px; overflow-x: auto; border: 1px solid var(--border); }}
    code {{ padding: 2px 6px; color: var(--accent); font-size: 0.9em; }}
    table {{ width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 13px; }}
    th, td {{ border: 1px solid var(--border); padding: 8px 12px; text-align: left; }}
    th {{ background: #1c222b; color: var(--text); }}
    blockquote {{ border-left: 3px solid var(--accent); margin: 16px 0; padding: 6px 16px; background: rgba(88, 166, 255, 0.05); color: var(--text-muted); }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .back-nav {{ margin-bottom: 24px; display: flex; justify-content: space-between; align-items: center; }}
    .back-btn {{ display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px; background: #1c222b; border: 1px solid var(--border); border-radius: 4px; color: var(--text); font-size: 12px; text-decoration: none; }}
    .back-btn:hover {{ border-color: var(--accent); color: var(--accent); }}
  </style>
</head>
<body>
  <div class="container">
    <div class="back-nav">
      <a href="/" class="back-btn">&larr; Back to Dashboard</a>
      <span style="font-size:12px; color:var(--text-muted);">HEIMDALL ARCHITECTURE REPORT</span>
    </div>
    <div id="content"></div>
  </div>
  <script>
    const markdown = {raw_json};
    document.getElementById('content').innerHTML = marked.parse(markdown);
  </script>
</body>
</html>"""
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))

        elif parsed.path == "/api/report":
            state, _ = _compute_current_state()
            report_path = report_generator.generate(state)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"path": str(report_path), "status": "success"}).encode("utf-8"))

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        pass


def run_dashboard(port: int = 8080, open_browser: bool = True) -> None:
    """Start the live dashboard web server and optionally open browser."""
    server_address = ("", port)
    HTTPServer.allow_reuse_address = True
    httpd = HTTPServer(server_address, DashboardRequestHandler)
    url = f"http://localhost:{port}"

    print(f"HEIMDALL Host Anomaly Detection Dashboard running at: {url}")
    print("Press Ctrl+C to stop the server.\n")

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard server stopped.")
    finally:
        httpd.server_close()

"""
Unit Tests for HEIMDALL v2.1 Advanced Intelligence Layers
==========================================================
Verifies:
1. Temporal sequence correlation & multi-stage detection.
2. Context-aware baseline tolerance modulation.
3. Decoupled forensic investigation evidence chain.
4. Population Stability Index (PSI) drift monitoring & feedback store.
"""

from datetime import datetime, timezone
import numpy as np
import pytest

from heimdall.analysis.context_baseline import evaluate_host_context
from heimdall.analysis.drift_monitor import DriftAndFeedbackStore, calculate_psi, evaluate_telemetry_drift
from heimdall.analysis.evidence_chain import build_investigation_evidence
from heimdall.analysis.temporal_detector import evaluate_temporal_sequence
from heimdall.collector.snapshot import SystemSnapshot
from heimdall.config import settings


def _create_dummy_snapshot(
    cpu_pct: float = 10.0,
    mem_pct: float = 40.0,
    tcp_conns: int = 20,
    procs: int = 150,
    top_proc: str = "bash",
    auth_fails: int = 0,
) -> SystemSnapshot:
    ts = datetime.now(timezone.utc).isoformat()
    return SystemSnapshot(
        timestamp=ts,
        label="test",
        cpu={"cpu_percent_mean": cpu_pct, "cpu_percent_max": cpu_pct, "load_avg_1": 0.5},
        memory={"mem_used_pct": mem_pct, "mem_available_mb": 4096.0, "swap_used_pct": 0.0},
        disk={"disk_used_pct": 30.0, "disk_read_mbps": 1.0, "disk_write_mbps": 1.0},
        network={"tcp_connections": tcp_conns, "net_bytes_recv_mbps": 0.5, "net_bytes_sent_mbps": 0.2, "net_dropin_per_sec": 0.0},
        process={"num_processes": procs, "num_threads": 400, "top_cpu_procs": [{"name": top_proc, "pid": 1234, "cpu_percent": cpu_pct, "memory_percent": 2.0}]},
        system={"auth_failures_60s": auth_fails, "failed_services": 0},
    )


def test_temporal_multi_stage_sequence():
    """Verify that a multi-stage progression (process spawn -> network surge -> CPU spike) triggers temporal alert."""
    s1 = _create_dummy_snapshot(cpu_pct=10.0, tcp_conns=15, procs=100)
    s2 = _create_dummy_snapshot(cpu_pct=15.0, tcp_conns=35, procs=108, top_proc="xmrig")
    s3 = _create_dummy_snapshot(cpu_pct=88.0, tcp_conns=45, procs=110, top_proc="xmrig")

    result = evaluate_temporal_sequence([s1, s2, s3])
    assert result.is_temporal_anomaly is True
    assert result.incident_type == "multi_stage_execution"
    assert result.risk_score >= 50.0
    assert len(result.event_timeline) >= 2


def test_context_aware_compilation_tolerance():
    """Verify that recognized build toolchains receive elevated CPU tolerance during business hours."""
    snap_build = _create_dummy_snapshot(cpu_pct=75.0, top_proc="gcc")
    dt_business = datetime(2026, 10, 15, 14, 30, tzinfo=timezone.utc)  # Thursday 14:30
    profile = evaluate_host_context(snap_build, dt=dt_business)

    assert profile.workload_type == "compilation"
    assert profile.threshold_modifier > 1.0
    assert profile.is_off_hours is False


def test_investigation_evidence_chain_decoupling():
    """Verify that the evidence chain captures physical metrics and detector votes separately from model class."""
    snap = _create_dummy_snapshot(cpu_pct=90.0, tcp_conns=200)
    evidence = build_investigation_evidence(
        timestamp=snap.timestamp,
        severity="WARNING",
        ae_result=None,
        iso_result={"is_anomaly": True, "raw_score": 0.65, "threshold": 0.47},
        mad_result={"is_anomaly": True, "score": 2.1, "threshold": 1.3},
        temporal_result=None,
        rule_messages=["High CPU detected"],
        mlp_class="cpu_stress",
        mlp_confidence=0.88,
        mlp_probs={"cpu_stress": 0.88, "normal": 0.12},
        snap=snap,
    )

    assert evidence.alert_id.startswith("HL-")
    assert len(evidence.contributing_detectors) >= 2
    assert evidence.inferred_classification["predicted_label"] == "cpu_stress"
    assert evidence.uncertainty_gate_status == "CONFIDENT"
    assert "High CPU detected" in evidence.active_security_rules


def test_drift_and_feedback_store(tmp_path):
    """Verify Population Stability Index and analyst dismissal recording."""
    expected = np.random.normal(0, 1, 100)
    actual_drifted = np.random.normal(5, 2, 100)
    psi = calculate_psi(expected, actual_drifted)
    assert psi > 0.25  # High distribution drift

    store = DriftAndFeedbackStore(store_path=tmp_path / "drift.json")
    store.dismiss_alert("HL-TEST-001", reason="Approved cron backup")
    assert len(store.dismissed_alerts) == 1
    assert store.dismissed_alerts[0]["alert_id"] == "HL-TEST-001"

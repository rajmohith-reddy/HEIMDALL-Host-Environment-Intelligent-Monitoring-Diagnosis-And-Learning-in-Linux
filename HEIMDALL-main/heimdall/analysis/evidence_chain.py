"""
Investigation Evidence Chain
============================
Constructs a forensic, audit-grade evidence dossier for every security or
operational alert.

Crucial Architectural Principle:
- Physical evidence (metrics, detector votes, sequence steps, baseline deviations)
  is strictly DECOUPLED from the supervised classifier's inferred attack label.
- The analyst sees what actually happened physically before seeing what the model thinks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class MetricEvidence:
    metric: str
    current_value: float
    baseline_value: float
    deviation_multiple: float
    sigma_deviation: float
    unit: str
    subsystem: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "current_value": round(self.current_value, 2),
            "baseline_value": round(self.baseline_value, 2),
            "deviation_multiple": round(self.deviation_multiple, 2),
            "sigma_deviation": round(self.sigma_deviation, 1),
            "unit": self.unit,
            "subsystem": self.subsystem,
        }


@dataclass
class DetectorVote:
    detector_name: str
    voted_anomaly: bool
    score: float
    threshold: float
    role: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector_name": self.detector_name,
            "voted_anomaly": self.voted_anomaly,
            "score": round(self.score, 3),
            "threshold": round(self.threshold, 3),
            "role": self.role,
        }


@dataclass
class InvestigationEvidence:
    alert_id: str
    timestamp: str
    severity: str
    plain_english_summary: str
    
    # Decoupled Physical Evidence
    triggering_metrics: list[MetricEvidence] = field(default_factory=list)
    contributing_detectors: list[DetectorVote] = field(default_factory=list)
    temporal_events: list[dict[str, str]] = field(default_factory=list)
    process_tree_context: list[dict[str, Any]] = field(default_factory=list)
    network_context: dict[str, Any] = field(default_factory=dict)
    active_security_rules: list[str] = field(default_factory=list)

    # Inferred Model Classification (Kept separate from raw evidence)
    inferred_classification: dict[str, Any] = field(default_factory=dict)
    uncertainty_gate_status: str = "CONFIDENT"  # "CONFIDENT" vs "UNCLASSIFIED_OOD"

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp,
            "severity": self.severity,
            "plain_english_summary": self.plain_english_summary,
            "triggering_metrics": [m.to_dict() for m in self.triggering_metrics],
            "contributing_detectors": [d.to_dict() for d in self.contributing_detectors],
            "temporal_events": self.temporal_events,
            "process_tree_context": self.process_tree_context,
            "network_context": self.network_context,
            "active_security_rules": self.active_security_rules,
            "inferred_classification": self.inferred_classification,
            "uncertainty_gate_status": self.uncertainty_gate_status,
        }


def build_investigation_evidence(
    timestamp: str,
    severity: str,
    ae_result: Any,
    iso_result: dict[str, Any] | None,
    mad_result: dict[str, Any] | None,
    temporal_result: Any | None,
    rule_messages: list[str],
    mlp_class: str,
    mlp_confidence: float,
    mlp_probs: dict[str, float],
    snap: Any,
) -> InvestigationEvidence:
    """
    Assembles the decoupled evidence dossier.
    """
    alert_id = f"HL-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    # 1. Detector Votes
    detectors: list[DetectorVote] = []
    if ae_result:
        detectors.append(DetectorVote(
            detector_name="Denoising Autoencoder",
            voted_anomaly=bool(ae_result.is_anomaly),
            score=float(ae_result.hybrid_score),
            threshold=float(ae_result.threshold),
            role="Reconstruction Error & XAI Attribution Engine",
        ))
    if iso_result:
        detectors.append(DetectorVote(
            detector_name="Isolation Forest",
            voted_anomaly=bool(iso_result.get("is_anomaly", False)),
            score=float(iso_result.get("raw_score", 0.0)),
            threshold=float(iso_result.get("threshold", 0.0)),
            role="Fast Manifold Screening",
        ))
    if mad_result:
        detectors.append(DetectorVote(
            detector_name="Robust Directional MAD",
            voted_anomaly=bool(mad_result.get("is_anomaly", False)),
            score=float(mad_result.get("score", 0.0)),
            threshold=float(mad_result.get("threshold", 0.0)),
            role="Physical Hardware Deviation & Drop Gate",
        ))
    if temporal_result and temporal_result.is_temporal_anomaly:
        detectors.append(DetectorVote(
            detector_name="Temporal Sequence Correlator",
            voted_anomaly=True,
            score=float(temporal_result.risk_score),
            threshold=45.0,
            role="Multi-Stage Kill-Chain & Creep Detection",
        ))
    if rule_messages:
        detectors.append(DetectorVote(
            detector_name="Security Rule Engine",
            voted_anomaly=True,
            score=float(len(rule_messages)),
            threshold=1.0,
            role="Deterministic Safety Thresholds",
        ))

    # 2. Triggering Physical Metrics from XAI & MAD
    metrics: list[MetricEvidence] = []
    if ae_result and hasattr(ae_result, "top_anomalous_features"):
        for f in ae_result.top_anomalous_features[:5]:
            feat_name = f.get("feature", "")
            z_err = f.get("z_error", 0.0)
            if z_err >= 0.5:
                metrics.append(MetricEvidence(
                    metric=feat_name,
                    current_value=float(f.get("value", 0.0)),
                    baseline_value=float(f.get("expected", 0.0)),
                    deviation_multiple=float(f.get("ratio", 1.0)),
                    sigma_deviation=float(z_err),
                    unit=f.get("unit", "units"),
                    subsystem=f.get("subsystem", "system"),
                ))

    # 3. Process & Network Context
    procs = []
    if hasattr(snap, "process"):
        for p in snap.process.get("top_cpu_procs", [])[:3]:
            procs.append({
                "pid": p.get("pid"),
                "name": p.get("name"),
                "cpu_percent": p.get("cpu_percent"),
                "memory_percent": p.get("memory_percent"),
            })

    net_ctx = {}
    if hasattr(snap, "network"):
        net_ctx = {
            "tcp_connections": snap.network.get("tcp_connections", 0),
            "rx_mbps": snap.network.get("net_bytes_recv_mbps", 0.0),
            "tx_mbps": snap.network.get("net_bytes_sent_mbps", 0.0),
            "packet_drops": snap.network.get("net_dropin_per_sec", 0.0),
        }

    # 4. Temporal Steps
    timeline = temporal_result.event_timeline if temporal_result else []

    # 5. Plain English Summary
    if rule_messages:
        summary = f"Deterministic security violation: {'; '.join(rule_messages)}."
    elif metrics:
        top_m = metrics[0]
        summary = (
            f"Physical anomaly driven primarily by {top_m.metric} ({top_m.current_value:.1f} {top_m.unit}, "
            f"{top_m.sigma_deviation:.1f}σ deviation from expected {top_m.baseline_value:.1f} {top_m.unit})."
        )
    elif temporal_result and temporal_result.is_temporal_anomaly:
        summary = f"Temporal correlation alert: {temporal_result.explanation}."
    else:
        summary = "Operational state within nominal baseline."

    # 6. Decoupled Inferred Classification
    uncertainty_status = "CONFIDENT" if mlp_confidence >= 0.80 else "UNCLASSIFIED_OOD"
    inferred = {
        "model": "Calibrated Classifier",
        "predicted_label": mlp_class,
        "confidence": round(mlp_confidence, 4),
        "class_probabilities": {k: round(v, 4) for k, v in mlp_probs.items()},
    }

    return InvestigationEvidence(
        alert_id=alert_id,
        timestamp=timestamp,
        severity=severity,
        plain_english_summary=summary,
        triggering_metrics=metrics,
        contributing_detectors=detectors,
        temporal_events=timeline,
        process_tree_context=procs,
        network_context=net_ctx,
        active_security_rules=rule_messages,
        inferred_classification=inferred,
        uncertainty_gate_status=uncertainty_status,
    )

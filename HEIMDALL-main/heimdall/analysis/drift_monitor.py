"""
Telemetry Drift Monitor & Analyst Feedback Engine
==================================================
Tracks physical telemetry distribution drift over operational lifecycles and
manages analyst alert dismissal / feedback without blindly auto-retraining on
potentially malicious activity.

Key Responsibilities:
1. Distribution Drift Monitoring:
   - Evaluates telemetry distributions across rolling windows (e.g. trailing 100 samples)
     against the baseline manifold using Population Stability Index (PSI).
   - If PSI > 0.25 on core features (e.g., after OS upgrade or legitimate infra change),
     flags that the baseline is drifting.
2. Safe Recalibration Safety Guard:
   - Does NOT automatically retrain on live anomalous telemetry to prevent
     adversarial baseline poisoning.
3. Analyst Feedback Store:
   - Logs alert dismissals, suppression rules for known scheduled cron jobs,
     and false-positive feedback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from heimdall.config import settings

DRIFT_STORE_PATH = settings.REPORTS_DIR / "drift_and_feedback.json"


@dataclass
class DriftReport:
    timestamp: str
    drift_detected: bool
    drift_score_psi: float
    drifted_features: list[dict[str, Any]]
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "drift_detected": self.drift_detected,
            "drift_score_psi": round(self.drift_score_psi, 4),
            "drifted_features": self.drifted_features,
            "recommendation": self.recommendation,
        }


def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
    """
    Computes Population Stability Index (PSI) between baseline and live window.
    PSI < 0.10: No significant drift
    PSI 0.10 - 0.25: Moderate drift
    PSI > 0.25: Significant operational drift
    """
    if len(expected) < 10 or len(actual) < 10:
        return 0.0

    eps = 1e-4
    # Compute quantiles from expected distribution
    percentiles = np.linspace(0, 100, num_buckets + 1)
    bucket_bounds = np.percentile(expected, percentiles)
    bucket_bounds[0] = -np.inf
    bucket_bounds[-1] = np.inf

    # Bin counts
    exp_counts = np.histogram(expected, bins=bucket_bounds)[0]
    act_counts = np.histogram(actual, bins=bucket_bounds)[0]

    # Convert to fractions
    exp_pct = exp_counts / max(len(expected), 1) + eps
    act_pct = act_counts / max(len(actual), 1) + eps

    psi_val = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(np.nan_to_num(psi_val, nan=0.0, posinf=1.0, neginf=0.0))


class DriftAndFeedbackStore:
    """Persisted store for drift events and analyst triage decisions."""

    def __init__(self, store_path: Path | str | None = None) -> None:
        self.store_path = Path(store_path) if store_path else DRIFT_STORE_PATH
        self.suppression_rules: list[dict[str, Any]] = []
        self.dismissed_alerts: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if self.store_path.exists():
            try:
                data = json.loads(self.store_path.read_text(encoding="utf-8"))
                self.suppression_rules = data.get("suppression_rules", [])
                self.dismissed_alerts = data.get("dismissed_alerts", [])
            except Exception:
                pass

    def _save(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "suppression_rules": self.suppression_rules,
            "dismissed_alerts": self.dismissed_alerts,
        }
        self.store_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def dismiss_alert(self, alert_id: str, reason: str, analyst: str = "soc_analyst") -> None:
        """Record analyst dismissal for false positive tracking."""
        self.dismissed_alerts.append({
            "alert_id": alert_id,
            "dismissed_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "analyst": analyst,
        })
        self._save()

    def add_suppression_rule(self, metric: str, condition: str, comment: str) -> None:
        """Add known maintenance or cron suppression rule."""
        self.suppression_rules.append({
            "metric": metric,
            "condition": condition,
            "comment": comment,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        self._save()

    def is_suppressed(self, metric: str) -> bool:
        return any(rule["metric"] == metric for rule in self.suppression_rules)


def evaluate_telemetry_drift(
    baseline_samples: np.ndarray,
    recent_window_samples: np.ndarray,
) -> DriftReport:
    """
    Evaluates whether the live host telemetry distribution has drifted from baseline.
    """
    ts = datetime.now(timezone.utc).isoformat()
    if len(baseline_samples) < 20 or len(recent_window_samples) < 20:
        return DriftReport(
            timestamp=ts,
            drift_detected=False,
            drift_score_psi=0.0,
            drifted_features=[],
            recommendation="Insufficient rolling history for distribution drift calculation.",
        )

    drifted = []
    psi_scores = []

    for idx, feat_name in enumerate(settings.FEATURE_NAMES):
        exp_col = baseline_samples[:, idx]
        act_col = recent_window_samples[:, idx]
        psi = calculate_psi(exp_col, act_col)
        psi_scores.append(psi)

        if psi > 0.25:
            drifted.append({
                "feature": feat_name,
                "psi": round(psi, 3),
                "severity": "HIGH_DRIFT",
            })

    mean_psi = float(np.mean(psi_scores)) if psi_scores else 0.0
    drift_detected = mean_psi > 0.20 or len(drifted) >= 3

    if drift_detected:
        rec = (
            f"Host distribution has drifted ({len(drifted)} features shifted, mean PSI={mean_psi:.3f}). "
            "Operator review recommended before recalibrating baseline (do not auto-train on active incidents)."
        )
    else:
        rec = f"Host telemetry distribution is stable within reference manifold (mean PSI={mean_psi:.3f})."

    return DriftReport(
        timestamp=ts,
        drift_detected=drift_detected,
        drift_score_psi=mean_psi,
        drifted_features=drifted,
        recommendation=rec,
    )

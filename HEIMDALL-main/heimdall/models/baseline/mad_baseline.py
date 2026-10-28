"""
Robust Non-Parametric MAD Baseline & Metric Drop Detector
=========================================================
Implements a non-parametric statistical baseline using Median and Median Absolute
Deviation (MAD) scaled by the standard consistency factor 1.4826:
    d_i = (x_i - median_i) / max(1.4826 * MAD_i, epsilon_i)

Also detects directional drops in normally active telemetry (e.g. network drops,
process collapses, thread vanishing) and ranks hardware impact independently of
alert suppression.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from heimdall.config import settings

BASELINE_PATH = settings.MODELS_DIR / "mad_baseline.json"

# Features where a sudden collapse indicates failure of core OS processes/threads
DROP_SENSITIVE_FEATURES = [
    "num_processes",
    "num_threads",
    "open_files",
]


def fit_baseline(
    raw_normal_dir: Path | str | None = None,
    save_path: Path | str | None = None,
) -> dict[str, Any]:
    """
    Fits Median and MAD baselines on unscaled raw normal host snapshots.
    Computes empirical 99.5th percentile score threshold on held-out normal data.
    """
    import glob
    from heimdall.collector.snapshot import SystemSnapshot
    from heimdall.preprocessor.feature_engineer import extract_features

    raw_dir = Path(raw_normal_dir) if raw_normal_dir else (settings.DATA_RAW_DIR / "normal")
    save_path = Path(save_path) if save_path else BASELINE_PATH

    print("── Robust MAD Baseline Calibrator ──────────────────────────")
    raw_files = sorted(glob.glob(str(raw_dir / "*.json")))
    if not raw_files:
        raise FileNotFoundError(f"No raw normal snapshots found in {raw_dir}")

    # Parse raw snapshots and extract true unscaled feature vectors
    snaps = []
    for f in raw_files:
        try:
            snaps.append(SystemSnapshot.from_dict(json.loads(Path(f).read_text(encoding="utf-8"))))
        except Exception:
            pass

    X_raw = np.array([
        extract_features(snaps[i], snaps[i - 1] if i > 0 else None)
        for i in range(len(snaps))
    ], dtype=np.float64)

    # 80/20 train/val split of raw snapshots
    split_idx = int(len(X_raw) * 0.8)
    X_train = X_raw[:split_idx]
    X_val = X_raw[split_idx:]

    medians = np.median(X_train, axis=0)
    abs_deviations = np.abs(X_train - medians)
    mads = np.median(abs_deviations, axis=0)

    # 1.4826 consistency scale factor for asymptotically normal estimator
    scaled_mads = 1.4826 * mads
    from heimdall.preprocessor.scaler import FEATURE_MIN_SCALE
    min_floors = np.array([FEATURE_MIN_SCALE.get(f, 0.5) for f in settings.FEATURE_NAMES], dtype=np.float64)
    denominators = np.maximum(scaled_mads, min_floors)

    # Compute validation scores: aggregate top-5 directional resource strains
    devs_val = (X_val - medians) / denominators
    strain_val = np.maximum(0.0, devs_val)
    for feat in DROP_SENSITIVE_FEATURES:
        if feat in settings.FEATURE_NAMES:
            idx = settings.FEATURE_NAMES.index(feat)
            strain_val[:, idx] = np.maximum(0.0, -devs_val[:, idx])

    top5_val = np.sort(strain_val, axis=1)[:, -5:]
    val_scores = np.mean(top5_val, axis=1)

    threshold_p995 = float(np.percentile(val_scores, 99.5))
    median_val_score = float(np.median(val_scores))

    meta = {
        "medians": {f: float(m) for f, m in zip(settings.FEATURE_NAMES, medians)},
        "mads": {f: float(mad) for f, mad in zip(settings.FEATURE_NAMES, mads)},
        "denominators": {f: float(d) for f, d in zip(settings.FEATURE_NAMES, denominators)},
        "threshold_p995": threshold_p995,
        "median_val_score": median_val_score,
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
    }

    with open(save_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"   Threshold (99.5th %-tile) : {threshold_p995:.4f}")
    print(f"   Median Normal Score       : {median_val_score:.4f}")
    print(f"   Saved MAD Baseline        : {save_path}")
    print("✓ MAD Baseline calibration complete!\n")

    return meta


class MADDetector:
    """Evaluates real-time feature vectors against the host's robust MAD baseline."""

    def __init__(self, baseline_path: Path | str | None = None) -> None:
        self.path = Path(baseline_path) if baseline_path else BASELINE_PATH
        if not self.path.exists():
            fit_baseline(save_path=self.path)

        with open(self.path) as f:
            self.meta = json.load(f)

        self.medians = np.array([self.meta["medians"][f] for f in settings.FEATURE_NAMES], dtype=np.float64)
        self.denominators = np.array([self.meta["denominators"][f] for f in settings.FEATURE_NAMES], dtype=np.float64)
        self.threshold = float(self.meta["threshold_p995"])

    def evaluate(self, raw_features: list[float] | np.ndarray) -> dict[str, Any]:
        x = np.asarray(raw_features, dtype=np.float64)
        deviations = (x - self.medians) / self.denominators
        strain = np.maximum(0.0, deviations)
        for feat in DROP_SENSITIVE_FEATURES:
            if feat in settings.FEATURE_NAMES:
                idx = settings.FEATURE_NAMES.index(feat)
                strain[idx] = max(0.0, -deviations[idx])

        # Check for directional drops in normally active signals
        drops: list[dict[str, Any]] = []
        for feat_name in DROP_SENSITIVE_FEATURES:
            if feat_name in settings.FEATURE_NAMES:
                idx = settings.FEATURE_NAMES.index(feat_name)
                med = self.medians[idx]
                val = x[idx]
                # If core OS handles collapse to less than 10% of typical baseline
                if med >= 50.0 and val < 0.10 * med:
                    drop_pct = (1.0 - (val / med)) * 100.0
                    drops.append({
                        "feature": feat_name,
                        "current": round(float(val), 2),
                        "baseline_median": round(float(med), 2),
                        "drop_pct": round(drop_pct, 1),
                    })

        # Top 5 deviating features by resource strain
        top_indices = np.argsort(strain)[::-1][:5]
        top_deviations = [
            {
                "feature": settings.FEATURE_NAMES[i],
                "subsystem": settings.FEATURE_TO_SUBSYSTEM.get(settings.FEATURE_NAMES[i], "unknown"),
                "current": round(float(x[i]), 2),
                "baseline_median": round(float(self.medians[i]), 2),
                "z_mad": round(float(strain[i]), 2),
                "direction": "HIGH" if deviations[i] > 0 else "LOW",
            }
            for i in top_indices
        ]

        top5_mean = float(np.mean(strain[top_indices]))
        score_multiple = float(top5_mean / max(self.threshold, 1e-4))
        is_anomaly = bool(top5_mean > self.threshold or len(drops) > 0)

        # Hardware impact level calculation (independent of anomaly veto)
        impact = self._compute_impact(x)

        return {
            "score": round(top5_mean, 4),
            "threshold": round(self.threshold, 4),
            "score_multiple": round(score_multiple, 2),
            "is_anomaly": is_anomaly,
            "top_deviating_features": top_deviations,
            "metric_drops": drops,
            "hardware_impact": impact,
        }

    def _compute_impact(self, x: np.ndarray) -> str:
        """Determines physical hardware impact level without suppressing alerts."""
        cpu_mean_idx = settings.FEATURE_NAMES.index("cpu_percent_mean") if "cpu_percent_mean" in settings.FEATURE_NAMES else None
        mem_used_idx = settings.FEATURE_NAMES.index("mem_used_pct") if "mem_used_pct" in settings.FEATURE_NAMES else None
        swap_idx = settings.FEATURE_NAMES.index("swap_used_pct") if "swap_used_pct" in settings.FEATURE_NAMES else None

        cpu = x[cpu_mean_idx] if cpu_mean_idx is not None else 0.0
        mem = x[mem_used_idx] if mem_used_idx is not None else 0.0
        swap = x[swap_idx] if swap_idx is not None else 0.0

        if cpu >= 90.0 or mem >= 95.0:
            return "CRITICAL"
        if cpu >= 75.0 or mem >= 85.0 or swap >= 25.0:
            return "HIGH"
        if cpu >= 50.0 or mem >= 70.0:
            return "MEDIUM"
        return "LOW"

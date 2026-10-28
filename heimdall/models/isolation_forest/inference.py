"""
Isolation Forest Inference Engine
=================================
Performs fast unsupervised anomaly scoring using the trained Isolation Forest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from heimdall.config import settings

_MODEL: Any = None
_THRESHOLD_DATA: dict[str, Any] | None = None


def load_model(
    model_path: Path | str | None = None,
    threshold_path: Path | str | None = None,
) -> tuple[Any, dict[str, Any]]:
    global _MODEL, _THRESHOLD_DATA
    m_path = Path(model_path) if model_path else (settings.MODELS_DIR / "isolation_forest.joblib")
    t_path = Path(threshold_path) if threshold_path else (settings.MODELS_DIR / "iso_threshold.json")

    if _MODEL is None or _THRESHOLD_DATA is None:
        if not m_path.exists() or not t_path.exists():
            raise FileNotFoundError(f"Isolation Forest model/threshold missing at {m_path}")
        _MODEL = joblib.load(m_path)
        with open(t_path) as f:
            _THRESHOLD_DATA = json.load(f)

    return _MODEL, _THRESHOLD_DATA


def detect(features_scaled: list[float] | np.ndarray) -> dict[str, Any]:
    """
    Score a single feature vector using the Isolation Forest.

    Returns:
    --------
    dict with:
        - raw_score: float (higher = more anomalous)
        - threshold: float (99.5th percentile threshold)
        - is_anomaly: bool (raw_score > threshold)
        - score_multiple: float (raw_score / threshold)
        - score_pct: float (score mapped to 0-100 range)
    """
    model, thresh_meta = load_model()
    x = np.asarray(features_scaled, dtype=np.float32).reshape(1, -1)

    # -score_samples yields positive values where higher is more anomalous
    raw_score = float(-model.score_samples(x)[0])
    threshold = float(thresh_meta.get("threshold", 0.5))
    p50 = float(thresh_meta.get("p50_val_score", threshold * 0.7))

    is_anomaly = bool(raw_score > threshold)
    score_multiple = float(raw_score / max(threshold, 1e-6))

    # Continuous scaling: 50.0 at threshold, 0 at p50 normal
    if raw_score <= threshold:
        pct = max(0.0, ((raw_score - p50) / max(threshold - p50, 1e-6)) * 50.0)
    else:
        # Above threshold: scales 50..100
        excess = (raw_score - threshold) / max(threshold, 1e-6)
        pct = min(100.0, 50.0 + excess * 50.0)

    return {
        "model": "isolation_forest",
        "raw_score": raw_score,
        "threshold": threshold,
        "is_anomaly": is_anomaly,
        "score_multiple": score_multiple,
        "anomaly_score": round(pct, 2),
    }

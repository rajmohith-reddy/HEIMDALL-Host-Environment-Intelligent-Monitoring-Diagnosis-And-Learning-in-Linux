"""
LightGBM Inference Engine
=========================
Fast probability prediction and tree-based explanation for live system telemetry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from heimdall.config import settings

_MODEL: Any = None
_META: dict[str, Any] | None = None


def load_model(
    model_path: Path | str | None = None,
    meta_path: Path | str | None = None,
) -> tuple[Any, dict[str, Any]]:
    global _MODEL, _META
    m_path = Path(model_path) if model_path else (settings.MODELS_DIR / "lightgbm_classifier.joblib")
    t_path = Path(meta_path) if meta_path else (settings.MODELS_DIR / "lightgbm_meta.json")

    if _MODEL is None or _META is None:
        if not m_path.exists():
            raise FileNotFoundError(f"LightGBM model missing at {m_path}")
        _MODEL = joblib.load(m_path)
        if t_path.exists():
            with open(t_path) as f:
                _META = json.load(f)
        else:
            _META = {}

    return _MODEL, _META


def classify(features_scaled: list[float] | np.ndarray) -> dict[str, Any]:
    """
    Predict workload/anomaly class and probability distribution.
    """
    model, meta = load_model()
    x = np.asarray(features_scaled, dtype=np.float32).reshape(1, -1)

    probs = model.predict_proba(x)[0]
    pred_idx = int(np.argmax(probs))
    pred_class = settings.CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    prob_dict = {
        name: float(p)
        for name, p in zip(settings.CLASS_NAMES, probs)
    }

    return {
        "model": "lightgbm",
        "predicted_class": pred_class,
        "predicted_index": pred_idx,
        "confidence": confidence,
        "probabilities": prob_dict,
    }

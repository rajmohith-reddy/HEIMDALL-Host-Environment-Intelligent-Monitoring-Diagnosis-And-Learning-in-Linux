"""
Unit Tests for HEIMDALL v2 AI Modules
=====================================
Verifies Isolation Forest, LightGBM, Temperature Scaling, and Robust MAD baseline.
"""

import numpy as np
import pytest

from heimdall.analysis.calibration import CALIBRATOR_PATH, calibrate_logits, compute_ece
from heimdall.config import settings
from heimdall.models.baseline.mad_baseline import MADDetector
from heimdall.models.isolation_forest.inference import detect as iso_detect
from heimdall.models.lightgbm.inference import classify as lgb_classify


def test_isolation_forest_inference():
    """Verify Isolation Forest outputs bounded scores and detects valid features."""
    sample_vec = [0.0] * settings.NUM_FEATURES
    result = iso_detect(sample_vec)

    assert "raw_score" in result
    assert "threshold" in result
    assert "is_anomaly" in result
    assert "anomaly_score" in result
    assert 0.0 <= result["anomaly_score"] <= 100.0


def test_lightgbm_inference():
    """Verify LightGBM classifier predicts valid class and probability distribution."""
    sample_vec = [0.0] * settings.NUM_FEATURES
    result = lgb_classify(sample_vec)

    assert result["predicted_class"] in settings.CLASS_NAMES
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["probabilities"]) == len(settings.CLASS_NAMES)
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-4


def test_temperature_calibration():
    """Verify temperature scaling rescales logits into valid probability simplex."""
    logits = np.array([[2.5, 1.0, 0.2, -0.5, 0.0, 0.1]])
    cal_probs = calibrate_logits(logits, CALIBRATOR_PATH)

    assert cal_probs.shape == (1, 6)
    assert np.all(cal_probs >= 0.0)
    assert np.isclose(np.sum(cal_probs), 1.0)


def test_mad_baseline_and_drops():
    """Verify non-parametric MAD baseline evaluates deviations and flags metric drops."""
    detector = MADDetector()
    sample_vec = [10.0] * settings.NUM_FEATURES

    result = detector.evaluate(sample_vec)
    assert "score" in result
    assert "threshold" in result
    assert "is_anomaly" in result
    assert "hardware_impact" in result
    assert result["hardware_impact"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert isinstance(result["top_deviating_features"], list)

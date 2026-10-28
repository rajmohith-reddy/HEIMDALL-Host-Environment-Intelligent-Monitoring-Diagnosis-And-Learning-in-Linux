"""
Tests — Preprocessor
=====================
Verifies feature extraction, scaler fit/transform/save/load, and basic
dataset-builder logic without requiring a full dataset on disk.
"""

import numpy as np
import pytest

from heimdall.collector.snapshot import collect
from heimdall.config import settings
from heimdall.preprocessor.feature_engineer import extract_features
from heimdall.preprocessor.scaler import Scaler


class TestFeatureEngineer:
    def test_returns_list(self):
        snap = collect()
        vec = extract_features(snap)
        assert isinstance(vec, list)

    def test_correct_length(self):
        snap = collect()
        vec = extract_features(snap)
        assert len(vec) == settings.NUM_FEATURES, (
            f"Expected {settings.NUM_FEATURES} features, got {len(vec)}"
        )

    def test_all_floats(self):
        snap = collect()
        vec = extract_features(snap)
        for i, v in enumerate(vec):
            assert isinstance(v, float), f"Feature {i} ({settings.FEATURE_NAMES[i]}) is not float: {v!r}"

    def test_no_nan_in_normal_snap(self):
        snap = collect()
        vec = extract_features(snap)
        for i, v in enumerate(vec):
            assert not (v != v), f"NaN at feature {i}: {settings.FEATURE_NAMES[i]}"

    def test_with_prev_snap(self):
        import time
        snap1 = collect()
        time.sleep(0.2)
        snap2 = collect()
        vec = extract_features(snap2, prev_snap=snap1)
        assert len(vec) == settings.NUM_FEATURES

    def test_delta_features_are_zero_without_prev(self):
        snap = collect()
        vec = extract_features(snap, prev_snap=None)
        # Last 4 features are deltas — should be 0.0 on first call
        delta_features = vec[-4:]
        for v in delta_features:
            assert v == 0.0


class TestScaler:
    def _make_data(self, n: int = 100) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.standard_normal((n, settings.NUM_FEATURES)).astype(np.float32)

    def test_fit_transform_shape(self):
        sc = Scaler()
        X = self._make_data()
        X_scaled = sc.fit_transform(X)
        assert X_scaled.shape == X.shape

    def test_transform_single_sample(self):
        sc = Scaler()
        X = self._make_data()
        sc.fit(X)
        single = X[0]
        result = sc.transform(single)
        assert result.shape == (settings.NUM_FEATURES,)

    def test_scaled_mean_approx_zero(self):
        sc = Scaler()
        X = self._make_data(n=1000)
        X_scaled = sc.fit_transform(X)
        col_means = X_scaled.mean(axis=0)
        assert np.allclose(col_means, 0.0, atol=0.1), "Scaled data mean should be near 0"

    def test_transform_before_fit_raises(self):
        sc = Scaler()
        with pytest.raises(RuntimeError):
            sc.transform(np.zeros((1, settings.NUM_FEATURES)))

    def test_save_load_roundtrip(self, tmp_path):
        sc = Scaler()
        X = self._make_data()
        sc.fit(X)

        save_path = tmp_path / "test_scaler.pkl"
        sc.save(save_path)

        sc2 = Scaler.load(save_path)
        X_orig = sc.transform(X[:5])
        X_load = sc2.transform(X[:5])
        assert np.allclose(X_orig, X_load, atol=1e-6)

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Scaler.load(tmp_path / "nonexistent.pkl")

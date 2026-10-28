"""
Tests — Model Inference (without saved checkpoints)
=====================================================
Tests that the model architectures initialise correctly, perform forward
passes without errors, and that inference wrappers handle missing checkpoint
files gracefully.
"""

import numpy as np
import pytest
import torch

from heimdall.config import settings
from heimdall.models.mlp.model import MLPClassifier
from heimdall.models.autoencoder.model import SystemAutoencoder


class TestMLPModel:
    def test_instantiation(self):
        model = MLPClassifier()
        assert model is not None

    def test_forward_pass_single(self):
        model = MLPClassifier()
        model.eval()
        x = torch.randn(1, settings.NUM_FEATURES)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, settings.NUM_CLASSES)

    def test_forward_pass_batch(self):
        model = MLPClassifier()
        model.eval()
        x = torch.randn(32, settings.NUM_FEATURES)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (32, settings.NUM_CLASSES)

    def test_predict_proba_sums_to_one(self):
        model = MLPClassifier()
        model.eval()
        x = torch.randn(10, settings.NUM_FEATURES)
        probs = model.predict_proba(x)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(10), atol=1e-5), \
            "Softmax probabilities should sum to 1"

    def test_predict_returns_valid_class(self):
        model = MLPClassifier()
        model.eval()
        x = torch.randn(5, settings.NUM_FEATURES)
        preds = model.predict(x)
        assert preds.shape == (5,)
        for p in preds:
            assert 0 <= int(p) < settings.NUM_CLASSES

    def test_predict_proba_single(self):
        model = MLPClassifier()
        model.eval()
        x = torch.randn(settings.NUM_FEATURES)   # 1D input
        probs = model.predict_proba(x)
        assert probs.shape == (settings.NUM_CLASSES,)

    def test_parameter_count_positive(self):
        model = MLPClassifier()
        assert model.count_parameters() > 0

    def test_inference_missing_checkpoint_raises(self):
        from heimdall.models.mlp import inference as mlp_inf
        mlp_inf.reload()   # clear cache
        dummy_path = "/tmp/nonexistent_model.pt"
        with pytest.raises(FileNotFoundError):
            mlp_inf.classify([0.0] * settings.NUM_FEATURES, model_path=dummy_path)


class TestAutoencoderModel:
    def test_instantiation(self):
        model = SystemAutoencoder()
        assert model is not None

    def test_forward_pass_single(self):
        model = SystemAutoencoder()
        model.eval()
        x = torch.randn(1, settings.NUM_FEATURES)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, settings.NUM_FEATURES), \
            "AE output shape must match input (reconstruction)"

    def test_forward_pass_batch(self):
        model = SystemAutoencoder()
        model.eval()
        x = torch.randn(16, settings.NUM_FEATURES)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (16, settings.NUM_FEATURES)

    def test_reconstruction_error_shape(self):
        model = SystemAutoencoder()
        model.eval()
        x = torch.randn(8, settings.NUM_FEATURES)
        with torch.no_grad():
            err = model.reconstruction_error(x)
        assert err.shape == (8,), "One error value per sample"

    def test_reconstruction_error_nonnegative(self):
        model = SystemAutoencoder()
        model.eval()
        x = torch.randn(8, settings.NUM_FEATURES)
        with torch.no_grad():
            err = model.reconstruction_error(x)
        assert (err >= 0).all()

    def test_encode_shape(self):
        model = SystemAutoencoder()
        model.eval()
        x = torch.randn(4, settings.NUM_FEATURES)
        with torch.no_grad():
            z = model.encode(x)
        assert z.shape == (4, settings.AE_LATENT_DIM)

    def test_ae_inference_missing_checkpoint_raises(self):
        from heimdall.models.autoencoder import inference as ae_inf
        ae_inf.reload()
        # Temporarily override path to force FileNotFoundError
        import heimdall.config.settings as cfg
        original = cfg.AE_MODEL_PATH
        from pathlib import Path
        cfg.AE_MODEL_PATH = Path("/tmp/nonexistent_ae.pt")
        try:
            with pytest.raises(FileNotFoundError):
                ae_inf.detect([0.0] * settings.NUM_FEATURES)
        finally:
            cfg.AE_MODEL_PATH = original
            ae_inf.reload()


class TestAnalysisRules:
    """Test the rules engine against synthetic snapshots."""

    def _make_snap(self, **overrides):
        """Create a minimal fake snapshot dict."""
        from heimdall.collector.snapshot import SystemSnapshot
        snap = SystemSnapshot(
            timestamp="2025-01-01T00:00:00+00:00",
            label="test",
            cpu={
                "cpu_percent_mean": overrides.get("cpu_pct", 5.0),
                "cpu_percent_max": 10.0,
                "load_avg_1": 0.1, "load_avg_5": 0.1, "load_avg_15": 0.1,
                "num_logical_cpus": 4,
            },
            memory={
                "mem_used_pct": overrides.get("mem_pct", 40.0),
                "mem_available_mb": overrides.get("mem_avail", 2000.0),
                "swap_used_pct": overrides.get("swap_pct", 5.0),
                "swap_out_per_sec": 0.0,
            },
            disk={
                "disk_used_pct": overrides.get("disk_pct", 30.0),
                "disk_read_mbps": 0.1, "disk_write_mbps": 0.1,
            },
            network={
                "net_bytes_recv_mbps": 0.01, "net_bytes_sent_mbps": 0.01,
                "tcp_connections": 10, "udp_connections": 2,
            },
            process={"num_processes": 100, "num_threads": 400, "num_zombies": overrides.get("zombies", 0), "open_files": 200},
            system={
                "failed_services": overrides.get("failed_svc", 0),
                "auth_failures_60s": overrides.get("auth_fails", 0),
                "logged_in_users": 1,
            },
        )
        return snap

    def test_no_rules_on_normal(self):
        from heimdall.analysis.rules import evaluate_rules
        snap = self._make_snap()
        msgs, sev = evaluate_rules(snap)
        assert sev == "OK"
        assert len(msgs) == 0

    def test_cpu_warning_triggered(self):
        from heimdall.analysis.rules import evaluate_rules
        snap = self._make_snap(cpu_pct=90.0)
        msgs, sev = evaluate_rules(snap)
        assert sev in ("WARNING", "CRITICAL")
        assert any("CPU" in m for m in msgs)

    def test_memory_critical_triggered(self):
        from heimdall.analysis.rules import evaluate_rules
        snap = self._make_snap(mem_avail=50.0)   # below CRITICAL threshold (100 MB)
        msgs, sev = evaluate_rules(snap)
        assert sev == "CRITICAL"

    def test_auth_failure_critical(self):
        from heimdall.analysis.rules import evaluate_rules
        snap = self._make_snap(auth_fails=10)
        msgs, sev = evaluate_rules(snap)
        assert sev == "CRITICAL"
        assert any("authentication" in m.lower() for m in msgs)

    def test_zombie_warning(self):
        from heimdall.analysis.rules import evaluate_rules
        snap = self._make_snap(zombies=5)
        msgs, sev = evaluate_rules(snap)
        assert sev == "WARNING"

"""
Autoencoder Inference (Production Multi-Signal + XAI Attribution)
==================================================================
Loads the trained `SystemAutoencoder` and calibrated `ae_threshold.json` once,
and exposes `detect()` returning an `AEResult` complete with:
  • `is_anomaly`             : bool (True if hybrid_score > threshold)
  • `anomaly_score`          : 0–100 normalised score (50 = threshold boundary)
  • `raw_error`              : mean MSE reconstruction error
  • `hybrid_score`           : Top-K + Global + Latent Mahalanobis score
  • `threshold`              : calibrated threshold τ
  • `subsystem_scores`       : dict[subsystem, float (0–100)]
  • `top_anomalous_features` : list of top-5 root-cause features with z-error & % share
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from heimdall.config import settings
from heimdall.models.autoencoder.model import SystemAutoencoder
from heimdall.models.autoencoder.trainer import compute_hybrid_score_np


# ── Cached state ──────────────────────────────────────────────────────────────
_model:      SystemAutoencoder | None = None
_calib:      dict[str, Any] | None    = None
_device:     torch.device             = torch.device("cpu")


def _get_model_and_calibration() -> tuple[SystemAutoencoder, dict[str, Any]]:
    global _model, _calib, _device

    if _model is not None and _calib is not None:
        return _model, _calib

    path = settings.AE_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Autoencoder not found at {path}. Run `heimdall train-ae` first."
        )
    _device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=_device, weights_only=False)
    _model     = SystemAutoencoder(
        num_features=checkpoint["num_features"],
        hidden_dims=checkpoint["hidden_dims"],
        latent_dim=checkpoint["latent_dim"],
    )
    _model.load_state_dict(checkpoint["model_state_dict"])
    _model.to(_device).eval()

    tpath = settings.AE_THRESHOLD_PATH
    if not tpath.exists():
        raise FileNotFoundError(
            f"Threshold file not found at {tpath}. Run `heimdall train-ae` first."
        )
    _calib = json.loads(tpath.read_text(encoding="utf-8"))
    return _model, _calib


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class AEResult:
    is_anomaly:             bool
    anomaly_score:          float                      # 0–100 normalised score (50 = threshold)
    raw_error:              float                      # mean MSE reconstruction error
    threshold:              float                      # calibrated threshold τ
    hybrid_score:           float = 0.0                # raw hybrid anomaly statistic
    latent_distance:        float = 0.0                # normalized distance in 8D latent space
    subsystem_scores:       dict[str, float] = field(default_factory=dict)
    top_anomalous_features: list[dict[str, Any]] = field(default_factory=list)
    dominant_subsystem:     str = "none"


# ── Public API ────────────────────────────────────────────────────────────────

def detect(
    feature_vector: list[float],
) -> AEResult:
    """
    Run multi-signal anomaly detection and XAI root-cause attribution on a
    single feature vector (scaled by `ae_scaler.pkl`).
    """
    model, calib = _get_model_and_calibration()

    x = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).to(_device)

    with torch.no_grad():
        x_hat, z_t = model.forward_with_latent(x)
        feat_err_t = (x - x_hat) ** 2
        raw_error  = float(feat_err_t.mean().item())
        feat_err   = feat_err_t.cpu().numpy()   # (1, F)
        z_np       = z_t.cpu().numpy()          # (1, D)

    threshold = float(calib.get("threshold", 1.6))
    top_k     = int(calib.get("top_k", settings.AE_TOP_K_FEATURES))

    # Check if full per-feature calibration exists
    if "feat_err_mean" in calib and "latent_mean" in calib:
        feat_err_mean = np.array(calib["feat_err_mean"], dtype=np.float32)
        feat_err_std  = np.array(calib["feat_err_std"],  dtype=np.float32)
        latent_mean   = np.array(calib["latent_mean"],   dtype=np.float32)
        latent_std    = np.array(calib["latent_std"],    dtype=np.float32)

        x_scaled = np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)
        hybrid_arr, z_feat_arr = compute_hybrid_score_np(
            feat_err, z_np,
            feat_err_mean, feat_err_std,
            latent_mean, latent_std,
            top_k=top_k,
            x_scaled=x_scaled,
        )
        hybrid_score = float(hybrid_arr[0])
        z_feat       = z_feat_arr[0]  # (F,)

        z_dist_dim   = np.maximum(0.0, np.abs(z_np[0] - latent_mean) / np.maximum(latent_std, 0.1) - 1.0)
        latent_dist  = float((z_dist_dim ** 2).mean())
    else:
        # Legacy fallback if old threshold file
        z_feat       = feat_err[0]
        hybrid_score = raw_error
        latent_dist  = 0.0

    is_anomaly = hybrid_score > threshold
    # Normalise: hybrid_score == threshold → 50.0; 2x threshold → 100.0
    anomaly_score = min(100.0, (hybrid_score / max(threshold, 1e-9)) * 50.0)

    # z_feat is already directional tail-gated
    attr_z = z_feat

    # Compute per-subsystem anomaly scores (0–100 scale) and unclamped stats
    feat_index = {name: idx for idx, name in enumerate(settings.FEATURE_NAMES)}
    subsystem_scores: dict[str, float] = {}
    raw_sub_stats: dict[str, float] = {}
    for sub_name, sub_feats in settings.SUBSYSTEM_FEATURES.items():
        idxs = [feat_index[f] for f in sub_feats if f in feat_index]
        if not idxs:
            subsystem_scores[sub_name] = 0.0
            raw_sub_stats[sub_name] = 0.0
            continue
        sub_vals = np.sort(attr_z[idxs])
        # Average top-2 features in each subsystem so localized subsystem spikes stand out
        sub_stat = float(sub_vals[-min(2, len(sub_vals)):].mean())
        raw_sub_stats[sub_name] = sub_stat
        sub_score = min(100.0, (sub_stat / max(threshold, 1e-9)) * 50.0)
        subsystem_scores[sub_name] = round(sub_score, 1)

    dominant_sub = "none"
    if is_anomaly and raw_sub_stats and max(subsystem_scores.values(), default=0.0) >= 35.0:
        dominant_sub = max(raw_sub_stats, key=lambda k: raw_sub_stats[k])
    elif not is_anomaly:
        subsystem_scores = {sub_name: 0.0 for sub_name in settings.SUBSYSTEM_FEATURES}

    # Top-5 Anomalous Features (XAI Attribution)
    top_features: list[dict[str, Any]] = []
    if is_anomaly:
        score_vec = attr_z
        total_z = float(score_vec.sum()) if float(score_vec.sum()) > 1e-9 else 1.0
        ranked_indices = np.argsort(score_vec)[::-1][:5]
        for idx in ranked_indices:
            if score_vec[int(idx)] <= 0.01:
                continue
            fname = settings.FEATURE_NAMES[int(idx)]
            z_val = float(score_vec[int(idx)])
            raw_f = float(feat_err[0, int(idx)])
            share = (z_val / total_z * 100.0)
            top_features.append({
                "feature":          fname,
                "subsystem":        settings.FEATURE_TO_SUBSYSTEM.get(fname, "system"),
                "z_error":          round(z_val, 2),
                "raw_sq_error":     round(raw_f, 4),
                "contribution_pct": round(share, 1),
            })

    return AEResult(
        is_anomaly=is_anomaly,
        anomaly_score=round(anomaly_score, 2),
        raw_error=round(raw_error, 6),
        threshold=round(threshold, 6),
        hybrid_score=round(hybrid_score, 4),
        latent_distance=round(latent_dist, 4),
        subsystem_scores=subsystem_scores,
        top_anomalous_features=top_features,
        dominant_subsystem=dominant_sub,
    )


def reload() -> None:
    """Force a reload from disk on next `detect()` call."""
    global _model, _calib
    _model = _calib = None

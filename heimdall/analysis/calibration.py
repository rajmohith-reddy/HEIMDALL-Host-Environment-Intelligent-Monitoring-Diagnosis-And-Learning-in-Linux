"""
Temperature Scaling & Probability Calibration
==============================================
Post-hoc probability calibration using Temperature Scaling (Guo et al., 2017).
Finds optimal scalar T > 0 on validation logits using Negative Log-Likelihood (NLL).
Computes Expected Calibration Error (ECE) before and after calibration.

Usage
-----
    from heimdall.analysis.calibration import fit_temperature_scaler, calibrate_logits
    meta = fit_temperature_scaler()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax

from heimdall.config import settings
from heimdall.preprocessor.scaler import Scaler

CALIBRATOR_PATH = settings.MODELS_DIR / "temperature_calibrator.json"


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    Divides confidence scores into n_bins and computes weighted difference
    between average confidence and accuracy per bin.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(labels)

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def fit_temperature_scaler(
    processed_dir: Path | str | None = None,
    save_path: Path | str | None = None,
) -> dict[str, Any]:
    """
    Fits optimal temperature T on validation logits from the trained MLP.
    """
    import torch
    from heimdall.models.mlp.model import MLPClassifier

    processed_dir = Path(processed_dir) if processed_dir else settings.DATA_PROCESSED_DIR
    save_path = Path(save_path) if save_path else CALIBRATOR_PATH

    print("── Temperature Scaling Calibrator ──────────────────────────")
    if not settings.MLP_MODEL_PATH.exists() or not settings.MLP_SCALER_PATH.exists():
        print("   [!] MLP model or scaler missing. Train MLP first.")
        return {}

    # 1. Load validation data
    val_df = pd.read_csv(processed_dir / "val.csv")
    scaler = Scaler.load(settings.MLP_SCALER_PATH)

    X_val_raw = val_df[settings.FEATURE_NAMES].values.astype(np.float32)
    y_val = val_df["label"].values.astype(int)
    X_val_scaled = scaler.transform(X_val_raw)

    # 2. Extract raw uncalibrated logits from MLP
    device = torch.device("cpu")
    ckpt = torch.load(settings.MLP_MODEL_PATH, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model = MLPClassifier(
        num_features=ckpt.get("num_features", settings.NUM_FEATURES) if isinstance(ckpt, dict) else settings.NUM_FEATURES,
        num_classes=ckpt.get("num_classes", settings.NUM_CLASSES) if isinstance(ckpt, dict) else settings.NUM_CLASSES,
        hidden_dims=ckpt.get("hidden_dims", settings.MLP_HIDDEN_DIMS) if isinstance(ckpt, dict) else settings.MLP_HIDDEN_DIMS,
        dropout_rates=ckpt.get("dropout_rates", settings.MLP_DROPOUT_RATES) if isinstance(ckpt, dict) else settings.MLP_DROPOUT_RATES,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        inputs = torch.tensor(X_val_scaled, dtype=torch.float32)
        logits = model(inputs).numpy()

    # 3. Compute baseline metrics before calibration
    raw_probs = softmax(logits, axis=1)
    uncalibrated_ece = compute_ece(raw_probs, y_val)

    # 4. Optimize temperature T to minimize Cross-Entropy (NLL)
    def nll_loss(t_param: np.ndarray) -> float:
        T = t_param[0]
        scaled = logits / max(T, 1e-4)
        # Log-sum-exp trick for numerical stability
        exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        p = exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)
        # Negative log-likelihood
        eps = 1e-15
        nll = -np.mean(np.log(p[np.arange(len(y_val)), y_val] + eps))
        return float(nll)

    res = minimize(nll_loss, x0=[1.5], bounds=[(0.05, 10.0)], method="L-BFGS-B")
    optimal_T = float(res.x[0])

    # 5. Compute calibrated metrics
    calibrated_probs = softmax(logits / optimal_T, axis=1)
    calibrated_ece = compute_ece(calibrated_probs, y_val)

    meta = {
        "temperature": round(optimal_T, 4),
        "uncalibrated_ece": round(uncalibrated_ece, 4),
        "calibrated_ece": round(calibrated_ece, 4),
        "ece_reduction_pct": round(max(0.0, (uncalibrated_ece - calibrated_ece) / max(uncalibrated_ece, 1e-6) * 100.0), 2),
        "val_samples": int(len(val_df)),
    }

    with open(save_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"   Optimal Temperature (T) : {optimal_T:.4f}")
    print(f"   Uncalibrated ECE        : {uncalibrated_ece:.4f}")
    print(f"   Calibrated ECE          : {calibrated_ece:.4f} (↓ {meta['ece_reduction_pct']}%)")
    print(f"   Saved Calibration Meta  : {save_path}")
    print("✓ Temperature calibration complete!\n")

    return meta


def calibrate_logits(logits: np.ndarray, temperature_path: Path | str | None = None) -> np.ndarray:
    """
    Applies calibrated temperature scaling to raw logits before softmax.
    """
    t_path = Path(temperature_path) if temperature_path else CALIBRATOR_PATH
    T = 1.0
    if t_path.exists():
        try:
            with open(t_path) as f:
                meta = json.load(f)
                T = float(meta.get("temperature", 1.0))
        except Exception:
            T = 1.0

    scaled = logits / max(T, 1e-4)
    return softmax(scaled, axis=-1)

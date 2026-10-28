"""
Autoencoder Trainer (Production Hybrid Calibration)
====================================================
Trains `SystemAutoencoder` exclusively on *normal* class snapshots scaled in
Normal-Only Z-score space (`ae_train.csv` / `ae_val.csv`), then calibrates a
production multi-signal anomaly detector:

1. Per-feature validation error distribution (μ_e,i and σ_e,i for i in 1..40)
2. Latent space normal cluster distribution (μ_z and σ_z for j in 1..8)
3. Hybrid Anomaly Score combining:
     - Top-K Standardized Feature Reconstruction Error (solves 40-feature dilution)
     - Global Mean Reconstruction Error
     - Latent Space Mahalanobis Distance
4. Empirical quantile threshold τ (97.5th percentile on validation normal data
   with a safety margin floor).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from heimdall.config import settings
from heimdall.models.autoencoder.model import SystemAutoencoder


def _load_normal_split(processed_dir: Path, split_name: str) -> torch.Tensor:
    """
    Load normal-only rows (label == 0) from `ae_<split>.csv` (preferred)
    or fallback to `<split>.csv`.
    """
    ae_csv = processed_dir / f"ae_{split_name}.csv"
    csv_path = ae_csv if ae_csv.exists() else (processed_dir / f"{split_name}.csv")
    df = pd.read_csv(csv_path)
    normal = df[df["label"] == 0]
    if normal.empty:
        raise ValueError(f"No 'normal' class samples found in {csv_path}.")
    return torch.tensor(normal[settings.FEATURE_NAMES].values, dtype=torch.float32)


def compute_hybrid_score_np(
    feat_errors: np.ndarray,
    z_latent: np.ndarray,
    feat_err_mean: np.ndarray,
    feat_err_std: np.ndarray,
    latent_mean: np.ndarray,
    latent_std: np.ndarray,
    top_k: int = settings.AE_TOP_K_FEATURES,
    x_scaled: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the production hybrid anomaly score for a batch of samples.
    Includes directional tail-gating: in physical telemetry, resources operating
    safely below or near normal (<= 1.5 sigma) are healthy baseline noise,
    not anomalies. Excess error activates smoothly as features enter the upper tail.
    """
    # 1. Standardized per-feature excess reconstruction error
    z_feat = np.maximum(0.0, (feat_errors - feat_err_mean) / np.maximum(feat_err_std, 0.05))

    if x_scaled is not None:
        tail_gate = np.maximum(0.0, np.minimum(1.0, (x_scaled - 1.5) / 1.0))
        z_feat = z_feat * tail_gate

    # 2. Top-K feature error (prevents 35 healthy features from diluting 3-5 anomalous ones)
    k = min(top_k, z_feat.shape[1])
    part_idx = np.argsort(z_feat, axis=1)[:, -k:]
    top_k_vals = np.take_along_axis(z_feat, part_idx, axis=1)
    top_k_score = top_k_vals.mean(axis=1)

    # 3. Global mean standardized error
    global_score = z_feat.mean(axis=1)

    # 4. Latent space normalized distance (excess beyond 1 std in latent space)
    z_dist_per_dim = np.maximum(0.0, np.abs(z_latent - latent_mean) / np.maximum(latent_std, 0.1) - 1.0)
    latent_score = (z_dist_per_dim ** 2).mean(axis=1)

    # Weighted fusion: 55% Top-K localized error + 25% Global error + 20% Latent distance
    hybrid = 0.55 * top_k_score + 0.25 * global_score + 0.20 * latent_score
    return hybrid, z_feat


def train(
    processed_dir: Path | str | None = None,
    model_path:    Path | str | None = None,
) -> dict:
    """
    Train the Denoising Autoencoder and persist model + multi-signal threshold stats.
    """
    processed_dir = Path(processed_dir) if processed_dir else settings.DATA_PROCESSED_DIR
    model_path    = Path(model_path)    if model_path    else settings.AE_MODEL_PATH

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("── Autoencoder Trainer (Production Denoising + Hybrid Score) ─")
    print(f"   Device    : {device}")
    print("   Training on 'normal' class in Normal-Only Z-Score space.")

    # 1. Load normal data
    X_train = _load_normal_split(processed_dir, "train").to(device)
    X_val   = _load_normal_split(processed_dir, "val").to(device)

    train_loader = DataLoader(
        TensorDataset(X_train, X_train),
        batch_size=settings.AE_BATCH_SIZE,
        shuffle=True,
    )

    print(f"   Train normal samples : {len(X_train)}")
    print(f"   Val   normal samples : {len(X_val)}")

    # 2. Model
    model = SystemAutoencoder().to(device)
    print(f"   {model}")

    criterion = nn.MSELoss()
    optimizer = AdamW(
        model.parameters(),
        lr=settings.AE_LEARNING_RATE,
        weight_decay=settings.AE_WEIGHT_DECAY,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # 3. Training loop
    history = {"train_loss": [], "val_loss": []}
    best_val_loss  = float("inf")
    patience_count = 0
    best_state     = None

    print(f"\n{'Epoch':>6} {'Train MSE':>11} {'Val MSE':>10}")
    print("─" * 32)

    start = time.time()
    for epoch in range(1, settings.AE_MAX_EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        for X_batch, _ in train_loader:
            optimizer.zero_grad()
            x_hat, z = model.forward_with_latent(X_batch)
            # Reconstruction loss + mild L2 latent regularization to keep normal cluster compact
            loss = criterion(x_hat, X_batch) + 1e-4 * (z ** 2).mean()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(X_batch)
        train_loss = train_loss_sum / len(X_train)

        # Validate (clean, no denoising noise)
        model.eval()
        with torch.no_grad():
            x_hat_val = model(X_val)
            val_loss  = criterion(x_hat_val, X_val).item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        print(f"{epoch:>6} {train_loss:>11.6f} {val_loss:>10.6f}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss  = val_loss
            patience_count = 0
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= settings.AE_PATIENCE:
                print(f"\n[Early Stopping] patience exceeded at epoch {epoch}.")
                break

    elapsed = time.time() - start
    print(f"\n── Training complete in {elapsed:.1f}s ─────────────────────────")

    # 4. Calibrate per-feature and latent statistics on combined normal (train + val)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        X_norm_all = torch.cat([X_train, X_val], dim=0)
        feat_errs_all = model.per_feature_error(X_norm_all).cpu().numpy()   # (N_all, F)
        z_all         = model.encode(X_norm_all).cpu().numpy()              # (N_all, D)

        feat_errs_val = model.per_feature_error(X_val).cpu().numpy()        # (N_val, F)
        z_val         = model.encode(X_val).cpu().numpy()                   # (N_val, D)
        raw_mse_val   = feat_errs_val.mean(axis=1)

    feat_err_mean = feat_errs_all.mean(axis=0)
    feat_err_std  = np.maximum(feat_errs_all.std(axis=0), 0.25)
    latent_mean   = z_all.mean(axis=0)
    latent_std    = np.maximum(z_all.std(axis=0), 0.15)

    hybrid_val, _ = compute_hybrid_score_np(
        feat_errs_val, z_val,
        feat_err_mean, feat_err_std,
        latent_mean, latent_std,
        top_k=settings.AE_TOP_K_FEATURES,
        x_scaled=X_val.cpu().numpy(),
    )

    q_thresh  = float(np.quantile(hybrid_val, settings.AE_THRESHOLD_QUANTILE))
    s_thresh  = float(hybrid_val.mean() + settings.AE_THRESHOLD_SIGMA * hybrid_val.std())
    # Ensure a practical minimum threshold floor (1.6) so minor normal desktop noise stays OK
    threshold = max(q_thresh, s_thresh, 1.60)

    raw_mse_thresh = float(
        max(
            raw_mse_val.mean() + settings.AE_THRESHOLD_SIGMA * raw_mse_val.std(),
            np.quantile(raw_mse_val, settings.AE_THRESHOLD_QUANTILE),
            0.25,
        )
    )

    print(f"   Val Raw MSE    : mean={raw_mse_val.mean():.6f}  std={raw_mse_val.std():.6f}")
    print(f"   Val Hybrid Scr : mean={hybrid_val.mean():.4f}   std={hybrid_val.std():.4f}")
    print(f"   Threshold τ    : {threshold:.4f} (hybrid) | {raw_mse_thresh:.6f} (raw MSE)")

    # 5. Save model + threshold calibration JSON
    model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": best_state or model.state_dict(),
        "num_features":     settings.NUM_FEATURES,
        "hidden_dims":      settings.AE_HIDDEN_DIMS,
        "latent_dim":       settings.AE_LATENT_DIM,
    }
    torch.save(checkpoint, model_path)
    print(f"   Model saved to    : {model_path}")

    threshold_data = {
        "threshold":       round(threshold, 6),
        "raw_mse_thresh":  round(raw_mse_thresh, 6),
        "val_mean":        round(float(hybrid_val.mean()), 6),
        "val_std":         round(float(hybrid_val.std()), 6),
        "raw_val_mean":    round(float(raw_mse_val.mean()), 6),
        "raw_val_std":     round(float(raw_mse_val.std()), 6),
        "sigma":           settings.AE_THRESHOLD_SIGMA,
        "quantile":        settings.AE_THRESHOLD_QUANTILE,
        "top_k":           settings.AE_TOP_K_FEATURES,
        "feat_err_mean":   [round(float(x), 6) for x in feat_err_mean],
        "feat_err_std":    [round(float(x), 6) for x in feat_err_std],
        "latent_mean":     [round(float(x), 6) for x in latent_mean],
        "latent_std":      [round(float(x), 6) for x in latent_std],
    }
    threshold_path = settings.AE_THRESHOLD_PATH
    threshold_path.write_text(json.dumps(threshold_data, indent=2), encoding="utf-8")
    print(f"   Threshold saved to: {threshold_path}")

    # 6. Plot training curve
    _plot_loss(history, settings.REPORTS_DIR)

    # Clear cached inference model if loaded in current process
    from heimdall.models.autoencoder import inference as ae_inf
    ae_inf.reload()

    return {"history": history, "threshold": threshold}


def _plot_loss(history: dict, reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(history["train_loss"]) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss (Denoising)", color="#4c8eda")
    plt.plot(epochs, history["val_loss"],   label="Val MSE (Clean)",        color="#e06c5a")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss")
    plt.title("HEIMDALL Production Autoencoder — Training Curve", fontweight="bold")
    plt.legend()
    plt.grid(alpha=0.3)
    out = reports_dir / "ae_training_curve.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"   AE training curve → {out}")

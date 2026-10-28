"""
Isolation Forest Anomaly Detector
=================================
Unsupervised tree-based anomaly detector for HEIMDALL.
Constructs an ensemble of isolation trees to partition telemetry feature space.
Anomalies require fewer splits to isolate and receive higher anomaly scores.

Usage
-----
    from heimdall.models.isolation_forest.trainer import train
    train()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from heimdall.config import settings
from heimdall.preprocessor.scaler import Scaler


def train(
    processed_dir: Path | str | None = None,
    model_path: Path | str | None = None,
    threshold_path: Path | str | None = None,
    scaler_path: Path | str | None = None,
) -> dict[str, Any]:
    """
    Train an Isolation Forest on normal baseline telemetry (ae_train.csv).
    Determines empirical threshold on ae_val.csv at the 99.5th percentile.
    """
    processed_dir = Path(processed_dir) if processed_dir else settings.DATA_PROCESSED_DIR
    model_path = Path(model_path) if model_path else (settings.MODELS_DIR / "isolation_forest.joblib")
    threshold_path = Path(threshold_path) if threshold_path else (settings.MODELS_DIR / "iso_threshold.json")
    scaler_path = Path(scaler_path) if scaler_path else settings.AE_SCALER_PATH

    print("── Isolation Forest Trainer ────────────────────────────────")
    print(f"   Target Model Path : {model_path}")
    print(f"   Features          : {settings.NUM_FEATURES}")

    # 1. Load normal-only training and validation sets
    raw_train_path = processed_dir / "raw_train.csv"
    raw_val_path = processed_dir / "raw_val.csv"
    scaler = Scaler.load(scaler_path) if scaler_path.exists() else None

    if raw_train_path.exists() and raw_val_path.exists() and scaler is not None:
        train_df = pd.read_csv(raw_train_path)
        val_df = pd.read_csv(raw_val_path)
        X_train_raw = train_df[train_df["label"] == 0][settings.FEATURE_NAMES].values.astype(np.float32)
        X_val_raw = val_df[val_df["label"] == 0][settings.FEATURE_NAMES].values.astype(np.float32)
        X_train = scaler.transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)
    else:
        # ae_train.csv and ae_val.csv are already scaled by ae_scaler
        train_df = pd.read_csv(processed_dir / "ae_train.csv")
        val_df = pd.read_csv(processed_dir / "ae_val.csv")
        X_train = train_df[train_df["label"] == 0][settings.FEATURE_NAMES].values.astype(np.float32)
        X_val = val_df[val_df["label"] == 0][settings.FEATURE_NAMES].values.astype(np.float32)


    # 3. Fit Isolation Forest
    # n_estimators=150, contamination='auto', max_samples=256 for fast robust partitioning
    iso_model = IsolationForest(
        n_estimators=150,
        max_samples="auto",
        contamination=0.01,
        random_state=settings.RANDOM_SEED,
        n_jobs=-1,
    )
    iso_model.fit(X_train)

    # 4. Calibrate threshold on validation normal set
    # score_samples returns opposite of anomaly score (lower is more anomalous)
    # We negate and shift so higher score = more anomalous
    val_scores_raw = -iso_model.score_samples(X_val)
    # Normalize score such that 0 is median normal, and threshold is calibrated at 99.5th percentile
    p50 = float(np.percentile(val_scores_raw, 50.0))
    p995 = float(np.percentile(val_scores_raw, 99.5))
    threshold = float(p995)

    meta = {
        "model_type": "IsolationForest",
        "n_estimators": 150,
        "p50_val_score": p50,
        "p995_threshold": threshold,
        "threshold": threshold,
        "train_samples": int(len(train_df)),
        "val_samples": int(len(val_df)),
    }

    # 5. Persist artifacts
    joblib.dump(iso_model, model_path)
    with open(threshold_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"   Train samples     : {len(X_train)}")
    print(f"   Val 99.5% Thresh  : {threshold:.4f}")
    print(f"   Saved Model       : {model_path}")
    print(f"   Saved Threshold   : {threshold_path}")
    print("✓ Isolation Forest training complete!\n")

    return meta

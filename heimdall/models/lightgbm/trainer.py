"""
LightGBM Classifier Trainer
===========================
Trains a Gradient Boosted Decision Tree ensemble for multi-class host workload
and anomaly diagnosis. Extremely fast, robust on tabular features, and provides
direct tree split importances for explainability.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from heimdall.config import settings
from heimdall.preprocessor.scaler import Scaler


def train(
    processed_dir: Path | str | None = None,
    model_path: Path | str | None = None,
    meta_path: Path | str | None = None,
    scaler_path: Path | str | None = None,
) -> dict[str, Any]:
    processed_dir = Path(processed_dir) if processed_dir else settings.DATA_PROCESSED_DIR
    model_path = Path(model_path) if model_path else (settings.MODELS_DIR / "lightgbm_classifier.joblib")
    meta_path = Path(meta_path) if meta_path else (settings.MODELS_DIR / "lightgbm_meta.json")
    scaler_path = Path(scaler_path) if scaler_path else settings.MLP_SCALER_PATH

    print("── LightGBM Classifier Trainer ─────────────────────────────")
    print(f"   Target Model Path : {model_path}")
    print(f"   Classes           : {settings.NUM_CLASSES} → {settings.CLASS_NAMES}")

    # 1. Load multi-class datasets
    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df = pd.read_csv(processed_dir / "val.csv")

    X_train_raw = train_df[settings.FEATURE_NAMES].values.astype(np.float32)
    y_train = train_df["label"].values.astype(int)

    X_val_raw = val_df[settings.FEATURE_NAMES].values.astype(np.float32)
    y_val = val_df["label"].values.astype(int)

    # 2. Scale features
    scaler = Scaler.load(scaler_path) if scaler_path.exists() else None
    if scaler is not None:
        X_train = scaler.transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)
    else:
        X_train = X_train_raw
        X_val = X_val_raw

    # 3. Fit LightGBM classifier
    clf = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        random_state=settings.RANDOM_SEED,
        objective="multiclass",
        num_class=settings.NUM_CLASSES,
        verbosity=-1,
        n_jobs=-1,
    )

    clf.fit(X_train, y_train)

    # 4. Evaluate on validation set
    val_preds = clf.predict(X_val)
    val_probs = clf.predict_proba(X_val)
    val_acc = float(accuracy_score(y_val, val_preds))
    val_f1 = float(f1_score(y_val, val_preds, average="macro"))

    # Feature importances
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    top_features = [
        {"feature": settings.FEATURE_NAMES[i], "importance": int(importances[i])}
        for i in sorted_idx[:10]
    ]

    meta = {
        "model_type": "LightGBMClassifier",
        "val_accuracy": val_acc,
        "val_macro_f1": val_f1,
        "train_samples": int(len(train_df)),
        "val_samples": int(len(val_df)),
        "classes": settings.CLASS_NAMES,
        "top_features": top_features,
    }

    # 5. Persist artifacts
    joblib.dump(clf, model_path)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"   Train samples     : {len(X_train)}")
    print(f"   Val Accuracy      : {val_acc * 100:.2f}%")
    print(f"   Val Macro-F1      : {val_f1:.4f}")
    print(f"   Saved Model       : {model_path}")
    print("✓ LightGBM training complete!\n")

    return meta

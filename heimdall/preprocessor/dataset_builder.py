"""
Dataset Builder
================
Reads raw JSON snapshots from `data/raw/<label>/*.json`, applies feature
engineering, performs chronological per-class splitting (preventing temporal
data leakage between consecutive snapshots), and fits TWO dedicated scalers:

  1. `models/scaler.pkl`    — Multi-class scaler (fit on all training classes)
                              → writes `train.csv`, `val.csv`, `test.csv` for MLP.
  2. `models/ae_scaler.pkl` — Normal-only scaler (fit ONLY on normal class train split)
                              → writes `ae_train.csv`, `ae_val.csv`, `ae_test.csv`
                              so the Autoencoder operates in true Normal Z-score space.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from heimdall.collector.snapshot import SystemSnapshot
from heimdall.config import settings
from heimdall.preprocessor.feature_engineer import extract_features
from heimdall.preprocessor.scaler import Scaler


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_snapshots_by_class(raw_dir: Path) -> dict[int, list[SystemSnapshot]]:
    """
    Load snapshots grouped and sorted chronologically per class index.
    """
    grouped: dict[int, list[SystemSnapshot]] = {}
    for label_idx, label in enumerate(settings.CLASS_NAMES):
        label_dir = raw_dir / label
        if not label_dir.is_dir():
            print(f"  [WARN] No directory found for label '{label}' at {label_dir} — skipping.")
            grouped[label_idx] = []
            continue

        files = sorted(label_dir.glob("*.json"))
        if not files:
            print(f"  [WARN] No snapshots found in {label_dir} — skipping.")
            grouped[label_idx] = []
            continue

        print(f"  Loading {len(files):>5} snapshots for class '{label}' …")
        snaps: list[SystemSnapshot] = []
        for f in files:
            try:
                data: dict[str, Any] = json.loads(f.read_text(encoding="utf-8"))
                snaps.append(SystemSnapshot.from_dict(data))
            except Exception as exc:
                print(f"  [WARN] Could not load {f.name}: {exc}")

        # Sort by ISO timestamp to preserve true temporal sequence
        snaps.sort(key=lambda s: s.timestamp)
        grouped[label_idx] = snaps

    return grouped


def _snaps_to_features(snaps: list[SystemSnapshot], label_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract features from a contiguous chronological sequence of snapshots.
    """
    if not snaps:
        return np.empty((0, settings.NUM_FEATURES), dtype=np.float32), np.empty((0,), dtype=np.int64)

    X_list: list[list[float]] = []
    for i, snap in enumerate(snaps):
        prev = snaps[i - 1] if i > 0 else None
        X_list.append(extract_features(snap, prev))

    X_arr = np.array(X_list, dtype=np.float32)
    y_arr = np.full((len(snaps),), label_idx, dtype=np.int64)
    return X_arr, y_arr


# ── Public API ────────────────────────────────────────────────────────────────

def build(
    raw_dir: Path | str | None = None,
    processed_dir: Path | str | None = None,
) -> dict[str, Path]:
    """
    Build chronological train/val/test CSV splits for both MLP and Autoencoder.
    """
    raw_dir       = Path(raw_dir)       if raw_dir       else settings.DATA_RAW_DIR
    processed_dir = Path(processed_dir) if processed_dir else settings.DATA_PROCESSED_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("── Dataset Builder (Chronological + Dual-Scaler) ───────────")
    print(f"   Raw data : {raw_dir}")
    print(f"   Output   : {processed_dir}")

    grouped = _load_snapshots_by_class(raw_dir)
    total_snaps = sum(len(v) for v in grouped.values())
    if total_snaps == 0:
        raise RuntimeError(
            f"No snapshots found in {raw_dir}. "
            "Run `python scripts/generate_realistic_data.py` or `bash scripts/collect_all.sh` first."
        )
    print(f"   Total snapshots loaded: {total_snaps}")

    # Chronological split within each class:
    # First 80% -> train, next 10% -> val, final 10% -> test
    train_X_parts, train_y_parts = [], []
    val_X_parts,   val_y_parts   = [], []
    test_X_parts,  test_y_parts  = [], []

    for label_idx, snaps in grouped.items():
        n = len(snaps)
        if n == 0:
            continue
        i_train = max(1, int(n * settings.TRAIN_RATIO))
        i_val   = max(i_train + 1, int(n * (settings.TRAIN_RATIO + settings.VAL_RATIO)))

        # Extract features independently per contiguous split so delta features don't leak across splits
        X_tr, y_tr = _snaps_to_features(snaps[:i_train], label_idx)
        X_va, y_va = _snaps_to_features(snaps[i_train:i_val], label_idx)
        X_te, y_te = _snaps_to_features(snaps[i_val:], label_idx)

        train_X_parts.append(X_tr); train_y_parts.append(y_tr)
        val_X_parts.append(X_va);   val_y_parts.append(y_va)
        test_X_parts.append(X_te);  test_y_parts.append(y_te)

    X_train = np.vstack(train_X_parts)
    y_train = np.concatenate(train_y_parts)
    X_val   = np.vstack(val_X_parts)
    y_val   = np.concatenate(val_y_parts)
    X_test  = np.vstack(test_X_parts)
    y_test  = np.concatenate(test_y_parts)

    # 1. Fit MLP multi-class scaler on all training classes
    mlp_scaler = Scaler(min_scale=1e-2)
    X_train_mlp = mlp_scaler.fit_transform(X_train)
    X_val_mlp   = mlp_scaler.transform(X_val)
    X_test_mlp  = mlp_scaler.transform(X_test)
    mlp_scaler_path = mlp_scaler.save(settings.MLP_SCALER_PATH)
    print(f"   MLP Scaler (all classes) saved to: {mlp_scaler_path}")

    # 2. Fit Autoencoder dedicated Normal-Only scaler ONLY on normal (label == 0) training rows
    X_train_normal = X_train[y_train == 0]
    if len(X_train_normal) == 0:
        raise RuntimeError("No 'normal' class samples found in training split to fit AE scaler.")

    ae_scaler = Scaler(min_scale=5e-2)
    ae_scaler.fit(X_train_normal)
    X_train_ae = ae_scaler.transform(X_train)
    X_val_ae   = ae_scaler.transform(X_val)
    X_test_ae  = ae_scaler.transform(X_test)
    ae_scaler_path = ae_scaler.save(settings.AE_SCALER_PATH)
    print(f"   AE Scaler (normal-only)  saved to: {ae_scaler_path}")

    def _to_df(X_scaled: np.ndarray, y_int: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame(X_scaled, columns=settings.FEATURE_NAMES)
        df["label"] = y_int
        return df

    splits: dict[str, Path] = {}
    # Save MLP splits
    for name, X_s, y_i in [
        ("train", X_train_mlp, y_train),
        ("val",   X_val_mlp,   y_val),
        ("test",  X_test_mlp,  y_test),
    ]:
        df   = _to_df(X_s, y_i)
        path = processed_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        splits[name] = path
        print(f"   {name:>8}.csv → {len(df):>5} rows  |  "
              + "  ".join(f"{settings.CLASS_NAMES[c]}={int((y_i==c).sum())}"
                          for c in range(settings.NUM_CLASSES) if int((y_i==c).sum()) > 0))

    # Save AE Normal-Z-Score splits (includes all classes in val/test for AE ROC-AUC evaluation!)
    for name, X_s, y_i in [
        ("ae_train", X_train_ae, y_train),
        ("ae_val",   X_val_ae,   y_val),
        ("ae_test",  X_test_ae,  y_test),
    ]:
        df   = _to_df(X_s, y_i)
        path = processed_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        splits[name] = path

    print("── Done ────────────────────────────────────────────────────")
    return splits

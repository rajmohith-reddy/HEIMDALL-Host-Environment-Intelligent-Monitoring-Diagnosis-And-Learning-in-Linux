"""
HEIMDALL Multi-Model AI Tournament
==================================
Runs an ablation benchmark across all AI model families under strict, honest
metrics without data leakage or circular synthetic evaluation.

Tournament Competitors:
1. Unsupervised Anomaly Detection:
   - Denoising Bottleneck Autoencoder (Deep Learning)
   - Isolation Forest (Tree Ensembles)
   - Robust Non-Parametric MAD Baseline (Statistical)
2. Supervised Diagnosis:
   - LightGBM (Gradient Boosted Trees)
   - PyTorch Multi-Layer Perceptron (Deep Neural Net)
   - Logistic Regression (Linear Baseline)

Usage
-----
    from heimdall.models.tournament import run_tournament
    results = run_tournament()
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from heimdall.analysis.calibration import compute_ece
from heimdall.config import settings
from heimdall.models.baseline.mad_baseline import MADDetector
from heimdall.preprocessor.scaler import Scaler

console = Console()


def run_tournament() -> dict[str, Any]:
    console.print("\n[bold cyan]╔══════════════════════════════════════════════════════════════╗[/bold cyan]")
    console.print("[bold cyan]║            HEIMDALL Multi-Model AI Tournament                ║[/bold cyan]")
    console.print("[bold cyan]╚══════════════════════════════════════════════════════════════╝[/bold cyan]\n")

    # 1. Load test datasets
    test_df = pd.read_csv(settings.DATA_PROCESSED_DIR / "test.csv")
    ae_test_df = pd.read_csv(settings.DATA_PROCESSED_DIR / "ae_test.csv")

    X_test_raw = test_df[settings.FEATURE_NAMES].values.astype(np.float32)
    y_test = test_df["label"].values.astype(int)

    # For unsupervised anomaly evaluation: label 0 is normal, labels 1..5 are anomalies
    y_binary_test = (y_test > 0).astype(int)

    # Scalers
    mlp_scaler = Scaler.load(settings.MLP_SCALER_PATH) if settings.MLP_SCALER_PATH.exists() else None
    ae_scaler = Scaler.load(settings.AE_SCALER_PATH) if settings.AE_SCALER_PATH.exists() else mlp_scaler

    X_test_scaled_mlp = mlp_scaler.transform(X_test_raw) if mlp_scaler else X_test_raw
    X_test_scaled_ae = ae_scaler.transform(X_test_raw) if ae_scaler else X_test_raw

    # ══════════════════════════════════════════════════════════════════════════
    # Part 1: Unsupervised Anomaly Detection Tournament
    # ══════════════════════════════════════════════════════════════════════════
    console.print("[bold yellow]► 1. Unsupervised Anomaly Detection Tournament[/bold yellow]")

    unsupervised_results = []

    # A. Isolation Forest
    iso_model_path = settings.MODELS_DIR / "isolation_forest.joblib"
    iso_meta_path = settings.MODELS_DIR / "iso_threshold.json"
    if iso_model_path.exists():
        iso_model = joblib.load(iso_model_path)
        with open(iso_meta_path) as f:
            iso_meta = json.load(f)
        thresh_iso = iso_meta["threshold"]

        t0 = time.perf_counter()
        raw_iso = -iso_model.score_samples(X_test_scaled_ae)
        iso_lat_us = ((time.perf_counter() - t0) / len(X_test_scaled_ae)) * 1e6

        preds_iso = (raw_iso > thresh_iso).astype(int)
        roc_iso = roc_auc_score(y_binary_test, raw_iso)
        f1_iso = f1_score(y_binary_test, preds_iso, zero_division=0)
        rec_iso = recall_score(y_binary_test, preds_iso, zero_division=0)
        # False Positive Rate on normal class (label == 0)
        norm_mask = y_binary_test == 0
        fpr_iso = np.mean(preds_iso[norm_mask]) if np.sum(norm_mask) > 0 else 0.0

        unsupervised_results.append({
            "model": "Isolation Forest (Ensemble Trees)",
            "roc_auc": roc_iso,
            "f1_score": f1_iso,
            "recall": rec_iso,
            "fpr_normal": fpr_iso,
            "latency_us": iso_lat_us,
        })

    # B. Denoising Deep Autoencoder
    if settings.AE_MODEL_PATH.exists() and settings.AE_THRESHOLD_PATH.exists():
        from heimdall.models.autoencoder.inference import detect as ae_detect

        scores_ae = []
        preds_ae = []
        t0 = time.perf_counter()
        for row in X_test_scaled_ae:
            res = ae_detect(row.tolist())
            scores_ae.append(res.hybrid_score)
            preds_ae.append(int(res.is_anomaly))
        ae_lat_us = ((time.perf_counter() - t0) / len(X_test_scaled_ae)) * 1e6

        scores_ae = np.array(scores_ae)
        preds_ae = np.array(preds_ae)
        roc_ae = roc_auc_score(y_binary_test, scores_ae)
        f1_ae = f1_score(y_binary_test, preds_ae, zero_division=0)
        rec_ae = recall_score(y_binary_test, preds_ae, zero_division=0)
        norm_mask = y_binary_test == 0
        fpr_ae = np.mean(preds_ae[norm_mask]) if np.sum(norm_mask) > 0 else 0.0

        unsupervised_results.append({
            "model": "Denoising Autoencoder (PyTorch NN)",
            "roc_auc": roc_ae,
            "f1_score": f1_ae,
            "recall": rec_ae,
            "fpr_normal": fpr_ae,
            "latency_us": ae_lat_us,
        })

    # C. Robust MAD Baseline
    mad_detector = MADDetector()
    scores_mad = []
    preds_mad = []
    t0 = time.perf_counter()
    for row in X_test_raw:
        res = mad_detector.evaluate(row)
        scores_mad.append(res["score"])
        preds_mad.append(int(res["is_anomaly"]))
    mad_lat_us = ((time.perf_counter() - t0) / len(X_test_raw)) * 1e6

    scores_mad = np.array(scores_mad)
    preds_mad = np.array(preds_mad)
    roc_mad = roc_auc_score(y_binary_test, scores_mad)
    f1_mad = f1_score(y_binary_test, preds_mad, zero_division=0)
    rec_mad = recall_score(y_binary_test, preds_mad, zero_division=0)
    norm_mask = y_binary_test == 0
    fpr_mad = np.mean(preds_mad[norm_mask]) if np.sum(norm_mask) > 0 else 0.0

    unsupervised_results.append({
        "model": "Robust MAD Baseline (Non-Parametric)",
        "roc_auc": roc_mad,
        "f1_score": f1_mad,
        "recall": rec_mad,
        "fpr_normal": fpr_mad,
        "latency_us": mad_lat_us,
    })

    # Render Unsupervised Table
    table_unsup = Table(title="Unsupervised Anomaly Detection Benchmark", show_header=True)
    table_unsup.add_column("Model / Architecture", style="cyan", no_wrap=True)
    table_unsup.add_column("ROC-AUC", justify="right")
    table_unsup.add_column("F1-Score", justify="right")
    table_unsup.add_column("Recall", justify="right")
    table_unsup.add_column("False Positive Rate", justify="right")
    table_unsup.add_column("Inference Latency", justify="right")

    for r in unsupervised_results:
        table_unsup.add_row(
            r["model"],
            f"{r['roc_auc']:.4f}",
            f"{r['f1_score']:.4f}",
            f"{r['recall'] * 100:.1f}%",
            f"{r['fpr_normal'] * 100:.2f}%",
            f"{r['latency_us']:.1f} µs/sample",
        )
    console.print(table_unsup)

    # ══════════════════════════════════════════════════════════════════════════
    # Part 2: Supervised Multi-Class Diagnosis Tournament
    # ══════════════════════════════════════════════════════════════════════════
    console.print("\n[bold yellow]► 2. Supervised Diagnosis Classifier Tournament[/bold yellow]")

    supervised_results = []

    # A. LightGBM
    lgb_model_path = settings.MODELS_DIR / "lightgbm_classifier.joblib"
    if lgb_model_path.exists():
        lgb_clf = joblib.load(lgb_model_path)
        t0 = time.perf_counter()
        lgb_probs = lgb_clf.predict_proba(X_test_scaled_mlp)
        lgb_lat_us = ((time.perf_counter() - t0) / len(X_test_scaled_mlp)) * 1e6
        lgb_preds = np.argmax(lgb_probs, axis=1)

        supervised_results.append({
            "model": "LightGBM (Gradient Boosted Trees)",
            "accuracy": accuracy_score(y_test, lgb_preds),
            "macro_f1": f1_score(y_test, lgb_preds, average="macro"),
            "log_loss": log_loss(y_test, lgb_probs),
            "ece": compute_ece(lgb_probs, y_test),
            "latency_us": lgb_lat_us,
        })

    # B. PyTorch Deep MLP
    if settings.MLP_MODEL_PATH.exists():
        from heimdall.analysis.calibration import CALIBRATOR_PATH, calibrate_logits
        from heimdall.models.mlp.model import MLPClassifier

        ckpt = torch.load(settings.MLP_MODEL_PATH, map_location="cpu", weights_only=False)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model = MLPClassifier(
            num_features=ckpt.get("num_features", settings.NUM_FEATURES) if isinstance(ckpt, dict) else settings.NUM_FEATURES,
            num_classes=ckpt.get("num_classes", settings.NUM_CLASSES) if isinstance(ckpt, dict) else settings.NUM_CLASSES,
            hidden_dims=ckpt.get("hidden_dims", settings.MLP_HIDDEN_DIMS) if isinstance(ckpt, dict) else settings.MLP_HIDDEN_DIMS,
            dropout_rates=ckpt.get("dropout_rates", settings.MLP_DROPOUT_RATES) if isinstance(ckpt, dict) else settings.MLP_DROPOUT_RATES,
        )
        model.load_state_dict(state_dict)
        model.eval()

        t0 = time.perf_counter()
        with torch.no_grad():
            raw_logits = model(torch.tensor(X_test_scaled_mlp, dtype=torch.float32)).numpy()
        mlp_lat_us = ((time.perf_counter() - t0) / len(X_test_scaled_mlp)) * 1e6

        # Calibrated probabilities via Temperature Scaling
        cal_probs = calibrate_logits(raw_logits, CALIBRATOR_PATH)
        mlp_preds = np.argmax(cal_probs, axis=1)

        supervised_results.append({
            "model": "PyTorch MLP (Calibrated with T-Scaling)",
            "accuracy": accuracy_score(y_test, mlp_preds),
            "macro_f1": f1_score(y_test, mlp_preds, average="macro"),
            "log_loss": log_loss(y_test, cal_probs),
            "ece": compute_ece(cal_probs, y_test),
            "latency_us": mlp_lat_us,
        })

    # C. Logistic Regression Baseline
    train_df = pd.read_csv(settings.DATA_PROCESSED_DIR / "train.csv")
    X_train_scaled = mlp_scaler.transform(train_df[settings.FEATURE_NAMES].values.astype(np.float32)) if mlp_scaler else train_df[settings.FEATURE_NAMES].values
    y_train = train_df["label"].values.astype(int)

    lr_clf = LogisticRegression(max_iter=300, random_state=settings.RANDOM_SEED)
    lr_clf.fit(X_train_scaled, y_train)

    t0 = time.perf_counter()
    lr_probs = lr_clf.predict_proba(X_test_scaled_mlp)
    lr_lat_us = ((time.perf_counter() - t0) / len(X_test_scaled_mlp)) * 1e6
    lr_preds = np.argmax(lr_probs, axis=1)

    supervised_results.append({
        "model": "Logistic Regression (Linear Baseline)",
        "accuracy": accuracy_score(y_test, lr_preds),
        "macro_f1": f1_score(y_test, lr_preds, average="macro"),
        "log_loss": log_loss(y_test, lr_probs),
        "ece": compute_ece(lr_probs, y_test),
        "latency_us": lr_lat_us,
    })

    # Render Supervised Table
    table_sup = Table(title="Supervised Classification & Calibration Benchmark", show_header=True)
    table_sup.add_column("Model / Architecture", style="green", no_wrap=True)
    table_sup.add_column("Accuracy", justify="right")
    table_sup.add_column("Macro-F1", justify="right")
    table_sup.add_column("Log Loss", justify="right")
    table_sup.add_column("ECE (Calibration)", justify="right")
    table_sup.add_column("Inference Latency", justify="right")

    for r in supervised_results:
        table_sup.add_row(
            r["model"],
            f"{r['accuracy'] * 100:.2f}%",
            f"{r['macro_f1']:.4f}",
            f"{r['log_loss']:.4f}",
            f"{r['ece']:.4f}",
            f"{r['latency_us']:.1f} µs/sample",
        )
    console.print(table_sup)

    # ══════════════════════════════════════════════════════════════════════════
    # Part 3: Tournament Verdict
    # ══════════════════════════════════════════════════════════════════════════
    best_unsup = max(unsupervised_results, key=lambda x: (x["roc_auc"], -x["fpr_normal"]))
    best_sup = max(supervised_results, key=lambda x: (x["macro_f1"], -x["ece"]))

    verdict_text = (
        f"★ [bold cyan]Best Anomaly Detector:[/bold cyan] [bold]{best_unsup['model']}[/bold]\n"
        f"   • ROC-AUC: {best_unsup['roc_auc']:.4f} | Recall: {best_unsup['recall']*100:.1f}% | FPR: {best_unsup['fpr_normal']*100:.2f}%\n"
        f"   • Latency: {best_unsup['latency_us']:.1f} µs/sample\n\n"
        f"★ [bold green]Best Classifier:[/bold green] [bold]{best_sup['model']}[/bold]\n"
        f"   • Macro-F1: {best_sup['macro_f1']:.4f} | Accuracy: {best_sup['accuracy']*100:.2f}%\n"
        f"   • Calibration ECE: {best_sup['ece']:.4f} (Lower is better) | Latency: {best_sup['latency_us']:.1f} µs/sample"
    )
    console.print(Panel(verdict_text, title="🏆 Tournament Verdict", border_style="bold gold1"))

    # Save tournament results to json
    results_path = settings.REPORTS_DIR / "tournament_results.json"
    full_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "unsupervised_tournament": unsupervised_results,
        "supervised_tournament": supervised_results,
        "best_unsupervised": best_unsup["model"],
        "best_supervised": best_sup["model"],
    }
    with open(results_path, "w") as f:
        json.dump(full_results, f, indent=2)

    console.print(f"[dim]Tournament results saved to {results_path}[/dim]\n")
    return full_results

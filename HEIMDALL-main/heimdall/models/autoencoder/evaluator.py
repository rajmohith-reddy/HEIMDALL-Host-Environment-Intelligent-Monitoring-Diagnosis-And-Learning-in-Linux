"""
Autoencoder Evaluator
======================
Evaluates the Production SystemAutoencoder on:
  1. The held-out HEIMDALL test set (`ae_test.csv` — normal vs. 5 anomaly classes)
  2. The real-world Server Machine Dataset (SMD) benchmark (`data/benchmarks/smd/`)
     if downloaded.

Outputs:
  • Binary anomaly detection metrics (Precision, Recall, F1, ROC-AUC)
  • Per-class anomaly detection rate (how many % of each anomaly class are caught)
  • Reconstruction error distribution histogram (`reports/ae_error_distribution.png`)
  • Markdown evaluation report (`reports/ae_evaluation.md`)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from heimdall.config import settings
from heimdall.models.autoencoder import inference as ae_inf


def evaluate(
    processed_dir: Path | str | None = None,
    reports_dir:   Path | str | None = None,
) -> dict[str, Any]:
    """
    Evaluate the saved Autoencoder on the test set and generate plots + report.
    """
    processed_dir = Path(processed_dir) if processed_dir else settings.DATA_PROCESSED_DIR
    reports_dir   = Path(reports_dir)   if reports_dir   else settings.REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("── Autoencoder Evaluator ───────────────────────────────────")
    ae_inf.reload()

    test_csv = processed_dir / "ae_test.csv"
    if not test_csv.exists():
        test_csv = processed_dir / "test.csv"

    df = pd.read_csv(test_csv)
    X_np   = df[settings.FEATURE_NAMES].values.astype(np.float32)
    y_cls  = df["label"].values.astype(np.int64)
    # Binary ground truth: 0 = Normal (label == 0), 1 = Anomaly (label > 0)
    y_true = (y_cls != 0).astype(np.int64)

    y_pred: list[int]       = []
    scores: list[float]     = []
    hybrids: list[float]    = []

    for row in X_np:
        res = ae_inf.detect(row.tolist())
        y_pred.append(1 if res.is_anomaly else 0)
        scores.append(res.anomaly_score)
        hybrids.append(res.hybrid_score)

    y_pred_np  = np.array(y_pred, dtype=np.int64)
    scores_np  = np.array(scores, dtype=np.float32)
    hybrids_np = np.array(hybrids, dtype=np.float32)

    acc  = float((y_pred_np == y_true).mean())
    prec = float(precision_score(y_true, y_pred_np, zero_division=0))
    rec  = float(recall_score(y_true, y_pred_np, zero_division=0))
    f1   = float(f1_score(y_true, y_pred_np, zero_division=0))
    try:
        roc_auc = float(roc_auc_score(y_true, hybrids_np))
    except ValueError:
        roc_auc = float("nan")

    report_str = classification_report(
        y_true, y_pred_np,
        target_names=["normal (0)", "anomaly (1..5)"],
        digits=4,
    )
    print("\n" + report_str)

    # Per-class recall / anomaly score breakdown
    print("   Per-Class Anomaly Detection Breakdown:")
    class_breakdown: list[dict[str, Any]] = []
    for c_idx, c_name in enumerate(settings.CLASS_NAMES):
        mask = (y_cls == c_idx)
        if not mask.any():
            continue
        flagged_rate = float(y_pred_np[mask].mean() * 100.0)
        mean_score   = float(scores_np[mask].mean())
        class_breakdown.append({
            "class": c_name,
            "samples": int(mask.sum()),
            "flagged_pct": round(flagged_rate, 1),
            "mean_anomaly_score": round(mean_score, 1),
        })
        print(f"     • {c_name:<17} (n={int(mask.sum()):>3}): "
              f"Flagged={flagged_rate:>5.1f}%  |  Mean Score={mean_score:>5.1f}/100")

    print(f"\n   Binary Accuracy  : {acc:.4f}")
    print(f"   Precision        : {prec:.4f}")
    print(f"   Recall           : {rec:.4f}")
    print(f"   F1-Score         : {f1:.4f}")
    print(f"   ROC-AUC          : {roc_auc:.4f}")

    # Optional: Check if SMD public benchmark metrics exist
    smd_metrics_path = settings.DATA_BENCHMARK_DIR / "smd_benchmark_results.json"
    smd_results = None
    if smd_metrics_path.exists():
        try:
            smd_results = json.loads(smd_metrics_path.read_text(encoding="utf-8"))
            print(f"\n   Public SMD Benchmark (Server Machine Dataset):")
            print(f"     • ROC-AUC={smd_results.get('roc_auc', 0):.4f}  "
                  f"F1={smd_results.get('f1', 0):.4f}  "
                  f"Precision={smd_results.get('precision', 0):.4f}  "
                  f"Recall={smd_results.get('recall', 0):.4f}")
        except Exception:
            pass

    _plot_distribution(scores_np, y_true, reports_dir)
    _write_markdown_report(
        report_str, acc, prec, rec, f1, roc_auc,
        class_breakdown, smd_results, reports_dir,
    )

    return {
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "roc_auc":   roc_auc,
        "breakdown": class_breakdown,
    }


def _plot_distribution(scores: np.ndarray, y_true: np.ndarray, reports_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    normal_scores  = scores[y_true == 0]
    anomaly_scores = scores[y_true == 1]

    ax.hist(normal_scores,  bins=25, alpha=0.75, label="Normal (Class 0)",       color="#10b981", edgecolor="black")
    ax.hist(anomaly_scores, bins=25, alpha=0.75, label="Anomalies (Classes 1–5)", color="#ef4444", edgecolor="black")
    ax.axvline(x=50.0, color="#f59e0b", linestyle="--", linewidth=2, label="Threshold Boundary (Score = 50)")

    ax.set_xlabel("Normalised Anomaly Score (0–100)", fontsize=11)
    ax.set_ylabel("Sample Count", fontsize=11)
    ax.set_title("HEIMDALL Autoencoder — Normal vs. Anomaly Score Distribution", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out = reports_dir / "ae_error_distribution.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"   Distribution plot → {out}")


def _write_markdown_report(
    report_str: str,
    acc: float,
    prec: float,
    rec: float,
    f1: float,
    roc_auc: float,
    breakdown: list[dict[str, Any]],
    smd_results: dict[str, Any] | None,
    reports_dir: Path,
) -> None:
    lines = [
        "# HEIMDALL Autoencoder — Anomaly Detection Evaluation\n\n",
        "## 1. Binary Classification Report (Normal vs. Anomaly)\n\n",
        "```\n", report_str, "```\n\n",
        f"- **Accuracy:** `{acc:.4f}`\n",
        f"- **Precision:** `{prec:.4f}`\n",
        f"- **Recall:** `{rec:.4f}`\n",
        f"- **F1-Score:** `{f1:.4f}`\n",
        f"- **ROC-AUC:** `{roc_auc:.4f}`\n\n",
        "## 2. Per-Class Anomaly Detection Breakdown\n\n",
        "| Operating Class | Test Samples | Flagged as Anomaly (%) | Mean Anomaly Score (0–100) |\n",
        "|-----------------|--------------|------------------------|----------------------------|\n",
    ]
    for item in breakdown:
        lines.append(
            f"| `{item['class']}` | {item['samples']} | {item['flagged_pct']:.1f}% | {item['mean_anomaly_score']:.1f} / 100 |\n"
        )

    if smd_results:
        lines += [
            "\n## 3. Public Server Machine Dataset (SMD) Benchmark Validation\n\n",
            f"- **Dataset:** `{smd_results.get('dataset', 'SMD machine-1-1')}` ({smd_results.get('n_test', 0)} test steps, {smd_results.get('n_features', 38)} server metrics)\n",
            f"- **ROC-AUC:** `{smd_results.get('roc_auc', 0):.4f}`\n",
            f"- **Point-Adjusted F1-Score:** `{smd_results.get('f1', 0):.4f}`\n",
            f"- **Precision:** `{smd_results.get('precision', 0):.4f}`\n",
            f"- **Recall:** `{smd_results.get('recall', 0):.4f}`\n",
        ]

    lines += [
        "\n## 4. Visualisations\n\n",
        "![Anomaly Score Distribution](ae_error_distribution.png)\n\n",
        "![AE Training Curve](ae_training_curve.png)\n",
    ]

    out = reports_dir / "ae_evaluation.md"
    out.write_text("".join(lines), encoding="utf-8")
    print(f"   Evaluation report → {out}")

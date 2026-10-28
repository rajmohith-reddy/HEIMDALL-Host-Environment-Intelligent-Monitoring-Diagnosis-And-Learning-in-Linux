"""
MLP Evaluator
=============
Loads the saved MLPClassifier checkpoint, runs it on the held-out test set,
and produces:
  - Confusion matrix (saved as PNG)
  - Per-class precision / recall / F1 (saved as Markdown report)
  - Overall accuracy and macro-averaged metrics
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from heimdall.config import settings
from heimdall.models.mlp.model import MLPClassifier


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_model(model_path: Path) -> MLPClassifier:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = MLPClassifier(
        num_features=checkpoint["num_features"],
        num_classes=checkpoint["num_classes"],
        hidden_dims=checkpoint["hidden_dims"],
        dropout_rates=checkpoint["dropout_rates"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _load_test(processed_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    df = pd.read_csv(processed_dir / "test.csv")
    X  = torch.tensor(df[settings.FEATURE_NAMES].values, dtype=torch.float32)
    y  = torch.tensor(df["label"].values,                dtype=torch.long)
    return X, y


# ── Main evaluate function ────────────────────────────────────────────────────

def evaluate(
    processed_dir: Path | str | None = None,
    model_path:    Path | str | None = None,
    reports_dir:   Path | str | None = None,
) -> dict:
    """
    Evaluate the saved MLP on the test set.

    Returns a dict with accuracy, macro F1, and per-class metrics.
    """
    processed_dir = Path(processed_dir) if processed_dir else settings.DATA_PROCESSED_DIR
    model_path    = Path(model_path)    if model_path    else settings.MLP_MODEL_PATH
    reports_dir   = Path(reports_dir)   if reports_dir   else settings.REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("── MLP Evaluator ───────────────────────────────────────────")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run `heimdall train` first."
        )

    model      = _load_model(model_path)
    X_test, y_true = _load_test(processed_dir)

    print(f"   Test samples: {len(y_true)}")
    print(f"   Model       : {model}")

    with torch.no_grad():
        probs  = model.predict_proba(X_test).numpy()   # (N, K)
        y_pred = probs.argmax(axis=1)

    y_true_np = y_true.numpy()

    # ── Classification report ─────────────────────────────────────────────
    report_str = classification_report(
        y_true_np, y_pred,
        target_names=settings.CLASS_NAMES,
        digits=4,
    )
    print("\n" + report_str)

    # ── ROC-AUC (macro, one-vs-rest) ─────────────────────────────────────
    try:
        roc_auc = roc_auc_score(y_true_np, probs, multi_class="ovr", average="macro")
    except ValueError:
        roc_auc = float("nan")

    # ── Confusion matrix ──────────────────────────────────────────────────
    cm = confusion_matrix(y_true_np, y_pred)
    _plot_confusion_matrix(cm, reports_dir)

    # ── Markdown report ───────────────────────────────────────────────────
    _write_markdown_report(report_str, roc_auc, cm, reports_dir)

    accuracy = float((y_pred == y_true_np).mean())
    print(f"\n   Overall Accuracy : {accuracy:.4f}")
    print(f"   Macro ROC-AUC    : {roc_auc:.4f}")
    print(f"\n   Reports saved to : {reports_dir}")

    return {
        "accuracy": accuracy,
        "roc_auc":  roc_auc,
        "report":   report_str,
        "cm":       cm.tolist(),
    }


def _plot_confusion_matrix(cm: np.ndarray, reports_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=settings.CLASS_NAMES,
        yticklabels=settings.CLASS_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label",      fontsize=11)
    ax.set_title("HEIMDALL MLP — Confusion Matrix", fontsize=13, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out = reports_dir / "mlp_confusion_matrix.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"   Confusion matrix → {out}")


def _write_markdown_report(
    report_str: str,
    roc_auc: float,
    cm: np.ndarray,
    reports_dir: Path,
) -> None:
    lines = [
        "# HEIMDALL MLP — Evaluation Report\n",
        "## Classification Report\n",
        "```\n", report_str, "```\n\n",
        f"**Macro ROC-AUC:** {roc_auc:.4f}\n\n",
        "## Confusion Matrix\n",
        "![Confusion Matrix](mlp_confusion_matrix.png)\n\n",
        "## Training Curves\n",
        "![Training Curves](mlp_training_curve.png)\n",
    ]
    out = reports_dir / "mlp_evaluation.md"
    out.write_text("".join(lines), encoding="utf-8")
    print(f"   Evaluation report → {out}")

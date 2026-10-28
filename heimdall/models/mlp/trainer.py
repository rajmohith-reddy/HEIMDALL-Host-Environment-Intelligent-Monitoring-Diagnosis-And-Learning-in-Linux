"""
MLP Trainer
============
Full supervised training loop for the MLPClassifier.

AI / NN Concepts Demonstrated
-------------------------------
• Backpropagation         — `loss.backward()` computes gradients for all params
• Adam optimiser          — adaptive moment estimation; first + second moment
                            running averages per parameter
• Categorical Cross-Entropy — training objective for multi-class classification
• Early Stopping          — halt training when val-loss stops improving
• ReduceLROnPlateau        — decays learning-rate when val-loss plateaus

Usage
-----
    from heimdall.models.mlp.trainer import train
    train()
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend to prevent GUI hangs in headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from heimdall.config import settings
from heimdall.models.mlp.model import MLPClassifier


# ── Dataset helpers ───────────────────────────────────────────────────────────

def _load_split(csv_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a CSV split into (X_tensor, y_tensor) on the active device."""
    df = pd.read_csv(csv_path)
    X  = torch.tensor(df[settings.FEATURE_NAMES].values, dtype=torch.float32)
    y  = torch.tensor(df["label"].values,                dtype=torch.long)
    return X, y


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    processed_dir: Path | str | None = None,
    model_path:    Path | str | None = None,
) -> dict[str, Any]:
    """
    Train the MLPClassifier and save the best checkpoint.

    Parameters
    ----------
    processed_dir : directory containing train.csv / val.csv
                    (default: settings.DATA_PROCESSED_DIR)
    model_path    : where to save the best model checkpoint
                    (default: settings.MLP_MODEL_PATH)

    Returns
    -------
    dict with training history (loss, accuracy per epoch) and best metrics.
    """
    processed_dir = Path(processed_dir) if processed_dir else settings.DATA_PROCESSED_DIR
    model_path    = Path(model_path)    if model_path    else settings.MLP_MODEL_PATH

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"── MLP Trainer ─────────────────────────────────────────────")
    print(f"   Device       : {device}")
    print(f"   Features     : {settings.NUM_FEATURES}")
    print(f"   Classes      : {settings.NUM_CLASSES} → {settings.CLASS_NAMES}")

    # 1. Load data
    X_train, y_train = _load_split(processed_dir / "train.csv")
    X_val,   y_val   = _load_split(processed_dir / "val.csv")

    train_ds = TensorDataset(X_train, y_train)
    val_ds   = TensorDataset(X_val,   y_val)

    train_loader = DataLoader(train_ds, batch_size=settings.MLP_BATCH_SIZE,
                              shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=settings.MLP_BATCH_SIZE,
                              shuffle=False, drop_last=False)

    print(f"   Train samples: {len(train_ds)}")
    print(f"   Val   samples: {len(val_ds)}")

    # 2. Build model
    model = MLPClassifier().to(device)
    print(f"   {model}")

    # 3. Loss, optimiser, scheduler
    criterion = nn.CrossEntropyLoss()           # Categorical Cross-Entropy
    optimizer = Adam(                           # Adam optimiser
        model.parameters(),
        lr=settings.MLP_LEARNING_RATE,
        weight_decay=settings.MLP_WEIGHT_DECAY,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=settings.MLP_LR_FACTOR,
        patience=settings.MLP_LR_PATIENCE,
    )

    # 4. Training loop
    history: dict[str, list[float]] = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "lr":         [],
    }

    best_val_loss   = float("inf")
    best_epoch      = 0
    patience_count  = 0
    best_state_dict = None

    print(f"\n{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9} {'LR':>10}")
    print("─" * 62)

    start_time = time.time()

    for epoch in range(1, settings.MLP_MAX_EPOCHS + 1):

        # ── Train phase ──────────────────────────────────────────────────────
        model.train()
        train_loss_sum = 0.0
        train_correct  = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()            # clear accumulated gradients

            logits = model(X_batch)          # forward pass
            loss   = criterion(logits, y_batch)  # compute cross-entropy loss

            loss.backward()                  # BACKPROPAGATION — compute ∂loss/∂params
            optimizer.step()                 # Adam parameter update step

            train_loss_sum += loss.item() * len(y_batch)
            preds = logits.argmax(dim=1)
            train_correct += (preds == y_batch).sum().item()

        train_loss = train_loss_sum / len(train_ds)
        train_acc  = train_correct  / len(train_ds)

        # ── Validation phase ─────────────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        val_correct  = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                logits  = model(X_batch)
                loss    = criterion(logits, y_batch)
                val_loss_sum += loss.item() * len(y_batch)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y_batch).sum().item()

        val_loss = val_loss_sum / len(val_ds)
        val_acc  = val_correct  / len(val_ds)

        current_lr = optimizer.param_groups[0]["lr"]

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(f"{epoch:>6} {train_loss:>11.4f} {train_acc:>9.2%} "
              f"{val_loss:>10.4f} {val_acc:>8.2%} {current_lr:>10.2e}")

        # ── LR scheduler step ────────────────────────────────────────────────
        scheduler.step(val_loss)

        # ── Early Stopping ───────────────────────────────────────────────────
        if val_loss < best_val_loss - 1e-6:
            best_val_loss   = val_loss
            best_epoch      = epoch
            patience_count  = 0
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= settings.MLP_PATIENCE:
                print(f"\n[Early Stopping] No improvement for {settings.MLP_PATIENCE} epochs. "
                      f"Best epoch: {best_epoch} (val_loss={best_val_loss:.4f})")
                break

    elapsed = time.time() - start_time
    print(f"\n── Training complete in {elapsed:.1f}s ─────────────────────────")

    # 5. Save the best checkpoint
    if best_state_dict is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": best_state_dict,
            "num_features":     settings.NUM_FEATURES,
            "num_classes":      settings.NUM_CLASSES,
            "hidden_dims":      settings.MLP_HIDDEN_DIMS,
            "dropout_rates":    settings.MLP_DROPOUT_RATES,
            "class_names":      settings.CLASS_NAMES,
            "best_epoch":       best_epoch,
            "best_val_loss":    best_val_loss,
        }
        torch.save(checkpoint, model_path)
        print(f"   Best model saved to: {model_path}")

    # 6. Plot training curves
    _plot_curves(history, best_epoch)

    return {
        "history":       history,
        "best_epoch":    best_epoch,
        "best_val_loss": best_val_loss,
    }


def _plot_curves(history: dict[str, list[float]], best_epoch: int) -> None:
    """Save loss and accuracy training curves to reports/."""
    settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("HEIMDALL MLP — Training Curves", fontsize=14, fontweight="bold")

    # Loss plot
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#4c8eda")
    ax1.plot(epochs, history["val_loss"],   label="Val Loss",   color="#e06c5a")
    ax1.axvline(x=best_epoch, color="gray", linestyle="--", alpha=0.7, label=f"Best epoch {best_epoch}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy plot
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Acc", color="#4c8eda")
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]],   label="Val Acc",   color="#e06c5a")
    ax2.axvline(x=best_epoch, color="gray", linestyle="--", alpha=0.7)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_path = settings.REPORTS_DIR / "mlp_training_curve.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"   Training curve saved to: {out_path}")

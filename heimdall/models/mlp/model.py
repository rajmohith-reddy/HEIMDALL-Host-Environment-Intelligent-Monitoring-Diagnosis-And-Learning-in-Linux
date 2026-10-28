"""
MLP Classifier — Model Definition
===================================
Implements a feed-forward Multi-Layer Perceptron for classifying Linux
system operating states.

Architecture
------------
    Input  (N features)
      ↓
    Linear(N → 128) → BatchNorm → ReLU → Dropout(0.30)
      ↓
    Linear(128 → 64) → BatchNorm → ReLU → Dropout(0.20)
      ↓
    Linear(64 → 32)  → BatchNorm → ReLU
      ↓
    Linear(32 → K classes)                 ← raw logits
      ↓
    Softmax                                 ← probability distribution
      ↓
    argmax → predicted class

The model returns raw logits from `forward()` so that
`torch.nn.CrossEntropyLoss` (which internally applies log-softmax) can be
used efficiently during training. The `predict_proba()` convenience method
applies Softmax for inference.

AI / NN Concepts Demonstrated
-------------------------------
• Feed-forward propagation     — data flows in one direction through layers
• ReLU activation              — non-linear hidden-layer activation
• Softmax activation           — converts logits to class probabilities
• Dropout regularisation       — randomly zeros hidden units during training
                                 to reduce co-adaptation / overfitting
• BatchNorm                    — stabilises and accelerates training
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from heimdall.config import settings


class MLPClassifier(nn.Module):
    """
    Feed-forward MLP for multi-class Linux system state classification.

    Parameters
    ----------
    num_features : int   — input feature vector dimension (default: settings.NUM_FEATURES)
    num_classes  : int   — number of output classes (default: settings.NUM_CLASSES)
    hidden_dims  : list  — hidden layer widths (default: settings.MLP_HIDDEN_DIMS)
    dropout_rates: list  — dropout probability per hidden layer
    """

    def __init__(
        self,
        num_features: int       = settings.NUM_FEATURES,
        num_classes: int        = settings.NUM_CLASSES,
        hidden_dims: list[int]  = settings.MLP_HIDDEN_DIMS,
        dropout_rates: list[float] = settings.MLP_DROPOUT_RATES,
    ) -> None:
        super().__init__()

        assert len(hidden_dims) == len(dropout_rates), (
            "hidden_dims and dropout_rates must have the same length."
        )

        self.num_features = num_features
        self.num_classes  = num_classes

        # ── Build hidden layers dynamically ──────────────────────────────────
        layers: list[nn.Module] = []
        in_dim = num_features

        for out_dim, drop_p in zip(hidden_dims, dropout_rates):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU(inplace=True))          # ReLU activation
            if drop_p > 0.0:
                layers.append(nn.Dropout(p=drop_p))       # Dropout regularisation
            in_dim = out_dim

        self.hidden = nn.Sequential(*layers)

        # ── Output layer ─────────────────────────────────────────────────────
        self.output = nn.Linear(in_dim, num_classes)

        # Weight initialisation (He / Kaiming — recommended with ReLU)
        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────────────
    def _init_weights(self) -> None:
        """Kaiming-normal init for all Linear layers; zeros for biases."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    # ── Forward pass ─────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feed-forward propagation.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, num_features)

        Returns
        -------
        torch.Tensor of shape (batch_size, num_classes)
            Raw logits (NOT softmax'd). Pass to CrossEntropyLoss during
            training, or call predict_proba() for probabilities.
        """
        h = self.hidden(x)           # pass through all hidden layers
        logits = self.output(h)      # final linear projection to class space
        return logits

    # ── Inference helpers ─────────────────────────────────────────────────────
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return Softmax class probabilities.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, num_features)
            or (num_features,) for a single sample

        Returns
        -------
        torch.Tensor of shape (batch_size, num_classes) — probabilities in [0,1]
        """
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)

        with torch.no_grad():
            self.eval()
            logits = self.forward(x)
            probs  = F.softmax(logits, dim=-1)   # Softmax activation

        return probs[0] if single else probs

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return the argmax class index (int64 tensor)."""
        return self.predict_proba(x).argmax(dim=-1)

    # ── Utility ───────────────────────────────────────────────────────────────
    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:  # noqa: D105
        params = self.count_parameters()
        return (
            f"MLPClassifier("
            f"in={self.num_features}, "
            f"hidden={settings.MLP_HIDDEN_DIMS}, "
            f"out={self.num_classes}, "
            f"params={params:,})"
        )

"""
Autoencoder — Production Model Definition
=========================================
Implements a Denoising Symmetric Encoder-Decoder architecture with LayerNorm,
GELU activations, and an unbounded linear latent bottleneck trained exclusively
on normal-scaled system snapshots (`ae_scaler.pkl`).

Key Production Enhancements
---------------------------
1. Denoising Input Regularisation: Injects mild Gaussian noise during training
   so the network learns true cross-feature telemetry correlations rather than
   memorizing identity mappings.
2. LayerNorm + GELU Hidden Blocks: Prevents dead-ReLU collapse on negative
   Z-scores and stabilises gradient flow.
3. Unbounded Linear Latent Bottleneck (8D): Preserves full sign and magnitude
   in latent space so Mahalanobis distance from the normal cluster center μ_z
   can be measured alongside reconstruction error.
4. Per-Feature Reconstruction Error (`per_feature_error`): Exposes individual
   feature squared errors (batch, 40) to prevent 40-feature dilution and enable
   root-cause feature attribution (XAI).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from heimdall.config import settings


class SystemAutoencoder(nn.Module):
    """
    Denoising Autoencoder for unsupervised anomaly detection and root-cause
    attribution on Linux telemetry.
    """

    def __init__(
        self,
        num_features: int      = settings.NUM_FEATURES,
        hidden_dims: list[int] = settings.AE_HIDDEN_DIMS,
        latent_dim: int        = settings.AE_LATENT_DIM,
        noise_std: float       = settings.AE_NOISE_STD,
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.hidden_dims  = list(hidden_dims)
        self.latent_dim   = latent_dim
        self.noise_std    = noise_std

        # ── Encoder ───────────────────────────────────────────────────────────
        encoder_layers: list[nn.Module] = []
        in_dim = num_features
        for out_dim in self.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, out_dim))
            encoder_layers.append(nn.LayerNorm(out_dim))
            encoder_layers.append(nn.GELU())
            in_dim = out_dim
        # Linear (unbounded) latent bottleneck — no ReLU clipping!
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # ── Decoder (mirror of encoder) ───────────────────────────────────────
        decoder_layers: list[nn.Module] = []
        in_dim = latent_dim
        for out_dim in reversed(self.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, out_dim))
            decoder_layers.append(nn.LayerNorm(out_dim))
            decoder_layers.append(nn.GELU())
            in_dim = out_dim
        # Final linear projection back to normal Z-score space
        decoder_layers.append(nn.Linear(in_dim, num_features))
        self.decoder = nn.Sequential(*decoder_layers)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    # ── Forward & Latent helpers ──────────────────────────────────────────────
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the unbounded 8D latent representation of *x*."""
        return self.encoder(x)

    def forward_with_latent(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode (with optional training denoising) then decode *x*, returning (x_hat, z).
        """
        if self.training and self.noise_std > 0.0:
            x_in = x + torch.randn_like(x) * self.noise_std
        else:
            x_in = x
        z     = self.encoder(x_in)
        x_hat = self.decoder(z)
        return x_hat, z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass returning reconstruction x_hat."""
        x_hat, _ = self.forward_with_latent(x)
        return x_hat

    def per_feature_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-feature squared reconstruction error `(x - x_hat) ** 2`.

        Returns
        -------
        torch.Tensor of shape (batch, num_features)
        """
        x_hat = self.forward(x)
        return (x - x_hat) ** 2

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute mean MSE reconstruction error per sample.

        Returns
        -------
        torch.Tensor of shape (batch,)
        """
        return self.per_feature_error(x).mean(dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"SystemAutoencoder("
            f"in={self.num_features}, "
            f"hidden={self.hidden_dims}, "
            f"latent={self.latent_dim}, "
            f"params={self.count_parameters():,})"
        )

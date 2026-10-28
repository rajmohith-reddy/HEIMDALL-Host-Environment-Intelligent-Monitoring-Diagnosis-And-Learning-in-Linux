"""
Scaler
======
Wraps scikit-learn's `StandardScaler` with variance flooring and a
cached save/load interface for both the multi-class MLP scaler (`scaler.pkl`)
and the dedicated Normal-Only Autoencoder scaler (`ae_scaler.pkl`).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from heimdall.config import settings


_cached_scalers: dict[str, "Scaler"] = {}


# Physical minimum standard deviation floors (domain-anchored to real Linux telemetry)
FEATURE_MIN_SCALE: dict[str, float] = {
    # CPU (9)
    "cpu_percent_mean":         15.0,   # at least 15% CPU std
    "cpu_percent_max":          25.0,   # at least 25% CPU max std
    "cpu_percent_std":          5.0,    # at least 5% CPU std dev
    "cpu_freq_mhz":             150.0,  # at least 150 MHz frequency fluctuation
    "load_avg_1":               0.50,   # at least 0.5 load units
    "load_avg_5":               0.40,
    "load_avg_15":              0.30,
    "cpu_ctx_switches_per_sec": 800.0,  # at least 800 context switches/s
    "cpu_interrupts_per_sec":   400.0,  # at least 400 interrupts/s
    # Memory (5)
    "mem_used_pct":             3.0,    # at least 3% RAM
    "mem_available_mb":         250.0,  # at least 250 MB
    "swap_used_pct":            3.0,    # at least 3% swap
    "swap_in_per_sec":          2.0,    # at least 2 pages/s
    "swap_out_per_sec":         2.0,
    # Disk (5)
    "disk_read_mbps":           2.5,    # at least 2.5 MB/s
    "disk_write_mbps":          3.0,    # at least 3.0 MB/s
    "disk_read_iops":           25.0,   # at least 25 IOPS
    "disk_write_iops":          30.0,
    "disk_used_pct":            2.0,    # at least 2% disk fill
    # Network (10)
    "net_bytes_sent_mbps":      2.0,    # at least 2.0 MB/s
    "net_bytes_recv_mbps":      2.5,    # at least 2.5 MB/s
    "net_packets_sent_per_sec": 200.0,  # at least 200 pkts/s
    "net_packets_recv_per_sec": 250.0,
    "net_errin_per_sec":        1.0,
    "net_errout_per_sec":       1.0,
    "net_dropin_per_sec":       2.0,
    "net_dropout_per_sec":      2.0,
    "tcp_connections":          35.0,   # at least 35 sockets
    "udp_connections":          15.0,
    # Process / System (7)
    "num_processes":            45.0,   # at least 45 procs
    "num_threads":              350.0,  # at least 350 threads
    "num_zombies":              2.0,
    "open_files":               150.0,  # realistic physical variance floor for host FDs
    "logged_in_users":          1.0,
    "failed_services":          2.0,
    "auth_failures_60s":        2.0,
    # Temporal deltas (4)
    "delta_cpu_percent":        15.0,   # at least 15% CPU rate change between snapshots
    "delta_mem_used_pct":       5.0,    # at least 5% RAM change
    "delta_disk_read_mbps":     12.0,   # at least 12.0 MB/s disk rate change
    "delta_net_recv_mbps":      12.0,   # at least 12.0 MB/s network rate change
}


class Scaler:
    """
    StandardScaler wrapper with physical domain-specific scale flooring to prevent
    routine fluctuations (e.g., a 1% CPU jitter or 0.5 MB/s disk read) from
    producing inflated Z-scores.
    """

    def __init__(self, min_scale: float = 1e-2) -> None:
        self._scaler: StandardScaler = StandardScaler()
        self._fitted: bool = False
        self.min_scale: float = min_scale

    # ── Fit ───────────────────────────────────────────────────────────────────
    def fit(self, X: Any) -> "Scaler":
        """
        Fit the scaler on training data and apply domain-anchored scale floors.
        """
        arr = np.array(X, dtype=np.float32)
        self._scaler.fit(arr)

        # Enforce physically grounded floors when fitting HEIMDALL 40-feature telemetry
        if arr.shape[1] == len(settings.FEATURE_NAMES):
            phys_floors = np.array(
                [FEATURE_MIN_SCALE.get(fname, self.min_scale) for fname in settings.FEATURE_NAMES],
                dtype=np.float32,
            )
        else:
            phys_floors = np.full((arr.shape[1],), self.min_scale, dtype=np.float32)

        mean_abs = np.abs(self._scaler.mean_)
        floor = np.maximum(phys_floors, 0.05 * mean_abs)
        self._scaler.scale_ = np.maximum(self._scaler.scale_, floor)
        self._scaler.var_ = self._scaler.scale_ ** 2

        self._fitted = True
        return self

    # ── Transform ─────────────────────────────────────────────────────────────
    def transform(self, X: Any) -> np.ndarray:
        """
        Scale *X* using the fitted parameters.
        """
        if not self._fitted:
            raise RuntimeError("Scaler has not been fitted yet. Call fit() first.")

        arr = np.array(X, dtype=np.float32)
        single = arr.ndim == 1
        if single:
            arr = arr.reshape(1, -1)

        scaled = self._scaler.transform(arr).astype(np.float32)
        # Clip extreme Z-scores at ±25σ to keep neural activations numerically stable
        scaled = np.clip(scaled, -25.0, 25.0)
        return scaled[0] if single else scaled

    def fit_transform(self, X: Any) -> np.ndarray:
        """Fit and then transform in one call."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: Any) -> np.ndarray:
        """Invert scaling back to original domain units."""
        if not self._fitted:
            raise RuntimeError("Scaler has not been fitted yet. Call fit() first.")
        arr = np.array(X, dtype=np.float32)
        single = arr.ndim == 1
        if single:
            arr = arr.reshape(1, -1)
        orig = self._scaler.inverse_transform(arr).astype(np.float32)
        return orig[0] if single else orig


    # ── Mean imputation (replaces NaN with feature mean) ─────────────────────
    def impute(self, X: Any) -> np.ndarray:
        """Replace NaN values with the column mean from training data."""
        if not self._fitted:
            raise RuntimeError("Scaler must be fitted before imputation.")

        arr = np.array(X, dtype=np.float32)
        nan_mask = np.isnan(arr)
        if nan_mask.any():
            arr = arr.copy()
            arr[nan_mask] = np.take(
                self._scaler.mean_,
                np.where(nan_mask)[0] % len(self._scaler.mean_),
            )
        return arr

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self, path: Path | str | None = None) -> Path:
        """Pickle the fitted scaler to disk."""
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted scaler.")

        save_path = Path(path) if path else settings.MLP_SCALER_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("wb") as fh:
            pickle.dump(self._scaler, fh)

        # Invalidate cache for this path
        _cached_scalers.pop(str(save_path.resolve()), None)
        return save_path

    @classmethod
    def load(cls, path: Path | str | None = None, use_cache: bool = True) -> "Scaler":
        """Load a previously saved scaler from disk (cached in memory by default)."""
        load_path = Path(path) if path else settings.MLP_SCALER_PATH
        key = str(load_path.resolve())

        if use_cache and key in _cached_scalers:
            return _cached_scalers[key]

        if not load_path.exists():
            raise FileNotFoundError(
                f"Scaler not found at {load_path}. "
                "Run `heimdall build-dataset` first to generate it."
            )
        instance = cls()
        with load_path.open("rb") as fh:
            instance._scaler = pickle.load(fh)
        instance._fitted = True

        if use_cache:
            _cached_scalers[key] = instance
        return instance

    @classmethod
    def reload_cache(cls) -> None:
        """Clear in-memory cached scalers."""
        _cached_scalers.clear()

    @property
    def mean_(self) -> np.ndarray:
        return self._scaler.mean_  # type: ignore[return-value]

    @property
    def scale_(self) -> np.ndarray:
        return self._scaler.scale_  # type: ignore[return-value]

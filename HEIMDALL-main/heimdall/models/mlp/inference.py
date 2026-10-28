"""
MLP Inference
=============
Lightweight inference wrapper: loads the checkpoint once (cached) and
exposes a simple `classify()` function for use by the analysis engine.

Returns
-------
InferenceResult namedtuple:
    predicted_class : str   — e.g. "cpu_stress"
    class_index     : int   — e.g. 1
    confidence      : float — probability of predicted class (0.0–1.0)
    probabilities   : dict  — {class_name: probability} for all classes
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from heimdall.config import settings
from heimdall.models.mlp.model import MLPClassifier


# ── Cached model ──────────────────────────────────────────────────────────────
_model: MLPClassifier | None = None
_device: torch.device = torch.device("cpu")


def _get_model(model_path: Path | None = None) -> MLPClassifier:
    """Load and cache the model. Only reads from disk on first call."""
    global _model, _device
    if _model is not None:
        return _model

    path = Path(model_path) if model_path else settings.MLP_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"MLP model not found at {path}. Run `heimdall train` first."
        )

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=_device, weights_only=False)
    _model = MLPClassifier(
        num_features=checkpoint["num_features"],
        num_classes=checkpoint["num_classes"],
        hidden_dims=checkpoint["hidden_dims"],
        dropout_rates=checkpoint["dropout_rates"],
    )
    _model.load_state_dict(checkpoint["model_state_dict"])
    _model.to(_device)
    _model.eval()
    return _model


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class MLPResult:
    predicted_class: str
    class_index:     int
    confidence:      float
    probabilities:   dict[str, float]   # {class_name: prob}


# ── Public API ────────────────────────────────────────────────────────────────

def classify(
    feature_vector: list[float],
    model_path: Path | None = None,
) -> MLPResult:
    """
    Classify a single feature vector.

    Parameters
    ----------
    feature_vector : list[float] of length settings.NUM_FEATURES (already scaled)
    model_path     : override path to .pt checkpoint

    Returns
    -------
    MLPResult with class name, index, confidence, and full probability dict
    """
    model = _get_model(model_path)

    x = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).to(_device)
    with torch.no_grad():
        logits = model(x).cpu().numpy()

    # Temperature Scaling (post-hoc probability calibration)
    try:
        from heimdall.analysis.calibration import CALIBRATOR_PATH, calibrate_logits
        probs = calibrate_logits(logits, CALIBRATOR_PATH)[0].tolist()
    except Exception:
        probs = model.predict_proba(x).squeeze(0).tolist()

    class_index     = int(max(range(len(probs)), key=lambda i: probs[i]))
    confidence      = probs[class_index]
    predicted_class = settings.CLASS_NAMES[class_index]
    prob_dict       = {name: round(p, 6) for name, p in zip(settings.CLASS_NAMES, probs)}

    return MLPResult(
        predicted_class=predicted_class,
        class_index=class_index,
        confidence=round(confidence, 6),
        probabilities=prob_dict,
    )


def reload() -> None:
    """Force a model reload from disk on the next `classify()` call."""
    global _model
    _model = None

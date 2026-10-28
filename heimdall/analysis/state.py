"""
System State
=============
Defines `SystemState` — the fully-analysed, annotated output of one
HEIMDALL analysis cycle. This is the central data structure passed between
the analysis engine, the reporter, and the report generator.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from heimdall.collector.snapshot import SystemSnapshot


# Severity levels (ordered: OK < WARNING < CRITICAL)
SeverityLevel = Literal["OK", "WARNING", "CRITICAL"]
_SEVERITY_ORDER = {"OK": 0, "WARNING": 1, "CRITICAL": 2}


def max_severity(a: SeverityLevel, b: SeverityLevel) -> SeverityLevel:
    """Return the higher of two severity levels."""
    return a if _SEVERITY_ORDER[a] >= _SEVERITY_ORDER[b] else b


@dataclass
class SystemState:
    """
    The complete analysed state of the system at one moment in time.
    """

    timestamp:              str
    snapshot:               "SystemSnapshot"
    # MLP
    mlp_class:              str                  = "unknown"
    mlp_confidence:         float                = 0.0
    mlp_probabilities:      dict[str, float]     = field(default_factory=dict)
    # Autoencoder + XAI Attribution
    is_anomaly:             bool                 = False
    anomaly_score:          float                = 0.0
    ae_raw_error:           float                = 0.0
    ae_threshold:           float                = 0.0
    ae_hybrid_score:        float                = 0.0
    ae_latent_distance:     float                = 0.0
    subsystem_scores:       dict[str, float]     = field(default_factory=dict)
    top_anomalous_features: list[dict[str, Any]] = field(default_factory=list)
    dominant_subsystem:     str                  = "none"
    # Extended Multi-Model & Audit Fields (v2)
    status_state:           str                  = "NORMAL"
    decision_source:        str                  = "baseline_consensus"
    open_set_status:        str                  = "known_pattern"
    hardware_impact:        str                  = "LOW"
    iso_forest_score:       float                = 0.0
    suggested_triage_command: str               = ""
    # Analysis output
    severity:               SeverityLevel        = "OK"
    triggered_rules:        list[str]            = field(default_factory=list)
    diagnosis:              list[str]            = field(default_factory=list)
    recommendations:        list[str]            = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["snapshot"] = self.snapshot.to_dict()
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    # ── Convenience accessors ─────────────────────────────────────────────────

    @property
    def is_ok(self) -> bool:
        return self.severity == "OK"

    @property
    def is_warning(self) -> bool:
        return self.severity == "WARNING"

    @property
    def is_critical(self) -> bool:
        return self.severity == "CRITICAL"

    @property
    def severity_color(self) -> str:
        """ANSI escape color for the severity level."""
        return {"OK": "green", "WARNING": "yellow", "CRITICAL": "red"}[self.severity]

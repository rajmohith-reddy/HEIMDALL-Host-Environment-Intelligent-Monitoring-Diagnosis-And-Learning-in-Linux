"""
Markdown Report Generator
==========================
Generates a structured incident report in Markdown format and saves it to
reports/heimdall_report_<timestamp>.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from heimdall.config import settings

if TYPE_CHECKING:
    from heimdall.analysis.state import SystemState


def generate(state: "SystemState", output_dir: Path | str | None = None) -> Path:
    """
    Write a Markdown incident report for *state*.
    """
    output_dir = Path(output_dir) if output_dir else settings.REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    snap = state.snapshot
    c, m, d, n, p, s = snap.cpu, snap.memory, snap.disk, snap.network, snap.process, snap.system

    lines: list[str] = [
        "# HEIMDALL — System Intelligence Report\n\n",
        f"**Generated:** {state.timestamp}  \n",
        f"**Overall Severity:** `{state.severity}`\n\n",
        "---\n\n",

        "## 1. System Snapshot Summary\n\n",
        "### CPU\n",
        "| Metric | Value |\n|--------|-------|\n",
        f"| Mean CPU Utilisation | {c.get('cpu_percent_mean', 0):.2f}% |\n",
        f"| Max Core Utilisation | {c.get('cpu_percent_max', 0):.2f}% |\n",
        f"| Load Average (1m)    | {c.get('load_avg_1', 0):.2f} |\n",
        f"| Load Average (5m)    | {c.get('load_avg_5', 0):.2f} |\n",
        f"| Load Average (15m)   | {c.get('load_avg_15', 0):.2f} |\n",
        f"| CPU Freq             | {c.get('cpu_freq_mhz', 0):.0f} MHz |\n",
        f"| Logical CPUs         | {c.get('num_logical_cpus', 0)} |\n\n",

        "### Memory\n",
        "| Metric | Value |\n|--------|-------|\n",
        f"| RAM Used          | {m.get('mem_used_pct', 0):.2f}% |\n",
        f"| RAM Available     | {m.get('mem_available_mb', 0):.0f} MB |\n",
        f"| Swap Used         | {m.get('swap_used_pct', 0):.2f}% |\n",
        f"| Swap-out rate     | {m.get('swap_out_per_sec', 0):.2f} pages/s |\n\n",

        "### Disk\n",
        "| Metric | Value |\n|--------|-------|\n",
        f"| Root Disk Used | {d.get('disk_used_pct', 0):.2f}% |\n",
        f"| Read           | {d.get('disk_read_mbps', 0):.3f} MB/s |\n",
        f"| Write          | {d.get('disk_write_mbps', 0):.3f} MB/s |\n",
        f"| Read IOPS      | {d.get('disk_read_iops', 0):.1f} |\n",
        f"| Write IOPS     | {d.get('disk_write_iops', 0):.1f} |\n\n",

        "### Network\n",
        "| Metric | Value |\n|--------|-------|\n",
        f"| RX              | {n.get('net_bytes_recv_mbps', 0):.4f} MB/s |\n",
        f"| TX              | {n.get('net_bytes_sent_mbps', 0):.4f} MB/s |\n",
        f"| TCP Connections | {n.get('tcp_connections', 0)} |\n",
        f"| Packet Drops/s  | {n.get('net_dropin_per_sec', 0):.2f} |\n\n",

        "### Processes & System\n",
        "| Metric | Value |\n|--------|-------|\n",
        f"| Total Processes    | {p.get('num_processes', 0)} |\n",
        f"| Total Threads      | {p.get('num_threads', 0)} |\n",
        f"| Zombie Procs       | {p.get('num_zombies', 0)} |\n",
        f"| Open Files         | {p.get('open_files', 0)} |\n",
        f"| Failed Services    | {s.get('failed_services', 0)} |\n",
        f"| Auth Failures/60s  | {s.get('auth_failures_60s', 0)} |\n",
        f"| Logged-in Users    | {s.get('logged_in_users', 0)} |\n\n",

        "---\n\n",

        "## 2. AI & XAI Model Output\n\n",
        "### MLP Classifier\n\n",
        "| Property | Value |\n|----------|-------|\n",
        f"| Effective Class | `{state.mlp_class}` |\n",
        f"| Confidence      | {state.mlp_confidence * 100:.2f}% |\n\n",

        "**Full Class Probabilities:**\n\n",
        "| Class | Probability |\n|-------|-------------|\n",
    ]

    for cls_name, prob in state.mlp_probabilities.items():
        lines.append(f"| `{cls_name}` | {prob * 100:.3f}% |\n")

    lines += [
        "\n",
        "### Production Autoencoder Anomaly Detection\n\n",
        "| Property | Value |\n|----------|-------|\n",
        f"| Anomaly Detected   | {'✓ YES' if state.is_anomaly else '✗ No'} |\n",
        f"| Anomaly Score      | {state.anomaly_score:.2f} / 100 |\n",
        f"| Hybrid Statistic   | {state.ae_hybrid_score:.4f} |\n",
        f"| Threshold (τ)      | {state.ae_threshold:.4f} |\n",
        f"| Raw MSE Error      | {state.ae_raw_error:.6f} |\n",
        f"| Dominant Subsystem | `{state.dominant_subsystem}` |\n\n",
    ]

    if state.subsystem_scores:
        lines += [
            "**Subsystem Anomaly Breakdown:**\n\n",
            "| Subsystem | Anomaly Score (0–100) |\n|-----------|-----------------------|\n",
        ]
        for sub_k, sub_v in state.subsystem_scores.items():
            lines.append(f"| `{sub_k}` | {sub_v:.1f} / 100 |\n")
        lines.append("\n")

    if state.top_anomalous_features:
        lines += [
            "**Top-5 Root-Cause Features (Explainable AI Attribution):**\n\n",
            "| Feature | Subsystem | Excess Z-Error (σ) | Contribution (%) |\n",
            "|---------|-----------|--------------------|------------------|\n",
        ]
        for f in state.top_anomalous_features:
            lines.append(
                f"| `{f['feature']}` | `{f['subsystem']}` | {f['z_error']:.2f}σ | {f['contribution_pct']:.1f}% |\n"
            )
        lines.append("\n")

    lines += [
        "---\n\n",
        "## 3. Rules Engine\n\n",
    ]

    if state.triggered_rules:
        for rule_msg in state.triggered_rules:
            lines.append(f"- ⚠️ {rule_msg}\n")
    else:
        lines.append("No rule violations detected.\n")

    lines += [
        "\n---\n\n",
        "## 4. Diagnosis\n\n",
    ]
    for sentence in state.diagnosis:
        lines.append(f"- {sentence}\n")

    lines += [
        "\n---\n\n",
        "## 5. Recommendations\n\n",
    ]
    for i, rec in enumerate(state.recommendations, 1):
        lines.append(f"{i}. {rec}\n")

    lines.append("\n---\n*Generated by HEIMDALL — Host Environment Intelligence for Monitoring, Diagnostics and Anomaly Learning*\n")

    safe_ts   = state.timestamp.replace(":", "-").replace("+", "p")[:23]
    out_path  = output_dir / f"heimdall_report_{safe_ts}.md"
    out_path.write_text("".join(lines), encoding="utf-8")
    return out_path

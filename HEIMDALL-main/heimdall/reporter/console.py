"""
Console Reporter
================
Rich-powered terminal output for HEIMDALL. Renders a colour-coded panel
showing:
  • System metrics summary (CPU / memory / disk / network at a glance)
  • MLP classification result with confidence bar
  • Autoencoder hybrid anomaly score + per-subsystem breakdown
  • Explainable AI (XAI) Top-5 root-cause telemetry features
  • Triggered rules (if any)
  • Diagnosis and recommendations
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from heimdall.analysis.state import SystemState

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

console = Console(highlight=False, force_interactive=False)

# ── Colour constants ──────────────────────────────────────────────────────────
_SEVERITY_STYLE = {
    "OK":       "bold green",
    "WARNING":  "bold yellow",
    "CRITICAL": "bold red",
}

_CLASS_STYLE = {
    "normal":          "green",
    "cpu_stress":      "red",
    "memory_pressure": "orange3",
    "disk_pressure":   "yellow",
    "network_flood":   "magenta",
    "system_anomaly":  "bold red",
    "unknown":         "dim",
}


# ── Gauge helper ──────────────────────────────────────────────────────────────

def _gauge(label: str, value: float, total: float = 100.0, warn: float = 70, crit: float = 90, unit: str = "%") -> str:
    """Return a mini text gauge like '[████░░░░]  42.3%'."""
    pct = min(value / max(total, 1e-9), 1.0)
    filled = int(pct * 20)
    bar = "█" * filled + "░" * (20 - filled)

    if (value / max(total, 1e-9)) * 100 >= crit:
        colour = "red"
    elif (value / max(total, 1e-9)) * 100 >= warn:
        colour = "yellow"
    else:
        colour = "green"

    return f"[{colour}][{bar}][/{colour}] {value:.1f}{unit}"


# ── Main render function ──────────────────────────────────────────────────────

def render(state: "SystemState", clear: bool = False) -> None:
    """
    Render a full HEIMDALL analysis panel to the terminal.
    """
    if clear:
        console.clear()

    severity_style = _SEVERITY_STYLE[state.severity]

    # ── Header ────────────────────────────────────────────────────────────────
    console.print(Rule(
        f"[bold cyan]HEIMDALL[/bold cyan]  [{severity_style}]● {state.severity}[/{severity_style}]  "
        f"[dim]{state.timestamp}[/dim]",
        style="dim"
    ))

    # ── Metrics summary table ─────────────────────────────────────────────────
    snap = state.snapshot
    c, m, d, n, p, s = snap.cpu, snap.memory, snap.disk, snap.network, snap.process, snap.system

    metrics = Table(box=box.SIMPLE, expand=True, show_header=True, header_style="bold cyan")
    metrics.add_column("Subsystem",  style="bold", width=14)
    metrics.add_column("Metric",     style="dim",  width=22)
    metrics.add_column("Value")

    # CPU
    cpu_pct  = c.get("cpu_percent_mean", 0)
    load1    = c.get("load_avg_1", 0)
    num_cpus = c.get("num_logical_cpus", 1)
    metrics.add_row("CPU",    "Utilisation",    _gauge("CPU", cpu_pct))
    metrics.add_row("",       "Load avg (1m)",  f"{load1:.2f}  [dim](/{num_cpus} CPUs)[/dim]")
    metrics.add_row("",       "Freq",           f"{c.get('cpu_freq_mhz', 0):.0f} MHz")

    # Memory
    mem_pct   = m.get("mem_used_pct", 0)
    avail_mb  = m.get("mem_available_mb", 0)
    swap_pct  = m.get("swap_used_pct", 0)
    metrics.add_row("Memory", "Used",           _gauge("RAM", mem_pct))
    metrics.add_row("",       "Available",      f"{avail_mb:.0f} MB")
    metrics.add_row("",       "Swap used",      _gauge("Swap", swap_pct, warn=40, crit=70))

    # Disk
    disk_pct   = d.get("disk_used_pct", 0)
    read_mbps  = d.get("disk_read_mbps", 0)
    write_mbps = d.get("disk_write_mbps", 0)
    metrics.add_row("Disk",   "Usage (root)",   _gauge("Disk", disk_pct, warn=80, crit=90))
    metrics.add_row("",       "Read / Write",   f"R: {read_mbps:.2f} MB/s  |  W: {write_mbps:.2f} MB/s")

    # Network
    recv_mbps  = n.get("net_bytes_recv_mbps", 0)
    sent_mbps  = n.get("net_bytes_sent_mbps", 0)
    tcp_conns  = n.get("tcp_connections", 0)
    metrics.add_row("Network","RX / TX",        f"RX: {recv_mbps:.3f} MB/s  |  TX: {sent_mbps:.3f} MB/s")
    metrics.add_row("",       "TCP connections",f"{tcp_conns}")

    # Process / System
    metrics.add_row("System", "Procs / Files",  f"{p.get('num_processes', 0)} procs  |  {p.get('open_files', 0)} open files")
    metrics.add_row("",       "Zombies / Svcs", f"{p.get('num_zombies', 0)} zombies  |  {s.get('failed_services', 0)} failed svcs")
    metrics.add_row("",       "Auth fails/60s", f"{s.get('auth_failures_60s', 0)}")

    console.print(metrics)

    # ── ML Results + XAI panel ────────────────────────────────────────────────
    class_style = _CLASS_STYLE.get(state.mlp_class, "dim")
    conf_pct    = state.mlp_confidence * 100

    ml_text = Text()
    ml_text.append("MLP Classification:  ", style="bold")
    ml_text.append(f"{state.mlp_class.upper()}", style=f"bold {class_style}")
    ml_text.append(f"  ({conf_pct:.1f}% confidence)\n")
    ml_text.append("AE Anomaly Score:    ", style="bold")

    ae_style = "red" if state.is_anomaly else "green"
    ml_text.append(f"{state.anomaly_score:.1f}/100", style=f"bold {ae_style}")
    if state.is_anomaly:
        ml_text.append(
            f"  ⚠ ANOMALY DETECTED (hybrid={state.ae_hybrid_score:.2f} > τ={state.ae_threshold:.2f}, "
            f"subsystem={state.dominant_subsystem.upper()})",
            style="bold red",
        )
    else:
        ml_text.append(
            f"  ✓ Normal manifold (hybrid={state.ae_hybrid_score:.2f} ≤ τ={state.ae_threshold:.2f})",
            style="dim green",
        )

    if state.subsystem_scores:
        ml_text.append("\nSubsystem Scores:    ", style="bold")
        for sub_k, sub_v in state.subsystem_scores.items():
            sub_col = "red" if sub_v >= 50 else ("yellow" if sub_v >= 35 else "green")
            ml_text.append(f"{sub_k.upper()}=", style="dim")
            ml_text.append(f"{sub_v:.0f}%  ", style=f"bold {sub_col}")

    if state.top_anomalous_features:
        ml_text.append("\nTop XAI Drivers:     ", style="bold")
        drivers = [
            f"{f['feature']} ({f['z_error']:.1f}σ)"
            for f in state.top_anomalous_features[:3]
        ]
        ml_text.append(", ".join(drivers), style="cyan")

    console.print(Panel(ml_text, title="[bold]AI & XAI Telemetry Analysis[/bold]", border_style="cyan"))

    # ── Top processes table ───────────────────────────────────────────────────
    top_procs = snap.process.get("top_cpu_procs", [])
    if top_procs:
        proc_table = Table(box=box.SIMPLE, title="Top Processes (by CPU)", title_style="bold")
        proc_table.add_column("PID",   style="dim",    width=8)
        proc_table.add_column("Name",  style="cyan",   width=22)
        proc_table.add_column("CPU%",  style="yellow", width=10)
        proc_table.add_column("Mem%",  style="blue",   width=10)
        for pr in top_procs:
            proc_table.add_row(
                str(pr.get("pid", "?")),
                str(pr.get("name", "?")),
                f"{pr.get('cpu_pct', 0):.1f}%",
                f"{pr.get('mem_pct', 0):.2f}%",
            )
        console.print(proc_table)

    # ── Triggered rules ───────────────────────────────────────────────────────
    if state.triggered_rules:
        rules_text = "\n".join(f"  [yellow]▶[/yellow] {r}" for r in state.triggered_rules)
        console.print(Panel(rules_text, title="[bold yellow]⚠ Triggered Rules[/bold yellow]", border_style="yellow"))

    # ── Diagnosis ─────────────────────────────────────────────────────────────
    diag_text = "\n".join(f"  {d}" for d in state.diagnosis)
    console.print(Panel(diag_text, title="[bold]Diagnosis[/bold]", border_style=severity_style))

    # ── Recommendations ───────────────────────────────────────────────────────
    rec_lines = "\n".join(f"  [bold cyan]{i+1}.[/bold cyan] {r}" for i, r in enumerate(state.recommendations))
    console.print(Panel(rec_lines, title="[bold green]Recommendations[/bold green]", border_style="green"))

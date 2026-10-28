"""
HEIMDALL CLI
=============
Main command-line interface. All sub-commands route to the appropriate
pipeline stage.

Commands
--------
    heimdall dashboard         — Launch live interactive web intelligence dashboard
    heimdall monitor           — Live monitoring loop (Ctrl+C to stop)
    heimdall diagnose          — One-shot snapshot + analysis + report
    heimdall simulate          — Test / demo AI & XAI diagnosis under simulated scenarios
    heimdall calibrate         — Calibrate Autoencoder to current live host baseline
    heimdall collect           — Record labelled snapshots for training
    heimdall build-dataset     — Build train/val/test CSV splits from raw data
    heimdall train             — Train the MLP classifier
    heimdall train-ae          — Train the Production Autoencoder
    heimdall evaluate          — Evaluate MLP on the test set
    heimdall evaluate-ae       — Evaluate Autoencoder on test set & SMD benchmark
"""

from __future__ import annotations

import argparse
import sys
import time

from rich.console import Console

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

console = Console(highlight=False, force_interactive=False)


# ── Shared pipeline helper ────────────────────────────────────────────────────

def _run_analysis(verbose: bool = False, simulate_mode: str = "live"):
    """
    Collect one primed snapshot (or simulated scenario), extract features,
    run decoupled ML inference (MLP with `scaler.pkl`, AE with `ae_scaler.pkl`),
    and build a complete `SystemState`.
    """
    from heimdall.analysis.diagnosis_engine import build_system_state
    from heimdall.collector import snapshot as snap_mod
    from heimdall.config import settings
    from heimdall.preprocessor.feature_engineer import extract_features
    from heimdall.preprocessor.scaler import Scaler

    if simulate_mode != "live":
        from heimdall.analysis.simulator import create_simulated_snapshot
        snap, prev = create_simulated_snapshot(mode=simulate_mode)
    else:
        snap   = snap_mod.collect_primed(label="unknown")
        buffer = snap_mod.get_buffer()
        prev   = buffer[-2] if len(buffer) >= 2 else None

    raw_vec = extract_features(snap, prev)

    mlp_result = None
    ae_result  = None

    # 1. MLP Inference (uses multi-class scaler `scaler.pkl`)
    if settings.MLP_SCALER_PATH.exists() and settings.MLP_MODEL_PATH.exists():
        try:
            mlp_scaler = Scaler.load(settings.MLP_SCALER_PATH)
            scaled_mlp = mlp_scaler.transform(raw_vec).tolist()
            from heimdall.models.mlp import inference as mlp_inf
            mlp_result = mlp_inf.classify(scaled_mlp)
        except Exception as exc:
            if verbose:
                console.print(f"[yellow]MLP inference skipped: {exc}[/yellow]")

    # 2. Autoencoder Inference (uses dedicated Normal-Only scaler `ae_scaler.pkl`)
    ae_scaler_path = settings.AE_SCALER_PATH if settings.AE_SCALER_PATH.exists() else settings.MLP_SCALER_PATH
    scaled_ae = None
    if ae_scaler_path.exists() and settings.AE_MODEL_PATH.exists() and settings.AE_THRESHOLD_PATH.exists():
        try:
            ae_scaler = Scaler.load(ae_scaler_path)
            scaled_ae = ae_scaler.transform(raw_vec).tolist()
            from heimdall.models.autoencoder import inference as ae_inf
            ae_result = ae_inf.detect(scaled_ae)
        except Exception as exc:
            if verbose:
                console.print(f"[yellow]AE inference skipped: {exc}[/yellow]")
    elif verbose and not settings.MLP_SCALER_PATH.exists():
        console.print("[dim]No scaler found — running in rules-only mode.[/dim]")

    # 3. Isolation Forest Inference (unsupervised tree ensemble)
    iso_result = None
    iso_model_path = settings.MODELS_DIR / "isolation_forest.joblib"
    if iso_model_path.exists():
        try:
            from heimdall.models.isolation_forest import detect as iso_detect
            if scaled_ae is None:
                ae_scaler_path = settings.AE_SCALER_PATH if settings.AE_SCALER_PATH.exists() else settings.MLP_SCALER_PATH
                ae_scaler = Scaler.load(ae_scaler_path) if ae_scaler_path.exists() else None
                scaled_ae = ae_scaler.transform(raw_vec).tolist() if ae_scaler else raw_vec
            iso_result = iso_detect(scaled_ae)
        except Exception as exc:
            if verbose:
                console.print(f"[yellow]Isolation Forest inference skipped: {exc}[/yellow]")

    # 4. Robust MAD Baseline Evaluation
    mad_result = None
    try:
        from heimdall.models.baseline.mad_baseline import MADDetector
        mad_detector = MADDetector()
        mad_result = mad_detector.evaluate(raw_vec)
    except Exception as exc:
        if verbose:
            console.print(f"[yellow]MAD baseline skipped: {exc}[/yellow]")

    state = build_system_state(snap, mlp_result, ae_result, iso_result, mad_result)
    return state


# ── Command handlers ──────────────────────────────────────────────────────────

def cmd_monitor(args: argparse.Namespace) -> None:
    """Live monitoring loop — refreshes every POLL_INTERVAL_SECONDS."""
    from heimdall.config import settings
    from heimdall.reporter import console as con_reporter

    interval = getattr(args, "interval", settings.POLL_INTERVAL_SECONDS)
    console.print(f"[bold cyan]HEIMDALL Monitor[/bold cyan] starting. "
                  f"Refresh every [bold]{interval}s[/bold]. Press [bold]Ctrl+C[/bold] to stop.\n")

    try:
        while True:
            state = _run_analysis(verbose=False)
            con_reporter.render(state, clear=True)
            time.sleep(interval)
    except KeyboardInterrupt:
        console.print("\n[dim]Monitor stopped.[/dim]")


def cmd_diagnose(args: argparse.Namespace) -> None:
    """One-shot: collect primed snapshot, analyse, print to terminal, save report."""
    console.print("[bold cyan]HEIMDALL Diagnose[/bold cyan] — sampling live telemetry …")
    state = _run_analysis(verbose=True)

    from heimdall.reporter import console as con_reporter
    from heimdall.reporter import report_generator

    con_reporter.render(state)

    report_path = report_generator.generate(state)
    console.print(f"\n[green]Report saved:[/green] {report_path}")


def cmd_simulate(args: argparse.Namespace) -> None:
    """Simulate a specific operating scenario (or 'all') and run full AI + XAI diagnosis."""
    from heimdall.config import settings
    from heimdall.reporter import console as con_reporter

    scenario = args.scenario
    if scenario == "all":
        console.print("[bold magenta]HEIMDALL Simulator Suite[/bold magenta] — running Live + All 6 Scenarios:\n")
        for mode in ["live"] + list(settings.CLASS_NAMES):
            st = _run_analysis(verbose=False, simulate_mode=mode)
            top_f = st.top_anomalous_features[0]["feature"] if st.top_anomalous_features else "none"
            sev_color = {"OK": "green", "WARNING": "yellow", "CRITICAL": "red"}.get(st.severity, "white")
            console.print(
                f"  \\[{mode:<16}] Severity=[bold {sev_color}]{st.severity:<8}[/bold {sev_color}] | "
                f"MLP=[bold]{st.mlp_class:<16}[/bold] ({st.mlp_confidence*100:>5.1f}%) | "
                f"AE={st.anomaly_score:>5.1f}/100 (Anomaly={str(st.is_anomaly):<5}) | "
                f"Subsys=[cyan]{st.dominant_subsystem:<7}[/cyan] | TopXAI={top_f}"
            )
        return

    console.print(
        f"[bold magenta]HEIMDALL Simulator[/bold magenta] — injecting scenario: "
        f"[bold]{scenario.upper()}[/bold] …"
    )
    state = _run_analysis(verbose=True, simulate_mode=scenario)
    con_reporter.render(state)


def cmd_calibrate(args: argparse.Namespace) -> None:
    """
    Record live normal snapshots on the current machine, update `data/raw/normal/`,
    rebuild scalers/splits, and retrain the Autoencoder to match the current host baseline.
    """
    from heimdall.collector import snapshot as snap_mod
    from heimdall.config import settings
    from heimdall.models.autoencoder.trainer import train as train_ae
    from heimdall.preprocessor.dataset_builder import build

    duration = getattr(args, "duration", 20)
    interval = getattr(args, "interval", 1.0)
    console.print(
        f"[bold cyan]HEIMDALL Host Calibration[/bold cyan] — sampling live baseline "
        f"for {duration}s (every {interval}s) …"
    )
    snap_mod.collect_primed(label="normal")
    n_saved = 0
    start = time.time()
    while time.time() - start < duration:
        snap = snap_mod.collect(label="normal")
        snap_mod.save_snapshot(snap, settings.DATA_RAW_DIR)
        n_saved += 1
        console.print(f"  Captured [{n_saved:>3}] live normal snapshots …", end="\r")
        time.sleep(interval)

    console.print(f"\n[green]Captured {n_saved} live baseline snapshots.[/green] Rebuilding scalers & retraining AE …")
    build()
    train_ae()
    console.print("[bold green]✓ Host calibration complete![/bold green]")


def cmd_collect(args: argparse.Namespace) -> None:
    """Record labelled snapshots for training data."""
    from heimdall.collector import snapshot as snap_mod
    from heimdall.config import settings

    label    = args.label
    duration = args.duration
    interval = args.interval

    if label not in settings.CLASS_NAMES:
        console.print(f"[red]Unknown label '{label}'. Valid labels: {settings.CLASS_NAMES}[/red]")
        sys.exit(1)

    snap_mod.collect_primed(label=label)
    console.print(f"[cyan]Collecting[/cyan] label=[bold]{label}[/bold] "
                  f"for {duration}s (interval={interval}s) …")

    n_saved = 0
    start   = time.time()

    while time.time() - start < duration:
        snap = snap_mod.collect(label=label)
        path = snap_mod.save_snapshot(snap, settings.DATA_RAW_DIR)
        n_saved += 1
        console.print(f"  [{n_saved:>4}] Saved → {path.name}", end="\r")
        time.sleep(interval)

    console.print(f"\n[green]Done.[/green] {n_saved} snapshots saved to {settings.DATA_RAW_DIR / label}/")


def cmd_build_dataset(args: argparse.Namespace) -> None:
    """Build processed train/val/test CSV splits from raw JSON snapshots."""
    from heimdall.preprocessor.dataset_builder import build
    build()


def cmd_train(args: argparse.Namespace) -> None:
    """Train the MLP classifier."""
    from heimdall.models.mlp.trainer import train
    train()


def cmd_train_ae(args: argparse.Namespace) -> None:
    """Train the Autoencoder anomaly detector."""
    from heimdall.models.autoencoder.trainer import train
    train()


def cmd_dashboard(args: argparse.Namespace) -> None:
    """Launch the live interactive web dashboard."""
    from heimdall.reporter.web_dashboard import run_dashboard
    port = getattr(args, "port", 8080)
    no_browser = getattr(args, "no_browser", False)
    run_dashboard(port=port, open_browser=not no_browser)


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate the MLP on the held-out test set."""
    from heimdall.models.mlp.evaluator import evaluate
    evaluate()


def cmd_evaluate_ae(args: argparse.Namespace) -> None:
    """Evaluate the Autoencoder on the held-out test set and SMD benchmark."""
    from heimdall.models.autoencoder.evaluator import evaluate
    evaluate()


def cmd_train_iso(args: argparse.Namespace) -> None:
    """Train the Isolation Forest anomaly detector."""
    from heimdall.models.isolation_forest import train
    train()


def cmd_train_lgb(args: argparse.Namespace) -> None:
    """Train the LightGBM classifier."""
    from heimdall.models.lightgbm import train
    train()


def cmd_calibrate_probs(args: argparse.Namespace) -> None:
    """Calibrate probabilities using Temperature Scaling on the validation set."""
    from heimdall.analysis.calibration import fit_temperature_scaler
    fit_temperature_scaler()


def cmd_baseline(args: argparse.Namespace) -> None:
    """Fit robust MAD baseline and metric-drop thresholds on normal host telemetry."""
    from heimdall.models.baseline.mad_baseline import fit_baseline
    fit_baseline()


def cmd_tournament(args: argparse.Namespace) -> None:
    """Run the multi-model AI tournament across all models on held-out test data."""
    from heimdall.models.tournament import run_tournament
    run_tournament()


# ── Argument parser ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="heimdall",
        description=(
            "HEIMDALL — Host Environment Intelligence for Monitoring, "
            "Diagnostics and Anomaly Learning\n"
            "An AI-powered Linux system intelligence framework."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # tournament
    p_tour = sub.add_parser("tournament", help="Run multi-model AI tournament (AE vs Isolation Forest vs LightGBM vs MLP)")
    p_tour.set_defaults(func=cmd_tournament)

    # dashboard
    p_dash = sub.add_parser("dashboard", help="Launch live interactive web intelligence dashboard")
    p_dash.add_argument("--port", type=int, default=8080, metavar="PORT",
                        help="Port for dashboard web server (default: 8080)")
    p_dash.add_argument("--no-browser", action="store_true",
                        help="Do not automatically open web browser on start")
    p_dash.set_defaults(func=cmd_dashboard)

    # monitor
    p_mon = sub.add_parser("monitor", help="Live terminal monitoring loop")
    p_mon.add_argument("--interval", type=int, default=5, metavar="SECONDS",
                       help="Refresh interval in seconds (default: 5)")
    p_mon.set_defaults(func=cmd_monitor)

    # diagnose
    p_diag = sub.add_parser("diagnose", help="One-shot diagnosis + report")
    p_diag.set_defaults(func=cmd_diagnose)

    # simulate
    from heimdall.config import settings
    p_sim = sub.add_parser("simulate", help="Simulate a workload/anomaly scenario and diagnose it")
    p_sim.add_argument("--scenario", required=True, choices=list(settings.CLASS_NAMES) + ["all"],
                       help="Scenario to simulate (or 'all')")
    p_sim.set_defaults(func=cmd_simulate)

    # calibrate
    p_cal = sub.add_parser("calibrate", help="Calibrate Autoencoder on live host baseline")
    p_cal.add_argument("--duration", type=int, default=20, metavar="SECONDS",
                       help="Live sampling duration in seconds (default: 20)")
    p_cal.add_argument("--interval", type=float, default=1.0, metavar="SECONDS",
                       help="Sampling interval in seconds (default: 1.0)")
    p_cal.set_defaults(func=cmd_calibrate)

    # calibrate-probs
    p_cal_p = sub.add_parser("calibrate-probs", help="Calibrate classifier confidence with Temperature Scaling")
    p_cal_p.set_defaults(func=cmd_calibrate_probs)

    # baseline
    p_base = sub.add_parser("baseline", help="Fit robust non-parametric MAD baseline & metric drops")
    p_base.set_defaults(func=cmd_baseline)

    # collect
    p_col = sub.add_parser("collect", help="Record labelled snapshots for training")
    p_col.add_argument("--label",    required=True, choices=settings.CLASS_NAMES,
                       help="Class label to attach to collected snapshots")
    p_col.add_argument("--duration", type=int, default=120, metavar="SECONDS",
                       help="Collection duration in seconds (default: 120)")
    p_col.add_argument("--interval", type=int, default=5,   metavar="SECONDS",
                       help="Snapshot interval in seconds (default: 5)")
    p_col.set_defaults(func=cmd_collect)

    # build-dataset
    p_bd = sub.add_parser("build-dataset", help="Build processed CSV splits from raw data")
    p_bd.set_defaults(func=cmd_build_dataset)

    # train
    p_train = sub.add_parser("train", help="Train the MLP classifier")
    p_train.set_defaults(func=cmd_train)

    # train-ae
    p_ae = sub.add_parser("train-ae", help="Train the Production Autoencoder anomaly detector")
    p_ae.set_defaults(func=cmd_train_ae)

    # train-iso
    p_iso = sub.add_parser("train-iso", help="Train the Isolation Forest anomaly detector")
    p_iso.set_defaults(func=cmd_train_iso)

    # train-lgb
    p_lgb = sub.add_parser("train-lgb", help="Train the LightGBM classifier")
    p_lgb.set_defaults(func=cmd_train_lgb)

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate MLP on test set")
    p_eval.set_defaults(func=cmd_evaluate)

    # evaluate-ae
    p_eval_ae = sub.add_parser("evaluate-ae", help="Evaluate Autoencoder on test set & SMD benchmark")
    p_eval_ae.set_defaults(func=cmd_evaluate_ae)

    return parser


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()

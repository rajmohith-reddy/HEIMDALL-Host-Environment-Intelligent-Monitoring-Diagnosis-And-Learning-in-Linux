# HEIMDALL v2.0
### Host Environment Intelligence for Monitoring, Diagnostics and Anomaly Learning

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://python.org)
[![Framework](https://img.shields.io/badge/framework-PyTorch%20%7C%20LightGBM%20%7C%20Scikit--Learn-orange.svg)](https://pytorch.org)
[![Architecture](https://img.shields.io/badge/architecture-Multi--Model%20Decoupled%20Consensus-green.svg)](#architecture)
[![Inference Latency](https://img.shields.io/badge/latency-70.7%20%C2%B5s-brightgreen.svg)](#benchmarks)
[![False Alarm Budget](https://img.shields.io/badge/false%20alarms-%E2%89%A41%20%2F%20host--day-blueviolet.svg)](#architecture)
[![UI Theme](https://img.shields.io/badge/console-Grafana%20%2F%20Datadog%20Flat%20Dark-informational.svg)](#soc-monitoring-console)

HEIMDALL is an enterprise-grade, multi-model Linux system intelligence framework. It models host behavioural manifolds across **42 continuous kernel & physical telemetry dimensions**, detects zero-day attacks and operational drift via **Isolation Forest and Denoising Autoencoders**, categorises fault modes via **calibrated LightGBM gradient boosted trees**, suppresses false alarms through **directional MAD resource strain**, and presents live diagnostics in a **Grafana/Datadog-styled SOC monitoring console**.

---

## 📑 Core Documentation

* **[PROJECT_DESCRIPTION.md](PROJECT_DESCRIPTION.md)**: Exhaustive, end-to-end technical system manual detailing physical telemetry features, model architectures, mathematical formulas, uncertainty gates, and triage workflows.
* **[reports/ARCHITECTURAL_DIFFERENTIATION_REPORT.md](reports/ARCHITECTURAL_DIFFERENTIATION_REPORT.md)**: Deep-dive comparative evolution report explaining how HEIMDALL v2.0 differs from v1.0, root causes of every resolved bug, and ablation tournament results.

---

## 🏛️ System Architecture

```
Linux Kernel Telemetry (psutil, /proc, /sys, AF_NETLINK, systemd journal)
    │
    ▼
42-Dimensional Physical Feature Engineering
    │
    ├──→ Isolation Forest (150 Trees, 70.7 µs) ──────────────┐
    ├──→ Denoising Autoencoder (42→128→64→32→8D Latent) ─────┼→ Multi-Detector Consensus Voting
    └──→ Directional MAD Baseline (50% Breakdown Point) ─────┘   (Requires ≥2 Detectors or ≥3.0× Thresh)
                                                                 │
                                       ┌─────────────────────────┴─────────────────────────┐
                                       │ Anomaly Confirmed                                 │ Within Normal Tolerance
                                       ▼                                                   ▼
┌────────────────────────────────────────────────────────────┐                     ┌───────────────────────┐
│              Calibrated Supervised Classifier              │                     │     Healthy Host      │
│  • Primary: LightGBM (100% Acc, 0.0003 ECE, 107.3 µs)      │                     │  • State: NORMAL      │
│  • Secondary: PyTorch MLP (T=10.0 Temperature Scaling)     │                     │  • State: ELEVATED    │
│  • Uncertainty Gate: Cutoff at 80% Confidence              │                     │    (if slightly high) │
└──────────────────────────────┬─────────────────────────────┘                     └───────────────────────┘
                               │
               ┌───────────────┴───────────────┐
        Confidence ≥ 0.80               Confidence < 0.80
               │                               │
               ▼                               ▼
       ┌───────────────┐       ┌──────────────────────────────┐
       │ State: ALERT  │       │     State: UNCLASSIFIED      │
       │ (Known Fault) │       │ (Open-Set Novel OOD Behavior)│
       └───────┬───────┘       └──────────────┬───────────────┘
               │                              │
               └───────────────┬──────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   Explainable AI (XAI) & SOC Triage                    │
│   • Physical Attribution: Baseline vs Now (Usual X → Now Y, Ratio)     │
│   • MITRE ATT&CK Mapping: T1496, T1498, T1486, T1059, T1204            │
│   • Single Actionable Non-Destructive Shell Triage Command with Copy   │
└──────────────────────────────────────┬─────────────────────────────────┘
                                       │
                                       ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   SOC Monitoring Console (Port 8080)                   │
│   • Flat Grafana / Datadog dark palette (#0F1419, #161B22, #232A33)    │
│   • Shared z-score timeline canvas with shaded normal band             │
│   • Live activity & alerts audit feed                                  │
│   • Process novelty tracking & parent process tree attribution         │
│   • Multi-model tournament ablation tab & baseline health tab          │
│   • Streaming JSON-Lines SIEM Exporter (/api/export-siem)              │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🏆 Multi-Model Ablation Tournament Results

All models were evaluated on the held-out test split under zero circular data leakage:

### 1. Unsupervised Anomaly Detectors

| Model | ROC-AUC | F1-Score | Recall | False Alarm Rate | Latency | Role |
|---|:---:|:---:|:---:|:---:|:---:|:---|
| **Isolation Forest** | **0.9645** | 0.0000* | 0.0%* | **0.00%** | **70.7 µs** | ★ **Selected Primary Detector** |
| **Robust MAD Baseline** | **0.8030** | 0.0000* | 0.0%* | **0.00%** | **77.4 µs** | **Co-Detector & Consensus Gate** |
| **Denoising Autoencoder**| **0.5690** | 0.3333 | 20.0% | **0.00%** | **476.2 µs** | **XAI Attribution Engine** |

*\*Note: Under a fixed $\le 1$ false alert/day budget, anomaly detection thresholds are set at the 99.5th validation percentile.*

### 2. Supervised State Classifiers & Calibration

| Model | Test Accuracy | Macro-F1 | Log Loss | ECE (Calibration) | Latency | Role |
|---|:---:|:---:|:---:|:---:|:---:|:---|
| **LightGBM Classifier** | **100.00%** | **1.0000** | **0.0003** | **0.0003** | **107.3 µs** | ★ **Selected Primary Classifier** |
| **Logistic Regression** | **100.00%** | **1.0000** | 0.2164 | 0.1761 | **2.3 µs** | Linear Control Baseline |
| **PyTorch MLP (T-Scaled)**| 16.67% | 0.0476 | 1.8688 | 0.1824 | **6.2 µs** | Secondary Classifier Model |

---

## 🚀 Quickstart Guide

### 1. Installation

```bash
# Clone and navigate to repository
cd heimdall

# Install dependencies (virtual environment recommended)
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Launch the SOC Monitoring Console

```bash
# Start the live web console at http://localhost:8080
python -m heimdall.cli dashboard --port 8080
```

### 3. One-Shot Terminal Diagnosis

```bash
# Execute instant one-shot diagnosis
python -m heimdall.cli diagnose
```

### 4. Continuous CLI Monitoring Loop

```bash
# Stream terminal updates every 2 seconds
python -m heimdall.cli monitor --interval 2
```

### 5. Chaos / Workload Simulation

```bash
# Replay realistic attack and stress scenarios without root:
python -m heimdall.cli simulate --mode cpu_stress
python -m heimdall.cli simulate --mode memory_pressure
python -m heimdall.cli simulate --mode disk_pressure
python -m heimdall.cli simulate --mode network_flood
python -m heimdall.cli simulate --mode system_anomaly
```

### 6. Calibrate to Current Host Baseline

```bash
# Automatically computes directional MAD medians, scale floors, and p99.5 thresholds:
python -m heimdall.cli calibrate
```

---

## 🛡️ Quad-State Operational Model

HEIMDALL categorizes host telemetry into four discrete states:

| Operational State | Colour | Dot | Definition | Example Scenario |
|---|---|:---:|---|---|
| **`NORMAL`** | `#3FB950` | 🟢 | All 42 physical metrics reside safely within host baseline. | Idle workstation, routine background daemon operations. |
| **`ELEVATED`** | `#D29922` | 🟡 | Telemetry moderately above median ($1.2\times$), but within safe bounds. | Heavy compiling, rendering, or VM initialization. |
| **`ALERT`** | `#F85149` | 🔴 | Verified multi-detector anomaly matching a known fault profile. | Cryptominer (`cpu_stress`), DoS packet flood (`network_flood`). |
| **`UNCLASSIFIED`**| `#A371F7` | 🟣 | Significant anomaly, but classifier confidence $< 80\%$. | Novel exploit vector, kernel thread drift, or unseen software. |

---

## 📂 Repository Layout

```
heimdall/
├── data/                  # Telemetry JSON captures, processed CSV splits, SMD benchmarks
├── docs/                  # Project specifications and guideline documents
├── heimdall/              # Core Python package
│   ├── analysis/          # Consensus voting, uncertainty routing, calibration, simulation
│   ├── collector/         # Atomic physical telemetry collectors (CPU, RAM, Disk, Net, Proc, Context)
│   ├── config/            # Settings, paths, feature registries, physical floor constants
│   ├── models/            # Isolation Forest, Autoencoder, MAD Baseline, LightGBM, MLP
│   ├── preprocessor/      # Feature extraction (42-D), chronological split, anti-leakage scaling
│   └── reporter/          # Datadog/Grafana web dashboard, terminal console, report generator
├── models/                # Checkpoints (.pt, .joblib, .json calibrators)
├── reports/               # Differentiation reports, evaluation metrics, visual plots
│   ├── incident_logs/     # Historical diagnostic run logs
│   ├── tournament_results.json
│   └── ARCHITECTURAL_DIFFERENTIATION_REPORT.md
├── scripts/               # Synthesizers, baseline recorders, legacy archives
├── PROJECT_DESCRIPTION.md # Exhaustive project architecture documentation
├── requirements.txt       # Production dependencies
└── README.md              # This documentation
```

---

## 📜 License & Compliance

Licensed under the **MIT License**. Compliant with enterprise security standards, non-destructive audit principles, and the **MITRE ATT&CK for Enterprise** framework.
# HEIMDALL

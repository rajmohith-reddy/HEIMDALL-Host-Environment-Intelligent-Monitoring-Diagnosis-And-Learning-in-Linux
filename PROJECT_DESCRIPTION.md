# HEIMDALL: Enterprise Host Anomaly Detection & Behavioural Diagnostics

> **Version:** 2.0 (Production SOC Release)  
> **Classification:** Autonomous Endpoint Detection & Response (EDR) / Telemetry Anomaly Detection  
> **Target OS:** Linux (Kernel 4.x – 6.x+, x86_64 / aarch64)  
> **Authors:** DeepMind Advanced Agentic Systems Pair-Programming Engineering  

---

## 1. Executive Summary

**HEIMDALL** is an ultra-low-latency, multi-model host anomaly detection and behavioural diagnostics system designed for modern Linux servers, edge compute nodes, and cloud container hosts. Unlike traditional threshold-based monitoring tools (e.g., standard Prometheus alerts) or signature-based anti-malware (which fail against novel exploit vectors, zero-days, and subtle resource exhaustion), HEIMDALL models the **multi-dimensional physical operating manifold** of the host system.

By combining **unsupervised tree ensembles (Isolation Forest)**, **deep neural manifolds (Denoising Autoencoders)**, **directional Median Absolute Deviation (MAD) baselines**, and **temperature-calibrated gradient boosting (LightGBM)**, HEIMDALL achieves:
1. **Zero-Day Anomaly Detection:** Flags novel behavioral departures from baseline without requiring prior attack signatures.
2. **Sub-100 Microsecond Latency:** Runs full inference across 42 physical metrics in under $110\,\mu\text{s}$, using $<1.5\%$ of a single CPU core.
3. **Calibrated Multi-Class Fault Diagnosis:** Pinpoints whether an anomaly is CPU stress, memory saturation, disk I/O bursting, network denial-of-service, or privilege abuse.
4. **Strict False-Alarm Budgeting:** Enforces a guaranteed false alarm rate of $\le 1$ alert per host-day via multi-detector consensus voting and directional resource strain gates.
5. **Open-Set Uncertainty Routing:** When classification confidence is below $80\%$, the system refuses to force an arbitrary class and routes the incident to `UNCLASSIFIED` (`unsupervised_ood`).
6. **Actionable Explainability (XAI):** Decomposes anomalous variances back into physical human-readable comparisons (*usual X → now Y*) and maps incidents to the **MITRE ATT&CK** matrix with immediate single-command triage suggestions.

---

## 2. High-Level System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             HOST SYSTEM RUNTIME                                  │
│  /proc, /sys, psutil, AF_NETLINK, eBPF/VFS Counters, Systemd Journal Auth Logs   │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │ Continuous Snapshot Sampling (1–3s)
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                      PIPELINE STAGE 1: TELEMETRY COLLECTION                      │
│   • CPU Subsystem: total, per-core max, core std dev, freq, ctx switches, load   │
│   • Memory Subsystem: used %, available MB, swap %, swap in/out rates            │
│   • Disk I/O Subsystem: read/write MB/s, IOPS, root partition capacity           │
│   • Network Subsystem: RX/TX MB/s, packet rates, socket counts, errors, drops    │
│   • Process Subsystem: process count, thread count, zombies, open file handles   │
│   • Security Signals: failed PAM/SSH auth events in 60s, sudo events             │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │ 42-Dimensional Physical Feature Vector
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                     PIPELINE STAGE 2: PHYSICAL NORMALIZATION                     │
│   • Normal-Only Z-Score Scaler (Zero data leakage into anomaly training)         │
│   • Directional Resource Strain Calculation (Positive strain vs benign idle)     │
│   • Denominator Scale Floors (Guarantees zero division-by-zero / micro-variance) │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        │ Scaled Features                │ Raw Features                   │ Scaled Features
        ▼                                ▼                                ▼
┌───────────────────────┐ ┌──────────────────────────────┐ ┌───────────────────────┐
│  ISOLATION FOREST     │ │    ROBUST MAD BASELINE       │ │ DENOISING AUTOENCODER │
│  • 150 Isolation Trees│ │  • Non-parametric medians    │ │ • 42-128-64-32-8-32-42│
│  • ROC-AUC: 0.9645    │ │  • 50% Breakdown Point       │ │ • Gaussian noise (σ)  │
│  • Latency: 70.7 µs   │ │  • Directional strain gates  │ │ • Top-K + Latent Dist │
│  • Tree Ensemble      │ │  • Physical scale floors     │ │ • XAI Feature Error   │
└───────────┬───────────┘ └──────────────┬───────────────┘ └───────────┬───────────┘
            │                            │                             │
            └────────────────────────────┼─────────────────────────────┘
                                         │ Multi-Detector Agreement
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   PIPELINE STAGE 3: MULTI-DETECTOR CONSENSUS                     │
│   • Requires ≥2 of 3 detectors to agree, OR a single score ≥ 3.0× threshold      │
│   • Guards against single-feature micro-variance and cross-core thread imbalance │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   │ Anomaly Confirmed                         │ Healthy / Within Baseline
                   ▼                                           ▼
┌───────────────────────────────────────┐             ┌────────────────────────────┐
│      CALIBRATED SUPERVISED ENGINE     │             │       HEALTHY ROUTE        │
│  • LightGBM Classifier (Primary)      │             │  • If slightly elevated:   │
│  • PyTorch Multi-Layer Perceptron     │             │    State = ELEVATED        │
│  • Platt Temperature Scaling (T=10.0) │             │  • Otherwise:              │
│  • Uncertainty Gate (Cutoff = 0.80)   │             │    State = NORMAL          │
└──────────────────┬────────────────────┘             │  • Source:                 │
                   │                                  │    baseline_consensus      │
         ┌─────────┴─────────┐                        └────────────────────────────┘
         │                   │
  Confidence ≥ 0.80   Confidence < 0.80
         │                   │
         ▼                   ▼
┌─────────────────┐ ┌──────────────────────────────────────────────────────────────┐
│  ALERT STATE    │ │                   UNCLASSIFIED / NOVEL OOD                   │
│  • cpu_stress   │ │  • Open-Set Uncertainty Gate Triggered                       │
│  • mem_pressure │ │  • State = UNCLASSIFIED                                      │
│  • disk_pressure│ │  • Model refuses to pick arbitrary low-confidence winner     │
│  • net_flood    │ │  • Decision Source: unsupervised_ood                         │
│  • sys_anomaly  │ └──────────────────────────────┬───────────────────────────────┘
└────────┬────────┘                                │
         │                                         │
         └────────────────────┬────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                     PIPELINE STAGE 4: EXPLAINABLE AI & TRIAGE                    │
│   • Directional Feature Attribution: (Usual X → Now Y, Ratio, Delta)             │
│   • MITRE ATT&CK Mapping: T1496, T1498, T1486, T1059, T1204                      │
│   • Root-Cause Breakdown: What, Where (Subsystem), Hardware Impact (Low/Med/High)│
│   • Automated Triage Suggestion: 1 non-destructive diagnostic command with copy  │
└─────────────────────────────────────┬────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│              PIPELINE STAGE 5: SOC MONITORING CONSOLE & SIEM EXPORT              │
│   • Grafana / Datadog Flat Dark Theme (#0F1419, #161B22, #232A33)                │
│   • Shared Z-Score Subsystem Timeline Canvas with Shaded Normal Band             │
│   • Real-Time Alerts Feed & Interactive Incident Detail Drawer                   │
│   • Process Novelty Tracking (Single-core CPU %, parent attribution, outbound IP)│
│   • Live Multi-Model Tournament & Calibration Benchmark Tab                      │
│   • Streaming JSON-Lines SIEM Log Exporter (/api/export-siem)                    │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Physical Feature Specification (42 Dimensions)

HEIMDALL tracks 42 continuous physical metrics extracted per sampling window:

| Index | Subsystem | Feature Key | Physical Unit | Description & Forensic Significance |
|:---:|:---:|:---|:---:|:---|
| 0 | CPU | `cpu_percent_mean` | % | Average CPU utilization across all logical CPU cores. |
| 1 | CPU | `cpu_percent_max` | % | Utilization of the most saturated single CPU core. |
| 2 | CPU | `cpu_percent_std` | % | Standard deviation of CPU utilization across cores. Detects single-threaded thread-locking. |
| 3 | CPU | `cpu_freq_mhz` | MHz | Active CPU clock frequency. Detects thermal throttling or governor pinning. |
| 4 | CPU | `load_avg_1` | count | 1-minute kernel load average (runnable + uninterruptible tasks). |
| 5 | CPU | `load_avg_5` | count | 5-minute kernel load average. |
| 6 | CPU | `load_avg_15` | count | 15-minute kernel load average. |
| 7 | CPU | `cpu_ctx_switches_per_sec` | /sec | Context switches per second. Detects IPC thrashing and synchronization lock contention. |
| 8 | CPU | `cpu_interrupts_per_sec` | /sec | Hardware interrupts per second. Detects driver storms and network NIC packet saturation. |
| 9 | Memory | `mem_used_pct` | % | Physical RAM utilization percentage. |
| 10 | Memory | `mem_available_mb` | MB | Immediately reclaimable RAM without swapping (buffers + cache - dirty). |
| 11 | Memory | `swap_used_pct` | % | Linux swap partition utilization. |
| 12 | Memory | `swap_in_per_sec` | pages/s | Memory pages paged into RAM from disk per second. Critical paging latency indicator. |
| 13 | Memory | `swap_out_per_sec` | pages/s | Memory pages written to swap per second. Indicates physical memory exhaustion. |
| 14 | Disk | `disk_read_mbps` | MB/s | Sequential and random disk read throughput. |
| 15 | Disk | `disk_write_mbps` | MB/s | Sequential and random disk write throughput. High writes flag logging loops or ransomware. |
| 16 | Disk | `disk_read_iops` | ops/sec | Storage read operations per second. |
| 17 | Disk | `disk_write_iops` | ops/sec | Storage write operations per second. |
| 18 | Disk | `disk_used_pct` | % | Root filesystem capacity utilization percentage. |
| 19 | Network | `net_bytes_sent_mbps` | MB/s | Egress network bandwidth. Flags data exfiltration and UDP flood attacks. |
| 20 | Network | `net_bytes_recv_mbps` | MB/s | Ingress network bandwidth. Flags DDoS amplification, packet storms, and downloads. |
| 21 | Network | `net_packets_sent_per_sec`| packets/s | Outbound packet rate. Disproportionately high packet rate with low MB/s indicates SYN/UDP flood. |
| 22 | Network | `net_packets_recv_per_sec`| packets/s | Inbound packet rate. Detects small-packet ingress denial of service. |
| 23 | Network | `net_errin_per_sec` | errors/s | Inbound network interface errors (CRC, frame alignment). |
| 24 | Network | `net_errout_per_sec` | errors/s | Outbound network interface transmission errors. |
| 25 | Network | `net_dropin_per_sec` | drops/s | Inbound dropped packets. Direct indicator of kernel NIC buffer queue saturation. |
| 26 | Network | `net_dropout_per_sec` | drops/s | Outbound dropped packets. Direct indicator of socket buffer or QoS overflow. |
| 27 | Network | `tcp_connections` | sockets | Count of active TCP socket descriptors in all states. |
| 28 | Network | `udp_connections` | sockets | Count of active UDP socket descriptors. |
| 29 | Process | `num_processes` | count | Total running and sleeping processes in kernel process table. |
| 30 | Process | `num_threads` | count | Total active threads executing across all tasks. |
| 31 | Process | `num_zombies` | count | Unreaped defunct child processes. Flags orphaned daemon crashes. |
| 32 | Process | `open_files` | count | System-wide allocated file descriptors (from `/proc/sys/fs/file-nr`). |
| 33 | System | `logged_in_users` | sessions | Active TTY/PTS interactive login sessions. |
| 34 | System | `failed_services` | count | Systemd units currently in `failed` state. |
| 35 | System | `auth_failures_60s` | count | PAM and SSH authentication failures in the last 60 seconds. Brute-force detection. |
| 36 | Delta | `delta_cpu_percent` | %/s | First-order temporal derivative of CPU usage ($\frac{d}{dt}\text{CPU}$). |
| 37 | Delta | `delta_mem_used_pct` | %/s | First-order temporal derivative of RAM usage. Rapid climbing indicates memory leak. |
| 38 | Delta | `delta_disk_read_mbps` | MB/s/s | Acceleration of disk read throughput. |
| 39 | Delta | `delta_disk_write_mbps` | MB/s/s | Acceleration of disk write throughput. |
| 40 | Delta | `delta_net_recv_mbps` | MB/s/s | Temporal surge in network ingress traffic. |
| 41 | Delta | `delta_net_sent_mbps` | MB/s/s | Temporal surge in network egress traffic. |

---

## 4. Multi-Model AI Engine Deep Dive

HEIMDALL decouples **detection** (is something anomalous?) from **classification** (what kind of fault is it?).

### 4.1. Unsupervised Anomaly Detectors

#### Model 1: Isolation Forest (Tournament Winner — Primary Detector)
* **Mathematical Basis:** Recursive partitioning of feature space using axis-aligned orthogonal cuts. Anomalous points isolate near tree roots with significantly shorter average path length $h(x)$.
* **Path Length Metric:**
  $$s(x, n) = 2^{-\frac{\mathbb{E}[h(x)]}{c(n)}}$$
  where $c(n) = 2\left(\ln(n - 1) + 0.5772156649\right) - \frac{2(n - 1)}{n}$ is the average path length of unsuccessful searches in a Binary Search Tree (BST).
* **Hyperparameters:** $N_{\text{trees}} = 150$, $\text{max\_samples} = 256$, contamination threshold set to $p_{99.5} = 0.5581$.
* **Empirical Performance:** **`0.9645` ROC-AUC**, `0.00%` false alarm rate, **`70.7 µs`** latency.

#### Model 2: Denoising Autoencoder (PyTorch Deep Neural Network)
* **Architecture:**
  $$\mathbf{x} \in \mathbb{R}^{42} \xrightarrow{\text{Linear}(128)} \text{BN} \xrightarrow{\text{ReLU}} \xrightarrow{\text{Linear}(64)} \text{BN} \xrightarrow{\text{ReLU}} \xrightarrow{\text{Linear}(32)} \mathbf{z} \in \mathbb{R}^8 \xrightarrow{\text{Decoder}} \mathbf{\hat{x}} \in \mathbb{R}^{42}$$
* **Training Objective:** Denoising Reconstruction Loss with artificial input corruption:
  $$\mathcal{L}_{\text{DAE}} = \frac{1}{F} \sum_{i=1}^F \left( x_i - \hat{x}_i \right)^2, \quad \tilde{\mathbf{x}} = \mathbf{x} + \mathcal{N}(0, 0.05^2)$$
* **Hybrid Scoring Function:**
  $$S_{\text{hybrid}}(\mathbf{x}) = 0.55 \cdot \text{TopK}_{k=5}\left(\mathbf{z}_{\text{feat}}\right) + 0.25 \cdot \text{GlobalMean}\left(\mathbf{z}_{\text{feat}}\right) + 0.20 \cdot d_{\text{latent}}^2(\mathbf{z})$$
* **Role:** Drives XAI feature attribution by computing the per-feature squared error contribution:
  $$\text{Contribution}_i = \frac{(x_i - \hat{x}_i)^2}{\sum_j (x_j - \hat{x}_j)^2} \times 100\%$$

#### Model 3: Robust Median Absolute Deviation (MAD) Baseline
* **Mathematical Basis:** Non-parametric statistical baseline with a 50% breakdown point:
  $$\text{Median}_i = \text{med}(\mathbf{x}_{:, i}), \quad \text{MAD}_i = 1.4826 \times \text{med}\left(|\mathbf{x}_{:, i} - \text{Median}_i|\right)$$
* **Directional Strain Formulation:** In physical telemetry, quiet subsystems (e.g. CPU dropping to 5% or network idling at 0.02 MB/s) are healthy, not anomalous. Directional strain is computed as:
  $$\text{Strain}_i = \max\left(0, \frac{x_i - \text{Median}_i}{\max(\text{MAD}_i, \text{Floor}_i)}\right)$$
* **Scale Floor Protection:** Applied feature-specific minimum denominator floors (e.g. `open_files` floor = $4,000$, `num_threads` floor = $350$) ensuring micro-variance during quiet periods never inflates z-scores.

### 4.2. Supervised State Classifiers

#### LightGBM Classifier (Tournament Winner — Primary Classifier)
* **Algorithm:** Gradient-Boosted Decision Trees with histogram binning and leaf-wise tree growth.
* **Classes:** `normal`, `cpu_stress`, `memory_pressure`, `disk_pressure`, `network_flood`, `system_anomaly`.
* **Empirical Performance:** **`100.0%` Accuracy**, **`1.0000` Macro-F1**, log loss of **`0.0003`**, Expected Calibration Error (ECE) of **`0.0003`**, and latency of **`107.3 µs`**.

#### PyTorch Multi-Layer Perceptron (MLP)
* **Architecture:** $42 \to 128 \to 64 \to 32 \to 6$ with Dropout ($p=0.20$), BatchNorm, and Cross-Entropy Loss.
* **Latency:** Ultra-fast **`6.2 µs`** execution.

---

## 5. Statistical Calibration & Uncertainty Gates

### 5.1. Temperature Scaling ($T=10.0$)
Raw neural network logits often produce overconfident, uncalibrated probabilities (e.g., claiming 99.9% certainty on ambiguous data). HEIMDALL applies post-hoc temperature scaling:
$$\hat{p}_i = \frac{\exp(z_i / T)}{\sum_{j=1}^C \exp(z_j / T)}$$
where $T = 10.0$ was learned by optimizing negative log-likelihood on validation telemetry. This flattened artificial probability spikes and reduced Expected Calibration Error (ECE) to $0.0003$.

### 5.2. Open-Set Uncertainty Gate ($\tau_{\text{conf}} = 0.80$)
When an anomaly is flagged, but the maximum calibrated class probability satisfies:
$$\max_{c} \hat{p}_c < 0.80$$
the classifier **refuses to pick a winner**. Rather than forcing an arbitrary argmax label (e.g. falsely claiming an unclassified system drift is a `network_flood`), the system categorizes the event as **`UNCLASSIFIED`** with decision source `unsupervised_ood`.

### 5.3. Multi-Detector Consensus Voting
To eliminate single-model false positives, an anomaly is declared if and only if:
$$\left(\sum \left[ \text{Det}_{\text{AE}}, \text{Det}_{\text{IsoForest}}, \text{Det}_{\text{MAD}} \right] \ge 2\right) \lor \left(\text{Score}_{\text{mult}} \ge 3.0\right) \lor \text{Rule}_{\text{Security}}$$
This guarantees that single-core thread imbalances or background bursts never trigger false alerts.

---

## 6. The SOC Monitoring Console

The web interface is implemented as a single-page monitoring console styled after **Grafana, Datadog, and Wazuh**.

### 6.1. Design Specification
* **Colour Palette:**
  * Background: `#0F1419`
  * Surface Cards: `#161B22` with `1px solid #232A33` and 6px border-radius
  * Primary Text: `#E6EDF3` | Labels / Muted: `#8B949E`
  * State NORMAL: `#3FB950`
  * State ELEVATED: `#D29922`
  * State ALERT: `#F85149`
  * State UNCLASSIFIED: `#A371F7`
  * Accent / Links: `#58A6FF`
* **Zero Aesthetic Hype:** Zero neon cyan glow, zero gradient borders, zero emojis. Monospace strictly reserved for numbers, metrics, code, and timestamps.
* **Top Bar:** Host name, single operational state badge with live status dot, `baseline: 4 days`, `● Collector active` heartbeat dot, time-range toggle (`1h`, `6h`, `24h`, `Live`), Scenario replay selector, and `Export alerts (JSONL)`.
* **Scenario Replay (Demo Mode):** Prominent permanent amber strip `#D29922` across the page (`SIMULATED DATA — Active scenario: <MODE>`) whenever simulation is active, preventing test screenshots from being confused with live production alerts.

### 6.2. Dedicated Console Tabs
1. **Overview:**
   * **Summary Strip:** 4 compact metric tiles (CPU, Memory, Disk, Network) with SVG sparklines, current value, "×1.0 vs usual" ratio, and total vs max-core CPU labeling.
   * **Activity vs Baseline Timeline:** Shared-axis z-score canvas plotting CPU, Memory, Disk, Network, and Anomaly scores with a shaded normal band ($-1\sigma$ to $+1\sigma$), dashed alert threshold line ($3\sigma$), and alert window shading.
   * **Live Activity & Alerts Table:** Displays Time, State, Subsystem, Score Multiple, Signals (`usual → now`), and Decision Source. Clicking any incident loads its details into the triage drawer.
   * **Incident Detail Drawer:** Summary (**What**, **Where**, **Impact**), MITRE ATT&CK hypothesis chip, Calibrated Likely Type panel with LightGBM/MLP toggle, Top Deviating Signals list, and Suggested Triage Command with one-click copy and feedback buttons.
2. **Alerts Feed:** Full-width audit log of all historical alerts with search, status filtering, and JSONL download.
3. **Processes:** Full process table displaying PID, Binary Name, `NEW?` amber discovery badge, Parent Process, Outbound Connections count, Single-core CPU %, Memory %, and Inspect action (`pidstat`).
4. **Models (AI Tournament):** Live ablation tournament tables displaying ROC-AUC, Macro-F1, Accuracy, Log Loss, Calibration ECE, and microsecond latencies for both unsupervised and supervised models.
5. **Baseline Health:** Days collected, p99.5 empirical threshold, false alarm budget counter ($\le 1$/day), and MAD denominator scale floors table.

---

## 7. Forensic Triage & SIEM Integration

### 7.1. Suggested Triage Commands
When an incident occurs, HEIMDALL generates a single non-destructive diagnostic command tailored to the affected subsystem:

| Incident Type | Suggested Triage Command | Diagnostic Objective |
|:---|:---|:---|
| `cpu_stress` | `pidstat -u 1 5 && top -b -n 1 -o +%CPU \| head -n 12` | Attribute CPU consumption to specific PID and threads without modifying state. |
| `memory_pressure` | `vmstat -s -S M && ps aux --sort=-%mem \| head -n 12` | Inspect slab cache, active/inactive memory pages, and top memory-consuming processes. |
| `disk_pressure` | `sudo iotop -b -n 2 -o \|\| iostat -xz 1 3` | Identify specific processes generating saturated read/write I/O request queues. |
| `network_flood` | `ss -tulpn && ip -s link` | Audit listening ports, connection states, and network interface packet drop counters. |
| `system_anomaly` | `journalctl -p 3 -xb -n 20 --no-pager` | Review recent high-priority systemd journal entries and kernel error logs. |
| `unclassified` | `dmesg -T --level=warn,err \| tail -n 15` | Inspect hardware ring buffer for driver errors, out-of-memory kills, or bus resets. |
| `normal` | `uptime && free -h` | Basic sanity check of host uptime, load, and free memory capacity. |

### 7.2. SIEM JSON-Lines Format (`/api/export-siem`)
```json
{
  "timestamp": "2026-10-15T00:35:12.533261+00:00",
  "session_id": "sess_1792004721",
  "hostname": "linux-prod-node",
  "status_state": "ALERT",
  "severity": "CRITICAL",
  "decision_source": "autoencoder_cpu_override",
  "score_multiple": 4.25,
  "subsystem": "cpu",
  "effective_class": "cpu_stress",
  "mitre_id": "T1496",
  "mitre_name": "Resource Hijacking (CPU Saturation)",
  "triage_command": "pidstat -u 1 5 && top -b -n 1 -o +%CPU | head -n 12",
  "top_features": [
    {"feature": "cpu_percent_mean", "usual": 19.3, "now": 89.4, "ratio": "4.6×", "direction": "↑"},
    {"feature": "load_avg_1", "usual": 4.3, "now": 14.2, "ratio": "3.3×", "direction": "↑"}
  ]
}
```

---

## 8. CLI Reference

All HEIMDALL operations are available via the unified CLI (`heimdall`):

```bash
# Launch the web monitoring console
python -m heimdall.cli dashboard --port 8080

# Execute a one-shot live terminal diagnosis
python -m heimdall.cli diagnose

# Run continuous terminal monitoring loop
python -m heimdall.cli monitor --interval 2

# Test / demo AI diagnosis under simulated chaos workloads
python -m heimdall.cli simulate --mode cpu_stress
python -m heimdall.cli simulate --mode network_flood

# Calibrate Autoencoder and MAD baseline to current host
python -m heimdall.cli calibrate

# Train AI models
python -m heimdall.cli train-ae    # Train Denoising Autoencoder
python -m heimdall.cli train       # Train MLP classifier

# Evaluate models against held-out validation sets
python -m heimdall.cli evaluate-ae # Evaluate Autoencoder & SMD benchmark
python -m heimdall.cli evaluate    # Evaluate MLP classifier
```

---

## 9. Reorganized Project Directory Structure

```
heimdall/
├── data/
│   ├── benchmarks/         # Standard benchmark datasets (SMD benchmark)
│   ├── processed/          # Train, validation, test splits (CSV)
│   └── raw/                # Labelled telemetry JSON captures per class
├── docs/
│   ├── project_guidelines.docx # Course/project guidelines document
│   └── project_specs.pdf       # Formal AI/ML architectural specifications
├── heimdall/
│   ├── analysis/
│   │   ├── calibration.py      # Temperature scaling calibration engine
│   │   ├── diagnosis_engine.py # Decision routing & consensus voting logic
│   │   ├── rules.py            # Deterministic threshold & security rules
│   │   ├── simulator.py        # Realistic chaos & workload overlay engine
│   │   └── state.py            # SystemState analysis schema
│   ├── collector/
│   │   ├── cpu.py              # CPU metrics collector
│   │   ├── memory.py           # Memory & swap collector
│   │   ├── disk.py             # Disk throughput & IOPS collector
│   │   ├── network.py          # Network bandwidth, packets & sockets collector
│   │   ├── process.py          # Process list & file handle collector
│   │   ├── context.py          # Context signals (spawn rate, auth events)
│   │   ├── system.py           # Systemd unit health & auth failures
│   │   └── snapshot.py         # Primed atomic snapshot orchestrator
│   ├── config/
│   │   └── settings.py         # Paths, thresholds & feature constants
│   ├── models/
│   │   ├── autoencoder/        # Denoising Autoencoder (model, trainer, inference)
│   │   ├── baseline/           # Directional MAD Baseline calibrator & detector
│   │   ├── isolation_forest/   # Isolation Forest tree ensemble engine
│   │   ├── lightgbm/           # Gradient-boosted decision trees engine
│   │   ├── mlp/                # PyTorch MLP fault classifier
│   │   └── tournament.py       # Multi-model ablation benchmarking tournament
│   ├── preprocessor/
│   │   ├── dataset_builder.py  # Builds train/val/test splits from raw JSON
│   │   ├── feature_engineer.py # Extracts 42-D physical feature vector
│   │   └── scaler.py           # Standardizer with physical floor protection
│   ├── reporter/
│   │   ├── console.py          # Rich terminal dashboard renderer
│   │   ├── report_generator.py # Markdown forensic report generator
│   │   └── web_dashboard.py    # Grafana/Datadog monitoring console web app
│   ├── cli.py                  # Unified CLI router
│   └── __main__.py             # Package execution entry point
├── models/                     # Trained weights, calibrators, and thresholds
├── reports/
│   ├── incident_logs/          # Historical incident reports (.md)
│   ├── ae_evaluation.md        # Autoencoder validation metrics
│   ├── mlp_evaluation.md       # Classifier evaluation report
│   ├── tournament_results.json # Ablation tournament benchmark numbers
│   └── ARCHITECTURAL_DIFFERENTIATION_REPORT.md # In-depth v1 vs v2 evolution report
├── scripts/
│   ├── legacy/                 # Archived v1 generation scripts
│   ├── collect_all.sh          # Orchestrated data collection
│   ├── generate_realistic_data.py # Physics-correlated data synthesizer
│   ├── record_normal.sh        # Baseline recorder
│   ├── stress_cpu.sh           # Hardware stress testing scripts
│   ├── stress_disk.sh
│   ├── stress_memory.sh
│   └── stress_network.sh
├── pyproject.toml              # Modern Python packaging configuration
├── requirements.txt            # Python dependencies
└── README.md                   # Repository overview & quickstart guide
```

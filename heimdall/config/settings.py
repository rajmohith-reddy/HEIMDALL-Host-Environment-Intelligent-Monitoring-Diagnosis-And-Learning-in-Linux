"""
HEIMDALL — Configuration / Settings
====================================
All tuneable parameters live here. Changing a value here propagates throughout
the entire codebase — no magic strings scattered across modules.
"""

from __future__ import annotations

from pathlib import Path

# ── Repo root (works regardless of cwd) ──────────────────────────────────────
ROOT_DIR: Path = Path(__file__).resolve().parents[2]

DATA_RAW_DIR:       Path = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR: Path = ROOT_DIR / "data" / "processed"
DATA_BENCHMARK_DIR: Path = ROOT_DIR / "data" / "benchmarks"
MODELS_DIR:         Path = ROOT_DIR / "models"
REPORTS_DIR:        Path = ROOT_DIR / "reports"
SCRIPTS_DIR:        Path = ROOT_DIR / "scripts"

# Create directories if they don't already exist
for _d in (DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_BENCHMARK_DIR, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Telemetry collection ──────────────────────────────────────────────────────
POLL_INTERVAL_SECONDS: int   = 5     # seconds between each snapshot in monitor mode
SNAPSHOT_BUFFER_SIZE:  int   = 12    # rolling window of snapshots kept in RAM
                                     # (12 × 5 s = 60 s of history)
COLD_START_WARMUP_SEC: float = 0.4   # sampling window to prime rate counters on cold start

# ── Dataset split ratios ──────────────────────────────────────────────────────
TRAIN_RATIO: float = 0.80
VAL_RATIO:   float = 0.10
TEST_RATIO:  float = 0.10
RANDOM_SEED: int   = 42

# ── MLP class definitions ─────────────────────────────────────────────────────
CLASS_NAMES: list[str] = [
    "normal",           # 0 — healthy idle / routine operation
    "cpu_stress",       # 1 — CPU-bound workload / runaway process
    "memory_pressure",  # 2 — high RAM usage / swap storm
    "disk_pressure",    # 3 — heavy I/O saturation
    "network_flood",    # 4 — high bandwidth / packet storm
    "system_anomaly",   # 5 — flagged generic abnormal / security state
]
NUM_CLASSES: int = len(CLASS_NAMES)

# ── MLP hyper-parameters ──────────────────────────────────────────────────────
MLP_HIDDEN_DIMS:      list[int]   = [128, 64, 32]
MLP_DROPOUT_RATES:    list[float] = [0.3, 0.2, 0.0]   # one per hidden layer
MLP_LEARNING_RATE:    float = 1e-3
MLP_WEIGHT_DECAY:     float = 1e-4
MLP_BATCH_SIZE:       int   = 64
MLP_MAX_EPOCHS:       int   = 200
MLP_PATIENCE:         int   = 18     # Early Stopping patience (epochs)
MLP_LR_FACTOR:        float = 0.5    # ReduceLROnPlateau factor
MLP_LR_PATIENCE:      int   = 7      # ReduceLROnPlateau patience

# ── Autoencoder hyper-parameters ─────────────────────────────────────────────
AE_HIDDEN_DIMS:     list[int] = [32, 16]      # encoder hidden dims (decoder is mirrored)
AE_LATENT_DIM:      int       = 8             # bottleneck dimension
AE_NOISE_STD:       float     = 0.05          # Gaussian denoising std during training
AE_LEARNING_RATE:   float     = 1e-3
AE_WEIGHT_DECAY:    float     = 1e-5
AE_BATCH_SIZE:      int       = 32
AE_MAX_EPOCHS:      int       = 150
AE_PATIENCE:        int       = 15
AE_THRESHOLD_SIGMA: float     = 3.2           # 3.2σ margin above validation normal hybrid score
AE_THRESHOLD_QUANTILE: float  = 0.99          # 99th percentile empirical threshold on val
AE_TOP_K_FEATURES:  int       = 5             # top-K anomalous features used in hybrid score

# ── Model file paths ──────────────────────────────────────────────────────────
MLP_MODEL_PATH:       Path = MODELS_DIR / "mlp_best.pt"
MLP_SCALER_PATH:      Path = MODELS_DIR / "scaler.pkl"
AE_MODEL_PATH:        Path = MODELS_DIR / "ae_best.pt"
AE_SCALER_PATH:       Path = MODELS_DIR / "ae_scaler.pkl"
AE_THRESHOLD_PATH:    Path = MODELS_DIR / "ae_threshold.json"

# ── Deterministic rule thresholds ────────────────────────────────────────────
RULE_CPU_WARNING_PCT:        float = 80.0
RULE_CPU_CRITICAL_PCT:       float = 92.0
RULE_MEM_FREE_WARNING_MB:    float = 512.0
RULE_MEM_FREE_CRITICAL_MB:   float = 200.0
RULE_MEM_USED_WARNING_PCT:   float = 85.0
RULE_MEM_USED_CRITICAL_PCT:  float = 94.0
RULE_SWAP_WARNING_PCT:       float = 50.0
RULE_DISK_UTIL_WARNING_PCT:  float = 80.0
RULE_DISK_USED_WARNING_PCT:  float = 88.0
RULE_DISK_IO_WARNING_MBPS:   float = 45.0
RULE_DISK_IO_CRITICAL_MBPS:  float = 120.0
RULE_NET_DROP_WARNING_PS:    float = 5.0
RULE_AUTH_FAIL_CRITICAL:     int   = 5     # failures in last 60 s
RULE_ZOMBIE_WARNING:         int   = 4
RULE_FAILED_SERVICES_WARN:   int   = 5

# ── Severity levels (used in SystemState) ────────────────────────────────────
SEVERITY_OK:       str = "OK"
SEVERITY_WARNING:  str = "WARNING"
SEVERITY_CRITICAL: str = "CRITICAL"

# ── Feature names (must match feature_engineer.py output order exactly) ───────
FEATURE_NAMES: list[str] = [
    # CPU (9)
    "cpu_percent_mean",
    "cpu_percent_max",
    "cpu_percent_std",
    "cpu_freq_mhz",
    "load_avg_1",
    "load_avg_5",
    "load_avg_15",
    "cpu_ctx_switches_per_sec",
    "cpu_interrupts_per_sec",
    # Memory (5)
    "mem_used_pct",
    "mem_available_mb",
    "swap_used_pct",
    "swap_in_per_sec",
    "swap_out_per_sec",
    # Disk (5)
    "disk_read_mbps",
    "disk_write_mbps",
    "disk_read_iops",
    "disk_write_iops",
    "disk_used_pct",
    # Network (10)
    "net_bytes_sent_mbps",
    "net_bytes_recv_mbps",
    "net_packets_sent_per_sec",
    "net_packets_recv_per_sec",
    "net_errin_per_sec",
    "net_errout_per_sec",
    "net_dropin_per_sec",
    "net_dropout_per_sec",
    "tcp_connections",
    "udp_connections",
    # Process / System (7)
    "num_processes",
    "num_threads",
    "num_zombies",
    "open_files",
    "logged_in_users",
    "failed_services",
    "auth_failures_60s",
    # Temporal deltas (4)
    "delta_cpu_percent",
    "delta_mem_used_pct",
    "delta_disk_read_mbps",
    "delta_net_recv_mbps",
]

NUM_FEATURES: int = len(FEATURE_NAMES)

# ── Subsystem feature groupings (for localized anomaly scoring & XAI) ─────────
SUBSYSTEM_FEATURES: dict[str, list[str]] = {
    "cpu": [
        "cpu_percent_mean", "cpu_percent_max", "cpu_percent_std", "cpu_freq_mhz",
        "load_avg_1", "load_avg_5", "load_avg_15",
        "cpu_ctx_switches_per_sec", "cpu_interrupts_per_sec", "delta_cpu_percent",
    ],
    "memory": [
        "mem_used_pct", "mem_available_mb", "swap_used_pct",
        "swap_in_per_sec", "swap_out_per_sec", "delta_mem_used_pct",
    ],
    "disk": [
        "disk_read_mbps", "disk_write_mbps", "disk_read_iops",
        "disk_write_iops", "disk_used_pct", "delta_disk_read_mbps",
    ],
    "network": [
        "net_bytes_sent_mbps", "net_bytes_recv_mbps",
        "net_packets_sent_per_sec", "net_packets_recv_per_sec",
        "net_errin_per_sec", "net_errout_per_sec",
        "net_dropin_per_sec", "net_dropout_per_sec",
        "tcp_connections", "udp_connections", "delta_net_recv_mbps",
    ],
    "process": [
        "num_processes", "num_threads", "num_zombies", "open_files",
    ],
    "system": [
        "logged_in_users", "failed_services", "auth_failures_60s",
    ],
}

FEATURE_TO_SUBSYSTEM: dict[str, str] = {
    feat: sub
    for sub, feats in SUBSYSTEM_FEATURES.items()
    for feat in feats
}

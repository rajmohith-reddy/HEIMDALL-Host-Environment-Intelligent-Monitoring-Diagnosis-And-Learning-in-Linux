#!/usr/bin/env bash
# =============================================================================
# HEIMDALL — Master Data Collection Script
# =============================================================================
# Runs all stress scenarios sequentially with cooldown periods between each.
# After completion, run `python -m heimdall build-dataset` to build the
# train/val/test CSV splits.
#
# Total estimated time (default): ~12 minutes
#
# Usage:
#   bash scripts/collect_all.sh
#
# Requirements:
#   stress-ng, fio, iperf3
#   pip install -r requirements.txt
# =============================================================================

set -euo pipefail

SCENARIO_DURATION=90      # seconds per scenario
COOLDOWN=30               # seconds to cool down between scenarios
NORMAL_DURATION=150       # more normal data for a better AE baseline

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

check_tool() {
    if ! command -v "$1" &>/dev/null; then
        echo "WARNING: '$1' not found. Scenario will be skipped."
        return 1
    fi
    return 0
}

echo "══════════════════════════════════════════════════════"
echo "  HEIMDALL — Full Dataset Collection"
echo "  $(date)"
echo "══════════════════════════════════════════════════════"
echo ""
echo "  Required tools: stress-ng, fio, iperf3"
echo ""

# 1. Normal baseline (longer — needed for AE training)
echo "── [1/5] Normal baseline (${NORMAL_DURATION}s) ──────────────────────"
bash "$SCRIPT_DIR/record_normal.sh" "$NORMAL_DURATION"
echo "  Cooling down (${COOLDOWN}s) …"; sleep "$COOLDOWN"

# 2. CPU Stress
echo "── [2/5] CPU Stress (${SCENARIO_DURATION}s) ──────────────────────────"
if check_tool stress-ng; then
    bash "$SCRIPT_DIR/stress_cpu.sh" "$SCENARIO_DURATION"
else
    echo "  Skipped."
fi
echo "  Cooling down (${COOLDOWN}s) …"; sleep "$COOLDOWN"

# 3. Memory Pressure
echo "── [3/5] Memory Pressure (${SCENARIO_DURATION}s) ─────────────────────"
if check_tool stress-ng; then
    bash "$SCRIPT_DIR/stress_memory.sh" "$SCENARIO_DURATION"
else
    echo "  Skipped."
fi
echo "  Cooling down (${COOLDOWN}s) …"; sleep "$COOLDOWN"

# 4. Disk Pressure
echo "── [4/5] Disk Pressure (${SCENARIO_DURATION}s) ───────────────────────"
if check_tool fio; then
    bash "$SCRIPT_DIR/stress_disk.sh" "$SCENARIO_DURATION"
else
    echo "  Skipped."
fi
echo "  Cooling down (${COOLDOWN}s) …"; sleep "$COOLDOWN"

# 5. Network Flood
echo "── [5/5] Network Flood (${SCENARIO_DURATION}s) ───────────────────────"
if check_tool iperf3; then
    bash "$SCRIPT_DIR/stress_network.sh" "$SCENARIO_DURATION"
else
    echo "  Skipped."
fi

echo ""
echo "══════════════════════════════════════════════════════"
echo "  Collection complete! $(date)"
echo ""
echo "  Next step — build processed dataset:"
echo "    python -m heimdall build-dataset"
echo ""
echo "  Then train:"
echo "    python -m heimdall train"
echo "    python -m heimdall train-ae"
echo "══════════════════════════════════════════════════════"

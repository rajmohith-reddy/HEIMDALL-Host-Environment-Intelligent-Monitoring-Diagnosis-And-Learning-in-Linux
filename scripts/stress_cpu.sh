#!/usr/bin/env bash
# =============================================================================
# HEIMDALL Data Collection Script — CPU Stress
# =============================================================================
# Induces CPU stress using stress-ng while recording snapshots.
# Produces labelled JSON files in data/raw/cpu_stress/
#
# Requires: stress-ng (apt install stress-ng)
#
# Usage:
#   bash scripts/stress_cpu.sh [duration_seconds]
# =============================================================================

set -euo pipefail

DURATION=${1:-90}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if ! command -v stress-ng &>/dev/null; then
    echo "ERROR: stress-ng not found. Install with: sudo apt install stress-ng"
    exit 1
fi

NUM_CPUS=$(nproc)

echo "──────────────────────────────────────────────────────"
echo "  HEIMDALL — Recording CPU_STRESS"
echo "  Duration   : ${DURATION}s"
echo "  Stressors  : ${NUM_CPUS} CPU workers (stress-ng)"
echo "──────────────────────────────────────────────────────"

cd "$PROJECT_ROOT"

# Start stress-ng in background
stress-ng --cpu "$NUM_CPUS" --timeout "${DURATION}s" --quiet &
STRESS_PID=$!

echo "  stress-ng started (PID $STRESS_PID). Recording …"

python -m heimdall collect --label cpu_stress --duration "$DURATION" --interval 5

# Ensure stress-ng is stopped
kill "$STRESS_PID" 2>/dev/null || true
wait "$STRESS_PID" 2>/dev/null || true

echo ""
echo "  Done. Snapshots saved to data/raw/cpu_stress/"

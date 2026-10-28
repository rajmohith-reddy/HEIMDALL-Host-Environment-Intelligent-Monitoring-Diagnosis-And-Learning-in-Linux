#!/usr/bin/env bash
# =============================================================================
# HEIMDALL Data Collection Script — Memory Pressure
# =============================================================================
# Induces memory pressure using stress-ng VM workers while recording snapshots.
# Produces labelled JSON files in data/raw/memory_pressure/
#
# Requires: stress-ng (apt install stress-ng)
#
# Usage:
#   bash scripts/stress_memory.sh [duration_seconds]
# =============================================================================

set -euo pipefail

DURATION=${1:-90}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if ! command -v stress-ng &>/dev/null; then
    echo "ERROR: stress-ng not found. Install with: sudo apt install stress-ng"
    exit 1
fi

echo "──────────────────────────────────────────────────────"
echo "  HEIMDALL — Recording MEMORY_PRESSURE"
echo "  Duration   : ${DURATION}s"
echo "  Stressors  : 2 VM workers using 75% of RAM"
echo "──────────────────────────────────────────────────────"

cd "$PROJECT_ROOT"

stress-ng --vm 2 --vm-bytes 75% --timeout "${DURATION}s" --quiet &
STRESS_PID=$!

echo "  stress-ng VM started (PID $STRESS_PID). Recording …"

python -m heimdall collect --label memory_pressure --duration "$DURATION" --interval 5

kill "$STRESS_PID" 2>/dev/null || true
wait "$STRESS_PID" 2>/dev/null || true

echo ""
echo "  Done. Snapshots saved to data/raw/memory_pressure/"

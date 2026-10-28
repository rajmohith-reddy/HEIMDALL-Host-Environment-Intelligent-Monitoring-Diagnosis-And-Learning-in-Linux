#!/usr/bin/env bash
# =============================================================================
# HEIMDALL Data Collection Script — Normal (idle) baseline
# =============================================================================
# Records system snapshots under normal idle conditions.
# Produces labelled JSON files in data/raw/normal/
#
# Usage:
#   bash scripts/record_normal.sh [duration_seconds]
#
# Default duration: 120 seconds (generates ~24 snapshots at 5s interval)
# =============================================================================

set -euo pipefail

DURATION=${1:-120}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "──────────────────────────────────────────────────────"
echo "  HEIMDALL — Recording NORMAL baseline"
echo "  Duration : ${DURATION}s"
echo "  Label    : normal"
echo "──────────────────────────────────────────────────────"
echo ""
echo "  Tip: Do NOT stress the system during this recording."
echo "  Let it sit idle for best baseline quality."
echo ""

cd "$PROJECT_ROOT"
python -m heimdall collect --label normal --duration "$DURATION" --interval 5

echo ""
echo "  Done. Snapshots saved to data/raw/normal/"

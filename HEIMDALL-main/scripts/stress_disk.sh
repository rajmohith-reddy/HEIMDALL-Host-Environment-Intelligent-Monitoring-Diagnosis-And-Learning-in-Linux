#!/usr/bin/env bash
# =============================================================================
# HEIMDALL Data Collection Script — Disk Pressure
# =============================================================================
# Induces heavy disk I/O using fio while recording snapshots.
# Produces labelled JSON files in data/raw/disk_pressure/
#
# Requires: fio (apt install fio)
#
# Usage:
#   bash scripts/stress_disk.sh [duration_seconds]
# =============================================================================

set -euo pipefail

DURATION=${1:-90}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if ! command -v fio &>/dev/null; then
    echo "ERROR: fio not found. Install with: sudo apt install fio"
    exit 1
fi

TMP_FILE="/tmp/heimdall_fio_test"

echo "──────────────────────────────────────────────────────"
echo "  HEIMDALL — Recording DISK_PRESSURE"
echo "  Duration   : ${DURATION}s"
echo "  Tool       : fio (random read+write, 512MB)"
echo "──────────────────────────────────────────────────────"

cd "$PROJECT_ROOT"

# Run fio for slightly longer than DURATION to ensure it covers the full window
FIO_DURATION=$(( DURATION + 10 ))

fio \
  --name=heimdall_stress \
  --rw=randrw \
  --rwmixread=50 \
  --bs=4k \
  --size=512m \
  --numjobs=2 \
  --runtime="${FIO_DURATION}" \
  --time_based \
  --filename="$TMP_FILE" \
  --output-format=normal \
  --group_reporting \
  --quiet &
FIO_PID=$!

echo "  fio started (PID $FIO_PID). Recording …"
sleep 2   # let fio ramp up

python -m heimdall collect --label disk_pressure --duration "$DURATION" --interval 5

kill "$FIO_PID" 2>/dev/null || true
wait "$FIO_PID" 2>/dev/null || true
rm -f "$TMP_FILE"

echo ""
echo "  Done. Snapshots saved to data/raw/disk_pressure/"

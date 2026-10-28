#!/usr/bin/env bash
# =============================================================================
# HEIMDALL Data Collection Script — Network Flood
# =============================================================================
# Induces high network throughput using iperf3 (loopback) while recording.
# Produces labelled JSON files in data/raw/network_flood/
#
# Requires: iperf3 (apt install iperf3)
#
# Usage:
#   bash scripts/stress_network.sh [duration_seconds]
# =============================================================================

set -euo pipefail

DURATION=${1:-90}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if ! command -v iperf3 &>/dev/null; then
    echo "ERROR: iperf3 not found. Install with: sudo apt install iperf3"
    exit 1
fi

echo "──────────────────────────────────────────────────────"
echo "  HEIMDALL — Recording NETWORK_FLOOD"
echo "  Duration   : ${DURATION}s"
echo "  Tool       : iperf3 (loopback, 4 parallel streams)"
echo "──────────────────────────────────────────────────────"

cd "$PROJECT_ROOT"

# Start iperf3 server in background
iperf3 --server --daemon --port 15201 --one-off 2>/dev/null || true
sleep 1

# Run iperf3 client — loopback, 4 parallel streams
iperf3 \
  --client 127.0.0.1 \
  --port 15201 \
  --parallel 4 \
  --time "$DURATION" \
  --zerocopy \
  --logfile /tmp/heimdall_iperf3.log &
IPERF_PID=$!

echo "  iperf3 client started (PID $IPERF_PID). Recording …"
sleep 2

python -m heimdall collect --label network_flood --duration "$DURATION" --interval 5

kill "$IPERF_PID" 2>/dev/null || true
wait "$IPERF_PID" 2>/dev/null || true
pkill -f "iperf3 --server" 2>/dev/null || true

echo ""
echo "  Done. Snapshots saved to data/raw/network_flood/"

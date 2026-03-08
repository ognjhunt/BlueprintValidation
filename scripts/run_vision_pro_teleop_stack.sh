#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RECORDER_SCRIPT="${ROOT_DIR}/scripts/run_isaac_record_teleop.sh"
RELAY_SCRIPT="${ROOT_DIR}/scripts/run_vision_pro_relay.sh"

cleanup() {
  jobs -p | xargs -r kill >/dev/null 2>&1 || true
}
trap cleanup EXIT

bash "$RECORDER_SCRIPT" &
RECORDER_PID=$!
sleep 2
bash "$RELAY_SCRIPT" &
RELAY_PID=$!

echo "Recorder PID: $RECORDER_PID"
echo "Relay PID: $RELAY_PID"
echo "Waiting for either process to exit..."
wait -n "$RECORDER_PID" "$RELAY_PID"

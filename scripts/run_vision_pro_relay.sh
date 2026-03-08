#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/same_facility_policy_uplift_openvla.yaml}"
BRIDGE_HOST="${BRIDGE_HOST:-0.0.0.0}"
BRIDGE_PORT="${BRIDGE_PORT:-49111}"
TARGET_HOST="${TARGET_HOST:-127.0.0.1}"
TARGET_PORT="${TARGET_PORT:-49110}"
ISAACLAB_PYTHON="${ISAACLAB_PYTHON:-}"
PACKET_LOG_PATH="${PACKET_LOG_PATH:-$ROOT_DIR/data/outputs/vision_pro_relay_packets.jsonl}"

resolve_isaac_python() {
  if [[ -n "$ISAACLAB_PYTHON" && -x "$ISAACLAB_PYTHON" ]]; then
    echo "$ISAACLAB_PYTHON"
    return 0
  fi
  if [[ -n "${ISAAC_SIM_ROOT:-}" && -x "${ISAAC_SIM_ROOT}/python.sh" ]]; then
    echo "${ISAAC_SIM_ROOT}/python.sh"
    return 0
  fi
  if command -v python.sh >/dev/null 2>&1; then
    command -v python.sh
    return 0
  fi
  return 1
}

PYTHON_BIN="$(resolve_isaac_python || true)"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "Could not resolve an Isaac runtime Python." >&2
  echo "Run scripts/bootstrap_isaac_teleop_runtime.sh first." >&2
  exit 1
fi

echo "=== Vision Pro Relay ==="
echo "LISTEN: ${BRIDGE_HOST}:${BRIDGE_PORT}"
echo "TARGET: ${TARGET_HOST}:${TARGET_PORT}"
echo "PACKET LOG: ${PACKET_LOG_PATH}"

"$PYTHON_BIN" -m blueprint_validation.cli --config "$CONFIG_PATH" run-vision-pro-relay \
  --listen-host "$BRIDGE_HOST" \
  --listen-port "$BRIDGE_PORT" \
  --target-host "$TARGET_HOST" \
  --target-port "$TARGET_PORT" \
  --packet-log-path "$PACKET_LOG_PATH"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/same_facility_policy_uplift_openvla.yaml}"
SCENE_ROOT="${SCENE_ROOT:-}"
TASK_ID="${TASK_ID:-}"
TASK_TEXT="${TASK_TEXT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/data/outputs/teleop}"
TELEOP_DEVICE="${TELEOP_DEVICE:-keyboard}"
TASK_PACKAGE="${TASK_PACKAGE:-}"
ENV_CFG_CLASS="${ENV_CFG_CLASS:-TeleopEnvCfg}"
SUCCESS_FLAG="${SUCCESS_FLAG:-auto}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
BRIDGE_HOST="${BRIDGE_HOST:-0.0.0.0}"
BRIDGE_PORT="${BRIDGE_PORT:-49110}"
BRIDGE_CONNECT_TIMEOUT_S="${BRIDGE_CONNECT_TIMEOUT_S:-120.0}"
BRIDGE_IDLE_TIMEOUT_S="${BRIDGE_IDLE_TIMEOUT_S:-10.0}"
ISAACLAB_PYTHON="${ISAACLAB_PYTHON:-}"

if [[ -f "$ROOT_DIR/scripts/runtime_env.local" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/scripts/runtime_env.local"
fi

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

if [[ -z "$SCENE_ROOT" || -z "$TASK_ID" || -z "$TASK_TEXT" ]]; then
  echo "Required env vars:" >&2
  echo "  SCENE_ROOT=/path/to/scene_root" >&2
  echo "  TASK_ID=pick_tote" >&2
  echo "  TASK_TEXT='Pick up the tote and place it on the shelf'" >&2
  exit 1
fi

echo "=== Isaac Teleop Dry Run ==="
echo "SCENE_ROOT:   $SCENE_ROOT"
echo "TASK_ID:      $TASK_ID"
echo "TELEOP_DEVICE:$TELEOP_DEVICE"
echo "OUTPUT_DIR:   $OUTPUT_DIR"
if [[ "$TELEOP_DEVICE" == "vision_pro" ]]; then
  echo "BRIDGE_HOST:  $BRIDGE_HOST"
  echo "BRIDGE_PORT:  $BRIDGE_PORT"
fi

"$PYTHON_BIN" -m blueprint_validation.cli --config "$CONFIG_PATH" \
  validate-scene-package --scene-root "$SCENE_ROOT"

CMD=(
  "$PYTHON_BIN" -m blueprint_validation.cli --config "$CONFIG_PATH"
  record-teleop
  --scene-root "$SCENE_ROOT"
  --task-id "$TASK_ID"
  --task-text "$TASK_TEXT"
  --output-dir "$OUTPUT_DIR"
  --teleop-device "$TELEOP_DEVICE"
  --env-cfg-class "$ENV_CFG_CLASS"
  --success-flag "$SUCCESS_FLAG"
  --max-attempts "$MAX_ATTEMPTS"
)

if [[ -n "$TASK_PACKAGE" ]]; then
  CMD+=(--task-package "$TASK_PACKAGE")
fi
if [[ "$TELEOP_DEVICE" == "vision_pro" ]]; then
  CMD+=(
    --bridge-host "$BRIDGE_HOST"
    --bridge-port "$BRIDGE_PORT"
    --bridge-connect-timeout-s "$BRIDGE_CONNECT_TIMEOUT_S"
    --bridge-idle-timeout-s "$BRIDGE_IDLE_TIMEOUT_S"
  )
fi

echo "Running:"
printf '  %q' "${CMD[@]}"
echo
"${CMD[@]}"

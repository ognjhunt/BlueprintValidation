#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ISAACLAB_PYTHON="${ISAACLAB_PYTHON:-}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/same_facility_policy_uplift_openvla.yaml}"

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
  echo "Could not resolve an Isaac Lab/Isaac Sim Python runtime." >&2
  echo "Set ISAACLAB_PYTHON=/path/to/python.sh or ISAAC_SIM_ROOT=/path/to/isaac-sim." >&2
  exit 1
fi

echo "=== BlueprintValidation Isaac Teleop Bootstrap ==="
echo "ROOT_DIR:      $ROOT_DIR"
echo "CONFIG_PATH:   $CONFIG_PATH"
echo "ISAAC PYTHON:  $PYTHON_BIN"

"$PYTHON_BIN" -m pip install -e "$ROOT_DIR"

"$PYTHON_BIN" - <<'PY'
checks = ["blueprint_validation", "isaaclab", "isaaclab_tasks", "torch", "cv2"]
for mod in checks:
    try:
        __import__(mod)
        print(f"{mod}: ok")
    except Exception as exc:
        raise SystemExit(f"{mod}: missing ({type(exc).__name__}: {exc})")
PY

echo ""
echo "Bootstrap complete."
echo "Next steps:"
echo "  1) source $ROOT_DIR/scripts/runtime_env.local  # optional"
echo "  2) $ROOT_DIR/scripts/run_isaac_record_teleop.sh"

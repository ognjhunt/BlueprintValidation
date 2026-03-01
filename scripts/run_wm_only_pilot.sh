#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/wm_only_pilot.cloud.yaml}"
WORK_DIR="${WORK_DIR:-/models/outputs/wm_only_pilot}"
RESUME="${RESUME:-true}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-false}"

# Use total instance hourly burn-rate (all GPUs combined) for budget guard.
BLUEPRINT_GPU_HOURLY_RATE_USD="${BLUEPRINT_GPU_HOURLY_RATE_USD:-4.0}"
export BLUEPRINT_GPU_HOURLY_RATE_USD

if [[ -f "$ROOT_DIR/scripts/runtime_env.local" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/scripts/runtime_env.local"
fi

if [[ -f "$ROOT_DIR/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.venv/bin/activate"
fi

if ! command -v blueprint-validate >/dev/null 2>&1; then
  echo "blueprint-validate not found in PATH. Activate the project venv first."
  exit 1
fi

echo "=== WM-Only Pilot Run ==="
echo "CONFIG_PATH: $CONFIG_PATH"
echo "WORK_DIR: $WORK_DIR"
echo "RESUME: $RESUME"
echo "SKIP_PREFLIGHT: $SKIP_PREFLIGHT"
echo "BLUEPRINT_GPU_HOURLY_RATE_USD: $BLUEPRINT_GPU_HOURLY_RATE_USD"

if [[ "$SKIP_PREFLIGHT" != "true" ]]; then
  blueprint-validate --config "$CONFIG_PATH" --work-dir "$WORK_DIR" preflight
fi

RUN_CMD=(blueprint-validate --config "$CONFIG_PATH" --work-dir "$WORK_DIR" run-all)
if [[ "$RESUME" == "true" ]]; then
  RUN_CMD+=(--resume)
fi

"${RUN_CMD[@]}"

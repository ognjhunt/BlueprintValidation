#!/usr/bin/env bash
# Prepare the same-facility OpenVLA claim runtime on a fresh GPU VM.
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/app}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/same_facility_policy_uplift_openvla.cloud.yaml}"
WORK_DIR="${WORK_DIR:-/models/outputs/same_facility_first}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/models/checkpoints}"
SHARED_OUTPUT_ROOT="${SHARED_OUTPUT_ROOT:-/models/outputs}"
OPENVLA_DATASET_ROOT="${OPENVLA_DATASET_ROOT:-/models/openvla_datasets}"
DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-true}"
INSTALL_VENDOR_CUDA_EXTRAS="${INSTALL_VENDOR_CUDA_EXTRAS:-${INSTALL_DREAMDOJO_EXTRA:-true}}"
DREAMDOJO_EXTRA="${DREAMDOJO_EXTRA:-cu128}"
FACILITY_A_SPLAT_SOURCE="${FACILITY_A_SPLAT_SOURCE:-}"
FACILITY_A_TASK_HINTS_SOURCE="${FACILITY_A_TASK_HINTS_SOURCE:-}"

if [ -f "$ROOT_DIR/scripts/runtime_env.local" ]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/scripts/runtime_env.local"
fi

if ! command -v blueprint-validate >/dev/null 2>&1; then
  echo "blueprint-validate not found in PATH. Activate the project venv first."
  exit 1
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is required for checkpoint download and gated asset access."
  exit 1
fi
if [ -z "${GOOGLE_GENAI_API_KEY:-}" ]; then
  echo "GOOGLE_GENAI_API_KEY is required for Stage 0 fallback and VLM scoring."
  exit 1
fi

mkdir -p "$ROOT_DIR/data/facilities/facility_a" "$WORK_DIR/facility_a/bootstrap" "$CHECKPOINT_DIR" "$SHARED_OUTPUT_ROOT" "$OPENVLA_DATASET_ROOT"

if [ ! -f "$ROOT_DIR/data/facilities/facility_a/splat.ply" ]; then
  if [ -z "$FACILITY_A_SPLAT_SOURCE" ]; then
    echo "Missing /app/data/facilities/facility_a/splat.ply."
    echo "Set FACILITY_A_SPLAT_SOURCE to a staged facility_a splat before running this script."
    exit 1
  fi
  cp "$FACILITY_A_SPLAT_SOURCE" "$ROOT_DIR/data/facilities/facility_a/splat.ply"
fi

if [ -n "$FACILITY_A_TASK_HINTS_SOURCE" ]; then
  cp "$FACILITY_A_TASK_HINTS_SOURCE" "$WORK_DIR/facility_a/bootstrap/task_targets.synthetic.json"
fi

ROOT_DIR="$ROOT_DIR" \
CONFIG_PATH="$CONFIG_PATH" \
WORK_DIR="$WORK_DIR" \
CHECKPOINT_DIR="$CHECKPOINT_DIR" \
SHARED_OUTPUT_ROOT="$SHARED_OUTPUT_ROOT" \
OPENVLA_DATASET_ROOT="$OPENVLA_DATASET_ROOT" \
DOWNLOAD_MODELS="$DOWNLOAD_MODELS" \
INSTALL_VENDOR_CUDA_EXTRAS="$INSTALL_VENDOR_CUDA_EXTRAS" \
DREAMDOJO_EXTRA="$DREAMDOJO_EXTRA" \
PREFLIGHT_AUDIT_MODE=false \
bash "$ROOT_DIR/scripts/provision_same_facility_claim_runtime.sh"

echo ""
echo "Same-facility cloud prep complete."
echo "Run full pipeline:"
echo "  blueprint-validate --config \"$CONFIG_PATH\" --work-dir \"$WORK_DIR\" run-all"

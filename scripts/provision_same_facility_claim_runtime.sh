#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/same_facility_policy_uplift_openvla.yaml}"
WORK_DIR="${WORK_DIR:-$ROOT_DIR/data/outputs/claim_runtime_check}"
DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-false}"
INSTALL_VENDOR_CUDA_EXTRAS="${INSTALL_VENDOR_CUDA_EXTRAS:-false}"
DREAMDOJO_EXTRA="${DREAMDOJO_EXTRA:-cu128}"
SYNC_MANIPULATION_EXTRA="${SYNC_MANIPULATION_EXTRA:-false}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT_DIR/data/.mplconfig}"

echo "== Provision Same-Facility Claim Runtime =="
echo "ROOT_DIR: $ROOT_DIR"
echo "CONFIG_PATH: $CONFIG_PATH"
echo "WORK_DIR: $WORK_DIR"
echo "DOWNLOAD_MODELS: $DOWNLOAD_MODELS"
echo "INSTALL_VENDOR_CUDA_EXTRAS: $INSTALL_VENDOR_CUDA_EXTRAS"
echo "DREAMDOJO_EXTRA: $DREAMDOJO_EXTRA"
echo "SYNC_MANIPULATION_EXTRA: $SYNC_MANIPULATION_EXTRA"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is required but not found on PATH." >&2
  exit 1
fi

mkdir -p "$MPLCONFIGDIR"

echo
echo "[1/5] Syncing repo-local environment"
SYNC_ARGS=(--extra rlds)
if [[ "$SYNC_MANIPULATION_EXTRA" == "true" ]]; then
  SYNC_ARGS+=(--extra manipulation)
fi
uv sync "${SYNC_ARGS[@]}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python interpreter not found at $PYTHON_BIN" >&2
  exit 1
fi

echo
echo "[2/5] Installing compatibility/runtime Python packages into the repo venv"
uv pip install --python "$PYTHON_BIN" \
  "transformers==4.51.3" \
  "peft==0.11.1" \
  "huggingface_hub==0.30.2" \
  "natsort>=8.4.0"
uv pip install --python "$PYTHON_BIN" \
  "sam2>=1.1.0" \
  "tensorflow==2.15.0" \
  "tensorflow-datasets==4.9.3"

if [[ "$INSTALL_VENDOR_CUDA_EXTRAS" == "true" ]]; then
  echo
  echo "[3/5] Installing DreamDojo/Cosmos CUDA extras into the active venv"
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.venv/bin/activate"
  for vendor_repo in "$ROOT_DIR/data/vendor/DreamDojo" "$ROOT_DIR/data/vendor/cosmos-transfer"; do
    if [[ ! -d "$vendor_repo" ]]; then
      echo "ERROR: vendor repo missing: $vendor_repo" >&2
      exit 1
    fi
    echo "Syncing $vendor_repo with --extra=$DREAMDOJO_EXTRA"
    (
      cd "$vendor_repo"
      uv sync --extra="$DREAMDOJO_EXTRA" --active --inexact
    )
  done
else
  echo
  echo "[3/5] Skipping vendor CUDA extras"
  echo "       Set INSTALL_VENDOR_CUDA_EXTRAS=true on a CUDA host to satisfy cosmos_predict2 imports."
fi

if [[ "$DOWNLOAD_MODELS" == "true" ]]; then
  echo
  echo "[4/5] Downloading model checkpoints"
  bash "$ROOT_DIR/scripts/download_models.sh" "$ROOT_DIR/data/checkpoints"
else
  echo
  echo "[4/5] Skipping model downloads"
  echo "       Set DOWNLOAD_MODELS=true to populate $ROOT_DIR/data/checkpoints"
fi

echo
echo "[5/5] Running repo-local preflight"
MPLCONFIGDIR="$MPLCONFIGDIR" "$PYTHON_BIN" -m blueprint_validation.cli \
  --config "$CONFIG_PATH" \
  --work-dir "$WORK_DIR" \
  preflight --audit-mode

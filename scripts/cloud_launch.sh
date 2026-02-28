#!/usr/bin/env bash
# Bootstrap helper for any cloud GPU instance (RunPod/Vast/Lambda/GCP/etc.).
# Assumes repository is available at /app and GPUs are provisioned.
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/app}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/interiorgs_kitchen_0787.cloud.yaml}"
WORK_DIR="${WORK_DIR:-/models/outputs}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/models/checkpoints}"
DATASET_DIR="${DATASET_DIR:-/models/openvla_datasets}"
DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-true}"
INSTALL_DREAMDOJO_EXTRA="${INSTALL_DREAMDOJO_EXTRA:-true}"
DREAMDOJO_EXTRA="${DREAMDOJO_EXTRA:-cu128}"

echo "=== BlueprintValidation Cloud Launch Setup ==="
echo "ROOT_DIR:        $ROOT_DIR"
echo "CONFIG_PATH:     $CONFIG_PATH"
echo "WORK_DIR:        $WORK_DIR"
echo "CHECKPOINT_DIR:  $CHECKPOINT_DIR"
echo "DATASET_DIR:     $DATASET_DIR"
echo "DOWNLOAD_MODELS: $DOWNLOAD_MODELS"
echo "INSTALL_DREAMDOJO_EXTRA: $INSTALL_DREAMDOJO_EXTRA"
echo "DREAMDOJO_EXTRA: $DREAMDOJO_EXTRA"

if [ ! -f "$ROOT_DIR/.venv/bin/activate" ]; then
  echo "Virtual environment missing: $ROOT_DIR/.venv/bin/activate"
  exit 1
fi

# shellcheck disable=SC1091
source "$ROOT_DIR/.venv/bin/activate"

if ! command -v blueprint-validate >/dev/null 2>&1; then
  echo "blueprint-validate CLI not found after activating venv."
  exit 1
fi

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU not detected")
props = torch.cuda.get_device_properties(0)
total_memory = getattr(props, "total_memory", None) or getattr(props, "total_mem", None)
vram = f"{total_memory / 1024**3:.0f}GB" if total_memory else "VRAM unknown"
print(f"GPU: {torch.cuda.get_device_name(0)} ({vram})")
PY

if [ -f "$ROOT_DIR/scripts/runtime_env.local" ]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/scripts/runtime_env.local"
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is required. Export it or place it in $ROOT_DIR/scripts/runtime_env.local"
  exit 1
fi
if [ -z "${GOOGLE_GENAI_API_KEY:-}" ]; then
  echo "GOOGLE_GENAI_API_KEY is required. Export it or place it in $ROOT_DIR/scripts/runtime_env.local"
  exit 1
fi

ROOT_DIR="$ROOT_DIR" \
CONFIG_PATH="$CONFIG_PATH" \
WORK_DIR="$WORK_DIR" \
CHECKPOINT_DIR="$CHECKPOINT_DIR" \
DATASET_DIR="$DATASET_DIR" \
DOWNLOAD_MODELS="$DOWNLOAD_MODELS" \
INSTALL_DREAMDOJO_EXTRA="$INSTALL_DREAMDOJO_EXTRA" \
DREAMDOJO_EXTRA="$DREAMDOJO_EXTRA" \
bash "$ROOT_DIR/scripts/cloud_prepare_0787.sh"

echo ""
echo "Setup complete. Run:"
echo "  blueprint-validate --config \"$CONFIG_PATH\" --work-dir \"$WORK_DIR\" run-all"

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

pip_install() {
  uv pip install --python "$PYTHON_BIN" "$@"
}

verify_dreamdojo_import() {
  DREAMDOJO_REPO="$ROOT_DIR/data/vendor/DreamDojo" "$PYTHON_BIN" - <<'PY'
import os
import sys
from pathlib import Path

repo = Path(os.environ["DREAMDOJO_REPO"])
text = str(repo)
if text not in sys.path:
    sys.path.insert(0, text)

from cosmos_predict2.action_conditioned import inference as _ac_inference  # noqa: F401
print("Verified cosmos_predict2.action_conditioned.inference import.")
PY
}

verify_torchcodec_import() {
  "$PYTHON_BIN" - <<'PY'
import traceback

try:
    import torchcodec  # noqa: F401
    print("Verified torchcodec import/runtime.")
except Exception:
    print("Torchcodec runtime check failed.")
    traceback.print_exc()
    raise SystemExit(1)
PY
}

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
pip_install \
  "numpy<2" \
  "transformers==4.51.3" \
  "peft==0.11.1" \
  "huggingface_hub==0.30.2" \
  "lightning" \
  "natsort>=8.4.0"
pip_install \
  "numpy<2" \
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
  if ! verify_dreamdojo_import; then
    echo "DreamDojo import failed; installing supplemental dependencies (piq, pytorch3d)..."
    pip_install --no-deps "piq==0.8.0"
    pip_install --no-build-isolation --no-deps "git+https://github.com/facebookresearch/pytorch3d.git"
    verify_dreamdojo_import
  fi
  if ! verify_torchcodec_import; then
    echo "WARNING: torchcodec runtime probe failed. Stage 3 can hang on video datasets if torchcodec/FFmpeg is incompatible."
    echo "         Fix runtime libs before running finetune, or set BLUEPRINT_SKIP_TORCHCODEC_CHECK=1 only for non-video action datasets."
  fi
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

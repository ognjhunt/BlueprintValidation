#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/same_facility_policy_uplift_openvla.yaml}"
WORK_DIR="${WORK_DIR:-$ROOT_DIR/data/outputs/claim_runtime_check}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT_DIR/data/checkpoints}"
SHARED_OUTPUT_ROOT="${SHARED_OUTPUT_ROOT:-$ROOT_DIR/data/outputs}"
OPENVLA_DATASET_ROOT="${OPENVLA_DATASET_ROOT:-$ROOT_DIR/data/openvla_datasets}"
DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-true}"
INSTALL_VENDOR_CUDA_EXTRAS="${INSTALL_VENDOR_CUDA_EXTRAS:-true}"
DREAMDOJO_EXTRA="${DREAMDOJO_EXTRA:-cu128}"
SYNC_MANIPULATION_EXTRA="${SYNC_MANIPULATION_EXTRA:-false}"
PREFLIGHT_AUDIT_MODE="${PREFLIGHT_AUDIT_MODE:-true}"
FACILITY_A_SPLAT_SOURCE="${FACILITY_A_SPLAT_SOURCE:-}"
FACILITY_A_TASK_HINTS_SOURCE="${FACILITY_A_TASK_HINTS_SOURCE:-}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT_DIR/data/.mplconfig}"

DREAMDOJO_REPO_URL="${DREAMDOJO_REPO_URL:-https://github.com/NVIDIA/DreamDojo.git}"
DREAMDOJO_REF="${DREAMDOJO_REF:-7f3379bcb831147c0cc170e79ba08471ad186497}"
COSMOS_REPO_URL="${COSMOS_REPO_URL:-https://github.com/nvidia-cosmos/cosmos-transfer2.5.git}"
COSMOS_REF="${COSMOS_REF:-c9ad44b7283613618d57c1e4c9991916907d4f4b}"
OPENVLA_REPO_URL="${OPENVLA_REPO_URL:-https://github.com/moojink/openvla-oft.git}"
OPENVLA_REF="${OPENVLA_REF:-e4287e94541f459edc4feabc4e181f537cd569a8}"

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

sync_vendor_optional() {
  local repo_path="$1"
  shift
  if ! (
    cd "$repo_path"
    uv sync "$@" --active --inexact
  ); then
    echo "WARNING: optional vendor sync failed for $repo_path with args: $*"
    return 1
  fi
}

ensure_repo() {
  local target="$1"
  local url="$2"
  local ref="$3"

  if [[ -d "$target/.git" ]]; then
    return 0
  fi
  if [[ -f "$target/pyproject.toml" ]]; then
    return 0
  fi

  mkdir -p "$(dirname "$target")"
  git clone "$url" "$target"
  git -C "$target" checkout "$ref"
  if [[ -f "$target/.gitmodules" ]]; then
    git -C "$target" submodule update --init --recursive
  fi
}

stage_optional_file() {
  local src="$1"
  local dest="$2"
  if [[ -z "$src" ]]; then
    return 0
  fi
  mkdir -p "$(dirname "$dest")"
  cp "$src" "$dest"
}

echo "== Provision Same-Facility Claim Runtime =="
echo "ROOT_DIR: $ROOT_DIR"
echo "CONFIG_PATH: $CONFIG_PATH"
echo "WORK_DIR: $WORK_DIR"
echo "CHECKPOINT_DIR: $CHECKPOINT_DIR"
echo "SHARED_OUTPUT_ROOT: $SHARED_OUTPUT_ROOT"
echo "OPENVLA_DATASET_ROOT: $OPENVLA_DATASET_ROOT"
echo "DOWNLOAD_MODELS: $DOWNLOAD_MODELS"
echo "INSTALL_VENDOR_CUDA_EXTRAS: $INSTALL_VENDOR_CUDA_EXTRAS"
echo "DREAMDOJO_EXTRA: $DREAMDOJO_EXTRA"
echo "SYNC_MANIPULATION_EXTRA: $SYNC_MANIPULATION_EXTRA"
echo "PREFLIGHT_AUDIT_MODE: $PREFLIGHT_AUDIT_MODE"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is required but not found on PATH." >&2
  exit 1
fi

mkdir -p "$MPLCONFIGDIR"
mkdir -p "$ROOT_DIR/data" "$ROOT_DIR/data/vendor" "$CHECKPOINT_DIR" "$SHARED_OUTPUT_ROOT" "$OPENVLA_DATASET_ROOT"

if [[ ! -e "$ROOT_DIR/data/checkpoints" ]]; then
  ln -s "$CHECKPOINT_DIR" "$ROOT_DIR/data/checkpoints"
fi
if [[ ! -e "$ROOT_DIR/data/outputs" ]]; then
  ln -s "$SHARED_OUTPUT_ROOT" "$ROOT_DIR/data/outputs"
fi
if [[ ! -e "$ROOT_DIR/data/openvla_datasets" ]]; then
  ln -s "$OPENVLA_DATASET_ROOT" "$ROOT_DIR/data/openvla_datasets"
fi

echo
echo "[0/5] Ensuring pinned vendor repos and same-facility assets"
ensure_repo "$ROOT_DIR/data/vendor/DreamDojo" "$DREAMDOJO_REPO_URL" "$DREAMDOJO_REF"
ensure_repo "$ROOT_DIR/data/vendor/cosmos-transfer" "$COSMOS_REPO_URL" "$COSMOS_REF"
ensure_repo "$ROOT_DIR/data/vendor/openvla-oft" "$OPENVLA_REPO_URL" "$OPENVLA_REF"

stage_optional_file \
  "$FACILITY_A_SPLAT_SOURCE" \
  "$ROOT_DIR/data/facilities/facility_a/splat.ply"
stage_optional_file \
  "$FACILITY_A_TASK_HINTS_SOURCE" \
  "$WORK_DIR/facility_a/bootstrap/task_targets.synthetic.json"

if [[ "$CONFIG_PATH" == *"same_facility_policy_uplift_openvla"* ]] && [[ ! -f "$ROOT_DIR/data/facilities/facility_a/splat.ply" ]]; then
  echo "ERROR: missing same-facility asset $ROOT_DIR/data/facilities/facility_a/splat.ply" >&2
  echo "Set FACILITY_A_SPLAT_SOURCE=/path/to/facility_a/splat.ply or copy the asset into place before provisioning." >&2
  exit 1
fi

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
  echo "[3/5] Installing vendor runtime dependencies into the active venv"
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.venv/bin/activate"
  cosmos_repo="$ROOT_DIR/data/vendor/cosmos-transfer"
  dreamdojo_repo="$ROOT_DIR/data/vendor/DreamDojo"
  for vendor_repo in "$cosmos_repo" "$dreamdojo_repo"; do
    if [[ ! -d "$vendor_repo" ]]; then
      echo "ERROR: vendor repo missing: $vendor_repo" >&2
      exit 1
    fi
  done
  echo "Syncing $cosmos_repo without CUDA extras to install Stage-2 dependencies"
  sync_vendor_optional "$cosmos_repo" || true
  echo "Syncing $dreamdojo_repo with --extra=$DREAMDOJO_EXTRA so DreamDojo owns the active cosmos CUDA runtime"
  if ! sync_vendor_optional "$dreamdojo_repo" --extra="$DREAMDOJO_EXTRA"; then
    echo "WARNING: DreamDojo CUDA extra sync failed; continuing with the base environment and import probes."
  fi
  if ! verify_dreamdojo_import; then
    echo "DreamDojo import failed; installing supplemental dependencies (piq, pytorch3d)..."
    pip_install --no-deps "piq==0.8.0"
    if ! pip_install --no-build-isolation --no-deps "git+https://github.com/facebookresearch/pytorch3d.git"; then
      echo "WARNING: pytorch3d install failed during bootstrap; retrying DreamDojo import without it."
    fi
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
  bash "$ROOT_DIR/scripts/download_models.sh" "$CHECKPOINT_DIR"
else
  echo
  echo "[4/5] Skipping model downloads"
  echo "       Set DOWNLOAD_MODELS=true to populate $CHECKPOINT_DIR"
fi

echo
echo "[5/5] Running repo-local preflight"
PRE_CMD=(
  "$PYTHON_BIN" -m blueprint_validation.cli
  --config "$CONFIG_PATH"
  --work-dir "$WORK_DIR"
  preflight
)
if [[ "$PREFLIGHT_AUDIT_MODE" == "true" ]]; then
  PRE_CMD+=(--profile audit)
fi
MPLCONFIGDIR="$MPLCONFIGDIR" "${PRE_CMD[@]}"

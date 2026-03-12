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
NEOVERSE_REPO_URL="${NEOVERSE_REPO_URL:-}"
NEOVERSE_REPO_REF="${NEOVERSE_REPO_REF:-main}"
NEOVERSE_REPO_PATH="${NEOVERSE_REPO_PATH:-$ROOT_DIR/data/vendor/neoverse}"
NEOVERSE_PYTHON_EXECUTABLE="${NEOVERSE_PYTHON_EXECUTABLE:-$PYTHON_BIN}"
NEOVERSE_CHECKPOINT_PATH="${NEOVERSE_CHECKPOINT_PATH:-$CHECKPOINT_DIR/neoverse}"
RUNTIME_ENV_LOCAL="${RUNTIME_ENV_LOCAL:-$ROOT_DIR/scripts/runtime_env.local}"

pip_install() {
  uv pip install --python "$PYTHON_BIN" "$@"
}

ensure_python_dev_headers() {
  if "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import sysconfig
from pathlib import Path
hdr = Path(sysconfig.get_paths()["include"]) / "Python.h"
raise SystemExit(0 if hdr.exists() else 1)
PY
  then
    return 0
  fi

  if ! command -v apt-get >/dev/null 2>&1; then
    echo "WARNING: Python.h missing and apt-get unavailable; continuing without auto-install."
    return 1
  fi

  local py_ver
  py_ver="$("$PYTHON_BIN" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
  echo "Installing Python development headers for ${py_ver}..."
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-dev \
    "python${py_ver}-dev" || DEBIAN_FRONTEND=noninteractive apt-get install -y python3-dev
}

verify_dreamdojo_import() {
  DREAMDOJO_REPO="$ROOT_DIR/data/vendor/DreamDojo" TRANSFORMERS_NO_TF=1 "$PYTHON_BIN" - <<'PY'
import os
import sys
from pathlib import Path

repo = Path(os.environ["DREAMDOJO_REPO"])
text = str(repo)
if text not in sys.path:
    sys.path.insert(0, text)

from cosmos_predict2.action_conditioned_config import (  # noqa: F401
    ActionConditionedInferenceArguments,
)
from cosmos_predict2._src.predict2.inference.video2world import (  # noqa: F401
    Video2WorldInference,
)
print("Verified modern DreamDojo action-conditioned imports.")
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
  local opt_src="${4:-}"

  if [[ -d "$target/.git" ]]; then
    git -C "$target" fetch origin
    if [[ -n "$ref" ]]; then
      git -C "$target" checkout "$ref"
      git -C "$target" pull --ff-only origin "$ref" || true
    fi
    if [[ -f "$target/.gitmodules" ]]; then
      git -C "$target" submodule update --init --recursive
    fi
    return 0
  fi
  if [[ -f "$target/pyproject.toml" ]]; then
    return 0
  fi

  mkdir -p "$(dirname "$target")"
  if [[ -n "$opt_src" && -d "$opt_src" ]]; then
    ln -s "$opt_src" "$target"
    return 0
  fi
  if [[ -z "$url" ]]; then
    echo "ERROR: cannot provision $target without a repo URL or preinstalled source." >&2
    exit 1
  fi
  git clone "$url" "$target"
  git -C "$target" checkout "$ref"
  if [[ -f "$target/.gitmodules" ]]; then
    git -C "$target" submodule update --init --recursive
  fi
}

upsert_runtime_env() {
  local key="$1"
  local value="$2"
  mkdir -p "$(dirname "$RUNTIME_ENV_LOCAL")"
  python - "$RUNTIME_ENV_LOCAL" "$key" "$value" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
key = sys.argv[2]
value = sys.argv[3]
lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
prefix = f"export {key}="
new_line = f'export {key}="{value}"'
for index, line in enumerate(lines):
    if line.startswith(prefix):
        lines[index] = new_line
        break
else:
    lines.append(new_line)
path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

persist_neoverse_runtime_env() {
  upsert_runtime_env "NEOVERSE_REPO_PATH" "$NEOVERSE_REPO_PATH"
  if [[ -n "$NEOVERSE_PYTHON_EXECUTABLE" ]]; then
    upsert_runtime_env "NEOVERSE_PYTHON_EXECUTABLE" "$NEOVERSE_PYTHON_EXECUTABLE"
  fi
  if [[ -n "$NEOVERSE_CHECKPOINT_PATH" ]]; then
    upsert_runtime_env "NEOVERSE_CHECKPOINT_PATH" "$NEOVERSE_CHECKPOINT_PATH"
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
if [[ ! -d "$NEOVERSE_REPO_PATH" && ! -d "/opt/neoverse" && -z "$NEOVERSE_REPO_URL" ]]; then
  echo "NeoVerse runtime not installed; set NEOVERSE_REPO_URL or preinstall /opt/neoverse." >&2
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
ensure_repo "$NEOVERSE_REPO_PATH" "$NEOVERSE_REPO_URL" "$NEOVERSE_REPO_REF" "/opt/neoverse"
persist_neoverse_runtime_env

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
ensure_python_dev_headers || true
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
  # Vendor sync can relax core compatibility constraints; restore the repo runtime pins before probing.
  pip_install "numpy<2" "protobuf<5"
  verify_dreamdojo_import
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

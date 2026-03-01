#!/usr/bin/env bash
# Prepare canonical InteriorGS kitchen scene (0787_841244) inside a GPU container.
# Intended for cloud execution (RunPod/Vast/GCP), not local laptop setup.
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/app}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/interiorgs_kitchen_0787.cloud.yaml}"
WORK_DIR="${WORK_DIR:-/models/outputs}"
SCENE_DIR="${SCENE_DIR:-$ROOT_DIR/data/interiorgs/0787_841244}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/models/checkpoints}"
DATASET_DIR="${DATASET_DIR:-/models/openvla_datasets}"
DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-true}"
INSTALL_DREAMDOJO_EXTRA="${INSTALL_DREAMDOJO_EXTRA:-true}"
DREAMDOJO_EXTRA="${DREAMDOJO_EXTRA:-cu128}"
INSTALL_OPENPI_DEPS="${INSTALL_OPENPI_DEPS:-true}"
INSTALL_COSMOS_RUNTIME_DEPS="${INSTALL_COSMOS_RUNTIME_DEPS:-true}"
FACILITY_ID="${FACILITY_ID:-}"

if [ -f "$ROOT_DIR/scripts/runtime_env.local" ]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/scripts/runtime_env.local"
fi

if ! command -v blueprint-validate >/dev/null 2>&1; then
  echo "blueprint-validate not found in PATH. Activate the project venv first."
  exit 1
fi

pip_install() {
  if command -v uv >/dev/null 2>&1; then
    uv pip install "$@"
  else
    python -m pip install "$@"
  fi
}

verify_dreamdojo_import() {
  DREAMDOJO_REPO="$ROOT_DIR/data/vendor/DreamDojo" python - <<'PY'
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
  python - <<'PY'
import traceback

try:
    import torchcodec  # noqa: F401
    print("Verified torchcodec import/runtime.")
except Exception as exc:  # pragma: no cover - runtime probe
    print("Torchcodec runtime check failed.")
    traceback.print_exc()
    raise SystemExit(1)
PY
}

if ! command -v hf >/dev/null 2>&1 && ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "HF CLI missing; installing huggingface_hub..."
  pip_install -U huggingface_hub
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is required for gated model and dataset downloads."
  exit 1
fi
if [ -z "${GOOGLE_GENAI_API_KEY:-}" ]; then
  echo "GOOGLE_GENAI_API_KEY is required for judge/spatial/cross-site evaluations."
  exit 1
fi

if command -v hf >/dev/null 2>&1; then
  HF_AUTH_CMD=(hf auth)
else
  HF_AUTH_CMD=(huggingface-cli)
fi

download_interiorgs_assets() {
  python - "$HF_TOKEN" "$ROOT_DIR/data/interiorgs" <<'PY'
import sys
from huggingface_hub import snapshot_download

token = sys.argv[1]
local_dir = sys.argv[2]
snapshot_download(
    repo_id="spatialverse/InteriorGS",
    repo_type="dataset",
    token=token,
    local_dir=local_dir,
    allow_patterns=[
        "0787_841244/3dgs_compressed.ply",
        "0787_841244/labels.json",
        "0787_841244/structure.json",
    ],
    local_dir_use_symlinks=False,
)
print("Downloaded InteriorGS scene assets.")
PY
}

mkdir -p "$SCENE_DIR" "$WORK_DIR" "$CHECKPOINT_DIR" "$DATASET_DIR"

# Keep legacy path available for tools that still expect /app/data/checkpoints.
mkdir -p "$ROOT_DIR/data"
if [ ! -e "$ROOT_DIR/data/checkpoints" ]; then
  ln -s "$CHECKPOINT_DIR" "$ROOT_DIR/data/checkpoints"
fi

# Keep legacy outputs path available for tools that still expect /app/data/outputs.
if [ ! -e "$ROOT_DIR/data/outputs" ]; then
  ln -s "$WORK_DIR" "$ROOT_DIR/data/outputs"
fi

# Keep legacy OpenVLA dataset path available.
if [ ! -e "$ROOT_DIR/data/openvla_datasets" ]; then
  ln -s "$DATASET_DIR" "$ROOT_DIR/data/openvla_datasets"
fi

ensure_repo() {
  local target="$1"
  local opt_src="$2"
  local url="$3"
  if [ -d "$target/.git" ] || [ -f "$target/pyproject.toml" ]; then
    return 0
  fi
  mkdir -p "$(dirname "$target")"
  if [ -d "$opt_src" ]; then
    ln -s "$opt_src" "$target"
    return 0
  fi
  git clone --depth 1 "$url" "$target"
}

echo "Ensuring vendor repos..."
ensure_repo "$ROOT_DIR/data/vendor/DreamDojo" "/opt/DreamDojo" "https://github.com/NVIDIA/DreamDojo.git"
ensure_repo "$ROOT_DIR/data/vendor/cosmos-transfer" "/opt/cosmos-transfer" "https://github.com/nvidia-cosmos/cosmos-transfer2.5.git"
ensure_repo "$ROOT_DIR/data/vendor/openvla-oft" "/opt/openvla-oft" "https://github.com/moojink/openvla-oft.git"
ensure_repo "$ROOT_DIR/data/vendor/openpi" "/opt/openpi" "https://github.com/Physical-Intelligence/openpi.git"

if [ "$INSTALL_COSMOS_RUNTIME_DEPS" = "true" ]; then
  echo "Installing Cosmos runtime dependencies (sam2, natsort)..."
  pip_install -U "sam2==1.1.0" "natsort>=8.4.0"
  python - <<'PY'
import importlib
importlib.import_module("sam2")
importlib.import_module("natsort")
print("Verified sam2 + natsort imports.")
PY
else
  echo "Skipping Cosmos runtime dependency install (INSTALL_COSMOS_RUNTIME_DEPS=false)."
fi

if [ "$INSTALL_OPENPI_DEPS" = "true" ]; then
  OPENPI_REPO="$ROOT_DIR/data/vendor/openpi"
  echo "Installing pi05 runtime dependency (lerobot)..."
  pip_install -U lerobot
  echo "Verifying openpi + lerobot imports..."
  OPENPI_REPO="$OPENPI_REPO" python - <<'PY'
import importlib
import os
import sys
from pathlib import Path

repo = Path(os.environ["OPENPI_REPO"])
for candidate in (repo, repo / "src"):
    text = str(candidate)
    if candidate.exists() and text not in sys.path:
        sys.path.insert(0, text)

importlib.import_module("openpi")
importlib.import_module("lerobot")
print("Verified openpi + lerobot imports.")
PY
else
  echo "Skipping openpi dependency install/verification (INSTALL_OPENPI_DEPS=false)."
fi

echo "Authenticating with Hugging Face..."
if ! "${HF_AUTH_CMD[@]}" whoami >/dev/null 2>&1; then
  if [ "${HF_AUTH_CMD[0]}" = "hf" ]; then
    hf auth login --token "$HF_TOKEN" --add-to-git-credential >/dev/null
  else
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential >/dev/null
  fi
fi

echo "Ensuring canonical InteriorGS scene assets..."
download_interiorgs_assets >/dev/null

if [ ! -f "$SCENE_DIR/task_targets.synthetic.json" ]; then
  echo "Generating task hints from labels/structure..."
  if [ -z "$FACILITY_ID" ]; then
    FACILITY_ID="$(python - "$CONFIG_PATH" <<'PY'
import sys
from pathlib import Path
import yaml

config_path = Path(sys.argv[1])
if not config_path.exists():
    print("")
    raise SystemExit(0)
raw = yaml.safe_load(config_path.read_text()) or {}
facilities = raw.get("facilities", {})
if isinstance(facilities, dict) and facilities:
    print(next(iter(facilities.keys())))
else:
    print("")
PY
)"
  fi
  FACILITY_ID="${FACILITY_ID:-kitchen_0787}"
  blueprint-validate --config "$CONFIG_PATH" --work-dir "$WORK_DIR" bootstrap-task-hints --facility "$FACILITY_ID"
fi

if [ "$DOWNLOAD_MODELS" = "true" ]; then
  echo "Ensuring checkpoints under $CHECKPOINT_DIR..."
  bash "$ROOT_DIR/scripts/download_models.sh" "$CHECKPOINT_DIR"
else
  echo "Skipping model downloads (DOWNLOAD_MODELS=false)."
fi

if [ "$INSTALL_DREAMDOJO_EXTRA" = "true" ]; then
  echo "Installing DreamDojo CUDA extra ($DREAMDOJO_EXTRA) into active environment..."
  pip_install -e "$ROOT_DIR/data/vendor/DreamDojo[$DREAMDOJO_EXTRA]"
  echo "Ensuring DreamDojo runtime dependency (lightning) is installed..."
  pip_install -U lightning

  if ! verify_dreamdojo_import; then
    echo "DreamDojo import failed; installing supplemental dependencies (piq, pytorch3d)..."
    # Avoid upgrading torch/vision here; DreamDojo installs a CUDA-matched stack.
    pip_install --no-deps "piq==0.8.0"
    pip_install --no-build-isolation --no-deps "git+https://github.com/facebookresearch/pytorch3d.git"
    verify_dreamdojo_import
  fi

  if ! verify_torchcodec_import; then
    echo "WARNING: torchcodec runtime probe failed. Stage 3 can hang on video datasets if torchcodec/FFmpeg is incompatible."
    echo "         Fix runtime libs before running finetune, or set BLUEPRINT_SKIP_TORCHCODEC_CHECK=1 only for non-video action datasets."
  fi
fi

echo "Running preflight..."
blueprint-validate --config "$CONFIG_PATH" --work-dir "$WORK_DIR" preflight

echo ""
echo "Cloud prep complete."
echo "Run full pipeline:"
echo "  blueprint-validate --config $CONFIG_PATH --work-dir $WORK_DIR run-all"

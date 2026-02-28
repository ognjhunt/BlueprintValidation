#!/usr/bin/env bash
# Prepare canonical InteriorGS kitchen scene (0787_841244) inside a GPU container.
# Intended for cloud execution (RunPod/Vast/GCP), not local laptop setup.
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/app}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/interiorgs_kitchen_0787.cloud.yaml}"
WORK_DIR="${WORK_DIR:-$ROOT_DIR/data/outputs}"
SCENE_DIR="${SCENE_DIR:-$ROOT_DIR/data/interiorgs/0787_841244}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/models/checkpoints}"
DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-true}"
INSTALL_DREAMDOJO_EXTRA="${INSTALL_DREAMDOJO_EXTRA:-true}"
DREAMDOJO_EXTRA="${DREAMDOJO_EXTRA:-cu128}"
FACILITY_ID="${FACILITY_ID:-}"

if [ -f "$ROOT_DIR/scripts/runtime_env.local" ]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/scripts/runtime_env.local"
fi

if ! command -v blueprint-validate >/dev/null 2>&1; then
  echo "blueprint-validate not found in PATH. Activate the project venv first."
  exit 1
fi

if ! command -v hf >/dev/null 2>&1 && ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "HF CLI missing; installing huggingface_hub..."
  python -m pip install -U huggingface_hub
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
  HF_DOWNLOAD_CMD=(hf download)
else
  HF_AUTH_CMD=(huggingface-cli)
  HF_DOWNLOAD_CMD=(huggingface-cli download)
fi

mkdir -p "$SCENE_DIR" "$WORK_DIR" "$CHECKPOINT_DIR"

# Keep legacy path available for tools that still expect /app/data/checkpoints.
mkdir -p "$ROOT_DIR/data"
if [ ! -e "$ROOT_DIR/data/checkpoints" ]; then
  ln -s "$CHECKPOINT_DIR" "$ROOT_DIR/data/checkpoints"
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

echo "Authenticating with Hugging Face..."
if ! "${HF_AUTH_CMD[@]}" whoami >/dev/null 2>&1; then
  if [ "${HF_AUTH_CMD[0]}" = "hf" ]; then
    hf auth login --token "$HF_TOKEN" --add-to-git-credential >/dev/null
  else
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential >/dev/null
  fi
fi

echo "Ensuring canonical InteriorGS scene assets..."
"${HF_DOWNLOAD_CMD[@]}" spatialverse/InteriorGS \
  --repo-type dataset \
  --local-dir "$ROOT_DIR/data/interiorgs" \
  0787_841244/3dgs_compressed.ply \
  0787_841244/labels.json \
  0787_841244/structure.json \
  --token "$HF_TOKEN" >/dev/null

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
  python -m pip install -e "$ROOT_DIR/data/vendor/DreamDojo[$DREAMDOJO_EXTRA]"
fi

echo "Running preflight..."
blueprint-validate --config "$CONFIG_PATH" --work-dir "$WORK_DIR" preflight

echo ""
echo "Cloud prep complete."
echo "Run full pipeline:"
echo "  blueprint-validate --config $CONFIG_PATH --work-dir $WORK_DIR run-all"

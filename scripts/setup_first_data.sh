#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAPTURE_PIPELINE_ROOT="${CAPTURE_PIPELINE_ROOT:-/Users/nijelhunt_1/workspace/BlueprintCapturePipeline}"
IOS_CAPTURE_ROOT="${IOS_CAPTURE_ROOT:-/Users/nijelhunt_1/Desktop/BlueprintCapture}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/pilot_validation.auto.yaml}"
WORK_DIR="${WORK_DIR:-$ROOT_DIR/data/outputs/pilot}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-true}"
PROVISION_REPOS="${PROVISION_REPOS:-true}"
PROVISION_OPENPI="${PROVISION_OPENPI:-true}"
INSTALL_OPENPI_DEPS="${INSTALL_OPENPI_DEPS:-false}"
VENDOR_ROOT="${VENDOR_ROOT:-$ROOT_DIR/data/vendor}"

echo "== BlueprintValidation First-Data Setup =="
echo "BlueprintValidation root: $ROOT_DIR"
echo "Capture pipeline root:   $CAPTURE_PIPELINE_ROOT"
echo "iOS capture root:        $IOS_CAPTURE_ROOT"

if [[ ! -d "$CAPTURE_PIPELINE_ROOT" ]]; then
  echo "ERROR: capture pipeline root not found: $CAPTURE_PIPELINE_ROOT" >&2
  exit 1
fi
if [[ ! -d "$IOS_CAPTURE_ROOT" ]]; then
  echo "WARNING: iOS capture root not found: $IOS_CAPTURE_ROOT"
fi

ensure_repo() {
  local target="$1"
  local url="$2"
  if [[ -d "$target/.git" ]]; then
    echo "Repo already present: $target"
    return 0
  fi
  echo "Cloning $url -> $target"
  mkdir -p "$(dirname "$target")"
  if ! git clone --depth 1 "$url" "$target"; then
    echo "WARNING: failed to clone $url (offline/DNS or auth issue)."
    echo "         Expected path remains: $target"
  fi
}

if [[ "$PROVISION_REPOS" == "true" ]]; then
  mkdir -p "$VENDOR_ROOT"
  if [[ ! -d "/opt/DreamDojo/.git" ]]; then
    ensure_repo "$VENDOR_ROOT/DreamDojo" "https://github.com/NVIDIA/DreamDojo.git"
  else
    echo "Using existing /opt/DreamDojo"
  fi
  if [[ ! -d "/opt/cosmos-transfer/.git" ]]; then
    ensure_repo "$VENDOR_ROOT/cosmos-transfer" "https://github.com/nvidia-cosmos/cosmos-transfer2.5.git"
  else
    echo "Using existing /opt/cosmos-transfer"
  fi
  if [[ ! -d "/opt/openvla-oft/.git" ]]; then
    ensure_repo "$VENDOR_ROOT/openvla-oft" "https://github.com/moojink/openvla-oft.git"
  else
    echo "Using existing /opt/openvla-oft"
  fi
  if [[ "$PROVISION_OPENPI" == "true" ]]; then
    if [[ ! -d "/opt/openpi/.git" ]]; then
      ensure_repo "$VENDOR_ROOT/openpi" "https://github.com/Physical-Intelligence/openpi.git"
    else
      echo "Using existing /opt/openpi"
    fi
  else
    echo "Skipping openpi clone (PROVISION_OPENPI=false)."
  fi
else
  echo "Skipping repo provisioning (PROVISION_REPOS=false)."
fi

python3 "$ROOT_DIR/scripts/generate_pilot_config.py" \
  --capture-pipeline-root "$CAPTURE_PIPELINE_ROOT" \
  --output-config "$CONFIG_PATH"

CONFIG_POLICY_ADAPTER="$(python3 - "$CONFIG_PATH" <<'PY'
import sys
from pathlib import Path
import yaml

cfg_path = Path(sys.argv[1])
raw = yaml.safe_load(cfg_path.read_text()) or {}
adapter = (((raw.get("policy_adapter") or {}).get("name")) or "openvla_oft").strip()
print(adapter)
PY
)"

CONFIG_OPENPI_REPO="$(python3 - "$CONFIG_PATH" <<'PY'
import sys
from pathlib import Path
import yaml

cfg_path = Path(sys.argv[1])
raw = yaml.safe_load(cfg_path.read_text()) or {}
repo = (((raw.get("policy_adapter") or {}).get("pi05") or {}).get("openpi_repo") or "").strip()
print(repo)
PY
)"
if [[ -z "$CONFIG_OPENPI_REPO" ]]; then
  CONFIG_OPENPI_REPO="$VENDOR_ROOT/openpi"
fi

if [[ "$INSTALL_OPENPI_DEPS" == "true" ]]; then
  echo "Installing pi05 runtime dependency (lerobot) and verifying imports..."
  python3 -m pip install -U lerobot
  OPENPI_REPO="$CONFIG_OPENPI_REPO" python3 - <<'PY'
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
elif [[ "$CONFIG_POLICY_ADAPTER" == "pi05" ]]; then
  if ! OPENPI_REPO="$CONFIG_OPENPI_REPO" python3 - <<'PY'
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
PY
  then
    echo "WARNING: policy_adapter=pi05 but openpi/lerobot imports are unavailable."
    echo "         Re-run with INSTALL_OPENPI_DEPS=true or install: python3 -m pip install -U lerobot"
  fi
fi

echo
echo "Pilot config generated at:"
echo "  $CONFIG_PATH"
echo

PRE_CMD=()
if command -v blueprint-validate >/dev/null 2>&1; then
  PRE_CMD=(blueprint-validate)
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PRE_CMD=("$ROOT_DIR/.venv/bin/python" -m blueprint_validation.cli)
fi

if [[ "${#PRE_CMD[@]}" -gt 0 && "$RUN_PREFLIGHT" == "true" ]]; then
  echo "Running preflight..."
  mkdir -p "$ROOT_DIR/data/.mplconfig"
  set +e
  MPLCONFIGDIR="$ROOT_DIR/data/.mplconfig" PYTHONPATH="$ROOT_DIR/src" \
    "${PRE_CMD[@]}" --config "$CONFIG_PATH" --work-dir "$WORK_DIR" preflight
  PRE_RC=$?
  set -e
  if [[ $PRE_RC -ne 0 ]]; then
    echo
    echo "Preflight reported failures. Fix those before long GPU runs."
  fi
else
  echo "Skipping preflight execution (CLI not found or RUN_PREFLIGHT=false)."
fi

echo
echo "Next commands (stage-by-stage):"
echo "0) warmup (CPU-only pre-computation — run before GPU allocation)"
echo "   blueprint-validate --config \"$CONFIG_PATH\" --work-dir \"$WORK_DIR\" warmup"
echo "1) preflight"
echo "   blueprint-validate --config \"$CONFIG_PATH\" --work-dir \"$WORK_DIR\" preflight"
echo "2) Stage 1 render"
echo "   blueprint-validate --config \"$CONFIG_PATH\" --work-dir \"$WORK_DIR\" render --facility facility_a"
echo "3) Stage 1b robot composite (optional)"
echo "   blueprint-validate --config \"$CONFIG_PATH\" --work-dir \"$WORK_DIR\" compose-robot --facility facility_a"
echo "4) Stage 1c Gemini polish (optional)"
echo "   blueprint-validate --config \"$CONFIG_PATH\" --work-dir \"$WORK_DIR\" polish-gemini --facility facility_a"
echo "5) Stage 1d RoboSplat-style augmentation (optional)"
echo "   blueprint-validate --config \"$CONFIG_PATH\" --work-dir \"$WORK_DIR\" augment-gaussian --facility facility_a"
echo "6) Stage 2 enrich"
echo "   blueprint-validate --config \"$CONFIG_PATH\" --work-dir \"$WORK_DIR\" enrich --facility facility_a"
echo "7) Stage 3 DreamDojo finetune"
echo "   blueprint-validate --config \"$CONFIG_PATH\" --work-dir \"$WORK_DIR\" finetune --facility facility_a"
echo "8) Stage 4 frozen policy eval"
echo "   blueprint-validate --config \"$CONFIG_PATH\" --work-dir \"$WORK_DIR\" eval-policy --facility facility_a"
echo "9) Stage 4a RLDS export (optional TFRecord path for Stage 3b)"
echo "   blueprint-validate --config \"$CONFIG_PATH\" --work-dir \"$WORK_DIR\" export-rlds --facility facility_a"
echo "10) Stage 3b policy finetune (optional OpenVLA-OFT finetune from RLDS)"
echo "   blueprint-validate --config \"$CONFIG_PATH\" --work-dir \"$WORK_DIR\" finetune-policy --facility facility_a"
echo "11) Stage 3c policy RL loop (optional World-VLA-Loop style)"
echo "   blueprint-validate --config \"$CONFIG_PATH\" --work-dir \"$WORK_DIR\" rl-loop-policy --facility facility_a"
echo "12) Stage 4e trained policy eval (optional)"
echo "   blueprint-validate --config \"$CONFIG_PATH\" --work-dir \"$WORK_DIR\" eval-trained-policy --facility facility_a"
echo "13) Stage 4b rollout export"
echo "   blueprint-validate --config \"$CONFIG_PATH\" --work-dir \"$WORK_DIR\" export-rollouts --facility facility_a"
echo "14) Stage 4c paired policy training"
echo "   blueprint-validate --config \"$CONFIG_PATH\" --work-dir \"$WORK_DIR\" train-policy-pair --facility facility_a"
echo "15) Stage 4d paired policy eval"
echo "    blueprint-validate --config \"$CONFIG_PATH\" --work-dir \"$WORK_DIR\" eval-policy-pair --facility facility_a"
echo "16) Stage 5 visual"
echo "    blueprint-validate --config \"$CONFIG_PATH\" --work-dir \"$WORK_DIR\" eval-visual --facility facility_a"
echo "17) Stage 6 spatial"
echo "    blueprint-validate --config \"$CONFIG_PATH\" --work-dir \"$WORK_DIR\" eval-spatial --facility facility_a"
echo "18) Stage 7 cross-site (requires facility_b in config)"
echo "    blueprint-validate --config \"$CONFIG_PATH\" --work-dir \"$WORK_DIR\" eval-crosssite"
echo "19) report"
echo "    blueprint-validate --config \"$CONFIG_PATH\" --work-dir \"$WORK_DIR\" report --format markdown --output \"$WORK_DIR/validation_report.md\""

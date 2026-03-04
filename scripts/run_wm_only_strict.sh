#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/wm_only_strict_fresh.cloud.yaml}"
WORK_DIR="${WORK_DIR:-/models/outputs/wm_only_strict_fresh}"
RESUME="${RESUME:-true}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-false}"

BLUEPRINT_GPU_HOURLY_RATE_USD="${BLUEPRINT_GPU_HOURLY_RATE_USD:-4.0}"
export BLUEPRINT_GPU_HOURLY_RATE_USD

PINNED_GIT_COMMIT="${PINNED_GIT_COMMIT:-}"

_sha256_file() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
  else
    shasum -a 256 "$path" | awk '{print $1}'
  fi
}

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

if ! git -C "$ROOT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Strict runs require a real git checkout at $ROOT_DIR (not an unversioned copy)." >&2
  exit 2
fi

if [[ -z "$PINNED_GIT_COMMIT" ]]; then
  echo "Set PINNED_GIT_COMMIT to the exact commit to run." >&2
  exit 2
fi

ACTUAL_COMMIT="$(git -C "$ROOT_DIR" rev-parse HEAD)"
if [[ "$ACTUAL_COMMIT" != "$PINNED_GIT_COMMIT"* ]]; then
  echo "Pinned commit mismatch: expected prefix '$PINNED_GIT_COMMIT' got '$ACTUAL_COMMIT'." >&2
  exit 2
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 2
fi

export BLUEPRINT_REQUIRE_PINNED_CHECKOUT=true
export BLUEPRINT_PINNED_GIT_COMMIT="$PINNED_GIT_COMMIT"
export BLUEPRINT_EXPECT_CONFIG_HASH="$(_sha256_file "$CONFIG_PATH")"
export BLUEPRINT_EXPECT_STAGE1_CODE_HASH="$(
  cat \
    "$ROOT_DIR/src/blueprint_validation/stages/s1_render.py" \
    "$ROOT_DIR/src/blueprint_validation/rendering/stage1_active_perception.py" \
    "$ROOT_DIR/src/blueprint_validation/rendering/camera_paths.py" \
    | if command -v sha256sum >/dev/null 2>&1; then sha256sum | awk '{print $1}'; else shasum -a 256 | awk '{print $1}'; fi
)"

echo "=== WM-Only Strict Run ==="
echo "CONFIG_PATH: $CONFIG_PATH"
echo "WORK_DIR: $WORK_DIR"
echo "RESUME: $RESUME"
echo "SKIP_PREFLIGHT: $SKIP_PREFLIGHT"
echo "PINNED_GIT_COMMIT: $PINNED_GIT_COMMIT"
echo "BLUEPRINT_GPU_HOURLY_RATE_USD: $BLUEPRINT_GPU_HOURLY_RATE_USD"
echo "BLUEPRINT_EXPECT_CONFIG_HASH: $BLUEPRINT_EXPECT_CONFIG_HASH"
echo "BLUEPRINT_EXPECT_STAGE1_CODE_HASH: $BLUEPRINT_EXPECT_STAGE1_CODE_HASH"

if [[ "$SKIP_PREFLIGHT" != "true" ]]; then
  blueprint-validate --config "$CONFIG_PATH" --work-dir "$WORK_DIR" preflight
fi

RUN_CMD=(blueprint-validate --config "$CONFIG_PATH" --work-dir "$WORK_DIR" run-all)
if [[ "$RESUME" == "true" ]]; then
  RUN_CMD+=(--resume)
fi

"${RUN_CMD[@]}"

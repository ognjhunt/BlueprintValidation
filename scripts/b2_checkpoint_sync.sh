#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Sync a local BlueprintValidation snapshot folder with Backblaze B2 via rclone.

Usage:
  bash scripts/b2_checkpoint_sync.sh push
  bash scripts/b2_checkpoint_sync.sh pull

Optional env:
  B2_BUCKET=blueprint-validation-checkpoints     # unless RCLONE_REMOTE_PATH is set
  INSTANCE_ID=<vast instance id>                 # default: unscoped
  SNAPSHOT_TAG=latest                            # use timestamp folder names if preferred
  LOCAL_BACKUP_ROOT=$HOME/BlueprintValidationBackups/vast/<instance_id>
  LOCAL_SNAPSHOT_DIR=<explicit local snapshot dir>
  B2_PREFIX=blueprint-validation/vast
  RCLONE_REMOTE=b2                               # preconfigured rclone remote name
  RCLONE_REMOTE_PATH=<full remote path override>
  B2_APPLICATION_KEY_ID=<scoped app key id>      # optional ephemeral rclone config
  B2_APPLICATION_KEY=<scoped app key secret>     # optional ephemeral rclone config
  RCLONE_TRANSFERS=8
  RCLONE_CHECKERS=16
  RCLONE_PROGRESS=false
  RCLONE_OPERATION=sync                        # sync (mirrors) or copy (append/update only)
EOF
}

MODE="${1:-push}"
if [[ "$MODE" != "push" && "$MODE" != "pull" ]]; then
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/runtime_env.local" ]]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/runtime_env.local"
fi

INSTANCE_ID="${INSTANCE_ID:-unscoped}"
SNAPSHOT_TAG="${SNAPSHOT_TAG:-latest}"
LOCAL_BACKUP_ROOT="${LOCAL_BACKUP_ROOT:-$HOME/BlueprintValidationBackups/vast/$INSTANCE_ID}"
LOCAL_SNAPSHOT_DIR="${LOCAL_SNAPSHOT_DIR:-$LOCAL_BACKUP_ROOT/$SNAPSHOT_TAG}"
B2_PREFIX="${B2_PREFIX:-blueprint-validation/vast}"
B2_BUCKET="${B2_BUCKET:-blueprint-validation-checkpoints}"
RCLONE_REMOTE="${RCLONE_REMOTE:-b2}"
RCLONE_REMOTE_PATH="${RCLONE_REMOTE_PATH:-}"
RCLONE_TRANSFERS="${RCLONE_TRANSFERS:-8}"
RCLONE_CHECKERS="${RCLONE_CHECKERS:-16}"
RCLONE_PROGRESS="${RCLONE_PROGRESS:-false}"
RCLONE_OPERATION="${RCLONE_OPERATION:-sync}"

if ! command -v rclone >/dev/null 2>&1; then
  echo "rclone is required. Install it first (https://rclone.org/downloads/)."
  exit 1
fi

if [[ "$RCLONE_OPERATION" != "sync" && "$RCLONE_OPERATION" != "copy" ]]; then
  echo "RCLONE_OPERATION must be 'sync' or 'copy' (got '$RCLONE_OPERATION')."
  exit 1
fi

if [[ "$MODE" == "push" && ! -e "$LOCAL_SNAPSHOT_DIR" ]]; then
  echo "Local snapshot path does not exist: $LOCAL_SNAPSHOT_DIR"
  exit 1
fi

if [[ "$MODE" == "pull" ]]; then
  mkdir -p "$LOCAL_SNAPSHOT_DIR"
fi

build_remote_path() {
  if [[ -n "$RCLONE_REMOTE_PATH" ]]; then
    echo "$RCLONE_REMOTE_PATH"
    return 0
  fi

  local prefix="${B2_PREFIX%/}/${INSTANCE_ID}/${SNAPSHOT_TAG}"

  # Prefer ephemeral config when key material is provided via env.
  if [[ -n "${B2_APPLICATION_KEY_ID:-}" && -n "${B2_APPLICATION_KEY:-}" ]]; then
    echo "b2ephem:${B2_BUCKET}/${prefix}"
    return 0
  fi

  echo "${RCLONE_REMOTE}:${B2_BUCKET}/${prefix}"
}

# Note: exports must happen in the parent shell (not inside command substitution).
if [[ -n "${B2_APPLICATION_KEY_ID:-}" && -n "${B2_APPLICATION_KEY:-}" ]]; then
  export RCLONE_CONFIG_B2EPHEM_TYPE="b2"
  export RCLONE_CONFIG_B2EPHEM_ACCOUNT="$B2_APPLICATION_KEY_ID"
  export RCLONE_CONFIG_B2EPHEM_KEY="$B2_APPLICATION_KEY"
fi

REMOTE_PATH="$(build_remote_path)"

rclone_args=(
  "--create-empty-src-dirs"
  "--transfers" "$RCLONE_TRANSFERS"
  "--checkers" "$RCLONE_CHECKERS"
)
if [[ "$RCLONE_PROGRESS" == "true" ]]; then
  rclone_args+=("--progress")
fi

if [[ "$MODE" == "push" ]]; then
  echo "[b2 push:${RCLONE_OPERATION}] ${LOCAL_SNAPSHOT_DIR}/ -> ${REMOTE_PATH}/"
  rclone "$RCLONE_OPERATION" "${LOCAL_SNAPSHOT_DIR}/" "${REMOTE_PATH}/" "${rclone_args[@]}"
else
  echo "[b2 pull:${RCLONE_OPERATION}] ${REMOTE_PATH}/ -> ${LOCAL_SNAPSHOT_DIR}/"
  rclone "$RCLONE_OPERATION" "${REMOTE_PATH}/" "${LOCAL_SNAPSHOT_DIR}/" "${rclone_args[@]}"
fi

echo "B2 sync complete (${MODE})."

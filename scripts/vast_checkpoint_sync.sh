#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Sync resumable BlueprintValidation run state between a Vast.ai instance and local disk.

Usage:
  bash scripts/vast_checkpoint_sync.sh pull
  bash scripts/vast_checkpoint_sync.sh push

Required env:
  INSTANCE_ID=<vast instance id>

Optional env:
  REMOTE_ROOT=/workspace/BlueprintValidation
  REMOTE_WORK_DIR=/models/outputs
  LOCAL_BACKUP_ROOT=$HOME/BlueprintValidationBackups/vast/<instance_id>
  SNAPSHOT_TAG=latest        # use "timestamp" to write YYYYmmddTHHMMSSZ folders
  STRICT_SYNC=false          # true: fail fast on first copy error
  SSH_HOST=<instance ssh host>  # auto-resolved from vastai if omitted
  SSH_PORT=<instance ssh port>  # auto-resolved from vastai if omitted
  ENABLE_B2_SYNC=auto        # auto|true|false
  B2_SYNC_MODE=auto          # auto|push|pull|off
EOF
}

MODE="${1:-pull}"
if [[ "$MODE" != "pull" && "$MODE" != "push" ]]; then
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/runtime_env.local" ]]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/runtime_env.local"
fi

INSTANCE_ID="${INSTANCE_ID:-}"
if [[ -z "$INSTANCE_ID" ]]; then
  echo "INSTANCE_ID is required."
  usage
  exit 1
fi

REMOTE_ROOT="${REMOTE_ROOT:-/workspace/BlueprintValidation}"
REMOTE_WORK_DIR="${REMOTE_WORK_DIR:-/models/outputs}"
LOCAL_BACKUP_ROOT="${LOCAL_BACKUP_ROOT:-$HOME/BlueprintValidationBackups/vast/$INSTANCE_ID}"
SNAPSHOT_TAG="${SNAPSHOT_TAG:-latest}"
STRICT_SYNC="${STRICT_SYNC:-false}"
ENABLE_B2_SYNC="${ENABLE_B2_SYNC:-auto}"
B2_SYNC_MODE="${B2_SYNC_MODE:-auto}"

if [[ "$SNAPSHOT_TAG" == "timestamp" ]]; then
  SNAPSHOT_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
fi

LOCAL_SNAPSHOT_DIR="${LOCAL_BACKUP_ROOT}/${SNAPSHOT_TAG}"
mkdir -p "$LOCAL_SNAPSHOT_DIR"

SYNC_ITEMS=(
  "${REMOTE_WORK_DIR}|work_dir"
  "/models/openvla_datasets|data/openvla_datasets_models"
  "${REMOTE_ROOT}/data/outputs|data/outputs_legacy"
  "${REMOTE_ROOT}/data/openvla_datasets|data/openvla_datasets"
  "${REMOTE_ROOT}/data/interiorgs/0787_841244/task_targets.synthetic.json|data/interiorgs/task_targets.synthetic.json"
)

failures=0

resolve_enable_b2_sync() {
  case "$ENABLE_B2_SYNC" in
    true) echo "true" ;;
    false) echo "false" ;;
    auto)
      if [[ -n "${RCLONE_REMOTE_PATH:-}" ]]; then
        echo "true"
      elif [[ -n "${B2_APPLICATION_KEY_ID:-}" && -n "${B2_APPLICATION_KEY:-}" ]]; then
        echo "true"
      elif [[ -n "${B2_BUCKET:-}" ]]; then
        echo "true"
      else
        echo "false"
      fi
      ;;
    *)
      echo "Invalid ENABLE_B2_SYNC='${ENABLE_B2_SYNC}'. Expected auto|true|false." >&2
      exit 1
      ;;
  esac
}

resolve_ssh_target() {
  if [[ -n "${SSH_HOST:-}" && -n "${SSH_PORT:-}" ]]; then
    return 0
  fi

  local meta
  meta="$(vastai show instance "$INSTANCE_ID" --raw)"
  SSH_HOST="$(
    python -c 'import json,sys; print(json.loads(sys.stdin.read()).get("ssh_host",""))' <<<"$meta"
  )"
  SSH_PORT="$(
    python -c 'import json,sys; print(json.loads(sys.stdin.read()).get("ssh_port",""))' <<<"$meta"
  )"
  if [[ -z "$SSH_HOST" || -z "$SSH_PORT" ]]; then
    echo "Failed to resolve SSH host/port for instance ${INSTANCE_ID}."
    exit 1
  fi
}

resolve_b2_mode() {
  case "$B2_SYNC_MODE" in
    auto)
      if [[ "$MODE" == "pull" ]]; then
        echo "push"
      else
        echo "pull"
      fi
      ;;
    push|pull)
      echo "$B2_SYNC_MODE"
      ;;
    off)
      echo ""
      ;;
    *)
      echo "Invalid B2_SYNC_MODE='${B2_SYNC_MODE}'. Expected auto|push|pull|off." >&2
      exit 1
      ;;
  esac
}

ssh_opts=()
ssh_target=""
ssh_rsync_cmd=""

remote_path_type() {
  local remote_path="$1"
  ssh "${ssh_opts[@]}" "$ssh_target" \
    "if [ -d '$remote_path' ]; then echo dir; elif [ -f '$remote_path' ]; then echo file; else echo missing; fi"
}

copy_one() {
  local remote_path="$1"
  local local_rel="$2"
  local local_abs="${LOCAL_SNAPSHOT_DIR}/${local_rel}"
  local remote_kind local_kind

  if [[ "$MODE" == "pull" ]]; then
    remote_kind="$(remote_path_type "$remote_path")"
    if [[ "$remote_kind" == "missing" ]]; then
      echo "[skip] Missing remote path: $remote_path"
      return 0
    fi
    mkdir -p "$(dirname "$local_abs")"
    if [[ "$remote_kind" == "dir" ]]; then
      mkdir -p "$local_abs"
      echo "[pull] ${ssh_target}:${remote_path}/ -> ${local_abs}/"
      rsync -az --delete -e "$ssh_rsync_cmd" \
        "${ssh_target}:${remote_path}/" \
        "${local_abs}/"
    else
      echo "[pull] ${ssh_target}:${remote_path} -> ${local_abs}"
      rsync -az -e "$ssh_rsync_cmd" \
        "${ssh_target}:${remote_path}" \
        "$local_abs"
    fi
  else
    if [[ ! -e "$local_abs" ]]; then
      echo "[skip] Missing local path for restore: $local_abs"
      return 0
    fi
    local_kind="file"
    if [[ -d "$local_abs" ]]; then
      local_kind="dir"
    fi
    ssh "${ssh_opts[@]}" "$ssh_target" "mkdir -p '$(dirname "$remote_path")'"
    if [[ "$local_kind" == "dir" ]]; then
      echo "[push] ${local_abs}/ -> ${ssh_target}:${remote_path}/"
      rsync -az --delete -e "$ssh_rsync_cmd" \
        "${local_abs}/" \
        "${ssh_target}:${remote_path}/"
    else
      echo "[push] ${local_abs} -> ${ssh_target}:${remote_path}"
      rsync -az -e "$ssh_rsync_cmd" \
        "$local_abs" \
        "${ssh_target}:${remote_path}"
    fi
  fi

  echo "[ok] $remote_path"
  return 0
}

resolve_ssh_target
ssh_opts=(-o StrictHostKeyChecking=no -o BatchMode=yes -p "$SSH_PORT")
ssh_target="root@${SSH_HOST}"
ssh_rsync_cmd="ssh -o StrictHostKeyChecking=no -o BatchMode=yes -p ${SSH_PORT}"

b2_enabled="$(resolve_enable_b2_sync)"
b2_mode=""
b2_pre_sync_ran="false"
if [[ "$b2_enabled" == "true" ]]; then
  b2_mode="$(resolve_b2_mode)"
  if [[ "$MODE" == "push" && "$b2_mode" == "pull" ]]; then
    echo "Pre-sync B2 restore (pull) before remote push..."
    INSTANCE_ID="$INSTANCE_ID" \
      SNAPSHOT_TAG="$SNAPSHOT_TAG" \
      LOCAL_BACKUP_ROOT="$LOCAL_BACKUP_ROOT" \
      LOCAL_SNAPSHOT_DIR="$LOCAL_SNAPSHOT_DIR" \
      bash "${SCRIPT_DIR}/b2_checkpoint_sync.sh" pull
    b2_pre_sync_ran="true"
  fi
fi

for item in "${SYNC_ITEMS[@]}"; do
  remote_path="${item%%|*}"
  local_rel="${item#*|}"
  if ! copy_one "$remote_path" "$local_rel"; then
    echo "[warn] Copy failed: $remote_path"
    failures=$((failures + 1))
    if [[ "$STRICT_SYNC" == "true" ]]; then
      exit 1
    fi
  fi
done

manifest_path="${LOCAL_SNAPSHOT_DIR}/sync_manifest.txt"
{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "instance_id=${INSTANCE_ID}"
  echo "mode=${MODE}"
  echo "remote_root=${REMOTE_ROOT}"
  echo "remote_work_dir=${REMOTE_WORK_DIR}"
  echo "ssh_host=${SSH_HOST}"
  echo "ssh_port=${SSH_PORT}"
  echo "snapshot_tag=${SNAPSHOT_TAG}"
  echo "strict_sync=${STRICT_SYNC}"
} >"$manifest_path"

if [[ "$failures" -gt 0 ]]; then
  echo "Completed with ${failures} copy warning(s). Manifest: $manifest_path"
  exit 1
fi

echo "Sync complete. Manifest: $manifest_path"

if [[ "$b2_enabled" == "true" ]]; then
  if [[ -n "${b2_mode:-}" && "$b2_pre_sync_ran" != "true" ]]; then
    echo "Chaining B2 sync (${b2_mode})..."
    INSTANCE_ID="$INSTANCE_ID" \
      SNAPSHOT_TAG="$SNAPSHOT_TAG" \
      LOCAL_BACKUP_ROOT="$LOCAL_BACKUP_ROOT" \
      LOCAL_SNAPSHOT_DIR="$LOCAL_SNAPSHOT_DIR" \
      bash "${SCRIPT_DIR}/b2_checkpoint_sync.sh" "$b2_mode"
  fi
fi

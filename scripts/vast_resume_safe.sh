#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INSTANCE_ID="${INSTANCE_ID:-}"
if [[ -z "$INSTANCE_ID" ]]; then
  echo "INSTANCE_ID is required."
  exit 1
fi

RESTORE_AFTER_START="${RESTORE_AFTER_START:-false}"
SNAPSHOT_TAG="${SNAPSHOT_TAG:-latest}"
LOCAL_BACKUP_ROOT="${LOCAL_BACKUP_ROOT:-$HOME/BlueprintValidationBackups/vast/$INSTANCE_ID}"

echo "Starting instance ${INSTANCE_ID}..."
vastai start instance "$INSTANCE_ID"

if [[ "$RESTORE_AFTER_START" == "true" ]]; then
  echo "Restoring checkpoint snapshot '${SNAPSHOT_TAG}'..."
  INSTANCE_ID="$INSTANCE_ID" \
    SNAPSHOT_TAG="$SNAPSHOT_TAG" \
    LOCAL_BACKUP_ROOT="$LOCAL_BACKUP_ROOT" \
    bash "$SCRIPT_DIR/vast_checkpoint_sync.sh" push
fi

echo "Resume flow complete."

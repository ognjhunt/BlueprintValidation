#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INSTANCE_ID="${INSTANCE_ID:-}"
if [[ -z "$INSTANCE_ID" ]]; then
  echo "INSTANCE_ID is required."
  exit 1
fi

SNAPSHOT_TAG="${SNAPSHOT_TAG:-timestamp}"
LOCAL_BACKUP_ROOT="${LOCAL_BACKUP_ROOT:-$HOME/BlueprintValidationBackups/vast/$INSTANCE_ID}"
SKIP_SYNC="${SKIP_SYNC:-false}"

echo "Preparing pause-safe stop for instance ${INSTANCE_ID}."

if [[ "$SKIP_SYNC" != "true" ]]; then
  echo "Running final checkpoint sync before stop..."
  INSTANCE_ID="$INSTANCE_ID" \
    SNAPSHOT_TAG="$SNAPSHOT_TAG" \
    LOCAL_BACKUP_ROOT="$LOCAL_BACKUP_ROOT" \
    bash "$SCRIPT_DIR/vast_checkpoint_sync.sh" pull
else
  echo "Skipping final sync (SKIP_SYNC=true)."
fi

echo "Stopping instance ${INSTANCE_ID}..."
vastai stop instance "$INSTANCE_ID"

echo "Instance stopped. Resume later with:"
echo "  vastai start instance ${INSTANCE_ID}"

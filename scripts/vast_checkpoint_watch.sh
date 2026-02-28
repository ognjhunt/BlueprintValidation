#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INSTANCE_ID="${INSTANCE_ID:-}"
if [[ -z "$INSTANCE_ID" ]]; then
  echo "INSTANCE_ID is required."
  exit 1
fi

INTERVAL_MINUTES="${INTERVAL_MINUTES:-30}"
SNAPSHOT_TAG="${SNAPSHOT_TAG:-latest}"
RUN_ONCE="${RUN_ONCE:-false}"
LOCAL_BACKUP_ROOT="${LOCAL_BACKUP_ROOT:-$HOME/BlueprintValidationBackups/vast/$INSTANCE_ID}"

if ! [[ "$INTERVAL_MINUTES" =~ ^[0-9]+$ ]] || [[ "$INTERVAL_MINUTES" -lt 1 ]]; then
  echo "INTERVAL_MINUTES must be a positive integer."
  exit 1
fi

echo "Starting periodic checkpoint sync for instance ${INSTANCE_ID}."
echo "Backup root: ${LOCAL_BACKUP_ROOT}"
echo "Interval: ${INTERVAL_MINUTES} minutes"
echo "Snapshot tag mode: ${SNAPSHOT_TAG}"

while true; do
  echo ""
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Sync cycle start"
  if INSTANCE_ID="$INSTANCE_ID" \
    SNAPSHOT_TAG="$SNAPSHOT_TAG" \
    LOCAL_BACKUP_ROOT="$LOCAL_BACKUP_ROOT" \
    bash "$SCRIPT_DIR/vast_checkpoint_sync.sh" pull; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Sync cycle complete"
  else
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Sync cycle failed"
  fi

  if [[ "$RUN_ONCE" == "true" ]]; then
    break
  fi

  sleep "$((INTERVAL_MINUTES * 60))"
done

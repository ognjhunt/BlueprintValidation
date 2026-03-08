#!/usr/bin/env bash
set -euo pipefail

REMOTE_TARGET="${REMOTE_TARGET:-}"
REMOTE_ROOT="${REMOTE_ROOT:-}"
SSH_PORT="${SSH_PORT:-22}"

if [[ -z "$REMOTE_TARGET" || -z "$REMOTE_ROOT" ]]; then
  echo "Required env vars:" >&2
  echo "  REMOTE_TARGET=user@host" >&2
  echo "  REMOTE_ROOT=/path/to/BlueprintValidation" >&2
  exit 1
fi

ssh -p "$SSH_PORT" "$REMOTE_TARGET" \
  "cd '$REMOTE_ROOT' && bash '$REMOTE_ROOT/scripts/run_isaac_record_teleop.sh'"

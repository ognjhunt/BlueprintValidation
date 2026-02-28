#!/usr/bin/env bash
# Backwards-compatible alias for older docs/scripts.
set -euo pipefail
echo "Notice: scripts/runpod_launch.sh is an alias; using scripts/cloud_launch.sh"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cloud_launch.sh" "$@"

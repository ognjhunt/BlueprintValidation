#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/app}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/qualified_opportunity_validation.yaml}"
WORK_DIR="${WORK_DIR:-/models/outputs}"
RUNTIME_ENV_LOCAL="${RUNTIME_ENV_LOCAL:-$ROOT_DIR/scripts/runtime_env.local}"
NEOVERSE_RUNTIME_SERVICE_URL="${NEOVERSE_RUNTIME_SERVICE_URL:-}"
NEOVERSE_RUNTIME_SERVICE_TIMEOUT_SECONDS="${NEOVERSE_RUNTIME_SERVICE_TIMEOUT_SECONDS:-120}"

if [ -f "$RUNTIME_ENV_LOCAL" ]; then
  # shellcheck disable=SC1090
  source "$RUNTIME_ENV_LOCAL"
fi

if ! command -v blueprint-validate >/dev/null 2>&1; then
  echo "blueprint-validate not found in PATH. Activate the project environment first."
  exit 1
fi

if [ -z "${NEOVERSE_RUNTIME_SERVICE_URL:-}" ]; then
  echo "NeoVerse runtime service URL not configured; set NEOVERSE_RUNTIME_SERVICE_URL."
  exit 1
fi

mkdir -p "$WORK_DIR"

upsert_runtime_env() {
  local key="$1"
  local value="$2"
  mkdir -p "$(dirname "$RUNTIME_ENV_LOCAL")"
  python - "$RUNTIME_ENV_LOCAL" "$key" "$value" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
key = sys.argv[2]
value = sys.argv[3]
lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
prefix = f"export {key}="
new_line = f'export {key}="{value}"'
for index, line in enumerate(lines):
    if line.startswith(prefix):
        lines[index] = new_line
        break
else:
    lines.append(new_line)
path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

persist_neoverse_runtime_env() {
  upsert_runtime_env "NEOVERSE_RUNTIME_SERVICE_URL" "$NEOVERSE_RUNTIME_SERVICE_URL"
  upsert_runtime_env "NEOVERSE_RUNTIME_SERVICE_TIMEOUT_SECONDS" "$NEOVERSE_RUNTIME_SERVICE_TIMEOUT_SECONDS"
}

persist_neoverse_runtime_env

echo "Running NeoVerse service preflight..."
blueprint-validate --config "$CONFIG_PATH" --work-dir "$WORK_DIR" preflight

echo
echo "Cloud prepare complete."
echo "NeoVerse runtime service: $NEOVERSE_RUNTIME_SERVICE_URL"
echo "Config: $CONFIG_PATH"
echo "Work dir: $WORK_DIR"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/qualified_opportunity_validation.yaml}"
WORK_DIR="${WORK_DIR:-$ROOT_DIR/data/outputs/runtime_service_check}"
RUNTIME_ENV_LOCAL="${RUNTIME_ENV_LOCAL:-$ROOT_DIR/scripts/runtime_env.local}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
NEOVERSE_RUNTIME_SERVICE_URL="${NEOVERSE_RUNTIME_SERVICE_URL:-}"
NEOVERSE_RUNTIME_SERVICE_TIMEOUT_SECONDS="${NEOVERSE_RUNTIME_SERVICE_TIMEOUT_SECONDS:-120}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python runtime not found at $PYTHON_BIN" >&2
  exit 1
fi

if [[ -f "$RUNTIME_ENV_LOCAL" ]]; then
  # shellcheck disable=SC1090
  source "$RUNTIME_ENV_LOCAL"
fi

if [[ -z "$NEOVERSE_RUNTIME_SERVICE_URL" ]]; then
  echo "NeoVerse runtime service URL not configured; set NEOVERSE_RUNTIME_SERVICE_URL." >&2
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

echo "== Provision NeoVerse Runtime Validation Environment =="
echo "ROOT_DIR: $ROOT_DIR"
echo "CONFIG_PATH: $CONFIG_PATH"
echo "WORK_DIR: $WORK_DIR"
echo "NEOVERSE_RUNTIME_SERVICE_URL: $NEOVERSE_RUNTIME_SERVICE_URL"

persist_neoverse_runtime_env

if command -v uv >/dev/null 2>&1; then
  echo "Syncing local environment..."
  (cd "$ROOT_DIR" && uv sync)
fi

echo "Running preflight..."
blueprint-validate --config "$CONFIG_PATH" --work-dir "$WORK_DIR" preflight

echo
echo "Provisioning complete."

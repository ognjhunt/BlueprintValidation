#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_CONFIG="${LOCAL_CONFIG:-$ROOT_DIR/configs/interiorgs_kitchen_0787.yaml}"
CLOUD_CONFIG="${CLOUD_CONFIG:-$ROOT_DIR/configs/interiorgs_kitchen_0787.cloud.yaml}"
WORK_DIR="${WORK_DIR:-$ROOT_DIR/data/outputs/pre_gpu_audit}"
RUN_CLOUD_PREFLIGHT="${RUN_CLOUD_PREFLIGHT:-auto}"  # auto|true|false

PYTEST_TARGETS=(
  tests/test_preflight.py
  tests/test_pipeline_smoke.py
  tests/test_pipeline_stages_order.py
  tests/test_config.py
  tests/test_dreamdojo_finetune.py
  tests/test_cosmos_runner.py
  tests/test_openvla_finetune.py
  tests/test_policy_rl_loop.py
  tests/test_cli_preflight.py
  tests/test_docker_contract.py
)

declare -a SUMMARY_LINES=()
HAS_FAILURE=0

ensure_python_module_cli() {
  local module="$1"
  local package="${2:-$module}"
  if python -m "$module" --version >/dev/null 2>&1; then
    return 0
  fi
  echo "Installing missing audit dependency: $package"
  python -m pip install "$package"
}

run_step() {
  local name="$1"
  shift
  echo ""
  echo "=== $name ==="
  if "$@"; then
    SUMMARY_LINES+=("PASS|$name")
  else
    SUMMARY_LINES+=("FAIL|$name")
    HAS_FAILURE=1
  fi
}

secret_scan() {
  local output
  if ! output="$(
    python - "$ROOT_DIR" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
pattern = re.compile(r"(AIza|hf_[A-Za-z0-9]{10,})")

ignored_roots = {
    root / ".git",
    root / ".venv",
    root / "data" / "vendor",
}
ignored_files = {
    root / "README.md",
    root / "scripts" / "pre_gpu_audit.sh",
    root / "scripts" / "runtime_env.local",
}

hits: list[str] = []
for path in root.rglob("*"):
    if path.is_dir():
        continue
    if any(str(path).startswith(str(ignored) + "/") for ignored in ignored_roots):
        continue
    if path in ignored_files:
        continue
    if "__pycache__" in path.parts:
        continue

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue

    rel = f"./{path.relative_to(root)}"
    for idx, line in enumerate(text.splitlines(), start=1):
        if pattern.search(line):
            hits.append(f"{rel}:{idx}:{line.strip()}")

if hits:
    print("\n".join(hits))
    raise SystemExit(1)
PY
  )"; then
    echo "Potential secrets detected:"
    if [[ -n "$output" ]]; then
      echo "$output"
    fi
    return 1
  fi

  echo "Secret scan clean."
  return 0
}

maybe_run_cloud_preflight() {
  local should_run=0
  case "$RUN_CLOUD_PREFLIGHT" in
    true) should_run=1 ;;
    false) should_run=0 ;;
    auto)
      if [[ -d /app ]]; then
        should_run=1
      else
        should_run=0
      fi
      ;;
    *)
      echo "Invalid RUN_CLOUD_PREFLIGHT='$RUN_CLOUD_PREFLIGHT' (expected auto|true|false)" >&2
      return 1
      ;;
  esac

  if [[ "$should_run" -eq 0 ]]; then
    SUMMARY_LINES+=("SKIP|Cloud preflight (--audit-mode)")
    return 0
  fi

  run_step "Cloud preflight (--audit-mode)" \
    blueprint-validate --config "$CLOUD_CONFIG" --work-dir "$WORK_DIR" preflight --audit-mode
}

echo "== Pre-GPU Audit =="
echo "Root: $ROOT_DIR"
echo "Local config: $LOCAL_CONFIG"
echo "Cloud config: $CLOUD_CONFIG"
echo "Work dir: $WORK_DIR"
echo "Run cloud preflight: $RUN_CLOUD_PREFLIGHT"

cd "$ROOT_DIR"

if ! command -v blueprint-validate >/dev/null 2>&1; then
  echo "blueprint-validate not found in PATH. Activate the project venv first."
  exit 1
fi

ensure_python_module_cli pytest pytest
ensure_python_module_cli ruff ruff

run_step "Secret scan" secret_scan

run_step "Targeted pytest" \
  env \
    -u BLUEPRINT_GPU_HOURLY_RATE_USD \
    -u BLUEPRINT_AUTO_SHUTDOWN_CMD \
    -u BLUEPRINT_POST_STAGE_SYNC_CMD \
    -u BLUEPRINT_POST_STAGE_SYNC_STRICT \
    python -m pytest "${PYTEST_TARGETS[@]}" -q

run_step "Lint (ruff check)" \
  python -m ruff check src tests scripts

run_step "Format check (ruff format --check)" \
  python -m ruff format --check src tests scripts

run_step "Local preflight (--audit-mode)" \
  blueprint-validate --config "$LOCAL_CONFIG" --work-dir "$WORK_DIR" preflight --audit-mode

maybe_run_cloud_preflight || HAS_FAILURE=1

echo ""
echo "=== Pre-GPU Audit Summary ==="
for line in "${SUMMARY_LINES[@]}"; do
  IFS="|" read -r status name <<< "$line"
  echo "[$status] $name"
done

if [[ "$HAS_FAILURE" -ne 0 ]]; then
  echo ""
  echo "Pre-GPU audit FAILED."
  exit 1
fi

echo ""
echo "Pre-GPU audit PASSED."

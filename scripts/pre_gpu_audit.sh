#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_CONFIG="${LOCAL_CONFIG:-$ROOT_DIR/configs/same_facility_policy_uplift_openvla.yaml}"
CLOUD_CONFIG="${CLOUD_CONFIG:-$ROOT_DIR/configs/same_facility_policy_uplift_openvla.cloud.yaml}"
WORK_DIR="${WORK_DIR:-$ROOT_DIR/data/outputs/pre_gpu_audit}"
DEFAULT_PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
if [[ -x "$DEFAULT_PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"
else
  PYTHON_BIN="${PYTHON_BIN:-$(command -v python || true)}"
fi
RUN_CLOUD_PREFLIGHT="${RUN_CLOUD_PREFLIGHT:-auto}"  # auto|true|false
RUN_LOCAL_PREFLIGHT="${RUN_LOCAL_PREFLIGHT:-true}"  # true|false|auto
RUN_SECRET_SCAN="${RUN_SECRET_SCAN:-true}"          # true|false|auto
RUN_TARGETED_PYTEST="${RUN_TARGETED_PYTEST:-true}"  # true|false|auto
RUN_LINT="${RUN_LINT:-true}"                        # true|false|auto
RUN_FORMAT_CHECK="${RUN_FORMAT_CHECK:-false}"       # true|false|auto
AUDIT_SCOPE="${AUDIT_SCOPE:-full}"                  # quick|full
AUTO_INSTALL_AUDIT_TOOLS="${AUTO_INSTALL_AUDIT_TOOLS:-false}"  # true|false

PYTEST_TARGETS=(
  tests/test_preflight.py
  tests/test_pipeline_smoke.py
  tests/test_pipeline_stages_order.py
  tests/test_config.py
  tests/test_dreamdojo_finetune.py
  tests/test_cosmos_runner.py
  tests/test_openvla_finetune.py
  tests/test_policy_rl_loop.py
  tests/test_dataset_builder_strict_video.py
  tests/test_dataset_builder_quality.py
  tests/test_manifest_validation.py
  tests/test_provenance.py
  tests/test_cli_preflight.py
  tests/test_docker_contract.py
)

declare -a SUMMARY_LINES=()
HAS_FAILURE=0
CLI_CMD=()

install_python_package() {
  local package="$1"
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$PYTHON_BIN" "$package"
    return 0
  fi
  "$PYTHON_BIN" -m pip install "$package"
}

ensure_python_module_cli() {
  local module="$1"
  local package="${2:-$module}"
  if "$PYTHON_BIN" -m "$module" --version >/dev/null 2>&1; then
    return 0
  fi
  case "$AUTO_INSTALL_AUDIT_TOOLS" in
    true|1|yes|on)
      echo "Installing missing audit dependency: $package"
      install_python_package "$package"
      ;;
    false|0|no|off)
      echo "Missing required audit dependency: $package" >&2
      echo "Install it first or set AUTO_INSTALL_AUDIT_TOOLS=true." >&2
      return 1
      ;;
    *)
      echo "Invalid AUTO_INSTALL_AUDIT_TOOLS='$AUTO_INSTALL_AUDIT_TOOLS' (expected true|false)" >&2
      return 1
      ;;
  esac
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

run_optional_step() {
  local flag="$1"
  local name="$2"
  shift 2

  case "$flag" in
    true|auto|1|yes|on)
      run_step "$name" "$@"
      ;;
    false|0|no|off)
      SUMMARY_LINES+=("SKIP|$name")
      ;;
    *)
      echo "Invalid toggle '$flag' for step '$name' (expected true|false|auto)" >&2
      SUMMARY_LINES+=("FAIL|$name")
      HAS_FAILURE=1
      ;;
  esac
}

secret_scan() {
  local output
  if ! output="$(
    "$PYTHON_BIN" - "$ROOT_DIR" <<'PY'
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
        raw = path.read_bytes()
    except Exception:
        continue
    if b"\x00" in raw:
        continue
    text = raw.decode("utf-8", errors="ignore")
    if not text:
        continue
    printable = sum(1 for ch in text if ch.isprintable() or ch in "\n\r\t")
    if (printable / max(1, len(text))) < 0.85:
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
    SUMMARY_LINES+=("SKIP|Cloud preflight (--profile audit)")
    return 0
  fi

  run_step "Cloud preflight (--profile audit)" \
    "${CLI_CMD[@]}" --config "$CLOUD_CONFIG" --work-dir "$WORK_DIR" preflight --profile audit
}

maybe_run_local_preflight() {
  local should_run=1
  case "$RUN_LOCAL_PREFLIGHT" in
    true|auto) should_run=1 ;;
    false) should_run=0 ;;
    *)
      echo "Invalid RUN_LOCAL_PREFLIGHT='$RUN_LOCAL_PREFLIGHT' (expected true|false|auto)" >&2
      return 1
      ;;
  esac

  if [[ "$should_run" -eq 0 ]]; then
    SUMMARY_LINES+=("SKIP|Local preflight (--profile audit)")
    return 0
  fi

  run_step "Local preflight (--profile audit)" \
    "${CLI_CMD[@]}" --config "$LOCAL_CONFIG" --work-dir "$WORK_DIR" preflight --profile audit
}

run_pytest_scope() {
  ensure_python_module_cli pytest pytest || return 1

  case "$AUDIT_SCOPE" in
    quick)
      env \
        -u BLUEPRINT_GPU_HOURLY_RATE_USD \
        -u BLUEPRINT_AUTO_SHUTDOWN_CMD \
        -u BLUEPRINT_POST_STAGE_SYNC_CMD \
        -u BLUEPRINT_POST_STAGE_SYNC_STRICT \
        "$PYTHON_BIN" -m pytest "${PYTEST_TARGETS[@]}" -q
      ;;
    full)
      env \
        -u BLUEPRINT_GPU_HOURLY_RATE_USD \
        -u BLUEPRINT_AUTO_SHUTDOWN_CMD \
        -u BLUEPRINT_POST_STAGE_SYNC_CMD \
        -u BLUEPRINT_POST_STAGE_SYNC_STRICT \
        "$PYTHON_BIN" -m pytest -m "not gpu" -q
      ;;
    *)
      echo "Invalid AUDIT_SCOPE='$AUDIT_SCOPE' (expected quick|full)" >&2
      return 1
      ;;
  esac
}

run_lint_check() {
  ensure_python_module_cli ruff ruff || return 1
  "$PYTHON_BIN" -m ruff check src tests scripts
}

run_format_check() {
  ensure_python_module_cli ruff ruff || return 1
  "$PYTHON_BIN" -m ruff format --check src tests scripts
}

echo "== Pre-GPU Audit =="
echo "Root: $ROOT_DIR"
echo "Python bin: $PYTHON_BIN"
echo "Local config: $LOCAL_CONFIG"
echo "Cloud config: $CLOUD_CONFIG"
echo "Work dir: $WORK_DIR"
echo "Run local preflight: $RUN_LOCAL_PREFLIGHT"
echo "Run cloud preflight: $RUN_CLOUD_PREFLIGHT"
echo "Run secret scan: $RUN_SECRET_SCAN"
echo "Run pytest: $RUN_TARGETED_PYTEST"
echo "Run lint: $RUN_LINT"
echo "Run format check: $RUN_FORMAT_CHECK"
echo "Audit scope: $AUDIT_SCOPE"
echo "Auto-install audit tools: $AUTO_INSTALL_AUDIT_TOOLS"

cd "$ROOT_DIR"

if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found. Set PYTHON_BIN or run uv sync first."
  exit 1
fi

if "$PYTHON_BIN" -m blueprint_validation.cli --help >/dev/null 2>&1; then
  CLI_CMD=("$PYTHON_BIN" -m blueprint_validation.cli)
elif command -v blueprint-validate >/dev/null 2>&1; then
  CLI_CMD=(blueprint-validate)
else
  echo "BlueprintValidation CLI is unavailable. Run 'uv sync --extra rlds' or use scripts/provision_same_facility_claim_runtime.sh first."
  exit 1
fi

run_optional_step "$RUN_SECRET_SCAN" "Secret scan" secret_scan

run_optional_step "$RUN_TARGETED_PYTEST" "Pytest ($AUDIT_SCOPE)" run_pytest_scope

run_optional_step "$RUN_LINT" "Lint (ruff check)" run_lint_check

run_optional_step "$RUN_FORMAT_CHECK" "Format check (ruff format --check)" run_format_check

maybe_run_local_preflight || HAS_FAILURE=1

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

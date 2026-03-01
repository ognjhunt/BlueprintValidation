# Operations Memory

This file tracks operational failures that occurred during real cloud runs and
the permanent fixes applied so we do not repeat them.

## 2026-02-28: L40S bring-up (`vast` instance `32182930`)

### 1) DreamDojo fallback upgraded Torch and broke `transformer_engine`
- Symptom:
  - `cloud_prepare_0787.sh` failed before preflight with:
    - `ImportError: ... transformer_engine_torch... undefined symbol ... c10_cuda_check_implementation`
- Root cause:
  - Fallback path used `pip_install -U piq`, which upgraded Torch/Torchvision/TRT
    away from the CUDA-matched DreamDojo stack.
- Permanent fix:
  - In `scripts/cloud_prepare_0787.sh`, fallback now installs supplemental deps
    without dependency upgrades:
    - `pip_install --no-deps "piq==0.8.0"`
    - `pip_install --no-build-isolation --no-deps "git+https://github.com/facebookresearch/pytorch3d.git"`

### 2) Audit script mutated runtime environment and reintroduced ABI mismatch
- Symptom:
  - `pre_gpu_audit.sh` uninstalled/reinstalled core ML packages, then preflight
    failed with the same `transformer_engine` symbol error.
- Root cause:
  - Audit used `uv run ...` for pytest/lint/preflight, which altered the active
    environment on the VM.
- Permanent fix:
  - In `scripts/pre_gpu_audit.sh`, switched to stable in-environment execution:
    - `python -m pytest`
    - `python -m ruff`
    - direct `blueprint-validate` calls
  - Added bootstrap checks to install missing `pytest`/`ruff` if absent.

### 3) Secret scan false positive on runtime secrets file
- Symptom:
  - Audit always failed secret scan because `scripts/runtime_env.local` contains
    valid runtime tokens by design.
- Root cause:
  - Secret scan inspected `scripts/runtime_env.local`.
- Permanent fix:
  - Replaced `rg`-dependent scan with Python-based scanner.
  - Explicitly excluded `scripts/runtime_env.local` (and existing audit script).

### 4) Budget guard ignored explicit disable (`max_cost_usd=0`)
- Symptom:
  - Pipeline smoke tests failed under cloud env vars; budget guard triggered even
    with `cloud.max_cost_usd = 0`.
- Root cause:
  - `ValidationPipeline._is_budget_exceeded()` compared estimated spend directly
    to config value, without honoring non-positive values as disabled.
- Permanent fix:
- In `src/blueprint_validation/pipeline.py`, return `False` early when
  `cloud.max_cost_usd <= 0`.

### 5) `run-all` Stage 2 can fail on missing `natsort`
- Symptom:
  - `s2_enrich` failed with `ModuleNotFoundError: No module named 'natsort'`.
- Root cause:
  - Cosmos runtime prep checked `sam2` but did not install/check `natsort`.
- Permanent fix:
  - `scripts/cloud_prepare_0787.sh` now installs `natsort` with `sam2` and verifies both imports.
  - `src/blueprint_validation/preflight.py` now requires `dep:natsort` before run-all.
  - `tests/test_preflight.py` updated to assert both dependencies are checked.

### 6) Backblaze ephemeral rclone config was dropped in subshell
- Symptom:
  - B2 sync failed with `didn't find section in config file` for `b2ephem:...`.
- Root cause:
  - `scripts/b2_checkpoint_sync.sh` exported `RCLONE_CONFIG_B2EPHEM_*` inside a
    command substitution helper, so exports were lost.
- Permanent fix:
- Export ephemeral rclone vars in parent shell before computing `REMOTE_PATH`.
- Keep watcher resilient with explicit status handling in `scripts/vast_checkpoint_watch.sh`.

### 7) Cosmos guardrail repo access can block Stage 2
- Symptom:
  - `s2_enrich` failed while trying to download `nvidia/Cosmos-Guardrail1` with
    `GatedRepoError: 403 ... not in the authorized list`.
- Root cause:
  - Token had access to core Cosmos model weights but not the guardrail repo.
- Permanent fix:
  - `src/blueprint_validation/enrichment/cosmos_runner.py` now passes
    `--disable-guardrails` by default (configurable) to avoid hard dependency on
    gated guardrail assets during enrichment.
  - `src/blueprint_validation/config.py` adds `enrich.disable_guardrails` (default `true`).

### 8) Stage 2 still needs Cosmos-Predict tokenizer repo access
- Symptom:
  - `s2_enrich` failed even with guardrails disabled while fetching
    `nvidia/Cosmos-Predict2.5-2B` `tokenizer.pth` (403 gated repo error).
- Root cause:
  - Cosmos Transfer runtime depends on Predict2.5 tokenizer assets that are not
    covered by only downloading `nvidia/Cosmos-Transfer2.5-2B`.
- Permanent fix:
  - `scripts/download_models.sh` now explicitly downloads/verifies
    `nvidia/Cosmos-Predict2.5-2B` `tokenizer.pth` during prep so access issues
    fail early before `run-all`.
  - HF license/access checklist now includes `nvidia/Cosmos-Predict2.5-2B`.

### 9) Local 2-minute sync loop silently failed under cron
- Symptom:
  - periodic local checkpoint sync log showed repeated
    `scripts/vast_checkpoint_sync.sh: line 100: vastai: command not found`.
  - later cycles also showed `rclone is required` despite manual sync working.
- Root cause:
  - cron environment lacked `vastai` in `PATH`, so auto SSH target resolution
    failed.
  - cron `PATH` also omitted `/opt/homebrew/bin`, so `rclone` was not found.
- Permanent fix:
  - cron entry now passes `SSH_HOST=ssh2.vast.ai` and `SSH_PORT=22930` explicitly
    so sync runs without `vastai` CLI lookup.
  - cron entry now sets explicit `PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin`
    so `rclone` is available for B2 mirroring.
  - one-shot validation confirmed local snapshot pull + chained B2 push succeed.

### 10) Stage 2 dynamic prompt fallback can silently rewrite scene type
- Symptom:
  - Cosmos enrichment outputs looked like warehouse/industrial footage instead of
    the kitchen scene.
- Root cause:
  - Dynamic Gemini variant generation failed, and code silently fell back to
    hardcoded industrial prompts.
- Permanent fix:
  - `src/blueprint_validation/enrichment/variant_specs.py` now supports strict
    mode that raises when dynamic variant generation fails.
  - `src/blueprint_validation/config.py` adds
    `enrich.allow_dynamic_variant_fallback` (default `true` for backward compatibility).
  - `configs/interiorgs_kitchen_0787.cloud.yaml` sets
    `allow_dynamic_variant_fallback: false` so this run fails fast instead of
    using wrong-scene fallback prompts.
  - Added tests in `tests/test_variant_specs.py`.

### 11) Do not assume Cosmos control-depth is 180° inverted
- Symptom:
  - `*_control_depth.mp4` looked visually upside-down during review.
- Root cause:
  - This was a perception mismatch. Direct frame alignment checks showed
    control-depth orientation matched the input depth orientation; forced 180°
    rotation worsened alignment.
- Permanent fix:
  - Keep scene-specific depth-rotation hook opt-in only (disabled by default in
    `src/blueprint_validation/stages/s2_enrich.py`).
  - Never hard-enable depth rotation for a facility without frame-level
    alignment verification first.

### 12) Nano-banana (Gemini image pre-pass) did not recover URDF arm in Cosmos output
- Symptom:
  - Even after Gemini image pre-pass (`gemini-2.5-flash-image`) before Cosmos,
    generated clips still dropped visible robot arm/end-effector.
- Root cause:
  - Prompt/image pre-pass alone was insufficient to enforce manipulator
    persistence through Cosmos transfer.
- Permanent fix:
  - For full Stage 2 runs, prefer baseline path over nano-banana pre-pass unless
    a new method shows measurable arm retention gains.
  - A/B acceptance gate for future attempts:
    - Same clip + prompt + depth input
    - Evaluate matched frames for arm visibility
    - Require improvement over baseline before rollout.

## Open follow-up
- `BLUEPRINT_AUTO_SHUTDOWN_CMD` is currently an echo placeholder on the VM.
  Replace with a real `vastai stop ...` or equivalent mechanism before long runs.

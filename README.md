# BlueprintValidation

`BlueprintValidation` is the downstream NeoVerse runtime consumer for built site-world packages.

## Runtime Role

This repo now has two explicit runtime backends:

- `neoverse_production`: the default production runtime service in this repo. It owns authoritative hosted-session state, reset/step lifecycle, render persistence, and NeoVerse model/checkpoint provenance. It requires a local NeoVerse runner integration plus model assets.
- `smoke_contract`: a local deterministic contract/smoke runtime for developer flows and interface validation only. It is not a production backend.

Preflight, hosted-session artifacts, exports, and reports record which runtime kind was used. A smoke endpoint can no longer silently pass as production.

## Scope

This repo owns the downstream runtime path:

- consume a built `site_world_spec.json`, `site_world_registration.json`, and `site_world_health.json`
- register the built package with a NeoVerse runtime backend
- preflight runtime kind, capabilities, and production readiness
- create, reset, step, batch-run, stop, and export hosted sessions
- produce runtime-aware validation reports and export manifests

This repo does not own package assembly or upstream qualification. Those stay upstream in `BlueprintCapturePipeline`.

## Install

```bash
git clone https://github.com/ognjhunt/BlueprintValidation.git
cd BlueprintValidation
uv sync
uv sync --extra vision
```

## Inputs

Expected upstream artifacts from `BlueprintCapturePipeline`:

- `pipeline/evaluation_prep/site_world_spec.json`
- `pipeline/evaluation_prep/site_world_registration.json`
- `pipeline/evaluation_prep/site_world_health.json`

## Runtime Services

Production runtime service:

```bash
export NEOVERSE_MODEL_ROOT="/path/to/neoverse/model"
export NEOVERSE_CHECKPOINT_PATH="/path/to/neoverse/checkpoint.pt"
export NEOVERSE_RUNNER_COMMAND="/path/to/neoverse-runner"

blueprint-neoverse-runtime
```

Smoke-contract runtime service:

```bash
blueprint-neoverse-smoke-runtime
```

The production service is the one intended to be surfaced behind a `runtime_base_url` for downstream tools or `Blueprint-WebApp`. The WebApp already expects a live runtime handle and proxies runtime calls through that URL; this repo now returns runtime-kind and production-readiness fields with that handle.

## Validation Workflow

```bash
export NEOVERSE_RUNTIME_SERVICE_URL="http://127.0.0.1:8787"

blueprint-validate \
  --config configs/example_validation.yaml \
  --required-runtime-kind neoverse_production \
  preflight \
  --site-world-registration /path/to/site_world_registration.json

blueprint-validate \
  --config configs/example_validation.yaml \
  --required-runtime-kind neoverse_production \
  session create \
  --session-id validation-session \
  --session-work-dir data/session-validation \
  --site-world-registration /path/to/site_world_registration.json \
  --robot-profile-id mobile_manipulator_rgb_v1 \
  --task-id task-1 \
  --scenario-id scenario-default \
  --start-state-id start-default

blueprint-validate session reset \
  --session-id validation-session \
  --session-work-dir data/session-validation

blueprint-validate session step \
  --session-work-dir data/session-validation \
  --episode-id episode-xxxxxxxx \
  --action-json '[0,0,0,0,0,0,0]'

blueprint-validate session run-batch \
  --session-work-dir data/session-validation \
  --num-episodes 5 \
  --max-steps 6

blueprint-validate session export \
  --session-id validation-session \
  --session-work-dir data/session-validation

blueprint-validate --work-dir data/session-validation report
```

## Outputs

- `session_state.json`
- `runtime_probe.json`
- `runtime_smoke.json` only when the smoke backend is used
- `runtime_batch_manifest.json`
- `rollouts/<episode>/...`
- `exports/raw_bundle/raw_session_bundle.json`
- `exports/rlds/rlds_manifest.json`
- `export_manifest.json`
- `validation_report.md`
- `standardized_eval_report.json`

## Production Backend Notes

- The production backend is local to this repo, but it still depends on external NeoVerse model assets and a local runner integration.
- The built-in `LocalNeoVerseRunnerAdapter` expects `NEOVERSE_RUNNER_COMMAND` to point to a command that accepts `request_json_path response_json_path` and writes a response manifest containing `camera_frames`.
- If model assets or checkpoint readiness are missing, `/v1/runtime` reports that explicitly and preflight fails for production validation.

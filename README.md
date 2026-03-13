# BlueprintValidation

`BlueprintValidation` is the downstream NeoVerse consumer for built site-world packages.

## Scope

This repo owns only the NeoVerse runtime path:

- consume a built `site_world_spec.json`, `site_world_registration.json`, and `site_world_health.json`
- preflight the NeoVerse runtime service against that package
- create, reset, step, batch-run, stop, and export hosted sessions
- build a minimal report from NeoVerse session and export artifacts

This repo does not own package assembly, world-model training, policy training, geometry preview, simulator fallback paths, teleop, or evaluation research flows.

## Install

```bash
git clone https://github.com/ognjhunt/BlueprintValidation.git
cd BlueprintValidation
uv sync
```

Optional video export support:

```bash
uv sync --extra vision
```

## Inputs

Expected upstream artifacts from `BlueprintCapturePipeline`:

- `pipeline/evaluation_prep/site_world_spec.json`
- `pipeline/evaluation_prep/site_world_registration.json`
- `pipeline/evaluation_prep/site_world_health.json`

## Runtime Workflow

```bash
export NEOVERSE_RUNTIME_SERVICE_URL="http://127.0.0.1:8787"

blueprint-validate --config configs/example_validation.yaml preflight \
  --site-world-registration /path/to/site_world_registration.json

blueprint-validate session create \
  --session-id smoke-session \
  --session-work-dir data/session-smoke \
  --site-world-registration /path/to/site_world_registration.json \
  --robot-profile-id mobile_manipulator_rgb_v1 \
  --task-id task-1 \
  --scenario-id scenario-default \
  --start-state-id start-default

blueprint-validate session reset \
  --session-id smoke-session \
  --session-work-dir data/session-smoke

blueprint-validate session step \
  --session-work-dir data/session-smoke \
  --episode-id episode-xxxxxxxx \
  --action-json '[0,0,0,0,0,0,0]'

blueprint-validate session run-batch \
  --session-work-dir data/session-smoke \
  --num-episodes 5 \
  --max-steps 6

blueprint-validate session export \
  --session-id smoke-session \
  --session-work-dir data/session-smoke

blueprint-validate --work-dir data/session-smoke report
```

## Outputs

- `session_state.json`
- `runtime_smoke.json`
- `runtime_batch_manifest.json`
- `rollouts/<episode>/...`
- `exports/raw_bundle/raw_session_bundle.json`
- `exports/rlds/rlds_manifest.json`
- `export_manifest.json`
- `validation_report.md`
- `standardized_eval_report.json`

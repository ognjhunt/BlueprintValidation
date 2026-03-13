# BlueprintValidation

Downstream NeoVerse site-world session, evaluation, rollout capture, and export tooling.

This repo is intentionally narrow:

- consume a built `site_world_spec.json`, `site_world_registration.json`, and `site_world_health.json`
- connect to a NeoVerse runtime service
- run `create -> reset -> step -> render -> batch -> export`
- inspect grounding, health, evaluation, and exported rollout data

It is not the source of truth for handoff schemas, site-world schemas, runtime-layer policy thresholds,
or canonical package versioning. Those contracts now live in the shared `BlueprintContracts` package.

## Supported Intake

Expected upstream artifacts from `BlueprintCapturePipeline`:

- `pipeline/evaluation_prep/site_world_spec.json`
- `pipeline/evaluation_prep/site_world_registration.json`
- `pipeline/evaluation_prep/site_world_health.json`

The registration file is the main handoff into this repo.

## Runtime Model

- `BlueprintCapturePipeline` prepares the canonical site-world package and, when configured, registers that package with NeoVerse.
- `BlueprintValidation` does not own NeoVerse process lifecycle. It talks to a NeoVerse runtime service over HTTP/WebSocket.
- The repo keeps local debug and export artifacts, but the site-specific world itself and its canonical package are upstream/runtime-owned.

Grounding should come from the upstream site-world spec, including:

- site video / keyframe
- ARKit poses
- ARKit intrinsics
- optional depth
- occupancy / collision geometry
- object index
- object geometry
- task / scenario / start-state catalogs
- robot profiles
- qualification references

## Install Profiles

Lean default for CPU-safe validation, config work, contract checks, and local CI:

```bash
git clone https://github.com/ognjhunt/BlueprintValidation.git
cd BlueprintValidation
uv sync
```

`uv sync` installs the lean runtime plus local dev/test tools. It does not install the heavy render,
vision, or training stacks by default.

Optional extras:

- `uv sync --extra vision` for OpenCV-backed frame/video handling, hosted session exports, and placeholder runtime render IO
- `uv sync --extra render` for Stage-1 gsplat rendering and visual-fidelity metrics
- `uv sync --extra training` for OpenVLA / DreamDojo policy and world-model training flows
- `uv sync --extra rlds` for TFRecord/RLDS export
- `uv sync --extra pi05` for LeRobot / OpenPI integration
- `uv sync --extra manipulation` for PyBullet-backed manipulation paths
- `uv sync --extra full` for the full heavyweight stack

If you want the lean runtime without local dev tooling, use `uv sync --no-dev`.

## Quick Start

Preferred downstream session path:

```bash
uv sync
export NEOVERSE_RUNTIME_SERVICE_URL="http://127.0.0.1:8787"
blueprint-validate session create \
  --session-id smoke-session \
  --session-work-dir data/session-smoke \
  --site-world-registration /path/to/site_world_registration.json \
  --robot-profile-id mobile_manipulator_rgb_v1 \
  --task-id task-1 \
  --scenario-id scenario-default \
  --start-state-id start-default
```

Audit / preflight path:

```bash
uv sync
export NEOVERSE_RUNTIME_SERVICE_URL="http://127.0.0.1:8787"
blueprint-validate --config configs/qualified_opportunity_validation.yaml preflight --profile audit
```

Placeholder local runtime service:

```bash
uv sync --extra vision
export NEOVERSE_RUNTIME_SERVICE_URL="http://127.0.0.1:8787"
blueprint-neoverse-runtime
```

Heavy render / training path:

```bash
uv sync --extra full
```

## Session Workflow

Create a local session workspace from a site-world registration:

```bash
blueprint-validate session create \
  --session-id smoke-session \
  --session-work-dir data/session-smoke \
  --site-world-registration /path/to/site_world_registration.json \
  --robot-profile-id mobile_manipulator_rgb_v1 \
  --task-id task-1 \
  --scenario-id scenario-default \
  --start-state-id start-default
```

Reset:

```bash
blueprint-validate session reset \
  --session-id smoke-session \
  --session-work-dir data/session-smoke
```

Manual step:

```bash
blueprint-validate session step \
  --session-id smoke-session \
  --session-work-dir data/session-smoke \
  --episode-id episode-xxxxxxxx \
  --action-json '[0,0,0,0,0,0,1]' \
  --no-auto-policy
```

Batch rollout:

```bash
blueprint-validate session run-batch \
  --session-id smoke-session \
  --session-work-dir data/session-smoke \
  --num-episodes 5
```

Export raw bundle + RLDS-style data:

```bash
blueprint-validate session export \
  --session-id smoke-session \
  --session-work-dir data/session-smoke
```

## Outputs

Validation writes local artifacts such as:

- `runtime_smoke.json`
- `runtime_batch_manifest.json`
- `rollouts/<episode>/...`
- `exports/raw_bundle/raw_session_bundle.json`
- `exports/rlds/rlds_manifest.json`
- `export_manifest.json`

These are derived interaction/export artifacts. The site world itself remains in the NeoVerse runtime service.

## Placeholder Service

This repo still includes a local placeholder NeoVerse-compatible service for contract testing and local
package registration:

```bash
blueprint-neoverse-runtime
```

Install `--extra vision` before using local render/frame endpoints or hosted session video exports.

Useful endpoints:

- `GET /healthz`
- `GET /v1/runtime`
- `POST /v1/site-worlds` registers a built site-world package
- `GET /v1/site-worlds/{site_world_id}`
- `GET /v1/site-worlds/{site_world_id}/health`
- `POST /v1/site-worlds/{site_world_id}/sessions`
- `POST /v1/sessions/{session_id}/reset`
- `POST /v1/sessions/{session_id}/step`
- `GET /v1/sessions/{session_id}/state`
- `GET /v1/sessions/{session_id}/render?camera_id=...`
- `WS /v1/sessions/{session_id}/stream`

## Notes

- The supported path is runtime-first and site-world-first.
- Older orchestration, scene-package, and training-heavy commands remain only as compatibility lanes and are not the primary supported path.

## Tests

Default CPU-safe subset:

```bash
uv run pytest -m "not integration and not gpu"
```

Integration tests are auto-marked from `tests/integration/` and skipped unless you opt in:

```bash
uv run pytest --run-integration -m integration tests/integration
uv run pytest --collect-only -m "not integration"
```

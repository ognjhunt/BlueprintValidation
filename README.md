# BlueprintValidation

NeoVerse site-world runtime validation, smoke testing, rollout capture, and export tooling.

This repo is now focused on one job:

- take a `site_world_registration.json` produced upstream
- connect to a NeoVerse runtime service
- run `create -> reset -> step -> render -> batch -> export`
- inspect grounding, health, and exported rollout data

## Supported Intake

Expected upstream artifacts from `BlueprintCapturePipeline`:

- `pipeline/evaluation_prep/site_world_spec.json`
- `pipeline/evaluation_prep/site_world_registration.json`
- `pipeline/evaluation_prep/site_world_health.json`

The registration file is the main handoff into this repo.

## Runtime Model

- `BlueprintCapturePipeline` prepares the grounding package and asks NeoVerse to build/register the site world.
- `BlueprintValidation` does not own NeoVerse process lifecycle. It talks to a NeoVerse runtime service over HTTP/WebSocket.
- The repo keeps local debug and export artifacts, but the site-specific world itself lives behind the NeoVerse runtime API.

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

## Quick Start

```bash
git clone https://github.com/ognjhunt/BlueprintValidation.git
cd BlueprintValidation
uv sync
```

Set the NeoVerse runtime service:

```bash
export NEOVERSE_RUNTIME_SERVICE_URL="http://127.0.0.1:8787"
```

Optional local env file:

```bash
cp scripts/runtime_env.example scripts/runtime_env.local
set -a && source scripts/runtime_env.local && set +a
```

Run preflight:

```bash
blueprint-validate --config configs/qualified_opportunity_validation.yaml preflight
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

This repo still includes a local placeholder NeoVerse-compatible service for contract testing:

```bash
blueprint-neoverse-runtime
```

Useful endpoints:

- `GET /healthz`
- `GET /v1/runtime`
- `POST /v1/site-worlds`
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
- Older training-first workflows are no longer the intended focus of this repo.

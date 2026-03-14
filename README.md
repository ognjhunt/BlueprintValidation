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

The `vision` extra is required for the production runtime path because frame IO and render/export flows depend on OpenCV.

## Inputs

Expected upstream artifacts from `BlueprintCapturePipeline`:

- `pipeline/evaluation_prep/site_world_spec.json`
- `pipeline/evaluation_prep/site_world_registration.json`
- `pipeline/evaluation_prep/site_world_health.json`

## Runtime Services

Bootstrap the NeoVerse environment for an A100 host:

```bash
blueprint-validate runtime bootstrap \
  --repo-root ./external/NeoVerse \
  --env-file ./scripts/runtime_env.local

source ./scripts/runtime_env.local
```

Production runtime service:

```bash
uv sync --extra vision
blueprint-neoverse-runtime
```

Smoke-contract runtime service:

```bash
blueprint-neoverse-smoke-runtime
```

Use the smoke runtime only for local interface smoke checks. It is intentionally non-production and should fail preflight whenever `--required-runtime-kind neoverse_production` is used.

The production service is the one intended to be surfaced behind a `runtime_base_url` for downstream tools or `Blueprint-WebApp`. The WebApp already expects a live runtime handle and proxies runtime calls through that URL; this repo now returns runtime-kind and production-readiness fields with that handle.

### Hosted Demo Setup

For a live hosted Blueprint demo, this repo is the runtime host. Redis and Firestore do not replace it; they only store app/session state after the web app knows which runtime URL to call.

1. Bootstrap the NeoVerse runtime environment on the runtime host.
2. Set the public runtime URLs before starting the service:

```bash
export NEOVERSE_RUNTIME_PUBLIC_BASE_URL="https://<live-runtime-host>"
export NEOVERSE_RUNTIME_PUBLIC_WS_BASE_URL="wss://<live-runtime-host>"
```

3. Preload the site-world package on boot so `GET /v1/site-worlds/<id>` works immediately:

```bash
export NEOVERSE_RUNTIME_BOOTSTRAP_REGISTRATION_PATH="/absolute/path/to/site_world_registration.json"
blueprint-neoverse-runtime
```

Use `NEOVERSE_RUNTIME_BOOTSTRAP_REGISTRATION_PATHS` for multiple registration files. The runtime service will load each built bundle and register it on startup.

4. Verify the runtime host:

```bash
curl https://<live-runtime-host>/v1/site-worlds/siteworld-f5fd54898cfb
curl https://<live-runtime-host>/v1/site-worlds/siteworld-f5fd54898cfb/health
```

5. If you prefer explicit registration instead of boot-time preload, register the bundle against a running runtime:

```bash
blueprint-validate runtime register-site-world \
  --service-url https://<live-runtime-host> \
  --site-world-registration /absolute/path/to/site_world_registration.json
```

That command returns the registered site-world payload plus the exact Blueprint web-app env values:

- `BLUEPRINT_HOSTED_DEMO_RUNTIME_BASE_URL`
- `BLUEPRINT_HOSTED_DEMO_RUNTIME_WEBSOCKET_BASE_URL`

6. Set those two env vars on the deployed Blueprint web service and redeploy it.

Run a live runtime smoke test against the production backend:

```bash
blueprint-validate runtime smoke-test \
  --site-world-registration /path/to/site_world_registration.json \
  --robot-profile-id mobile_manipulator_rgb_v1 \
  --task-id task-1 \
  --scenario-id scenario-default \
  --start-state-id start-default
```

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
- `blueprint-validate runtime bootstrap` clones the public NeoVerse repo, creates a dedicated `.venv`, downloads model assets from Hugging Face, discovers a checkpoint, and writes an env file with the runtime variables this service expects.
- The built-in `LocalNeoVerseRunnerAdapter` expects `NEOVERSE_RUNNER_COMMAND` to point to a command that accepts `request_json_path response_json_path` and writes a response manifest containing `camera_frames`. The bootstrap command configures this to `python -m blueprint_validation.neoverse_runner_wrapper`.
- `blueprint_validation.neoverse_runner_wrapper` invokes the official NeoVerse `inference.py` entrypoint, extracts the first frame of each generated video, and returns camera-frame paths back to the runtime service.
- The production runtime also requires the `vision` extra so image/frame IO is available at runtime.
- If model assets or checkpoint readiness are missing, `/v1/runtime` reports that explicitly and preflight fails for production validation.

# Platform Context

## System Framing

- `BlueprintCapture` owns raw evidence capture.
- `BlueprintCapturePipeline` owns deterministic assembly of the canonical site-world package.
- `Blueprint-WebApp` owns business/workflow state and routing.
- `BlueprintValidation` starts after the package exists and consumes it to run sessions, evaluate, and export.

## Contract Boundary

Shared cross-repo contract code lives in `BlueprintContracts`. That shared package owns:

- qualified opportunity handoff validation
- site-world bundle/spec/health loading and structural validation
- runtime-layer grounding policy schemas and thresholds
- canonical package version computation and verification

`BlueprintValidation` must not become the source of truth for those contracts.

## What This Repo Owns

`BlueprintValidation` owns downstream runtime consumption:

- connect to NeoVerse runtime services
- create/reset/step/render/run-batch/export hosted sessions
- capture rollout artifacts and downstream exports
- run downstream evaluation and scoring over built site-world packages

## What This Repo Consumes

The supported intake is the built site-world package produced upstream:

- `site_world_spec.json`
- `site_world_registration.json`
- `site_world_health.json`

Qualified-opportunity handoffs and scene-memory artifacts are upstream inputs or references, not canonical truth this repo creates.

## Maintenance Guidance

- Changes to portable schemas, versioning rules, and runtime-layer thresholds belong in `BlueprintContracts`.
- Changes to deterministic handoff/package assembly belong upstream in `BlueprintCapturePipeline`.
- Local placeholder runtime code here may cache or mirror built artifacts for testing, but it must not become the canonical source of truth.

## Practical Rule For Agents

When making changes here:

1. keep Validation clearly downstream of package creation
2. keep portable artifact schemas in `BlueprintContracts`
3. preserve session/eval/export flows
4. treat upstream build/orchestration paths as legacy compatibility, not the primary product path

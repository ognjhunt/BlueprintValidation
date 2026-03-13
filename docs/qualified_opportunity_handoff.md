# Qualified Opportunity Handoff

`BlueprintValidation` now sits after qualification, not before it, and after canonical package creation for the supported runtime/session flow.

The preferred intake for qualification context is a rich qualified opportunity handoff plus the
smallest set of downstream assets needed for evaluation:

- scoped task definition
- site constraints
- optional target robot/team
- preferred scene-memory bundle
- optional preview simulation package
- optional geometry bundle
- optional scene package when simulator-backed evaluation is justified

For hosted NeoVerse runtime sessions, the supported runtime handoff remains the built
`site_world_registration.json` with adjacent `site_world_spec.json` and `site_world_health.json`.

## Supported Handoffs

`BlueprintValidation` accepts two `schema_version: "v1"` handoff shapes.

### Rich downstream handoff

Required top-level fields:

- `site_submission_id`
- `opportunity_id`
- `qualification_state`
- `downstream_evaluation_eligibility`
- `operator_approved_summary`
- `scoped_task_definition`
- `site_constraints`

Required nested fields:

- `scoped_task_definition.task_id`
- `scoped_task_definition.scoped_task_statement`
- `scoped_task_definition.success_criteria`
- `scoped_task_definition.in_scope_zone`
- `site_constraints.operating_constraints`
- `site_constraints.privacy_security_constraints`
- `site_constraints.known_blockers`

Accepted `qualification_state` values:

- `ready`
- `risky`
- `not_ready_yet`

When `target_robot_team` is omitted, the handoff is still valid as a neutral qualification-first handoff but only supports lightweight downstream advisory review. Full downstream execution requires `target_robot_team`.

`BlueprintValidation` preflight only accepts the handoff for full downstream execution when:

- `qualification_state == "ready"`
- `downstream_evaluation_eligibility == true`

### Legacy thin BlueprintCapturePipeline handoff

This shape is compatibility-only. New qualification flows should prefer the rich handoff plus linked evidence artifacts, and runtime execution should still begin from the built site-world package.

The current upstream pipeline writes a slimmer handoff under `.../pipeline/opportunity_handoff.json`.

Minimum working fields:

- `schema_version`
- `scene_id`
- `capture_id`
- `readiness_state`
- `match_ready`

Optional fields such as `summary`, `constraints`, `routing_status`, and `recommended_lane` are preserved when present. Validation normalizes this payload into the canonical downstream fields:

- `opportunity_id <- scene_id`
- `site_submission_id <- capture_id`
- `qualification_state <- readiness_state`
- `downstream_evaluation_eligibility <- match_ready`
- `operator_approved_summary <- summary` or a generated fallback

## Preferred Scene-Memory Bundle

Preferred when the upstream pipeline emitted canonical scene-memory artifacts:

- `scene_memory/scene_memory_manifest.json`
- `scene_memory/conditioning_bundle.json`
- optional `scene_memory/adapter_manifests/gen3c.json`
- optional `scene_memory/adapter_manifests/neoverse.json`
- optional `scene_memory/adapter_manifests/cosmos_transfer.json`
- optional `preview_simulation/preview_simulation_manifest.json`

These inputs are preferred because they preserve the qualification-backed canonical scene substrate while allowing multiple downstream generation backends.

Default downstream runtime order in `BlueprintValidation`:

- `neoverse`
- `gen3c`
- `cosmos_transfer`

`3dsceneprompt` is treated as a watchlist backend and is not selected by default until its public runtime is mature enough for production use.

`scene_memory_package` is a first-class optional handoff mapping and should be present whenever the upstream pipeline emitted the canonical bundle.

## Legacy Geometry Bundle

When geometry is justified, prefer an InteriorGS-like bundle instead of a naked PLY:

- `3dgs_compressed.ply`
- `labels.json`
- `structure.json`
- `task_targets.synthetic.json`

This bundle remains a supported adapter because downstream evaluation may still need object locations and structure context for:

- targeted camera planning
- task-local bootstrapping
- stack-specific evaluation and adaptation

Minimum working bundle contents:

- Required for geometry-backed evaluation: `3dgs_compressed.ply`
- Strongly preferred: `labels.json`
- Strongly preferred: `structure.json`
- Optional when labels/structure are present: `task_targets.synthetic.json`

If `task_targets.synthetic.json` is missing but `labels.json` and `structure.json` are present, Stage 0 can still bootstrap task hints.

## Rich Handoff Example

```json
{
  "schema_version": "v1",
  "site_submission_id": "site-sub-0787",
  "opportunity_id": "warehouse_tote_pick_0787",
  "qualification_state": "ready",
  "downstream_evaluation_eligibility": true,
  "operator_approved_summary": "Qualified tote-pick opportunity for downstream stack evaluation.",
  "scoped_task_definition": {
    "task_id": "tote_pick_shelf_bay_3",
    "scoped_task_statement": "Pick a tote from shelf bay 3 and stage it at the outbound handoff point.",
    "success_criteria": ["Acquire tote", "Clear shelf", "Reach handoff point"],
    "in_scope_zone": "shelf_bay_3_and_outbound_handoff"
  },
  "site_constraints": {
    "operating_constraints": ["Night shift only"],
    "privacy_security_constraints": ["No worker faces in shareable outputs"],
    "known_blockers": ["Reflective pallet wrap"]
  },
  "target_robot_team": {
    "team_name_or_id": "team_alpha",
    "robot_platform": "franka_panda",
    "embodiment_notes": "Fixed-base arm on a mobile cart with wrist RGB camera."
  },
  "scene_memory_package": {
    "bundle_path": "../pipeline/scene_memory",
    "scene_memory_manifest_path": "../pipeline/scene_memory/scene_memory_manifest.json"
  },
  "geometry_package": {
    "bundle_path": "../data/interiorgs/0787_841244"
  },
  "scene_package": {
    "scene_package_path": "../data/scene_packages/warehouse_tote_pick_0787"
  }
}
```

## Config Usage

Use `qualified_opportunities` as the preferred top-level config key:

```yaml
qualified_opportunities:
  warehouse_tote_pick_0787:
    opportunity_handoff_path: ./opportunity_handoff.example.json
    # Preferred when the handoff already points at ./scene_memory/
    # scene_memory_bundle_path: ../pipeline/scene_memory
    # Optional legacy geometry adapter when the handoff already sits beside ./advanced_geometry/
    geometry_bundle_path: ../data/interiorgs/0787_841244
```

Legacy compatibility remains available:

- `facilities` still works as a top-level alias
- `ply_path` and `task_hints_path` still work as direct overrides
- `--facility` still works as a CLI alias for `--opportunity`

## Minimal Working Intake

For a handoff produced by `BlueprintCapturePipeline`, the preferred working layout is:

```text
pipeline/
  opportunity_handoff.json
  scene_memory/
    scene_memory_manifest.json
    conditioning_bundle.json
    adapter_manifests/
      neoverse.json
      gen3c.json
      cosmos_transfer.json
  preview_simulation/
    preview_simulation_manifest.json   # optional
  advanced_geometry/
    3dgs_compressed.ply           # legacy adapter only
    labels.json
    structure.json
    task_targets.synthetic.json   # optional if labels/structure are present
```

With that layout, the config can be as small as:

```yaml
qualified_opportunities:
  warehouse_tote_pick_0787:
    opportunity_handoff_path: ./pipeline/opportunity_handoff.json
```

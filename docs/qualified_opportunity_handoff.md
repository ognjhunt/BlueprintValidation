# Qualified Opportunity Handoff

`BlueprintValidation` now sits after qualification, not before it.

The preferred intake is a qualified opportunity handoff plus the smallest set of downstream assets needed for evaluation:

- scoped task definition
- site constraints
- target robot/team
- optional geometry bundle
- optional scene package when simulator-backed evaluation is justified

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
- `target_robot_team`

Required nested fields:

- `scoped_task_definition.task_id`
- `scoped_task_definition.scoped_task_statement`
- `scoped_task_definition.success_criteria`
- `scoped_task_definition.in_scope_zone`
- `site_constraints.operating_constraints`
- `site_constraints.privacy_security_constraints`
- `site_constraints.known_blockers`
- `target_robot_team.team_name_or_id`
- `target_robot_team.robot_platform`
- `target_robot_team.embodiment_notes`

Accepted `qualification_state` values:

- `ready`
- `risky`
- `not_ready_yet`

`BlueprintValidation` preflight only accepts the handoff for downstream execution when:

- `qualification_state == "ready"`
- `downstream_evaluation_eligibility == true`

### Current BlueprintCapturePipeline handoff

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

## Preferred Geometry Bundle

When geometry is justified, prefer an InteriorGS-like bundle instead of a naked PLY:

- `3dgs_compressed.ply`
- `labels.json`
- `structure.json`
- `task_targets.synthetic.json`

This bundle is preferred because downstream evaluation needs object locations and structure context for:

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
    # Optional when the handoff already sits beside ./advanced_geometry/
    geometry_bundle_path: ../data/interiorgs/0787_841244
```

Legacy compatibility remains available:

- `facilities` still works as a top-level alias
- `ply_path` and `task_hints_path` still work as direct overrides
- `--facility` still works as a CLI alias for `--opportunity`

## Minimal Working Intake

For a handoff produced by `BlueprintCapturePipeline`, the simplest working layout is:

```text
pipeline/
  opportunity_handoff.json
  advanced_geometry/
    3dgs_compressed.ply
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

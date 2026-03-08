# Low-Cost PolaRiS Pilot

This is the cheapest useful shape for a directional single-scene pilot:

- one facility
- no RL loop
- no WM refresh loop
- one Stage-2 variant per render
- hard-capped DreamDojo time
- short OpenVLA fine-tune
- only a few world-model rollouts
- only a few PolaRiS rollouts

Config:

- [`configs/low_cost_polaris_pilot.yaml`](/Users/nijelhunt_1/workspace/BlueprintValidation/configs/low_cost_polaris_pilot.yaml)

## What It Optimizes For

- Cheap first-pass signal on whether scene adaptation helps.
- Keeping the full training path alive: scan -> enrich -> DreamDojo -> synthetic rollouts -> policy fine-tune.
- Using PolaRiS for a small final directional check instead of a full expensive benchmark.

## Teleop

The config keeps Stage 1f / Stage 1g ready but disabled by default:

- `external_interaction.enabled=false`
- `external_rollouts.enabled=false`

Enable them only if you already have:

- `teleop_stage1_source_manifest.json` for Stage 1f
- `teleop_session_manifest.json` for Stage 1g

Once those exist, flip both blocks to `enabled: true` and point `manifest_path` to the absolute files.

## Current Default Inputs

The checked-in pilot config now defaults to:

- `ply_path: /Users/nijelhunt_1/Downloads/3dgs_compressed.ply`

Alternate local scan:

- `/Users/nijelhunt_1/Downloads/3dgs_compressed (1).ply`

PolaRiS is disabled in this default pilot because raw PLYs alone are not enough for the strict final gate. To enable PolaRiS later, add a matching `scene_package_path` for the same scene and then flip `eval_polaris.enabled: true`.

## Commands

```bash
blueprint-validate --config configs/low_cost_polaris_pilot.yaml preflight
blueprint-validate --config configs/low_cost_polaris_pilot.yaml --work-dir data/outputs/low_cost_polaris_pilot run-all
```

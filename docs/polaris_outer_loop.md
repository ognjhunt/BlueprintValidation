# PolaRiS Outer Loop

This repo now treats the world model as the inner-loop adaptation engine and PolaRiS as the default outer-loop deployment gate.

## What Changed

- DreamDojo, Stage 3d refresh, teleop ingest, RLDS export, policy fine-tuning, and policy RL stay in place.
- Stage 4f (`s4f_polaris_eval`) compares frozen OpenVLA vs the latest adapted OpenVLA candidate and produces the default deployment recommendation when `eval_polaris.default_as_primary_gate=true`.
- Stage 4, Stage 4e, and Stage 4d remain supporting evidence unless PolaRiS is disabled.

## Config

Add `scene_package_path` per facility and enable `eval_polaris`:

```yaml
facilities:
  facility_a:
    name: Facility A
    ply_path: /path/to/scene_splat.ply
    scene_package_path: /path/to/scene_package

eval_polaris:
  enabled: true
  repo_path: /opt/PolaRiS
  hub_path: /opt/PolaRiS-Hub
  environment_mode: scene_package_bridge
  default_as_primary_gate: true
  use_for_claim_gate: true
  num_rollouts: 16
  device: cuda
  policy_client: OpenVLA
  observation_mode: external_only
  action_mode: native
  export_dir: ./data/outputs/polaris
  require_scene_package: true
  require_success_correlation_metadata: true
```

## Scene Modes

- `scene_package_bridge`: strict path for Blueprint scene packages. Primary-gate eligible only when the package and task metadata are present.
- `native_bundle`: use an already-packaged PolaRiS environment under `hub_path`.
- `scan_only_bridge`: research/smoke only. Never primary-gate eligible.

Raw scan inputs like `3dgs_compressed*.ply` stay upstream of PolaRiS until they are converted into a scene package or a native PolaRiS bundle.

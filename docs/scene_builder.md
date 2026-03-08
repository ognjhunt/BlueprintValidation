# Direct Scene Builder

This repo can now build a minimal scene package directly from:

- one raw scene PLY
- a small config-driven set of local USD/USDZ task assets

The builder is intended for the v1 path:

- static room/background from the PLY
- only a few explicit imported manipulable assets
- enough package structure for teleop and the strict PolaRiS scene bridge

## Command

```bash
blueprint-validate --config <config.yaml> build-scene-package
```

## Required Config Block

```yaml
scene_builder:
  enabled: true
  source_ply_path: /abs/path/to/scene.ply
  output_scene_root: /abs/path/to/output_scene
  static_collision_mode: simple
  asset_manifest_path: /abs/path/to/scene_assets.json
  robot_type: franka
  task_template: pick_place_v1
  emit_isaac_lab: true
  emit_polaris_metadata: true
```

## Asset Manifest Shape

```json
{
  "schema_version": "v1",
  "scene_id": "demo_scene",
  "task": {
    "task_id": "pick_place_demo",
    "task_text": "Pick up the mug and place it on the tray",
    "task_type": "pick_place",
    "target_object_id": "mug_001",
    "goal_object_id": "tray_001",
    "goal_region_label": "tray_surface"
  },
  "assets": [
    {
      "object_id": "mug_001",
      "label": "mug",
      "asset_type": "rigid",
      "asset_path": "/abs/path/to/mug.usda",
      "pose": {
        "position": [0.1, 0.2, 0.3],
        "rotation_quaternion": [1.0, 0.0, 0.0, 0.0]
      },
      "scale": [1.0, 1.0, 1.0],
      "task_role": "target_object"
    }
  ]
}
```

## Output

The builder emits:

```text
<scene_root>/
  assets/scene_manifest.json
  usd/scene.usda
  geniesim/task_config.json
  isaac_lab/
```

## Current V1 Constraints

- Imported task assets must already be local USD/USDZ files.
- Raw PLY is used only for the static background/collision shell.
- Full scan-to-object reconstruction is out of scope for v1.
- Franka is the only supported robot type in the generated task package.

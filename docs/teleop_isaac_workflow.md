# Isaac Teleop Workflow

This repo now supports a simple loop for teleop-video+actions:

1. Validate a scene handoff directory.
2. Record one teleop demo in Isaac Sim / Isaac Lab.
3. Build teleop manifests from the resulting video, actions, and state files.
4. Feed the teleop video manifest into Stage 1f for world-model conditioning.
5. Feed the teleop session manifest into the external-rollout path for policy training.

## Expected Scene Package

Minimum required structure:

```text
<scene_root>/
  assets/scene_manifest.json
  usd/scene.usda
  isaac_lab/              # optional but recommended
  geniesim/task_config.json  # optional
```

Validate it with:

```bash
blueprint-validate validate-scene-package --scene-root /path/to/scene_root
```

## Franka Defaults

Use the included Franka Isaac config:

- [configs/robots/franka_panda_isaac.json](/Users/nijelhunt_1/workspace/BlueprintValidation/configs/robots/franka_panda_isaac.json)

The v1 action contract is:

- `ee_delta_pose + gripper`
- action dimension `7`

## Record One Local Teleop Demo

The first working path is keyboard teleop against an Isaac Lab task package already available under the scene handoff.

```bash
blueprint-validate record-teleop \
  --scene-root /path/to/scene_root \
  --task-id pick_mug_to_sink \
  --task-text "Pick up the mug from the prep counter and place it in the sink" \
  --output-dir /path/to/teleop_bundle \
  --teleop-device keyboard \
  --task-package my_scene_task \
  --env-cfg-class TeleopEnvCfg \
  --camera-key wrist_rgb \
  --state-key policy \
  --confirm-success \
  --max-attempts 3
```

Keyboard controls:

- `w/s`: x translation
- `a/d`: y translation
- `r/f`: z translation
- `i/k`: roll
- `j/l`: pitch
- `u/o`: yaw
- `g/h`: gripper open/close
- `q`: finish recording

SpaceMouse is also supported for local recording:

```bash
blueprint-validate record-teleop \
  --scene-root /path/to/scene_root \
  --task-id pick_mug_to_sink \
  --task-text "Pick up the mug from the prep counter and place it in the sink" \
  --output-dir /path/to/teleop_bundle \
  --teleop-device spacemouse \
  --task-package my_scene_task \
  --env-cfg-class TeleopEnvCfg \
  --confirm-success \
  --max-attempts 3
```

SpaceMouse controls:

- translation and rotation map directly to end-effector deltas
- button 1 opens the gripper
- button 2 closes the gripper
- both buttons together finish the recording

The recorder now also tries to auto-discover camera observations from Isaac Lab task outputs. Use `--camera-key` only if auto-discovery picks the wrong streams.

The command writes:

- `actions.json`
- `states.json`
- per-camera MP4s
- per-camera calibration JSON
- `teleop_session_manifest.json`
- `teleop_stage1_source_manifest.json`
- `teleop_quality_report.json`

## Build Teleop Manifests

Once you have:

- one MP4 per required camera
- one action-sequence JSON file
- one state-sequence JSON file
- a LeRobot dataset root

package them into manifests:

```bash
blueprint-validate build-teleop-manifests \
  --scene-id kitchen_scene_a \
  --task-id pick_mug_to_sink \
  --task-text "Pick up the mug from the prep counter and place it in the sink" \
  --demo-index 0 \
  --robot-type franka \
  --robot-asset-ref robot/franka/franka.usd \
  --teleop-device spacemouse \
  --sim-backend isaac_sim \
  --action-space ee_delta_pose_gripper \
  --action-dim 7 \
  --lerobot-root /path/to/lerobot_dataset \
  --episode-ref episode_000000 \
  --action-sequence-path /path/to/actions.json \
  --state-sequence-path /path/to/states.json \
  --video wrist=/path/to/wrist.mp4 \
  --calibration wrist=/path/to/wrist_calibration.json \
  --state-key joint_positions \
  --state-key joint_velocities \
  --state-key end_effector_pose \
  --state-key gripper_state \
  --output-dir /path/to/teleop_bundle
```

This writes:

- `teleop_session_manifest.json`
- `teleop_stage1_source_manifest.json`
- `teleop_quality_report.json`

## Use Teleop Videos For The World Model

Point Stage 1f at the derived Stage 1 source manifest:

```yaml
external_interaction:
  enabled: true
  manifest_path: /path/to/teleop_bundle/teleop_stage1_source_manifest.json
  source_name: teleop
```

## Use Teleop Actions For Policy Training

Point the new external rollout path at the teleop session manifest:

```yaml
external_rollouts:
  enabled: true
  manifest_path: /path/to/teleop_bundle/teleop_session_manifest.json
  source_name: teleop
  mode: wm_and_policy
```

With `mode: wm_and_policy`:

- teleop videos can be used for world-model conditioning
- teleop actions are merged into policy training datasets

## Notes

- The current implementation assumes Franka-first and action-labeled teleop data already recorded in Isaac.
- It does not yet implement a live Isaac device loop inside this repo.
- The important new piece is the manifest contract and the downstream ingest path, so teleop data can be consumed without cross-repo glue.

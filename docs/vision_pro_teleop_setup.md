# Vision Pro Teleop Setup

This repo now supports `teleop_device=vision_pro` as a remote fallback path.

The implementation here is deliberately simple:

- the remote GPU box runs `record-teleop`
- `record-teleop` opens a TCP JSON-lines bridge
- a Vision Pro-side process sends control packets to that bridge

This is separate from video streaming. The headset display path still depends on NVIDIA's official CloudXR / Isaac XR teleop setup.

## What is officially required

The official NVIDIA path for Apple Vision Pro teleoperation uses:

- Isaac Lab / Isaac Sim on a remote NVIDIA workstation
- CloudXR Runtime on that workstation
- the **Isaac XR Teleop Sample Client** built and installed on Apple Vision Pro from a Mac using Xcode

References:

- [Isaac Lab CloudXR teleoperation docs](https://isaac-sim.github.io/IsaacLab/main/source/how-to/cloudxr_teleoperation.html)
- [CloudXR Framework for visionOS](https://docs.nvidia.com/cloudxr-sdk/latest/usr_guide/cloudxr_framework/index.html)

## What this repo expects from the Vision Pro side

The in-repo bridge expects newline-delimited JSON packets on the remote GPU box:

```json
{"action":[0.0,0.0,0.0,0.0,0.0,0.0,0.0],"done":false}
```

or

```json
{"ee_delta_pose":[0.0,0.0,0.0,0.0,0.0,0.0],"gripper_delta":0.0,"done":false}
```

To stop the recording:

```json
{"done":true}
```

This means the missing piece on the Vision Pro side is a small relay from the NVIDIA sample-client control stream into this JSON bridge contract.

The exact Swift-side hook is defined here:

- [docs/vision_pro_sample_client_hook.md](/Users/nijelhunt_1/workspace/BlueprintValidation/docs/vision_pro_sample_client_hook.md)

The reusable Swift helper file is here:

- [examples/vision_pro/BlueprintVisionProRelayClient.swift](/Users/nijelhunt_1/workspace/BlueprintValidation/examples/vision_pro/BlueprintVisionProRelayClient.swift)

## Remote run flow

On the GPU box:

```bash
cd /path/to/BlueprintValidation
ISAACLAB_PYTHON=/path/to/isaac-sim/python.sh \
bash scripts/bootstrap_isaac_teleop_runtime.sh
```

Then set:

```bash
export SCENE_ROOT=/path/to/scene_root
export TASK_ID=pick_tote
export TASK_TEXT="Pick up the tote and place it on the shelf"
export TELEOP_DEVICE=vision_pro
export TASK_PACKAGE=my_scene_task
export ISAACLAB_PYTHON=/path/to/isaac-sim/python.sh
export BRIDGE_HOST=0.0.0.0
export BRIDGE_PORT=49110
```

Then start the recorder:

```bash
bash scripts/run_isaac_record_teleop.sh
```

In a second shell on the GPU box, start the relay:

```bash
bash scripts/run_vision_pro_relay.sh
```

Or start both together:

```bash
bash scripts/run_vision_pro_teleop_stack.sh
```

The recorder + relay pair will:

- validate the scene package
- start `record-teleop --teleop-device vision_pro`
- wait for a bridge client on `BRIDGE_HOST:BRIDGE_PORT`
- record videos, actions, and state
- write the teleop manifests

## Triggering from your Mac

From your local machine:

```bash
export REMOTE_TARGET=user@remote-gpu-host
export REMOTE_ROOT=/path/to/BlueprintValidation
bash scripts/ssh_vision_pro_record_teleop.sh
```

The relay can be started in a second SSH session:

```bash
ssh "$REMOTE_TARGET" "cd '$REMOTE_ROOT' && bash scripts/run_vision_pro_relay.sh"
```

## Important limitation

This repo now has the **remote recorder-side Vision Pro bridge**.

It does **not** contain the Vision Pro headset app or the NVIDIA sample-client modifications needed to emit these control packets. You still need the official NVIDIA Vision Pro teleop stack plus a small relay on top of it.

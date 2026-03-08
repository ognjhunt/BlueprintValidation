# Remote Isaac Teleop Setup

Use this when your local machine cannot run Isaac Sim / Isaac Lab and you need a remote NVIDIA box.

## What this setup assumes

- this repo is already present on the remote machine
- the remote machine has an NVIDIA GPU and an Isaac Sim / Isaac Lab installation
- you can SSH into the machine from your laptop

## Step 1: Bootstrap the repo inside the Isaac runtime

On the remote machine:

```bash
cd /path/to/BlueprintValidation
ISAACLAB_PYTHON=/path/to/isaac/python.sh \
bash scripts/bootstrap_isaac_teleop_runtime.sh
```

That script:

- installs this repo into the Isaac runtime
- checks imports for:
  - `blueprint_validation`
  - `isaaclab`
  - `isaaclab_tasks`
  - `torch`
  - `cv2`

## Step 2: Set the run inputs

On the remote machine, export at least:

```bash
export SCENE_ROOT=/path/to/scene_root
export TASK_ID=pick_tote
export TASK_TEXT="Pick up the tote and place it on the shelf"
export TELEOP_DEVICE=keyboard   # or spacemouse
export TASK_PACKAGE=my_scene_task
export ISAACLAB_PYTHON=/path/to/isaac/python.sh
```

Optional:

```bash
export CONFIG_PATH=/path/to/BlueprintValidation/configs/same_facility_policy_uplift_openvla.yaml
export OUTPUT_DIR=/path/to/output/teleop
export SUCCESS_FLAG=auto
export MAX_ATTEMPTS=3
```

## Step 3: Run teleop on the remote machine

```bash
bash scripts/run_isaac_record_teleop.sh
```

This will:

1. validate the scene package
2. launch `record-teleop`
3. write:
   - `actions.json`
   - `states.json`
   - teleop videos
   - calibration JSON
   - `teleop_session_manifest.json`
   - `teleop_stage1_source_manifest.json`
   - `teleop_quality_report.json`

## Step 4: Trigger it over SSH from your laptop

From your local machine:

```bash
export REMOTE_TARGET=user@host
export REMOTE_ROOT=/path/to/BlueprintValidation
ssh -t "$REMOTE_TARGET" "cd '$REMOTE_ROOT' && bash scripts/run_isaac_record_teleop.sh"
```

Or with the helper wrapper:

```bash
export REMOTE_TARGET=user@host
export REMOTE_ROOT=/path/to/BlueprintValidation
bash scripts/ssh_isaac_record_teleop.sh
```

## After recording

Use:

- `teleop_stage1_source_manifest.json` for Stage 1f / world-model conditioning
- `teleop_session_manifest.json` for action-labeled policy seeding

The existing workflow in [docs/teleop_isaac_workflow.md](/Users/nijelhunt_1/workspace/BlueprintValidation/docs/teleop_isaac_workflow.md) stays the same after the remote recording step.

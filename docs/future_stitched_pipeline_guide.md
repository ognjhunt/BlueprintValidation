# Future Stitched Pipeline Guide

Version: 2026-03-08

## What This Document Is

This is a planning note for a future pipeline, not a recommendation to change the current stack now.

The idea is simple: keep the parts of this repo that already work, then add the missing middle layer that current papers only solve in pieces. That middle layer is where the robot gets grounded into the scene, where the task contact zone becomes simulator-ready, and where closed-loop rollouts become more action-valid than the current world-model-only path.

For now, the repo should still treat the qualified-handoff plus scene-memory-first pipeline as the default, with geometry bundles only as legacy adapters and scene packages only as stricter fallback paths.

## The Main Idea In One Line

Take the current legacy geometry-forward pipeline:

`scene_memory / geometry adapter -> render -> enrich -> world-model finetune -> synthetic rollouts -> policy finetune/eval`

Then add three missing capabilities in the middle:

- BridgeV2W-style robot grounding after the scan
- DISCOVERSE or GSWorld-style task-local physics
- World-VLA-Loop-style closed-loop rollout improvement

That stitched system is the closest thing to the "ideal paper" that does not exist yet.

## The Stitched Stages

### 1. Capture The Site

We make a 3D picture of the place from a phone walkthrough so the system has a site-specific scene to work from.

What the repo already has:

- Gaussian splat input
- Stage 1 rendering over the site
- task and camera-path scaffolding around manipulation zones

What is still needed:

- nothing major for the current future plan

### 2. Find The Small Part Of The Scene That Matters

We do not try to turn the whole building into a physics simulator. We only pick the small area where the robot will touch things, like one shelf, one bin, or one table.

What the repo already has:

- manipulation zones
- scene-aware rendering hooks
- task hints and task-local camera paths

What is still needed:

- stronger task-local object and support-surface extraction
- cleaner automation for defining the interaction zone from the splat

### 3. Put The Robot Into The Scene After The Scan

We tell the system what robot we have, where it stands, and what the camera sees, so it can place the robot into the already-scanned scene.

What the repo already has:

- optional URDF-based visual compositing in [s1b_robot_composite.py](../src/blueprint_validation/stages/s1b_robot_composite.py)

What is still needed:

- robot base pose in world coordinates
- camera intrinsics and extrinsics
- a real calibration artifact path, not just visual overlay
- a BridgeV2W-like representation that ties robot actions to rendered observations

### 4. Turn Only The Touch Zone Into Simulator Assets

The room can stay a splat. The things the robot touches cannot. Those have to become real simulator pieces with collision and contact.

What the repo already has:

- splat rendering
- robot description inputs
- a place to add future external interaction data

What is still needed:

- task-local collision geometry
- support-surface proxies
- 1-3 task objects with sim-ready geometry
- object poses in the same world frame as the robot
- MuJoCo-ready robot and scene assembly assets
- contact parameters like mass and friction

### 5. Run Closed-Loop Rollouts In That Hybrid Scene

Now the robot can try an action, see what happened, and try the next action. This is the point where the system stops being "nice robot video" and starts becoming useful for manipulation training.

What the repo already has:

- rollout and evaluation stages
- policy adapters
- world-model rollout loops

What is still needed:

- a simulator-backed rollout driver
- reset logic for task scenes
- reward and task-success signals tied to simulator state
- export of rollout trajectories from the simulator path

### 6. Keep Good, Almost-Good, And Bad Attempts

We save successful tries, near-misses, and failures because each teaches something different. Good tries show what worked, near-misses show what almost worked, and failures show what to avoid.

What the repo already has:

- rollout dataset stages
- success and near-miss curriculum logic
- policy RL loop scaffolding

What is still needed:

- direct ingestion of simulator rollouts into the same dataset contracts
- simulator-native success and failure labeling
- checks that action dimensions and rollout metadata match policy expectations

### 7. Fine-Tune The Policy First

The first thing to train on this better rollout data should be the policy. That is where the extra physics and action validity are most likely to help.

What the repo already has:

- policy fine-tune stages
- RLDS and rollout export paths
- trained-policy evaluation stages

What is still needed:

- a clean simulator-to-dataset bridge
- one stable policy-training recipe for simulator-generated manipulation data

### 8. Optionally Feed The World Model Later

Once the simulator rollouts look good enough, we can render them back into videos and use them to refresh the world model too. This is optional and should come after policy training, not before.

What the repo already has:

- world-model finetuning
- world-model refresh loops

What is still needed:

- a render-back path from simulator rollouts to video clips
- quality gates to avoid poisoning the world model with bad synthetic videos

### 9. Compare It Against The Current Baseline

The last step is to test the new stitched path against the current world-model-only baseline on the same held-out tasks. If it does not beat the current path, the extra simulator work is not worth carrying.

What the repo already has:

- same-facility evaluation framing
- held-out policy comparison stages
- policy uplift claim logic

What is still needed:

- a fixed A/B benchmark for "current WM-only" vs "stitched hybrid path"
- clear reporting that separates world-model uplift from future IRL uplift

## What Stays Splat-Only Vs What Must Become Sim-Ready

Splat-only is fine for:

- room and background appearance
- distant geometry used only for rendering
- non-interactive clutter
- scene realism outside the contact zone

Must become sim-ready:

- robot embodiment used for control
- grasped, pushed, opened, lifted, or placed objects
- support surfaces and receptacles near contact
- articulated objects
- geometry that decides reachability, collision, or placement success

This is the part worth remembering: a robot drawn into a splat is enough for visual conditioning, but not for action-valid manipulation.

## What The Repo Already Has Vs What Is Missing

The current codebase already covers the front half and the back half.

Front half already present:

- scan or splat input
- rendering
- video enrichment
- world-model finetuning

Back half already present:

- rollout dataset building
- policy finetuning
- policy evaluation and A/B comparison

The missing middle is:

- robot grounding
- task-local simulator assets
- simulator-backed closed-loop rollouts

That missing middle is the reason there is still no clean end-to-end paper for this workflow.

## Best Current Research Pieces To Stitch

Use these as design references, not as a direct drop-in stack:

- BridgeV2W for post-hoc robot grounding into visual world models
- DISCOVERSE or GSWorld for task-local 3DGS plus physics
- World-VLA-Loop for closed-loop policy and world-model improvement
- the current repo for dataset export, policy fine-tuning, and fixed-world evaluation

## Expected Uplift

### Source-Backed Facts

There is no single paper that benchmarks this exact stitched pipeline against the current repo.

What the source papers do show is narrower:

- DISCOVERSE reports stronger zero-shot Sim2Real transfer than weaker simulator baselines on its benchmark tasks. Source: [DISCOVERSE paper](https://arxiv.org/abs/2507.21981).
- World-VLA-Loop reports closed-loop gains from iteratively improving policy and world model together, but its setup depends on a different data regime. Source: [World-VLA-Loop paper](https://arxiv.org/abs/2602.06508).
- BridgeV2W shows that post-hoc robot grounding can make action-conditioned visual prediction substantially more robot-aware. Source: [BridgeV2W paper](https://arxiv.org/abs/2602.03793).

These are useful pieces. They are not an end-to-end proof for this repo's workflow.

### Internal Planning Estimate

Treat these as planning priors, not claims.

If a future stitched pipeline actually works well:

- world-model improvement: `+12 to +25` absolute points
- future same-facility IRL improvement: `+6 to +18` absolute points

Why the world-model estimate is higher:

- the stitched system directly improves the quality of synthetic rollouts inside its own training substrate
- IRL still pays the cost of calibration error, transfer loss, sensing noise, and imperfect contacts

Failure case:

If the task-local simulator assets are weak, the stitched path may add a lot of complexity and almost no gain.

## When To Revisit This

Revisit the stitched pipeline only if most of these are true:

- one manipulation task is stable and well-specified
- robot pose and camera calibration can be collected reliably
- at least one object and one support surface can be turned into usable simulator assets
- the current world-model-only uplift has flattened enough to justify a larger system change
- the team is willing to maintain a simulator path in addition to the current world-model path

If those conditions are not true, stay with the existing stack.

## Sources

- [BlueprintValidation README](../README.md)
- [No-New-Data Scale Playbook](./no_new_data_scaling_playbook.md)
- [Scan-Only Capture Research Memo](../reports/scan_only_capture_research_20260307.md)
- [BridgeV2W paper](https://arxiv.org/abs/2602.03793)
- [DISCOVERSE paper](https://arxiv.org/abs/2507.21981)
- [DISCOVERSE repository](https://github.com/TATP-233/DISCOVERSE)
- [DISCOVERSE Real2Sim repository](https://github.com/GuangyuWang99/DISCOVERSE-Real2Sim)
- [GSWorld paper](https://arxiv.org/abs/2510.20813)
- [World-VLA-Loop paper](https://arxiv.org/abs/2602.06508)

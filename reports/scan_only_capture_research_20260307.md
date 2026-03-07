# Scan-Only Capture Research Memo

Date: 2026-03-07

## Constraint Checklist

This memo evaluates papers against the actual BlueprintCapture constraint set:

- capture operator has only a phone or walkthrough video
- no robot present during capture
- no extra hardware required at capture time
- build a Gaussian splat first
- generate extra synthetic data from that capture to fine-tune downstream policies
- target use case is site-specific manipulation data generation, not just benchmarking or world-model demos

## Verdict

No paper reviewed here is a direct fit for BlueprintCapture's phone-only walkthrough workflow. Several adjacent papers are useful references, but all of them fail at least one critical requirement: robot-aware capture, extra calibration hardware, action-conditioned robot interaction data, or object/task-specific demonstrations instead of site-scale capture.

## Reviewed Papers

### Does Not Fit

#### Interactive World Simulator

- Verdict: `does not fit`
- Sources: [project page](https://www.yixuanwang.me/interactive_world_sim/), [paper](https://www.yixuanwang.me/interactive_world_sim/texts/main.pdf), [code](https://github.com/WangYixuan12/interactive_world_sim)

Interactive World Simulator is an action-conditioned video world model trained from robot interaction episodes. Its core assumption is paired robot observations and actions, and the released code/data path is built around ALOHA episodes rather than phone-only scene capture. That makes it useful as a reference for world-model data generation and policy evaluation, but not as a drop-in way to turn a walkthrough video into a site-specific manipulation simulator.

#### SplatSim

- Verdict: `does not fit`
- Sources: [paper](https://arxiv.org/abs/2409.10161), [project page](https://splatsim.github.io/), [code](https://github.com/qureshinomaan/SplatSim)

SplatSim is directly adjacent to the desired workflow because it uses Gaussian splatting plus a simulator to generate synthetic manipulation data. However, its published method requires an initial static scene capture with the robot already in the scene, manual robot segmentation/alignment, and robot-aware simulator calibration. It removes the need for extra real interaction collection in the new scene, but it does not satisfy the stricter BlueprintCapture requirement of robot-free phone-only capture.

#### GSWorld

- Verdict: `does not fit`
- Sources: [paper](https://arxiv.org/abs/2510.20813), [project page](https://3dgsworld.github.io/), [code](https://github.com/luccachiang/GSWorld)

GSWorld is a stronger simulator platform than SplatSim and supports zero-shot sim2real training, DAgger, benchmarking, and virtual teleoperation. But its real2sim pipeline still assumes robot-aware capture: metric scaling with ArUco markers, recorded robot joint pose during capture, manual robot crop, and sim-to-real alignment to transfer robot semantics into the reconstruction. That makes it valuable for future robot-aware digital twins, but not a fit for pure walkthrough capture with no robot and no extra capture hardware.

### Partial

#### PhysTwin

- Verdict: `partial`
- Sources: [paper](https://arxiv.org/abs/2503.17973)

PhysTwin is a strong real-to-sim paper for deformable objects. It reconstructs a physically meaningful digital twin from sparse interaction videos and supports interactive simulation and planning. The mismatch is scope: it is object-centric and interaction-centric, not a general site-scale walkthrough-to-splat system for generating robot policy data across a location. It also depends on dynamic interaction videos rather than passive scene capture.

#### Gen2Real

- Verdict: `partial`
- Sources: [paper](https://arxiv.org/abs/2509.14178)

Gen2Real is relevant because it reduces dependence on human robot demonstrations by generating a video first and then extracting usable trajectories for dexterous manipulation. But it is not a scene reconstruction or digital-twin paper. It learns skills from generated or human-like demonstration videos, not from a site walkthrough, so it does not solve the environment-capture side of the BlueprintCapture problem.

#### Actron3D

- Verdict: `partial`
- Sources: [paper](https://arxiv.org/abs/2510.12971)

Actron3D learns transferable manipulation skills from a few monocular, uncalibrated human RGB videos. That is attractive from a data-efficiency perspective, but the videos are still task demonstrations rather than a generic walkthrough of the environment. The method is object- and action-centric, not environment-bootstrap-centric, so it does not provide a way to start from a robot-free site scan and then generate synthetic policy data for that site.

#### GaussTwin

- Verdict: `partial`
- Sources: [paper](https://arxiv.org/abs/2603.05108)

GaussTwin is a real-time robotic digital twin that combines physically grounded simulation with Gaussian splatting and visual correction. It is promising for closed-loop digital twins and online correction, but it is not a phone-only bootstrap pipeline. The setup still assumes a robot-specific digital twin with simulation/correction tightly coupled to the robot and task, so it does not match a robot-free walkthrough capture workflow.

### Additional Adjacent Results From the Broader Search

#### OK-Robot

- Verdict: `partial`
- Sources: [paper](https://arxiv.org/abs/2401.12202)

OK-Robot is notable because it can operate in a new home environment seeded by an iPhone scan. That is the closest result in the broader search to the "phone-only capture" constraint. However, it is not a site-specific synthetic data generation or fine-tuning pipeline based on Gaussian splats. It is a systems integration result for open-vocabulary mobile manipulation, so it does not replace the BlueprintValidation scan-to-synthetic-data path.

#### PhysWorld

- Verdict: `partial`
- Sources: [paper](https://arxiv.org/abs/2510.21447)

PhysWorld is another object-centric digital twin paper for deformable objects. It synthesizes demonstrations inside a simulator after fitting a physics-aware twin from videos. That is useful as a reference for data synthesis once a twin exists, but it still focuses on deformable object modeling from interaction videos rather than phone-only site capture for general manipulation data generation.

## Recommendation

- No current paper reviewed here is a direct drop-in for BlueprintCapture's phone-only walkthrough workflow.
- The current repo should continue treating scan/video-first world-model augmentation as the baseline.
- If a future direction is explored, the strongest adjacent paths are:
  - robot-aware digital twins such as GSWorld, but only if the capture protocol changes
  - task-demonstration-first methods such as Actron3D or Gen2Real, but only if the data model changes away from environment-first capture

For the current BlueprintCapture constraint, the most defensible conclusion is still negative: no exact-fit method was found.

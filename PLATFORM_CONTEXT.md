# Platform Context

<!-- SHARED_PLATFORM_CONTEXT_START -->
## Shared Platform Doctrine

### System Framing

- `BlueprintCapture` captures raw evidence packages.
- `BlueprintCapturePipeline` converts evidence plus intake into qualification artifacts, readiness decisions, and handoffs.
- `Blueprint-WebApp` is the operating and commercial system around qualification records and derived downstream lanes.
- `BlueprintValidation` performs post-qualification scene derivation, robot evaluation, adaptation, and tuning work.

This platform is qualification-first.

### Truth Hierarchy

- qualification records, readiness decisions, and supporting evidence links are authoritative
- capture-backed scene memory is the preferred downstream substrate when deeper technical work is justified
- preview simulations, world-model outputs, and world-model-trained policies are derived downstream assets; they do not rewrite qualification truth

### Product Stack

1. primary product: site qualification / readiness pack
2. secondary product: qualified opportunity exchange for robot teams
3. third product: scene memory / preview simulation / robot eval package
4. fourth product: world-model-based adaptation, managed tuning, training data, licensing

### Downstream Training Rule

- world-model RL and world-model-based post-training are first-class downstream paths for site adaptation, checkpoint ranking, synthetic rollout generation, and bounded robot-team evaluation
- those paths sit behind qualification and do not by themselves replace stricter validation for contact-critical, safety-critical, or contractual deployment claims
- Isaac-backed, physics-backed, or otherwise stricter validation remains the higher-trust lane when reproducibility, contact fidelity, or formal signoff matters

### Data Rule

- passive site capture and walkthrough evidence are valuable context for scene memory, preview simulation, and downstream conditioning
- strong robot adaptation gains usually require action-conditioned robot interaction data such as play, teleop logs, or task rollouts; site video alone is usually not enough for reliable policy training from scratch
- derived assets may inform routing and downstream work, but they must not mutate qualification state or source-of-truth readiness records
<!-- SHARED_PLATFORM_CONTEXT_END -->

This repo is the post-qualification evaluation and adaptation engine.

## What This Repo Owns

`BlueprintValidation` starts after qualification.

Its main jobs are:

- consume qualified opportunity handoffs
- consume scene-memory bundles, preview simulations, or legacy geometry bundles
- generate synthetic frames, trajectories, rollouts, datasets, and evaluation packages
- compare downstream robot-stack performance under bounded protocols

This repo does not own site-readiness truth, intake, or upstream capture policy.

## Product Context

The correct stack is:

1. primary product: site qualification / readiness pack
2. secondary product: qualified opportunity exchange
3. third product: scene memory / preview simulation / evaluation package
4. fourth product: world-model-based adaptation / managed tuning / training data / licensing

This repo lives in steps 3 and 4.

## Practical Rule For Agents

When making changes here, optimize for:

1. post-qualification evaluation discipline
2. scene-memory-first intake compatibility
3. clean dataset and rollout contracts
4. explicit separation between downstream world-model evidence and stronger validation gates

Do not let generated rollouts replace qualification evidence or authoritative readiness records.

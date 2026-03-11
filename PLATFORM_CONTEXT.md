# Platform Context

This repo is one part of a four-repo system.

## System Framing

- `BlueprintCapture` creates the raw evidence package.
- `BlueprintCapturePipeline` creates the qualification record and canonical scene-memory bundle.
- `Blueprint-WebApp` is the operating system around qualification records and derived assets.
- `BlueprintValidation` is the post-qualification scene-derivation and evaluation engine.

This platform is qualification-first.

Canonical scene doctrine:

- capture-backed scene memory is the preferred downstream input
- generated rollouts and world-model outputs are supporting evidence by default
- stricter validation paths such as PolaRiS or Isaac-backed scene packages remain the place for contact-critical claims

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
4. fourth product: scenario generation / managed tuning / licensing

This repo lives in steps 3 and 4.

## Practical Rule For Agents

When making changes here, optimize for:

1. post-qualification evaluation discipline
2. scene-memory-first intake compatibility
3. clean dataset and rollout contracts
4. explicit separation between supporting world-model evidence and stronger validation gates

Do not let generated rollouts replace qualification evidence or authoritative readiness records.

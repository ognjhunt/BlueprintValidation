# No-New-Data Scale Playbook

Version: 2026-03-01

## Purpose

This guide is the working playbook for future sessions when constraints are strict:

- no new real-world data collection
- no teleop collection
- no extra robot hardware in the loop
- only existing site captures ("splay") + synthetic videos/trajectories/demos/world-model rollouts
- target operating mode: scale to 100+ locations per week

Current repo outputs are world-model-only. Any IRL numbers below are planning priors for future matched robot runs, not metrics produced by the current pipeline.

Canonical question this repo can answer today:

- Within one fixed site-adapted world model of the target facility, can site-specific policy training beat the frozen baseline on disjoint starts/tasks in that same facility?

Future same-facility deployment follow-up, not answered here:

- If that uplift appears in the adapted world model, does it also carry over IRL in the exact same facility?

Treat any IRL number below as future deployment planning only, not as a current repo output or headline claim.

## Scoring Convention

All uplift estimates in this guide are **absolute points on a 100-point success-rate scale**.

- Example: 42% to 55% success = +13 absolute points
- 10-scale conversion: divide by 10 (e.g., +13 on 100-scale = +1.3 on 10-scale)

## Current Baseline Assumption (This Repo)

Pipeline stages in scope:

- Stage 1: coverage rendering from site splat
- Stage 2: cosmos enrichment
- Stage 3: world-model finetune
- Stage 3d (default-on in WM-only scope): world-model refresh loop from success/near-miss rollouts
- Stage 4+: policy finetune/eval loops

Recently implemented fidelity + memory hooks (already in this repo) should primarily improve data quality and consistency (orientation, duration fidelity, quality, context hooks).

Expected immediate impact from these already-applied changes:

- world-model metrics in this repo: **+6 to +12 absolute points**
- future same-facility IRL planning prior once matched robot runs exist: **+2 to +6 absolute points**

These are priors, not guarantees. Validate the current repo claim per-location with the fixed-world Stage 4d claim path and held-out tasks inside the adapted world model.

## Decision Matrix (No-New-Real-Data Compatible)

| Approach | Code available now? | Fits no-new-real-data rule? | Stack change | Extra data needed beyond current splay/sim outputs | Estimated uplift (WM now / future same-facility IRL planning prior, absolute points) |
|---|---|---|---|---|---|
| A. Existing fidelity + context hooks (already merged) | Yes (in-repo) | Yes | Small | None | +6 to +12 / +2 to +6 |
| B. Stage 1 coverage densification + longer windows | Yes (in-repo knobs) | Yes | Small | None | +3 to +8 / +1 to +4 |
| C. Stage 2 multi-view + scene-index retrieval at scale | Yes (in-repo scaffolding) | Yes | Small-Med | None | +2 to +6 / +1 to +3 |
| D. Synthetic hard-negative loop (WMPO-style, but fully synthetic rollouts) | Yes (build in current stack) | Yes | Medium | Synthetic failures only | +5 to +12 / +3 to +8 |
| E. SplatSim integration path | Yes ([repo](https://github.com/qureshinomaan/SplatSim)) | Yes (if seeded from existing captures + synthetic trajectories) | Medium-Large | Simulation trajectories/demos (synthetic acceptable) | +4 to +12 / +3 to +10 |
| F. GSWorld integration path | Yes ([repo](https://github.com/luccachiang/GSWorld)) | Yes (can run from existing captures) | Large | Existing site images/splats + synthetic policies | +6 to +15 / +4 to +12 |
| G. DISCOVERSE integration path | Yes ([repo](https://github.com/TATP-233/DISCOVERSE)) | Yes (no additional real rollouts required) | Large | Scene assets + synthetic training loops | +6 to +16 / +4 to +12 |
| H. MimicGen/RoboCasa synthetic demo expansion | Yes ([MimicGen](https://github.com/NVlabs/mimicgen), [RoboCasa docs](https://robocasa.ai/docs/use_cases/mimicgen.html)) | Yes (if source demos are synthetic/existing) | Medium | Source demonstrations (can be synthetic/programmatic) | +5 to +14 / +3 to +9 |
| I. WMPO full stack adoption | Yes ([repo](https://github.com/WM-PO/WMPO)) | Mostly (can train without new real interaction after base) | Large | Heavy compute + large released data/checkpoints | +6 to +18 / +4 to +12 |
| J. World-VLA-Loop direct reproduction | **Not fully** (repo says "In preparation") | Not ideal under current constraints | Large | SANS-like near-success + teleop in original method | N/A until code/data mature |

## What To Prioritize First (Highest ROI Under Constraints)

1. Fully exploit in-repo knobs before external framework integration.
2. Add synthetic hard-negative loops (self-play failures -> retrain) inside current pipeline.
3. Add one external sim stack (choose only one first: GSWorld or DISCOVERSE) to avoid fragmentation.
4. Use MimicGen-style expansion only after source demo quality gates are strong.

## Recommended 3-Phase Execution Plan

### Phase 1 (1-2 weeks): Max out current stack, no architecture fork

- Expand Stage 1 coverage density for task-critical paths
- Keep Stage 2 full-duration input (`max_input_frames=0`) unless memory bound
- Turn on multi-view context for task-critical clips
- Enable scene-index retrieval for context augmentation
- Add hard-negative mining from failed policy rollouts in world model

Target uplift in current repo:

- world model: +4 to +10

Future same-facility IRL planning prior (not measured here):

- IRL transfer: +2 to +5

### Phase 2 (2-4 weeks): Synthetic closed-loop training

- Generate synthetic rollouts from current policy in world model
- Rank failures with VLM/judge + reward proxy
- Retrain world model and policy on success + near-success + hard failures
- Use acceptance gates to prevent distribution drift

Target uplift in current repo:

- world model: +6 to +14

Future same-facility IRL planning prior (not measured here):

- IRL transfer: +3 to +8

### Phase 3 (4-8+ weeks): Add one photo-realistic simulator backbone

Pick one implementation path first:

- GSWorld if you want stronger closed-loop + ManiSkill ecosystem alignment
- DISCOVERSE if you want a broad modular 3DGS + MuJoCo pipeline
- SplatSim if you want a narrower, faster-to-prototype sim2real path

Target uplift in current repo (incremental over Phases 1-2):

- world model: +4 to +12

Future same-facility IRL planning prior (not measured here):

- IRL transfer: +3 to +10

## Practical Constraints and Risks

- Orientation/time/quality regressions can erase most gains; keep automated visual checks mandatory.
- Synthetic-only loops can overfit to simulator artifacts; require held-out site checks.
- Large external stacks can stall velocity; run one integration at a time.
- Reward model drift is a hidden failure mode; keep periodic human spot-audits on sampled clips.

## Minimal KPI Pack For Every New Location

Track these per location and aggregate weekly:

- Stage 2 artifact pass rate (orientation, duration fidelity, quality gate)
- World-model success on held-out scripted tasks
- Policy success in adapted world model
- Future same-facility IRL proxy score and final IRL success, once matched robot runs exist
- Gap metric: world-model success minus same-facility IRL success (future)

Decision thresholds:

- If world-model improves but future matched same-facility IRL runs do not: prioritize sim2real robustness (domain randomization, retrieval context, hard negatives).
- If both stall: increase Stage 1 coverage and task-conditional diversity before adding new frameworks.

## Future Session Template

Use this template to keep work consistent across sessions:

1. Confirm constraints: no new real data, no teleop, scalable to 100+ locations/week.
2. Pick one phase objective (Phase 1, 2, or 3).
3. Define expected uplift range in absolute points (100-scale).
4. Specify exact config/code changes.
5. Run A/B on at least one held-out task set.
6. Report uplift with confidence band and failure modes.

## Research Notes (Verified 2026-03-01)

### World-VLA-Loop

- Project page outlines a 4-phase loop and explicitly includes SANS curation mainly via manual teleoperation ([project](https://showlab.github.io/World-VLA-Loop/)).
- Public GitHub currently states source code is "In preparation" ([repo](https://github.com/showlab/World-VLA-Loop)).
- Conclusion: use as design inspiration, not as immediate drop-in.

### SplatSim

- Public repository and installation docs are available ([repo](https://github.com/qureshinomaan/SplatSim)).
- README describes dependency on trajectories and real2sim assets for examples.
- Conclusion: implementable now; needs careful adapter layer to fit this repo.

### GSWorld

- Official code is public with setup docs and real2sim pipeline scripts ([repo](https://github.com/luccachiang/GSWorld), [paper](https://arxiv.org/abs/2510.20813)).
- Conclusion: high potential, but larger integration effort.

### DISCOVERSE

- Open-source 3DGS-based framework with docs and training workflows ([repo](https://github.com/TATP-233/DISCOVERSE), [paper](https://arxiv.org/abs/2507.21981)).
- Conclusion: strong candidate for scalable synthetic expansion.

### MimicGen / RoboCasa

- MimicGen full generation code is publicly released and intended for large-scale synthetic demo creation ([repo](https://github.com/NVlabs/mimicgen)).
- RoboCasa provides a documented MimicGen integration path ([docs](https://robocasa.ai/docs/use_cases/mimicgen.html)).
- Conclusion: strong fit for synthetic data scaling if source demos are high quality.

### WMPO

- Official code, scripts, and checkpoints are public ([repo](https://github.com/WM-PO/WMPO), [project](https://wm-po.github.io/)).
- Repository notes large dataset/checkpoint footprint.
- Conclusion: implementable now but compute-heavy; good reference for synthetic policy optimization loops.

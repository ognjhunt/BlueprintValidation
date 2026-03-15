# Platform Context

<!-- SHARED_PLATFORM_CONTEXT_START -->
## Shared Platform Doctrine

### System Framing

- `BlueprintCapture` is the contributor evidence-capture tool inside Blueprint's three-sided marketplace.
- `BlueprintCapturePipeline` is the authoritative qualification, provenance, and provider-routing service.
- `Blueprint-WebApp` is the three-sided marketplace and operating system connecting capturers, robot teams, and site operators around qualification records and downstream work.
- `BlueprintValidation` is optional downstream infrastructure for provider benchmarking, runtime-backed demos, and deeper robot evaluation after qualification.

This platform is qualification-first.

### Three-Sided Marketplace

- **Capturers** supply evidence packages from real sites.
- **Robot teams** are the primary demand-side buyers of trusted qualification outcomes and downstream technical work.
- **Site operators** control access, rights, and commercialization boundaries for their facilities.

### Truth Hierarchy

- qualification records, readiness decisions, trust signals, and provenance links are authoritative
- capture-backed scene memory and evaluation-prep packages are preferred downstream technical substrates once qualification justifies them
- preview simulations, provider outputs, advanced-geometry bundles, and trained policies are derived downstream assets; they do not rewrite qualification truth

### Product Stack

1. primary product: qualification record / readiness decision / buyer-safe evidence bundle
2. secondary product: qualified opportunity exchange and provider-backed preview lane
3. third product: scene memory / evaluation-prep / runtime-backed robot evaluation
4. fourth product: world-model-based adaptation, managed tuning, training data, licensing

### Downstream Training Rule

- world-model RL and world-model-based post-training are first-class downstream paths for site adaptation, checkpoint ranking, synthetic rollout generation, and bounded robot-team evaluation
- those paths sit behind qualification and do not by themselves replace stricter validation for contact-critical, safety-critical, or contractual deployment claims
- Isaac-backed, physics-backed, or otherwise stricter validation remains the higher-trust lane when reproducibility, contact fidelity, or formal signoff matters

### Data Rule

- passive site capture and walkthrough evidence are valuable context for qualification, scene memory, preview simulation, and downstream conditioning
- strong robot adaptation gains usually require action-conditioned robot interaction data such as play, teleop logs, or task rollouts; site video alone is usually not enough for reliable policy training from scratch
- derived assets may inform routing and downstream work, but they must not mutate qualification truth
<!-- SHARED_PLATFORM_CONTEXT_END -->

This repo is optional downstream benchmarking and evaluation infrastructure.

## Local Doctrine

- This repo is not launch-critical for alpha.
- Qualification, trust, rights/compliance, and buyer-facing readiness stay upstream.
- This repo is used after qualification when Blueprint needs provider comparison, runtime-backed demos, or deeper robot evaluation.

## What This Repo Owns

- provider preview benchmarking and comparison
- runtime-backed demo flows and exportable reports
- deeper robot evaluation after a site has already been qualified
- downstream artifacts that help technical teams compare preview/runtime options

## What This Repo Does Not Own

- raw capture intake
- qualification state
- buyer trust score
- rights/compliance status
- payout or contributor approval state
- marketplace routing or buyer review UX

## Practical Rule For Agents In This Repo

When making changes here, optimize for:

1. optional downstream utility
2. benchmark clarity and provenance
3. explicit separation from upstream qualification truth
4. safe failure modes that do not block the alpha capture-to-review loop

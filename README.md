# BlueprintValidation

Gaussian splat to robot world model validation pipeline. Proves that scanning a facility and turning it into training data makes robot policies perform better in a world model that knows that specific site.

## Pipeline

```
PLY file (from BlueprintCapturePipeline)
  → Stage 1: Render video clips at robot-height via gsplat
  → Stage 1b (optional): Composite URDF robot arm with camera extrinsics
  → Stage 1c (optional): Gemini image polish on composited clips
  → Stage 1d (optional): Full RoboSplat-default 3D Gaussian augmentation (hybrid fallback)
  → Stage 1e (optional): Minimal SplatSim interaction clips (PyBullet, fallback-safe)
  → Stage 2: Enrich with Cosmos Transfer 2.5 (5-10 variants per clip)
  → Stage 3: Fine-tune DreamDojo-2B on enriched video
  → Stage 4: Frozen policy rollouts (baseline vs adapted world model) + VLM scoring
  → Stage 4a: Export adapted rollouts to RLDS TFRecords
  → Stage 3b: OFT-oriented policy fine-tuning on generated rollouts
  → Stage 3c: World-VLA-Loop-style iterative policy RL + world refresh
  → Stage 4e: Evaluate trained policy in adapted world model
  → Stage 4b: Export paired rollouts to RLDS-style train/heldout datasets
  → Stage 4c: Train policy_base vs policy_site from same initialization/budget
  → Stage 4d: Heldout policy A/B evaluation in same world model
  → Stage 5: Visual fidelity metrics (PSNR/SSIM/LPIPS)
  → Stage 6: Spatial accuracy (VLM layout verification)
  → Stage 7: Cross-site discrimination (requires 2 facilities)
  → Final validation report (Markdown + JSON)
```

## Quick Start

```bash
# Install
git clone https://github.com/ognjhunt/BlueprintValidation.git
cd BlueprintValidation
uv sync

# Download model weights (~30GB)
bash scripts/download_models.sh

# Place PLY files
cp /path/to/facility_a.ply data/facilities/facility_a/splat.ply
cp /path/to/facility_b.ply data/facilities/facility_b/splat.ply

# Configure
cp configs/example_validation.yaml validation.yaml
# Edit validation.yaml with your facility details

# Run preflight checks
blueprint-validate preflight

# Auto-generate a fast pilot config from BlueprintCapturePipeline runs
bash scripts/setup_first_data.sh

# Optional: source runtime secrets from a local untracked file
# cp scripts/runtime_env.example scripts/runtime_env.local
# set -a && source scripts/runtime_env.local && set +a

# Run full pipeline
blueprint-validate run-all

# Or run stages individually
blueprint-validate render --facility facility_a
blueprint-validate compose-robot --facility facility_a    # optional
blueprint-validate polish-gemini --facility facility_a    # optional
blueprint-validate augment-gaussian --facility facility_a # optional Stage 1d (full RoboSplat default)
blueprint-validate augment-robosplat --facility facility_a # alias
blueprint-validate simulate-interaction --facility facility_a # optional Stage 1e
blueprint-validate enrich --facility facility_a
blueprint-validate finetune --facility facility_a
blueprint-validate eval-policy --facility facility_a
blueprint-validate export-rlds --facility facility_a      # optional Stage 4a
blueprint-validate finetune-policy --facility facility_a  # optional
blueprint-validate rl-loop-policy --facility facility_a   # optional Stage 3c
blueprint-validate eval-trained-policy --facility facility_a  # optional Stage 4e
blueprint-validate export-rollouts --facility facility_a
blueprint-validate train-policy-pair --facility facility_a
blueprint-validate eval-policy-pair --facility facility_a
blueprint-validate eval-visual --facility facility_a
blueprint-validate eval-spatial --facility facility_a
blueprint-validate eval-crosssite
blueprint-validate report
```

## Cloud GPU (RunPod/Lambda)

```bash
# Build Docker image
docker build -f docker/runpod.Dockerfile -t blueprint-validation:latest .

# On RunPod: SSH in and run
bash /app/scripts/runpod_launch.sh
blueprint-validate --config /app/configs/example_validation.yaml run-all
```

### Docker Snapshot

Build a reusable runtime image snapshot (includes DreamDojo/Cosmos/OpenVLA repos):

```bash
bash scripts/build_runtime_snapshot.sh
```

Build + push to Docker Hub:

```bash
DOCKERHUB_IMAGE=<dockerhub-user>/blueprint-validation PUSH=true bash scripts/build_runtime_snapshot.sh
```

Provision local vendor repos for non-Docker runs (auto-invoked by `setup_first_data.sh`):

```bash
PROVISION_REPOS=true bash scripts/setup_first_data.sh
```

## Components

| Component | Source | Purpose |
|-----------|--------|---------|
| [DreamDojo 2B](https://github.com/NVIDIA/DreamDojo) | NVIDIA | World model for fine-tuning |
| [Cosmos Transfer 2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5) | NVIDIA | Video enrichment |
| RoboSplat (pinned vendor + native completion) | SHOWLab + local | Default S1d 3D Gaussian augmentation backend |
| [OpenVLA-OFT](https://github.com/moojink/openvla-oft) | OpenVLA team | Default policy adapter/fine-tuning recipe |
| [OpenVLA 7B](https://github.com/openvla/openvla) | Stanford | Backward-compatible baseline policy path |
| [gsplat](https://github.com/nerfstudio-project/gsplat) | Nerfstudio | Gaussian splat rendering |
| Gemini 3 Flash Preview | Google | VLM judge with Agentic Vision |

## Requirements

- Python 3.10+
- CUDA GPU (H100 recommended for full pipeline)
- `uv` package manager
- (Optional) `pybullet` for Stage 1e SplatSim interaction generation (`pip install "blueprint-validation[manipulation]"`)
- (Optional) pinned RoboSplat vendor repo at `./vendor/robosplat` for vendor backend path
- Google Gemini API key (for VLM judge)
- HuggingFace account (for model downloads)
- Local clones or container images with:
  - DreamDojo repo (and importable `cosmos_predict2`)
  - Cosmos Transfer repo (`examples/inference.py`)
  - OpenVLA-OFT repo (`vla-scripts/finetune.py` compatible entrypoint)
- HuggingFace auth (`huggingface-cli login` or `HF_TOKEN`) with accepted licenses for:
  - `nvidia/DreamDojo`
  - `nvidia/Cosmos-Transfer2.5-2B`
  - `openvla/openvla-7b`

## Manipulation-Focused Setup

- Use manipulation-centric tasks in `eval_policy.tasks` (pick/place/regrasp/tote handling).
- Reuse task inference from `BlueprintCapturePipeline` by setting `facilities.<id>.task_hints_path`
  to that run's `task_targets.json` (Gemini+heuristic video analysis output).
- Use close-range capture paths around task-relevant objects (totes, bins, shelf faces).
- For synthetic robot-context data:
  - enable `robot_composite.enabled=true` and set `robot_composite.urdf_path`
  - optionally enable `gemini_polish.enabled=true` for photoreal blending polish
  - keep geometry filters on (`robot_composite.min_visible_joint_ratio`, `min_consistency_score`)
- For policy improvement claim:
  - `policy_finetune.enabled` and `rollout_dataset.enabled` are on by default in current templates
  - `policy_rl_loop.enabled` and `policy_compare.enabled` stay off by default
  - enable `policy_compare.enabled` when you want paired policy A/B training/eval
  - run `export-rollouts`, `train-policy-pair`, `eval-policy-pair`
- Capture checklist: `configs/capture/manipulation_capture_checklist.md`

## Testing

```bash
uv run pytest tests/ -v
```

## License

MIT

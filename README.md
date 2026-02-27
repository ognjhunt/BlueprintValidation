# BlueprintValidation

Gaussian splat to robot world model validation pipeline. Proves that scanning a facility and turning it into training data makes robot policies perform better in a world model that knows that specific site.

## Pipeline

```
PLY file (from BlueprintCapturePipeline)
  → Stage 1: Render video clips at robot-height via gsplat
  → Stage 2: Enrich with Cosmos Transfer 2.5 (5-10 variants per clip)
  → Stage 3: Fine-tune DreamDojo-2B on enriched video
  → Stage 3b (optional): Fine-tune OpenVLA policy (LoRA/OFT)
  → Stage 4: OpenVLA policy eval (baseline vs adapted) + VLM judge scoring
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

# Run full pipeline
blueprint-validate run-all

# Or run stages individually
blueprint-validate render --facility facility_a
blueprint-validate enrich --facility facility_a
blueprint-validate finetune --facility facility_a
blueprint-validate finetune-policy --facility facility_a  # optional
blueprint-validate eval-policy --facility facility_a
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

## Components

| Component | Source | Purpose |
|-----------|--------|---------|
| [DreamDojo 2B](https://github.com/NVIDIA/DreamDojo) | NVIDIA | World model for fine-tuning |
| [Cosmos Transfer 2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5) | NVIDIA | Video enrichment |
| [OpenVLA 7B](https://github.com/openvla/openvla) | Stanford | Robot policy for evaluation |
| [gsplat](https://github.com/nerfstudio-project/gsplat) | Nerfstudio | Gaussian splat rendering |
| Gemini 3 Flash Preview | Google | VLM judge with Agentic Vision |

## Requirements

- Python 3.10+
- CUDA GPU (H100 recommended for full pipeline)
- `uv` package manager
- Google Gemini API key (for VLM judge)
- HuggingFace account (for model downloads)

## Manipulation-Focused Setup

- Use manipulation-centric tasks in `eval_policy.tasks` (pick/place/regrasp/tote handling).
- Use close-range capture paths around task-relevant objects (totes, bins, shelf faces).
- If you want policy weight updates, enable `policy_finetune.enabled=true` and point
  `policy_finetune.data_root_dir` to an OpenVLA-compatible dataset root.
- Capture checklist: `configs/capture/manipulation_capture_checklist.md`

## Testing

```bash
uv run pytest tests/ -v
```

## License

MIT

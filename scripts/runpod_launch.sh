#!/usr/bin/env bash
# Helper script for setting up BlueprintValidation on a RunPod pod.
# Run this after SSH-ing into the pod.
set -euo pipefail

echo "=== BlueprintValidation RunPod Setup ==="

# Activate venv
if [ ! -f "/app/.venv/bin/activate" ]; then
    echo "Virtual environment not found at /app/.venv/bin/activate"
    exit 1
fi
source /app/.venv/bin/activate

if ! command -v blueprint-validate >/dev/null 2>&1; then
    echo "blueprint-validate CLI is not installed in /app/.venv"
    exit 1
fi

# Verify GPU
python -c "import torch; props=torch.cuda.get_device_properties(0); mem=getattr(props,'total_memory',None) or getattr(props,'total_mem',0); print(f'GPU: {torch.cuda.get_device_name(0)} ({mem / 1024**3:.0f}GB)')"

# Set up HuggingFace auth (interactive)
if [ -z "${HF_TOKEN:-}" ]; then
    echo "Set HF_TOKEN for model downloads:"
    echo "  export HF_TOKEN=hf_..."
    echo "  huggingface-cli login --token \$HF_TOKEN"
fi

# Download models if not present
if [ ! -d "/app/data/checkpoints/DreamDojo" ]; then
    echo "Downloading model weights..."
    bash /app/scripts/download_models.sh /app/data/checkpoints
fi

# Run preflight
echo ""
echo "Running preflight checks..."
blueprint-validate --config /app/configs/example_validation.yaml preflight || true

echo ""
echo "=== Setup complete ==="
echo "To run the full pipeline:"
echo "  blueprint-validate --config /app/configs/example_validation.yaml --work-dir /app/data/outputs run-all"
echo ""
echo "Or run stages individually:"
echo "  blueprint-validate --config /app/configs/example_validation.yaml render --facility facility_a"

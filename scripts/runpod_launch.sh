#!/usr/bin/env bash
# Helper script for setting up BlueprintValidation on a RunPod pod.
# Run this after SSH-ing into the pod.
set -euo pipefail

echo "=== BlueprintValidation RunPod Setup ==="

# Activate venv
source /app/.venv/bin/activate

# Verify GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_mem / 1024**3:.0f}GB)')"

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
blueprint-validate preflight --config /app/configs/example_validation.yaml || true

echo ""
echo "=== Setup complete ==="
echo "To run the full pipeline:"
echo "  blueprint-validate run-all --config /app/configs/example_validation.yaml --work-dir /app/data/outputs"
echo ""
echo "Or run stages individually:"
echo "  blueprint-validate render --facility facility_a --config /app/configs/example_validation.yaml"

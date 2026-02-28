#!/usr/bin/env bash
# Download model weights for BlueprintValidation pipeline.
# Usage: bash scripts/download_models.sh [DATA_DIR]
set -euo pipefail

DATA_DIR="${1:-./data/checkpoints}"
mkdir -p "$DATA_DIR"

echo "=== Downloading model weights to $DATA_DIR ==="
echo ""

# Check for huggingface-cli
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface_hub CLI..."
    pip install -U "huggingface_hub[cli]"
fi

# Check for HF auth
if ! huggingface-cli whoami &> /dev/null; then
    if [ -n "${HF_TOKEN:-}" ]; then
        echo "Using HF_TOKEN for non-interactive HuggingFace login..."
        huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential >/dev/null
    fi
fi

if ! huggingface-cli whoami &> /dev/null; then
    echo "Please authenticate with HuggingFace:"
    echo "  huggingface-cli login"
    echo "or set:"
    echo "  export HF_TOKEN=hf_..."
    echo ""
    echo "You also need to accept the model licenses on HuggingFace for:"
    echo "  - nvidia/DreamDojo"
    echo "  - nvidia/Cosmos-Transfer2.5-2B"
    echo "  - openvla/openvla-7b (base weights for OpenVLA-OFT)"
    exit 1
fi

echo "[1/3] Downloading DreamDojo-2B pretrained..."
huggingface-cli download nvidia/DreamDojo \
    --local-dir "$DATA_DIR/DreamDojo/2B_pretrain/" \
    --resume-download \
    --include "2B_pretrain/*"

echo ""
echo "[2/3] Downloading Cosmos Transfer 2.5 (2B)..."
huggingface-cli download nvidia/Cosmos-Transfer2.5-2B \
    --local-dir "$DATA_DIR/cosmos-transfer-2.5-2b/" \
    --resume-download

echo ""
echo "[3/3] Downloading OpenVLA-OFT base 7B weights..."
huggingface-cli download openvla/openvla-7b \
    --local-dir "$DATA_DIR/openvla-7b/" \
    --resume-download

echo ""
echo "=== All models downloaded ==="
echo "DreamDojo:      $DATA_DIR/DreamDojo/2B_pretrain/"
echo "Cosmos Transfer: $DATA_DIR/cosmos-transfer-2.5-2b/"
echo "OpenVLA-OFT base: $DATA_DIR/openvla-7b/"
echo ""
echo "Total disk usage:"
du -sh "$DATA_DIR"

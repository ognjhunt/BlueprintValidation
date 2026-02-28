#!/usr/bin/env bash
# Download model weights for BlueprintValidation pipeline.
# Usage: bash scripts/download_models.sh [DATA_DIR]
set -euo pipefail

DATA_DIR="${1:-${CHECKPOINT_DIR:-./data/checkpoints}}"
mkdir -p "$DATA_DIR"

# Load repo-local env defaults when available.
if [ -z "${HF_TOKEN:-}" ] && [ -f "./scripts/runtime_env.local" ]; then
    # shellcheck disable=SC1091
    source "./scripts/runtime_env.local"
fi

echo "=== Downloading model weights to $DATA_DIR ==="
echo ""

# Resolve HF CLI command (new `hf` preferred; keep `huggingface-cli` fallback).
HF_CLI=""
if command -v hf &> /dev/null; then
    HF_CLI="hf"
elif command -v huggingface-cli &> /dev/null; then
    HF_CLI="huggingface-cli"
else
    echo "Installing huggingface_hub..."
    pip install -U huggingface_hub
    if command -v hf &> /dev/null; then
        HF_CLI="hf"
    elif command -v huggingface-cli &> /dev/null; then
        HF_CLI="huggingface-cli"
    fi
fi

if [ -z "$HF_CLI" ]; then
    echo "Could not find HF CLI after install. Ensure your Python bin is in PATH."
    exit 1
fi

# Check for HF auth
if [ "$HF_CLI" = "hf" ]; then
    WHOAMI_CMD=(hf auth whoami)
    LOGIN_CMD=(hf auth login)
    DOWNLOAD_CMD=(hf download)
else
    WHOAMI_CMD=(huggingface-cli whoami)
    LOGIN_CMD=(huggingface-cli login)
    DOWNLOAD_CMD=(huggingface-cli download)
fi

if ! "${WHOAMI_CMD[@]}" &> /dev/null; then
    if [ -n "${HF_TOKEN:-}" ]; then
        echo "Using HF_TOKEN for non-interactive HuggingFace login..."
        "${LOGIN_CMD[@]}" --token "${HF_TOKEN}" --add-to-git-credential >/dev/null
    fi
fi

if ! "${WHOAMI_CMD[@]}" &> /dev/null; then
    echo "Please authenticate with HuggingFace:"
    if [ "$HF_CLI" = "hf" ]; then
        echo "  hf auth login"
    else
        echo "  huggingface-cli login"
    fi
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
"${DOWNLOAD_CMD[@]}" nvidia/DreamDojo \
    --local-dir "$DATA_DIR/DreamDojo/2B_pretrain/" \
    --resume-download \
    --include "2B_pretrain/*"

echo ""
echo "[2/3] Downloading Cosmos Transfer 2.5 (2B)..."
"${DOWNLOAD_CMD[@]}" nvidia/Cosmos-Transfer2.5-2B \
    --local-dir "$DATA_DIR/cosmos-transfer-2.5-2b/" \
    --resume-download

echo ""
echo "[3/3] Downloading OpenVLA-OFT base 7B weights..."
"${DOWNLOAD_CMD[@]}" openvla/openvla-7b \
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

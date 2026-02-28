# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/opt/hf
ENV UV_CACHE_DIR=/opt/uv_cache

ARG VENDOR_STRATEGY=auto
ARG DREAMDOJO_REPO_URL=https://github.com/NVIDIA/DreamDojo.git
ARG DREAMDOJO_REF=7f3379bcb831147c0cc170e79ba08471ad186497
ARG COSMOS_REPO_URL=https://github.com/nvidia-cosmos/cosmos-transfer2.5.git
ARG COSMOS_REF=c9ad44b7283613618d57c1e4c9991916907d4f4b
ARG OPENVLA_REPO_URL=https://github.com/moojink/openvla-oft.git
ARG OPENVLA_REF=e4287e94541f459edc4feabc4e181f537cd569a8

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-venv python-is-python3 \
    ffmpeg git git-lfs curl wget openssh-server \
    build-essential cmake ninja-build \
    libgl1 libglib2.0-0 nano vim htop \
    && rm -rf /var/lib/apt/lists/*

# SSH for RunPod
RUN mkdir -p /var/run/sshd && \
    ssh-keygen -A && \
    sed -i 's/#PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    echo 'root:root' | chpasswd

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Install HuggingFace CLI for model downloads
RUN uv tool install -U "huggingface_hub[cli]"

WORKDIR /app

# Application code
COPY pyproject.toml uv.lock README.md ./
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/

# Install Python dependencies
RUN uv venv /app/.venv && \
    . /app/.venv/bin/activate && \
    uv sync --frozen --no-dev --extra rlds && \
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 && \
    uv pip install "sam2==1.1.0" && \
    uv pip install setuptools wheel psutil && \
    if [ "$(uname -m)" = "x86_64" ]; then uv pip install flash-attn --no-build-isolation; else echo "Skipping flash-attn on $(uname -m)"; fi && \
    rm -rf /opt/uv_cache /root/.cache/uv /root/.cache/pip
ENV PATH="/app/.venv/bin:$PATH"

# Vendor repo strategy:
#   - auto: use /data/vendor/<repo> from build context when present, else clone pinned ref.
#   - vendored: require build-context /data/vendor/<repo>.
#   - clone: always clone pinned ref.
RUN --mount=type=bind,source=.,target=/buildctx,readonly <<EOF
set -euo pipefail
mkdir -p /app/data/vendor /opt /models/checkpoints

resolve_vendor_or_clone() {
    local repo_name="\$1"
    local repo_url="\$2"
    local repo_ref="\$3"
    local context_path="/buildctx/data/vendor/\${repo_name}"
    local dest_path="/app/data/vendor/\${repo_name}"
    local use_vendored=0

    case "${VENDOR_STRATEGY}" in
        vendored)
            use_vendored=1
            ;;
        clone)
            use_vendored=0
            ;;
        auto)
            if [ -d "\${context_path}" ]; then
                use_vendored=1
            fi
            ;;
        *)
            echo "Unsupported VENDOR_STRATEGY='${VENDOR_STRATEGY}' (expected auto|vendored|clone)." >&2
            exit 1
            ;;
    esac

    if [ "\${use_vendored}" -eq 1 ]; then
        if [ ! -d "\${context_path}" ]; then
            echo "VENDOR_STRATEGY=vendored but missing build-context path: \${context_path}" >&2
            exit 1
        fi
        echo "Using vendored repo from build context: \${repo_name}"
        cp -a "\${context_path}" "\${dest_path}"
        rm -rf "\${dest_path}/.git"
        return 0
    fi

    echo "Cloning pinned repo: \${repo_name} @ \${repo_ref}"
    git clone "\${repo_url}" "\${dest_path}"
    git -C "\${dest_path}" checkout "\${repo_ref}"
    if [ -f "\${dest_path}/.gitmodules" ]; then
        git -C "\${dest_path}" submodule update --init --recursive
    fi
    rm -rf "\${dest_path}/.git"
}

resolve_vendor_or_clone "DreamDojo" "${DREAMDOJO_REPO_URL}" "${DREAMDOJO_REF}"
resolve_vendor_or_clone "cosmos-transfer" "${COSMOS_REPO_URL}" "${COSMOS_REF}"
resolve_vendor_or_clone "openvla-oft" "${OPENVLA_REPO_URL}" "${OPENVLA_REF}"

ln -sfn /app/data/vendor/DreamDojo /opt/DreamDojo
ln -sfn /app/data/vendor/cosmos-transfer /opt/cosmos-transfer
ln -sfn /app/data/vendor/openvla-oft /opt/openvla-oft
EOF

RUN if [ "$(uname -m)" = "x86_64" ]; then . /app/.venv/bin/activate && uv pip install -e /opt/DreamDojo; else echo "Skipping DreamDojo editable install on $(uname -m)"; fi

ENV DREAMDOJO_ROOT=/opt/DreamDojo
ENV COSMOS_ROOT=/opt/cosmos-transfer
ENV OPENVLA_ROOT=/opt/openvla-oft

EXPOSE 22

# Hint: mount outputs + checkpoints at runtime:
#   -v /host/data/outputs:/app/data/outputs
#   -v /host/models:/models
# Then on first SSH: blueprint-validate warmup

# RunPod entrypoint: SSH daemon + keep alive
ENTRYPOINT ["/bin/bash", "-c", "/usr/sbin/sshd && sleep infinity"]

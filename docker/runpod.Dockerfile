FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/opt/hf
ENV UV_CACHE_DIR=/opt/uv_cache

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
COPY data/vendor/ /app/data/vendor/

# Install Python dependencies
RUN uv venv /app/.venv && \
    . /app/.venv/bin/activate && \
    uv sync --frozen --no-dev --extra rlds && \
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 && \
    uv pip install setuptools wheel psutil && \
    if [ "$(uname -m)" = "x86_64" ]; then uv pip install flash-attn --no-build-isolation; else echo "Skipping flash-attn on $(uname -m)"; fi && \
    rm -rf /opt/uv_cache /root/.cache/uv /root/.cache/pip
ENV PATH="/app/.venv/bin:$PATH"

# Use vendored copies in this repo to avoid upstream HEAD drift at image build time.
RUN mkdir -p /opt /models/checkpoints && \
    ln -sfn /app/data/vendor/DreamDojo /opt/DreamDojo && \
    ln -sfn /app/data/vendor/cosmos-transfer /opt/cosmos-transfer && \
    ln -sfn /app/data/vendor/openvla-oft /opt/openvla-oft
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

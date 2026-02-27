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
ENV PATH="/root/.cargo/bin:$PATH"

# Install HuggingFace CLI for model downloads
RUN uv tool install -U "huggingface_hub[cli]"

WORKDIR /app

# Application code
COPY pyproject.toml README.md ./
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/

# Install Python dependencies
RUN uv venv /app/.venv && \
    . /app/.venv/bin/activate && \
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 && \
    uv pip install flash-attn --no-build-isolation && \
    uv pip install -e /app
ENV PATH="/app/.venv/bin:$PATH"

# Clone DreamDojo and Cosmos Transfer 2.5
RUN git clone --depth 1 https://github.com/NVIDIA/DreamDojo.git /opt/DreamDojo && \
    git clone --depth 1 https://github.com/nvidia-cosmos/cosmos-transfer2.5.git /opt/cosmos-transfer
RUN . /app/.venv/bin/activate && uv pip install -e /opt/DreamDojo
RUN git clone --depth 1 https://github.com/openvla/openvla.git /opt/openvla

ENV DREAMDOJO_ROOT=/opt/DreamDojo
ENV COSMOS_ROOT=/opt/cosmos-transfer
ENV OPENVLA_ROOT=/opt/openvla

EXPOSE 22

# RunPod entrypoint: SSH daemon + keep alive
ENTRYPOINT ["/bin/bash", "-c", "/usr/sbin/sshd && sleep infinity"]

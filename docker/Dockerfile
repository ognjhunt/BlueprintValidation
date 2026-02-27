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
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

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
    uv pip install -e /app
ENV PATH="/app/.venv/bin:$PATH"

# Clone and install DreamDojo
RUN git clone --depth 1 https://github.com/NVIDIA/DreamDojo.git /opt/DreamDojo
ENV DREAMDOJO_ROOT=/opt/DreamDojo
RUN . /app/.venv/bin/activate && uv pip install -e /opt/DreamDojo

# Clone and install Cosmos Transfer 2.5
RUN git clone --depth 1 https://github.com/nvidia-cosmos/cosmos-transfer2.5.git /opt/cosmos-transfer
ENV COSMOS_ROOT=/opt/cosmos-transfer

# Clone OpenVLA repo for fine-tuning entrypoints (vla-scripts/finetune.py).
RUN git clone --depth 1 https://github.com/openvla/openvla.git /opt/openvla
ENV OPENVLA_ROOT=/opt/openvla

ENTRYPOINT ["blueprint-validate"]
CMD ["--help"]

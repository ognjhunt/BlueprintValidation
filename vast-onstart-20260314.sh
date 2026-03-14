#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

apt-get update -o Acquire::Retries=5 --fix-missing
apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  ffmpeg \
  git \
  git-lfs \
  jq \
  libgl1 \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender1 \
  openssh-client \
  python3 \
  python3-pip \
  python3-venv \
  rsync \
  tmux \
  unzip \
  wget

mkdir -p /workspace /data /logs
git lfs install --skip-repo

cat >/etc/profile.d/blueprint_runtime.sh <<'EOF'
export HF_HOME=/workspace/.cache/huggingface
export PYTHONUNBUFFERED=1
EOF

mkdir -p /workspace/.cache/huggingface

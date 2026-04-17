# ─────────────────────────────────────────────────────────────────
# WhisperX Stage 2 — Forced Alignment Only
# RunPod Serverless Docker Image
#
# Base: NVIDIA CUDA 12.4.1 Ubuntu 22.04 Setup
# Models baked in: wav2vec2 English, silero-vad
# Dependencies baked in: Python 3.11, PyTorch 2.4.0 (cu124)
# ─────────────────────────────────────────────────────────────────

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# ── 1. System dependencies & Python 3.11 ─────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Map python to python3.11 safely
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# Ensure latest pip
RUN python -m pip install --upgrade pip

# ── 2. Explicit PyTorch (CUDA 12.4) ──────────────────────────────
# We install torch and torchaudio targeting cu124 explicitly
RUN pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# ── 3. App Requirements ──────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── 4. WhisperX Source ───────────────────────────────────────────
# Install the exact official repo of WhisperX
RUN pip install --no-cache-dir "git+https://github.com/m-bain/whisperX.git"

# ── 5. Copy source code ──────────────────────────────────────────
COPY download_models.py .
COPY handler.py .

# ── 6. Pre-bake models into image layer ──────────────────────────
# This runs during build — models are saved to HF & torch caches
ENV HF_HOME=/app/hf_cache
ENV TORCH_HOME=/app/torch_cache

RUN python download_models.py

# ── Entrypoint ────────────────────────────────────────────────────
CMD ["python", "-u", "handler.py"]


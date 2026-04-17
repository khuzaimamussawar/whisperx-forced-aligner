# ─────────────────────────────────────────────────────────────────
# WhisperX Stage 2 — Forced Alignment Only
# RunPod Serverless Docker Image
#
# Base: runpod/pytorch (CUDA 11.8, Python 3.10, PyTorch 2.1)
# Models baked in: wav2vec2 English, silero-vad
# ─────────────────────────────────────────────────────────────────

FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# ── System dependencies ───────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────
# Install WhisperX from GitHub (latest stable)
RUN pip install --no-cache-dir \
    "git+https://github.com/m-bain/whisperX.git" \
    runpod \
    num2words \
    requests \
    silero-vad

# ── Copy source ───────────────────────────────────────────────────
COPY download_models.py .
COPY handler.py .

# ── Pre-bake models into image layer ──────────────────────────────
# This runs during build — models are saved to HuggingFace cache
# Cold starts will be fast because models are already on disk
ENV HF_HOME=/app/hf_cache
ENV TORCH_HOME=/app/torch_cache

RUN python download_models.py

# ── Entrypoint ────────────────────────────────────────────────────
CMD ["python", "-u", "handler.py"]

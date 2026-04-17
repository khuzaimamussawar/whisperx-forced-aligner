"""
Pre-download all required models during Docker build.
This bakes models into the image so cold starts are faster.
Run: python download_models.py
"""

import torch
import whisperx
from silero_vad import load_silero_vad

print("=" * 60)
print("Pre-downloading models for WhisperX Stage 2 Forced Aligner")
print("=" * 60)

# ── 1. wav2vec2 English alignment model ──────────────────────
print("\n[1/2] Downloading wav2vec2 English alignment model...")
print("  Source: HuggingFace (jonatasgrosman/wav2vec2-large-960h-lv60-self)")
try:
    align_model, metadata = whisperx.load_align_model(
        language_code="en",
        device="cpu"   # CPU during build — CUDA not available in builder
    )
    print("  ✅ wav2vec2 English model downloaded.")
    del align_model, metadata   # free memory
except Exception as e:
    print(f"  ❌ Failed: {e}")
    raise

# ── 2. Silero VAD ────────────────────────────────────────────
print("\n[2/2] Downloading silero-vad model...")
try:
    vad_model = load_silero_vad()
    print("  ✅ silero-vad model downloaded.")
    del vad_model
except Exception as e:
    print(f"  ❌ Failed: {e}")
    raise

print("\n" + "=" * 60)
print("All models pre-baked successfully. Image build complete.")
print("=" * 60)

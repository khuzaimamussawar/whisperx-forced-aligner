"""
Pre-download all required models during Docker build.
This bakes models into the image so cold starts are faster.
Run: python download_models.py

WhisperX English alignment model uses torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
(NOT HuggingFace) — confirmed from whisperx/alignment.py DEFAULT_ALIGN_MODELS_TORCH.
"""

import os
import torch
import torchaudio
import whisperx
import nltk

print("=" * 60)
print("Pre-downloading models for WhisperX Stage 2 Forced Aligner")
print("=" * 60)

# ── 1. torchaudio WAV2VEC2_ASR_BASE_960H (English alignment) ─
# This is what whisperx.load_align_model("en") downloads internally.
# Confirmed in whisperx/alignment.py line 37:
#   DEFAULT_ALIGN_MODELS_TORCH = { "en": "WAV2VEC2_ASR_BASE_960H", ... }
print("\n[1/4] Downloading torchaudio WAV2VEC2_ASR_BASE_960H (English aligner)...")
try:
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model(dl_kwargs={"model_dir": os.environ.get("TORCH_HOME", "/app/torch_cache")})
    print("  ✅ WAV2VEC2_ASR_BASE_960H downloaded.")
    del model, bundle
except Exception as e:
    print(f"  ❌ Failed: {e}")
    raise

# ── 2. Load via whisperx to warm the full pipeline ───────────
print("\n[2/4] Warming whisperx.load_align_model('en') pipeline...")
try:
    align_model, metadata = whisperx.load_align_model(
        language_code="en",
        device="cpu",
        model_dir=os.environ.get("TORCH_HOME", "/app/torch_cache")
    )
    print("  ✅ whisperx align pipeline warmed.")
    del align_model, metadata
except Exception as e:
    print(f"  ❌ Failed: {e}")
    raise

# ── 3. NLTK punkt tokenizer (required by whisperx.align()) ───
print("\n[3/4] Downloading NLTK punkt_tab tokenizer...")
try:
    nltk.download("punkt_tab", quiet=False)
    print("  ✅ NLTK punkt_tab downloaded.")
except Exception as e:
    print(f"  ❌ Failed: {e}")
    raise


# ── 5. Whisper Transcription Model ────────────────────────────
print("\n[5/5] WARMING Whisper large-v3 ASR model...")
try:
    asr_model = whisperx.load_model("large-v3", device="cpu", compute_type="float32", download_root=os.environ.get("TORCH_HOME", "/app/torch_cache"))
    print("  ✅ Whisper large-v3 downloaded.")
    del asr_model
except Exception as e:
    print(f"  ❌ Failed: {e}")
    raise

print("\n" + "=" * 60)
print("All models pre-baked successfully. Image build complete.")
print("=" * 60)


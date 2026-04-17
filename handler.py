"""
WhisperX — Transcription + Forced Alignment
RunPod Serverless Handler

Input:
  audio_url   : string  — public URL to audio file (mp3/wav/m4a)
  segments    : array   — [{text, start, end}] from SceneBuilder script (IGNORED during transcription, retained for backwards compat)
  language    : string  — default "en"

Output:
  words       : array   — [{word, start, end}] — perfectly transcribed and word-level aligned outputs
  duration    : float   — total audio duration in seconds
  word_count  : int     — number of real words aligned
"""

import os
import tempfile
import requests
import runpod
import torch
import whisperx

# ─────────────────────────────────────────────
# GLOBAL MODEL LOAD (once, stays hot between calls)
# ─────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# `float16` is typically much faster and perfectly safe on RunPod GPUs.
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "float32"

print(f"[INIT] Using device: {DEVICE}")

print("[INIT] Loading Whisper large-v3 ASR model...")
_asr_model = whisperx.load_model(
    "large-v3", 
    device=DEVICE, 
    compute_type=COMPUTE_TYPE,
    download_root=os.environ.get("TORCH_HOME", "/app/torch_cache")
)
print("[INIT] Whisper ASR loaded.")

print("[INIT] Loading wav2vec2 English alignment model...")
_align_model_en, _align_metadata_en = whisperx.load_align_model(
    language_code="en", device=DEVICE
)
print("[INIT] wav2vec2 loaded.")

print("[INIT] All models ready. Handler starting.")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def download_audio(url: str) -> str:
    """Download audio from URL to a temp file. Returns local path."""
    ext = ".mp3"
    for suffix in [".wav", ".m4a", ".ogg", ".flac"]:
        if url.lower().endswith(suffix):
            ext = suffix
            break

    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    resp = requests.get(url, timeout=60, stream=True)
    resp.raise_for_status()
    for chunk in resp.iter_content(chunk_size=8192):
        tmp.write(chunk)
    tmp.flush()
    tmp.close()
    return tmp.name


def interpolate_missing_timestamps(words: list) -> list:
    """Fill in null start/end timestamps for unaligned words (e.g. numbers)."""
    for i, w in enumerate(words):
        if w["start"] is None or w["end"] is None:
            prev = next((x for x in reversed(words[:i]) if x.get("end") is not None), None)
            nxt  = next((x for x in words[i + 1:]    if x.get("start") is not None), None)
            w["start"] = prev["end"] if prev else (nxt["start"] if nxt else 0.0)
            w["end"]   = nxt["start"] if nxt else (prev["end"] if prev else 0.0)
    return words


# ─────────────────────────────────────────────
# MAIN HANDLER
# ─────────────────────────────────────────────

def handler(job):
    inp = job.get("input", {})

    audio_url    = inp.get("audio_url")
    language     = inp.get("language", "en")

    # ── Validate ──────────────────────────────
    if not audio_url:
        return {"error": "audio_url is required"}

    # ── 1. Download audio ────────────────────
    try:
        audio_path = download_audio(audio_url)
        audio = whisperx.load_audio(audio_path)
        audio_duration = round(len(audio) / 16000, 3)
        os.unlink(audio_path)
        print(f"[ALIGN] Audio loaded: {audio_duration}s")
    except Exception as e:
        return {"error": f"Failed to load audio: {e}"}

    # ── 2. Whisper Transcription (Standard Pipeline) ────
    try:
        print("[ALIGN] Running Whisper Transcription...")
        # batch_size=16 utilizes VRAM heavily for fast processing of long audio
        transcribe_result = _asr_model.transcribe(audio, batch_size=16, language=language)
        print(f"[ALIGN] Transcription done. Segments found: {len(transcribe_result.get('segments', []))}")
    except Exception as e:
        return {"error": f"Transcription failed: {e}"}

    # ── 3. Forced Alignment ───
    try:
        print("[ALIGN] Running Alignment...")
        if language == "en":
            align_model, align_meta = _align_model_en, _align_metadata_en
        else:
            # Load on-demand (slower first call for non-EN)
            align_model, align_meta = whisperx.load_align_model(
                language_code=language, device=DEVICE
            )

        result = whisperx.align(
            transcribe_result["segments"],
            align_model,
            align_meta,
            audio,
            DEVICE,
            return_char_alignments=False
        )
        print("[ALIGN] Alignment complete.")
    except Exception as e:
        return {"error": f"Alignment failed: {e}"}

    # ── 4. Flatten words + interpolate nulls ────────
    aligned_words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            aligned_words.append({
                "word":  w.get("word", ""),
                "start": w.get("start"),
                "end":   w.get("end")
            })

    aligned_words = interpolate_missing_timestamps(aligned_words)

    # ── 5. Build output ─
    output_words = []
    for word in aligned_words:
        w_start = word["start"] or 0.0
        w_end   = word["end"]   or 0.0

        output_words.append({
            "word":  word["word"],
            "start": round(w_start, 3),
            "end":   round(w_end, 3)
        })

    real_word_count = len(output_words)
    print(f"[DONE] Words: {real_word_count} aligned.")

    return {
        "words":      output_words,
        "duration":   audio_duration,
        "word_count": real_word_count
    }


runpod.serverless.start({"handler": handler})

"""
WhisperX — Transcription + Forced Alignment
RunPod Serverless Handler

Input:
  source_records : array - ordered timeline clips with source slices + keep_ranges
  audio_url      : string - legacy single-audio fallback
  language       : string - default "en"

Output:
  words       : array   — [{word, start, end}] — transcribed and word-level aligned
  duration    : float   — total audio duration in seconds
  word_count  : int     — number of words returned
  source_record_count : int
  chunk_seconds, interpolated_words, align_errors, voiced_holes — diagnostics

Long audio (minutes to 30+ min) is cut in pauses into ~CHUNK_SECONDS pieces; each piece is
transcribed and aligned on its own, so Whisper never drops text and WhisperX never squeezes
words into a wrong window. Any voiced stretch left without words is re-transcribed once with
extra context, and reported in `voiced_holes` if it is still empty.
"""

import os
import tempfile
from pathlib import Path
import requests
import runpod
import torch
import whisperx

from audio_pipeline import build_canonical_wav
from chunking import repair_holes, tidy, transcribe_range

# ─────────────────────────────────────────────
# GLOBAL MODEL LOAD (once, stays hot between calls)
# ─────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# `float16` is typically much faster and perfectly safe on RunPod GPUs.
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "float32"
ASR_BATCH_SIZE = int(os.environ.get("WHISPER_BATCH_SIZE", "8"))
CHUNK_SECONDS = float(os.environ.get("CHUNK_SECONDS", "20"))  # target piece length, cut in a pause
CHUNK_MAX_SECONDS = float(os.environ.get("CHUNK_MAX_SECONDS", "24"))  # a piece is never longer than this

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

def download_audio(url: str, destination: Path) -> Path:
    """Download the legacy single-audio input into the current job workspace."""
    with requests.get(url, timeout=(20, 180), stream=True) as response:
        response.raise_for_status()
        with destination.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)
    return destination


def interpolate_missing_timestamps(words: list) -> list:
    """Fill in null start/end timestamps for unaligned words (e.g. numbers)."""
    for i, w in enumerate(words):
        if w["start"] is None or w["end"] is None:
            prev = next((x for x in reversed(words[:i]) if x.get("end") is not None), None)
            nxt  = next((x for x in words[i + 1:]    if x.get("start") is not None), None)
            w["start"] = prev["end"] if prev else (nxt["start"] if nxt else 0.0)
            w["end"]   = nxt["start"] if nxt else (prev["end"] if prev else 0.0)
    return words


def spread_words(segments: list) -> list:
    """Fallback if alignment fails for a piece: spread each segment's words evenly over its window."""
    words = []
    for seg in segments:
        tokens = seg.get("text", "").split()
        step = (seg["end"] - seg["start"]) / max(1, len(tokens))
        for i, token in enumerate(tokens):
            words.append({"word": token, "start": seg["start"] + i * step, "end": seg["start"] + (i + 1) * step})
    return words


def make_transcribe_align(language: str, stats: dict):
    """Returns f(chunk_audio) -> [{word, start, end}] relative to the chunk, using the hot models."""
    if language == "en":
        align_model, align_meta = _align_model_en, _align_metadata_en
    else:
        # Load on-demand (slower first call for non-EN)
        align_model, align_meta = whisperx.load_align_model(language_code=language, device=DEVICE)

    def run(chunk):
        segments = _asr_model.transcribe(chunk, batch_size=ASR_BATCH_SIZE, language=language).get("segments", [])
        if not segments:
            return []
        try:
            aligned = whisperx.align(segments, align_model, align_meta, chunk, DEVICE, return_char_alignments=False)
            raw = [w for seg in aligned.get("segments", []) for w in seg.get("words", [])]
        except Exception as e:
            print(f"[ALIGN] piece alignment failed ({e}); spreading words over the segment windows")
            stats["align_errors"] += 1
            raw = spread_words(segments)
        words = [{"word": w.get("word", "").strip(), "start": w.get("start"), "end": w.get("end")} for w in raw]
        words = [w for w in words if w["word"]]
        stats["interpolated"] += sum(1 for w in words if w["start"] is None or w["end"] is None)
        return interpolate_missing_timestamps(words)

    return run


# ─────────────────────────────────────────────
# MAIN HANDLER
# ─────────────────────────────────────────────

def handler(job):
    inp = job.get("input", {})

    source_records = inp.get("source_records") or inp.get("sourceRecords")
    audio_url = inp.get("audio_url")
    language = str(inp.get("language", "en")).split("-", 1)[0].lower()

    if not source_records and not audio_url:
        return {"error": "source_records or audio_url is required"}

    # 1. Build one canonical mono 16 kHz WAV, then load it once for WhisperX.
    try:
        with tempfile.TemporaryDirectory(prefix="whisperx-job-") as job_dir:
            work_dir = Path(job_dir)
            if source_records:
                audio_path, audio_duration, source_record_count = build_canonical_wav(
                    source_records, work_dir
                )
            else:
                audio_path = download_audio(audio_url, work_dir / "legacy-audio")
                audio_duration = None
                source_record_count = 1

            audio = whisperx.load_audio(str(audio_path))
            loaded_duration = round(len(audio) / 16000, 3)
            if audio_duration is None:
                audio_duration = loaded_duration
            elif abs(audio_duration - loaded_duration) > 0.05:
                print(
                    f"[CANONICAL] Duration corrected from {audio_duration}s "
                    f"to decoded duration {loaded_duration}s"
                )
                audio_duration = loaded_duration
            print(
                f"[ALIGN] Canonical audio loaded: {audio_duration}s, "
                f"records={source_record_count}"
            )
    except Exception as e:
        return {"error": f"Failed to build canonical audio: {e}"}

    # ── 2. Chunked transcription + alignment, then hole repair ────
    stats = {"interpolated": 0, "align_errors": 0}
    split = {"target": CHUNK_SECONDS, "hi": CHUNK_MAX_SECONDS}
    try:
        run = make_transcribe_align(language, stats)
        print(f"[ALIGN] Transcribing in pieces of ~{CHUNK_SECONDS:.0f}s cut in pauses...")
        words = transcribe_range(audio, 0.0, audio_duration, run, **split)
        if not words:
            return {"error": "No speech was detected in the audio"}
        words, holes = repair_holes(audio, words, run, **split)
        words = tidy(words)
        if holes:
            print(f"[ALIGN] WARNING voiced stretches still without words: {holes}")
    except Exception as e:
        return {"error": f"Transcription/alignment failed: {e}"}

    # ── 3. Build output ─
    output_words = [
        {"word": w["word"], "start": round(w["start"], 3), "end": round(w["end"], 3)}
        for w in words
    ]
    print(f"[DONE] Words: {len(output_words)} aligned.")

    return {
        "words":      output_words,
        "duration":   audio_duration,
        "word_count": len(output_words),
        "source_record_count": source_record_count,
        "chunk_seconds": CHUNK_SECONDS,
        "interpolated_words": stats["interpolated"],
        "align_errors": stats["align_errors"],
        "voiced_holes": holes,
    }


runpod.serverless.start({"handler": handler})

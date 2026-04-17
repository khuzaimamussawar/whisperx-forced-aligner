"""
WhisperX Stage 2 — Forced Alignment Only
RunPod Serverless Handler

Input:
  audio_url   : string  — public URL to audio file (mp3/wav/m4a)
  segments    : array   — [{text, start, end}] from SceneBuilder script snippets
  language    : string  — default "en"
  vad_gap_min : float   — minimum gap (seconds) to check for unmatched speech (default 0.3)

Output:
  words       : array   — [{word, start, end}] — includes [UNMATCHED] tokens for VAD-detected speech gaps
  duration    : float   — total audio duration in seconds
  word_count  : int     — number of real (non-[UNMATCHED]) words aligned
"""

import os
import re
import tempfile
import requests
import runpod
import torch
import whisperx
from num2words import num2words as _num2words
from silero_vad import load_silero_vad, get_speech_timestamps

# ─────────────────────────────────────────────
# GLOBAL MODEL LOAD (once, stays hot between calls)
# ─────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INIT] Using device: {DEVICE}")

print("[INIT] Loading wav2vec2 English alignment model...")
_align_model_en, _align_metadata_en = whisperx.load_align_model(
    language_code="en", device=DEVICE
)
print("[INIT] wav2vec2 loaded.")

print("[INIT] Loading silero-vad...")
_vad_model = load_silero_vad()
print("[INIT] silero-vad loaded.")

print("[INIT] All models ready. Handler starting.")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def normalize_numbers(text: str) -> str:
    """Convert digit sequences to spoken English words for wav2vec2 alignment."""
    def replace_num(m):
        n = int(m.group())
        try:
            return _num2words(n, lang="en")
        except Exception:
            return m.group()
    # Replace standalone numbers (not part of larger tokens)
    return re.sub(r"\b\d+\b", replace_num, text)


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


def compute_proportional_times(segments: list, audio_duration: float) -> list:
    """
    Assign rough start/end estimates to segments proportional to text length.
    WhisperX uses these as search windows — proportional is better than equal slices.
    """
    total_chars = sum(len(s["text"]) for s in segments)
    if total_chars == 0:
        return segments

    cursor = 0.0
    for s in segments:
        ratio = len(s["text"]) / total_chars
        dur = ratio * audio_duration
        s["start"] = round(cursor, 3)
        s["end"]   = round(cursor + dur, 3)
        cursor += dur

    return segments


# ─────────────────────────────────────────────
# MAIN HANDLER
# ─────────────────────────────────────────────

def handler(job):
    inp = job.get("input", {})

    audio_url    = inp.get("audio_url")
    raw_segments = inp.get("segments", [])
    language     = inp.get("language", "en")
    vad_gap_min  = float(inp.get("vad_gap_min", 0.3))

    # ── Validate ──────────────────────────────
    if not audio_url:
        return {"error": "audio_url is required"}
    if not raw_segments:
        return {"error": "segments array is required and must not be empty"}

    # ── 1. Download audio ────────────────────
    try:
        audio_path = download_audio(audio_url)
        audio = whisperx.load_audio(audio_path)
        audio_duration = round(len(audio) / 16000, 3)
        os.unlink(audio_path)
        print(f"[ALIGN] Audio loaded: {audio_duration}s")
    except Exception as e:
        return {"error": f"Failed to load audio: {e}"}

    # ── 2. Normalize numbers + assign time windows ────
    segments = []
    for s in raw_segments:
        text = s.get("text", "").strip()
        if not text:
            continue
        segments.append({
            "text":  " " + normalize_numbers(text),   # leading space required by WhisperX
            "start": s.get("start", 0.0),
            "end":   s.get("end",   0.0)
        })

    if not segments:
        return {"error": "All segments were empty after filtering"}

    segments = compute_proportional_times(segments, audio_duration)

    # ── 3. Forced alignment (Stage 2 — NO Whisper) ───
    try:
        # English model pre-loaded; extend here for other languages if needed
        if language == "en":
            align_model, align_meta = _align_model_en, _align_metadata_en
        else:
            # Load on-demand (slower first call for non-EN)
            align_model, align_meta = whisperx.load_align_model(
                language_code=language, device=DEVICE
            )

        result = whisperx.align(
            segments,
            align_model,
            align_meta,
            audio,
            DEVICE,
            return_char_alignments=False
        )
        print(f"[ALIGN] Alignment complete. Segments: {len(result.get('segments', []))}")
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

    # ── 5. VAD — detect all speech regions ──────────
    audio_tensor = torch.FloatTensor(audio)
    speech_ts = get_speech_timestamps(
        audio_tensor,
        _vad_model,
        sampling_rate=16000,
        threshold=0.5,
        return_seconds=True
    )
    print(f"[VAD] Speech segments detected: {len(speech_ts)}")

    def has_speech_in_range(t_start: float, t_end: float) -> bool:
        return any(
            s["end"] > t_start and s["start"] < t_end
            for s in speech_ts
        )

    # ── 6. Build output — insert [UNMATCHED] tokens ─
    output_words = []
    prev_end = 0.0

    for word in aligned_words:
        w_start = word["start"] or prev_end
        w_end   = word["end"]   or prev_end
        gap     = w_start - prev_end

        # Gap with speech → [UNMATCHED] token
        if gap > vad_gap_min and has_speech_in_range(prev_end, w_start):
            output_words.append({
                "word":  "[UNMATCHED]",
                "start": round(prev_end, 3),
                "end":   round(w_start, 3)
            })

        output_words.append({
            "word":  word["word"],
            "start": round(w_start, 3),
            "end":   round(w_end, 3)
        })
        prev_end = w_end

    # Trailing gap after last aligned word
    trailing = audio_duration - prev_end
    if trailing > vad_gap_min and has_speech_in_range(prev_end, audio_duration):
        output_words.append({
            "word":  "[UNMATCHED]",
            "start": round(prev_end, 3),
            "end":   round(audio_duration, 3)
        })

    real_word_count = sum(1 for w in output_words if w["word"] != "[UNMATCHED]")
    print(f"[DONE] Words: {real_word_count} aligned, "
          f"{len(output_words) - real_word_count} [UNMATCHED] tokens")

    return {
        "words":      output_words,
        "duration":   audio_duration,
        "word_count": real_word_count
    }


runpod.serverless.start({"handler": handler})

"""
Silence-aware chunking for long audio.

Why: Whisper large-v3 in batched mode loses text, and WhisperX squeezes words into
Whisper's own segment windows, when it is fed minutes of continuous speech at once.
Fed ~20 s pieces cut in pauses, it is accurate. This module has no GPU or WhisperX
imports, so it is unit-tested on its own (tests/test_chunking.py).
"""

import numpy as np

SAMPLE_RATE = 16_000
HOP = 400  # 25 ms analysis frames


def frame_db(audio):
    """Loudness (dB) of every 25 ms frame."""
    n = len(audio) // HOP
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    frames = np.asarray(audio[: n * HOP], dtype=np.float32).reshape(n, HOP)
    return 20 * np.log10(np.sqrt((frames * frames).mean(axis=1)) + 1e-9)


def silence_mask(db):
    """Frames far below the speech level (relative threshold, so it works for quiet and loud files)."""
    if len(db) == 0:
        return np.zeros(0, dtype=bool)
    return db < max(np.percentile(db, 90) - 30.0, -70.0)


def runs(mask, min_frames=1, merge_gap=0):
    """(start, end) frame ranges where mask is True; runs closer than merge_gap frames are joined."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        if out and i - out[-1][1] <= merge_gap:
            out[-1] = (out[-1][0], j)
        else:
            out.append((i, j))
        i = j
    return [(a, b) for a, b in out if b - a >= min_frames]


def split_points(audio, target=20.0, lo=6.0, hi=24.0):
    """Cut times in seconds, each in the middle of a pause; the quietest frame if a stretch has no pause."""
    db = frame_db(audio)
    dur = len(audio) / SAMPLE_RATE
    mids = [(a + b) / 2 * HOP / SAMPLE_RATE for a, b in runs(silence_mask(db), min_frames=6)]  # pauses >= 150 ms
    cuts = [0.0]
    while dur - cuts[-1] > hi:
        a, b = cuts[-1] + lo, cuts[-1] + hi
        near = [m for m in mids if a <= m <= b]
        if near:
            cut = min(near, key=lambda m: abs(m - (cuts[-1] + target)))
        else:
            i, j = int(a * SAMPLE_RATE / HOP), min(int(b * SAMPLE_RATE / HOP), len(db))
            cut = (i + int(np.argmin(db[i:j]))) * HOP / SAMPLE_RATE
        cuts.append(cut)
    cuts.append(dur)
    return cuts


def transcribe_range(audio, t0, t1, transcribe_align, **split_kw):
    """Words for audio[t0:t1] (seconds). transcribe_align(chunk) -> [{word,start,end}] relative to the chunk."""
    piece = audio[int(t0 * SAMPLE_RATE): int(t1 * SAMPLE_RATE)]
    cuts = split_points(piece, **split_kw)
    words = []
    for a, b in zip(cuts, cuts[1:]):
        chunk = piece[int(a * SAMPLE_RATE): int(b * SAMPLE_RATE)]
        for w in transcribe_align(chunk):
            words.append({**w, "start": round(w["start"] + t0 + a, 3), "end": round(w["end"] + t0 + a, 3)})
    return words


def voiced_holes(audio, words, min_len=1.5):
    """Voiced stretches of at least min_len seconds that have no word on them (a silent loss, not a pause)."""
    db = frame_db(audio)
    voiced = ~silence_mask(db)
    covered = np.zeros(len(db), dtype=bool)
    for w in words:
        a = max(0, int((w["start"] - 0.3) * SAMPLE_RATE / HOP))
        b = min(len(db), int((w["end"] + 0.3) * SAMPLE_RATE / HOP) + 1)
        covered[a:b] = True
    bad = voiced & ~covered
    need = int(min_len * SAMPLE_RATE / HOP)
    return [(round(a * HOP / SAMPLE_RATE, 2), round(b * HOP / SAMPLE_RATE, 2))
            for a, b in runs(bad, min_frames=need, merge_gap=int(0.4 * SAMPLE_RATE / HOP))]


def snap_to_pause(audio, t, win=3.0):
    """Middle of the pause nearest to t (within win seconds), else t: a cut there never splits a word."""
    mids = [(a + b) / 2 * HOP / SAMPLE_RATE for a, b in runs(silence_mask(frame_db(audio)), min_frames=6)]
    near = [m for m in mids if abs(m - t) <= win]
    return min(near, key=lambda m: abs(m - t)) if near else t


def repair_holes(audio, words, transcribe_align, passes=2, margin=6.0, **split_kw):
    """Re-transcribe each hole with extra context, replacing every word in that region. Returns (words, holes_left)."""
    dur = len(audio) / SAMPLE_RATE
    for _ in range(passes):
        holes = voiced_holes(audio, words)
        if not holes:
            break
        for h0, h1 in holes:
            t0 = snap_to_pause(audio, max(0.0, h0 - margin))
            t1 = snap_to_pause(audio, min(dur, h1 + margin))
            new = transcribe_range(audio, t0, t1, transcribe_align, **split_kw)
            words = [w for w in words if not (t0 <= w["start"] < t1)] + new
            words.sort(key=lambda w: w["start"])
    return words, voiced_holes(audio, words)


def tidy(words):
    """Monotonic, non-overlapping times."""
    words = sorted(words, key=lambda w: w["start"])
    for a, b in zip(words, words[1:]):
        if b["start"] < a["end"]:
            a["end"] = max(a["start"], b["start"])
    return words

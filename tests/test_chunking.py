import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chunking import SAMPLE_RATE as SR, frame_db, repair_holes, silence_mask, split_points, tidy, transcribe_range, voiced_holes

rng = np.random.default_rng(7)


def speechlike(bursts, pause=0.4):
    """Noise bursts (seconds) separated by digital silence; returns (audio, [(start, end)] voiced spans)."""
    parts, spans, t = [], [], 0.0
    for d in bursts:
        parts.append((rng.standard_normal(int(d * SR)) * 0.1).astype(np.float32))
        spans.append((t, t + d))
        t += d
        parts.append(np.zeros(int(pause * SR), dtype=np.float32))
        t += pause
    return np.concatenate(parts), spans


def fake_transcribe_align(chunk, step=0.5):
    """One word per `step` s on every voiced stretch of the chunk: what a perfect aligner would return."""
    voiced = ~silence_mask(frame_db(chunk))
    words, i, n = [], 0, len(voiced)
    while i < n:
        if voiced[i]:
            j = i
            while j < n and voiced[j]:
                j += 1
            a, b = i * 400 / SR, j * 400 / SR
            t = a
            while t + step <= b + 1e-6:
                words.append({"word": "w", "start": t, "end": t + step * 0.9})
                t += step
            i = j
        else:
            i += 1
    return words


class Chunking(unittest.TestCase):
    def test_cuts_land_in_pauses_and_pieces_stay_short(self):
        audio, _ = speechlike([4, 6, 5, 7, 3, 8, 6, 5, 4, 9, 6, 7, 5, 4, 6])
        cuts = split_points(audio)
        self.assertTrue(all(b - a <= 24.01 for a, b in zip(cuts, cuts[1:])))
        silent = silence_mask(frame_db(audio))
        for c in cuts[1:-1]:
            self.assertTrue(silent[int(c * SR / 400)], f"cut at {c:.2f}s is inside speech")

    def test_continuous_speech_is_still_cut(self):
        audio = (rng.standard_normal(70 * SR) * 0.1).astype(np.float32)
        cuts = split_points(audio)
        self.assertTrue(all(b - a <= 24.01 for a, b in zip(cuts, cuts[1:])))
        self.assertGreaterEqual(len(cuts), 4)

    def test_offsets_and_order_over_a_long_file(self):
        audio, spans = speechlike([5, 6, 7, 4, 8, 6, 5, 7, 6, 4, 8, 5, 6, 7, 5, 6, 4, 8, 7, 5])  # ~2 min
        words = tidy(transcribe_range(audio, 0.0, len(audio) / SR, fake_transcribe_align))
        self.assertAlmostEqual(len(words), sum(b - a for a, b in spans) / 0.5, delta=len(spans) * 2)
        starts = [w["start"] for w in words]
        self.assertEqual(starts, sorted(starts))
        for w in words:  # every word sits on a voiced span
            self.assertTrue(any(a - 0.1 <= w["start"] and w["end"] <= b + 0.1 for a, b in spans), w)
        self.assertEqual(voiced_holes(audio, words), [])

    def test_a_silent_hole_is_found_and_repaired(self):
        audio, spans = speechlike([6, 7, 8, 9, 7, 6, 8, 5, 7])
        full = tidy(transcribe_range(audio, 0.0, len(audio) / SR, fake_transcribe_align))
        a, b = spans[3]
        lost = [w for w in full if a - 0.2 <= w["start"] <= b]
        broken = [w for w in full if w not in lost]
        self.assertEqual(len(voiced_holes(audio, broken)), 1)
        fixed, left = repair_holes(audio, broken, fake_transcribe_align)
        self.assertEqual(left, [])
        self.assertEqual(len(fixed), len(full))

    def test_squeezed_words_are_replaced_not_duplicated(self):
        audio, spans = speechlike([6, 22, 6])  # a long stretch, like the chapter 5/6 one
        full = tidy(transcribe_range(audio, 0.0, len(audio) / SR, fake_transcribe_align))
        a, b = spans[1]
        inside = [w for w in full if a <= w["start"] < b]
        squeezed = [{"word": "w", "start": a + i * (6.0 / len(inside)), "end": a + (i + 0.8) * (6.0 / len(inside))}
                    for i in range(len(inside))]  # all words crammed into the first 6 s
        broken = [w for w in full if not (a <= w["start"] < b)] + squeezed
        fixed, left = repair_holes(audio, broken, fake_transcribe_align)
        self.assertEqual(left, [])
        self.assertEqual(len(fixed), len(full))


if __name__ == "__main__":
    unittest.main()

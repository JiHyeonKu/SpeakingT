"""Synthetic signal checks; they do not validate pronunciation assessment."""

import json
from pathlib import Path
import tempfile
import unittest
import wave

import numpy as np

from speakingt.audio import analyze_audio


class AudioTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def wav(self, samples, rate=16000, width=2, filename="sample.wav"):
        samples = np.asarray(samples)
        channels = samples.shape[1] if samples.ndim == 2 else 1
        clipped = np.clip(samples.reshape(-1), -1, 1 - 1 / (2 ** (8 * width - 1)))
        integer = np.rint(clipped * (2 ** (8 * width - 1))).astype(np.int64)
        if width == 1:
            raw = (integer + 128).astype(np.uint8).tobytes()
        elif width == 2:
            raw = integer.astype("<i2").tobytes()
        elif width == 3:
            raw = np.stack([integer & 255, (integer >> 8) & 255, (integer >> 16) & 255], axis=1).astype(np.uint8).tobytes()
        else:
            raw = integer.astype("<i4").tobytes()
        path = self.root / filename
        with wave.open(str(path), "wb") as output:
            output.setnchannels(channels)
            output.setsampwidth(width)
            output.setframerate(rate)
            output.writeframes(raw)
        return path

    def tone(self, frequency=200, seconds=1.2, rate=16000):
        return 0.25 * np.sin(2 * np.pi * frequency * np.arange(round(seconds * rate)) / rate)

    def test_known_sine_pitch_at_multiple_rates(self):
        for rate in (8000, 16000, 44100, 48000):
            with self.subTest(rate=rate):
                result = analyze_audio(self.wav(self.tone(193, rate=rate), rate=rate))
                self.assertAlmostEqual(result["pitch"]["median_hz"], 193, delta=1)
                self.assertTrue(result["pitch"]["reliable"])
                self.assertGreater(result["pitch"]["voiced_fraction"], 0.95)
                self.assertAlmostEqual(result["signal"]["rms"], 0.25 / np.sqrt(2), delta=0.001)
                self.assertLess(abs(np.median([row["semitones_from_median"] for row in result["pitch"]["contour"]])), 0.01)
                json.dumps(result, allow_nan=False)

    def test_harmonics_recover_fundamental(self):
        time = np.arange(19200) / 16000
        samples = 0.12 * np.sin(2 * np.pi * 127 * time) + 0.25 * np.sin(2 * np.pi * 254 * time) + 0.08 * np.sin(2 * np.pi * 381 * time)
        result = analyze_audio(self.wav(samples))
        self.assertAlmostEqual(result["pitch"]["median_hz"], 127, delta=1)

    def test_silence_has_no_pitch_activity_or_internal_pauses(self):
        result = analyze_audio(self.wav(np.zeros(32000)))
        self.assertEqual(result["pitch"]["contour"], [])
        self.assertIsNone(result["pitch"]["median_hz"])
        self.assertFalse(result["pitch"]["reliable"])
        self.assertEqual(result["activity"]["active_fraction"], 0)
        self.assertEqual(result["activity"]["internal_pauses"], [])
        self.assertEqual(result["signal"]["rms_dbfs"], -120)
        json.dumps(result, allow_nan=False)

    def test_white_noise_does_not_pass_periodicity_check(self):
        samples = np.random.default_rng(42).normal(0, 0.1, 32000)
        result = analyze_audio(self.wav(samples))
        self.assertFalse(result["pitch"]["reliable"])
        self.assertLess(result["pitch"]["voiced_fraction"], 0.05)

    def test_only_internal_long_pause_counted(self):
        samples = np.r_[np.zeros(8000), self.tone(seconds=0.8), np.zeros(6400), self.tone(seconds=0.8), np.zeros(11200)]
        result = analyze_audio(self.wav(samples))
        pauses = result["activity"]["internal_pauses"]
        self.assertEqual(len(pauses), 1)
        self.assertAlmostEqual(pauses[0]["start_s"], 1.3, delta=0.025)
        self.assertAlmostEqual(pauses[0]["end_s"], 1.7, delta=0.025)
        self.assertAlmostEqual(pauses[0]["duration_s"], 0.4, delta=0.025)
        self.assertAlmostEqual(result["activity"]["speech_span_s"], 2.0, delta=0.025)
        self.assertAlmostEqual(result["activity"]["internal_pause_ratio"], 0.2, delta=0.025)

    def test_short_gap_is_not_a_hesitation(self):
        result = analyze_audio(self.wav(np.r_[self.tone(seconds=0.6), np.zeros(1280), self.tone(seconds=0.6)]))
        self.assertEqual(result["activity"]["internal_pauses"], [])

    def test_stereo_is_downmixed_and_input_clipping_is_preserved(self):
        left = self.tone()
        right = -left
        result = analyze_audio(self.wav(np.column_stack((left, right))))
        self.assertEqual(result["channels"], 2)
        self.assertLess(result["signal"]["rms"], 0.0001)
        self.assertTrue(result["warnings"])
        clipped = np.column_stack((np.ones(16000), -np.ones(16000)))
        result = analyze_audio(self.wav(clipped))
        self.assertEqual(result["signal"]["clipping_fraction"], 1.0)
        self.assertFalse(result["pitch"]["reliable"])

    def test_pcm_widths(self):
        for width in (1, 2, 3, 4):
            with self.subTest(width=width):
                result = analyze_audio(self.wav(self.tone(220), width=width))
                self.assertAlmostEqual(result["pitch"]["median_hz"], 220, delta=1)
                self.assertAlmostEqual(result["signal"]["rms"], 0.25 / np.sqrt(2), delta=0.005)

    def test_short_audio_never_claims_reliable_pitch(self):
        result = analyze_audio(self.wav(self.tone(seconds=0.2)))
        self.assertFalse(result["pitch"]["reliable"])
        with self.assertRaisesRegex(ValueError, "duration"):
            analyze_audio(self.wav(self.tone(seconds=0.02)))

    def test_invalid_format_rate_truncation_and_missing_file(self):
        bad = self.root / "bad.wav"
        bad.write_bytes(b"this is not a WAV")
        with self.assertRaises(ValueError):
            analyze_audio(bad)
        with self.assertRaisesRegex(ValueError, "sample rate"):
            analyze_audio(self.wav(np.zeros(4000), rate=4000))
        valid = self.wav(self.tone())
        raw = valid.read_bytes()
        valid.write_bytes(raw[:-100])
        with self.assertRaisesRegex(ValueError, "truncated"):
            analyze_audio(valid)
        with self.assertRaises(FileNotFoundError):
            analyze_audio(self.root / "missing.wav")


if __name__ == "__main__":
    unittest.main()

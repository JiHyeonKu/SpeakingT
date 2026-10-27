"""CLI/report integration checks using synthetic fixtures, never human labels."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import wave

import numpy as np

from speakingt.cli import clip_wav, main
from speakingt.report import build_report, feature_vector, render_html, save_report


class ReportIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.rate = 16000

    def tearDown(self):
        self.directory.cleanup()

    def wav(self, name="fixture.wav", seconds=1.5, silent=False):
        time = np.arange(round(seconds * self.rate)) / self.rate
        samples = np.zeros_like(time) if silent else 0.2 * np.sin(2 * np.pi * 173 * time)
        path = self.root / name
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(self.rate)
            output.writeframes((samples * 32767).astype("<i2").tobytes())
        return path

    def call_cli(self, arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = main([str(argument) for argument in arguments])
        return status, stdout.getvalue(), stderr.getvalue()

    def test_module_cli_demo_produces_explicitly_synthetic_portable_report(self):
        destination = self.root / "demo"
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        result = subprocess.run(
            [sys.executable, "-m", "speakingt", "demo", "--output-dir", str(destination)],
            cwd=Path(__file__).resolve().parents[1], env=environment,
            capture_output=True, text=True, encoding="utf-8", timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads((destination / "synthetic_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["stage"], "synthetic_pipeline_demo_not_speech")
        self.assertEqual(report["pronunciation_assessment"]["status"], "not_assessed")
        self.assertEqual(len(report["audio"]["activity"]["internal_pauses"]), 1)
        self.assertAlmostEqual(report["audio"]["activity"]["internal_pauses"][0]["duration_s"], 0.5, delta=0.04)
        self.assertTrue(report["audio"]["pitch"]["reliable"])
        json.dumps(report, allow_nan=False)
        html = (destination / "synthetic_report.html").read_text(encoding="utf-8")
        self.assertIn("합성 신호", html)
        self.assertIn("<svg", html)
        self.assertNotIn("<script", html)
        self.assertNotIn("https://", html)

    def test_silent_report_is_valid_but_training_features_are_refused(self):
        report = build_report(self.wav(silent=True))
        self.assertEqual(report["pronunciation_assessment"]["status"], "not_assessed")
        with self.assertRaises(ValueError):
            feature_vector(report["audio"])
        output_json, output_html = save_report(report, self.root / "silent-report")
        self.assertTrue(output_json.is_file())
        self.assertTrue(output_html.is_file())
        json.dumps(json.loads(output_json.read_text(encoding="utf-8")), allow_nan=False)

    def test_reference_requires_common_text_and_comparison_remains_descriptive(self):
        learner = self.wav()
        reference = self.wav("reference.wav")
        for common_text in (None, "", "   "):
            with self.subTest(common_text=common_text):
                with self.assertRaises(ValueError):
                    build_report(learner, reference_audio=reference, reference_text=common_text)
        report = build_report(learner, reference_audio=reference, reference_text="This is a fixture.")
        comparison = report["reference_comparison"]
        self.assertEqual(comparison["status"], "descriptive_only")
        self.assertEqual(comparison["same_words"], "user_asserted_not_verified")
        self.assertEqual(comparison["alignment"], "none")
        self.assertTrue(all(abs(value) < 1e-9 for value in comparison["feature_differences_learner_minus_reference"].values()))
        silent = self.wav("silent.wav", silent=True)
        report = build_report(learner, reference_audio=silent, reference_text="This is a fixture.")
        self.assertEqual(report["reference_comparison"]["status"], "insufficient_periodic_audio")

    def test_untrusted_strings_are_escaped_in_all_html_sections(self):
        payload = '<script>alert("unsafe")</script><img src=x onerror=alert(1)>'
        report = build_report(self.wav(), transcript="um 어 " + payload, reference_text=payload)
        report["recording"]["name"] = payload
        report["feedback"].append({"kind": "test_fixture", "message": payload})
        report["limitations"].append(payload)
        rendered = render_html(report)
        self.assertNotIn("<script", rendered)
        self.assertNotIn("<img", rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertIn("&quot;unsafe&quot;", rendered)
        self.assertIn("filler_candidate", rendered)
        self.assertIn("korean_text_observed", rendered)

    def test_analyze_keeps_reference_text_separate_from_observed_transcript(self):
        audio = self.wav()
        text_path = self.root / "observed.txt"
        text_path.write_text("um I I 어 agree", encoding="utf-8-sig")
        output = self.root / "analysis.json"
        original_hash = hashlib.sha256(audio.read_bytes()).hexdigest()
        status, _, errors = self.call_cli([
            "analyze", audio, "--transcript-file", text_path,
            "--reference-text", "I agree.", "--output", output,
        ])
        self.assertEqual(status, 0, errors)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["reference_text"], "I agree.")
        self.assertEqual(report["transcript"]["text"], "um I I 어 agree")
        self.assertEqual(report["transcript"]["source"], "user_supplied")
        self.assertEqual(report["recording"]["sha256"], original_hash)
        self.assertEqual(hashlib.sha256(audio.read_bytes()).hexdigest(), original_hash)

    def test_clip_exact_pcm_region_and_original_preservation(self):
        source = self.wav(seconds=2)
        before = source.read_bytes()
        target = self.root / "clips" / "excerpt.wav"
        clip_wav(str(source), str(target), 0.25, 1.25)
        with wave.open(str(source), "rb") as original, wave.open(str(target), "rb") as excerpt:
            self.assertEqual(excerpt.getnframes(), self.rate)
            self.assertEqual(excerpt.getframerate(), original.getframerate())
            original.setpos(self.rate // 4)
            self.assertEqual(excerpt.readframes(self.rate), original.readframes(self.rate))
        self.assertEqual(source.read_bytes(), before)
        with self.assertRaises(ValueError):
            clip_wav(str(source), str(target), 0, 1)
        with self.assertRaises(ValueError):
            clip_wav(str(source), str(source), 0, 1)
        self.assertEqual(source.read_bytes(), before)

    def test_clip_rejects_out_of_range_and_nonfinite_boundaries(self):
        source = self.wav()
        target = self.root / "invalid-clip.wav"
        for start, end in ((-0.1, 1), (0, 2), (1, 0), (0.4, 0.4), (float("nan"), 1), (0, float("inf"))):
            with self.subTest(start=start, end=end):
                with self.assertRaises(ValueError):
                    clip_wav(str(source), str(target), start, end)
                self.assertFalse(target.exists())

    def test_clip_rejects_interval_with_no_whole_pcm_sample(self):
        source = self.wav()
        target = self.root / "zero-frame.wav"
        with self.assertRaises(ValueError):
            clip_wav(str(source), str(target), 0, 1e-9)
        self.assertFalse(target.exists())

    def test_prepare_label_refuses_synthetic_demo(self):
        destination = self.root / "demo"
        status, _, errors = self.call_cli(["demo", "--output-dir", destination])
        self.assertEqual(status, 0, errors)
        output = self.root / "labels.jsonl"
        status, _, errors = self.call_cli([
            "prepare-label", destination / "synthetic_report.json",
            "--speaker-id", "fixture", "--recording-id", "fixture-recording",
            "--split", "train", "--output", output,
        ])
        self.assertEqual(status, 2)
        self.assertIn("합성", errors)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

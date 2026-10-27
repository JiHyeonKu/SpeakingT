"""Audio import safety checks using a small mocked streaming decoder."""

from fractions import Fraction
import hashlib
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import wave

from speakingt.import_audio import import_audio


def frame(samples=3200, *, value=123, pts=0, rate=16000, channels=1, width=2):
    return SimpleNamespace(
        samples=samples, sample_rate=rate, pts=pts, time_base=Fraction(1, rate),
        layout=SimpleNamespace(channels=tuple(range(channels))),
        format=SimpleNamespace(bytes=width, name="s16"), is_corrupt=False,
        planes=[struct.pack("<h", value) * samples + b"padding"],
    )


class AudioImportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.source = self.root / "input.m4a"
        self.source.write_bytes(b"mock encoded audio")
        self.output = self.root / "output.wav"

    def tearDown(self):
        self.directory.cleanup()

    def decoder(self, frames=None, flush=None, audio=True):
        decoded = [frame()] if frames is None else frames
        stream = SimpleNamespace(codec_context=SimpleNamespace(name="aac"), index=0)
        container = Mock()
        container.streams = SimpleNamespace(audio=[stream] if audio else [])
        container.decode.return_value = iter(decoded)
        context = Mock()
        context.__enter__ = Mock(return_value=container)
        context.__exit__ = Mock(return_value=False)
        resampler = Mock()
        resampler.resample.side_effect = lambda incoming: (flush or []) if incoming is None else [incoming]
        av = SimpleNamespace(open=Mock(return_value=context), AudioResampler=Mock(return_value=resampler), __version__="mock")
        return av, container, resampler

    def run_import(self, av):
        with patch.dict("sys.modules", {"av": av}):
            return import_audio(self.source, self.output)

    def assert_no_output(self):
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".speakingt-import-*")), [])

    def test_pcm_output_provenance_and_original_preserved(self):
        original = self.source.read_bytes()
        av, _, _ = self.decoder()
        result = self.run_import(av)
        with wave.open(str(self.output), "rb") as reader:
            self.assertEqual((reader.getnchannels(), reader.getsampwidth(), reader.getframerate()), (1, 2, 16000))
            self.assertEqual(reader.getnframes(), 3200)
            self.assertEqual(reader.readframes(3200), struct.pack("<h", 123) * 3200)
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual(result["source"]["sha256"], hashlib.sha256(original).hexdigest())
        self.assertEqual(result["output"]["sha256"], hashlib.sha256(self.output.read_bytes()).hexdigest())
        self.assertEqual(result["source"]["codec"], "aac")
        self.assertFalse(result["processing"]["normalized"])
        self.assertFalse(result["processing"]["trimmed"])
        self.assertFalse(result["processing"]["vad_applied"])
        av.AudioResampler.assert_called_once_with(format="s16", layout="mono", rate=16000)
        json.dumps(result, allow_nan=False)

    def test_resampler_flush_and_boundary_silence_are_preserved(self):
        av, _, resampler = self.decoder([frame(value=0), frame(1600, value=222, pts=3200)], flush=[frame(1600, value=0, pts=4800)])
        result = self.run_import(av)
        self.assertEqual(result["output"]["samples"], 6400)
        self.assertIsNone(resampler.resample.call_args_list[-1].args[0])
        with wave.open(str(self.output), "rb") as reader:
            raw = reader.readframes(6400)
        self.assertEqual(raw[:6400], b"\x00" * 6400)
        self.assertEqual(raw[-3200:], b"\x00" * 3200)

    def test_source_format_and_resampling_are_recorded(self):
        original = frame(9600, rate=48000, channels=2, width=4)
        original.format.name = "fltp"
        av, _, resampler = self.decoder([original])
        resampler.resample.side_effect = lambda incoming: [] if incoming is None else [frame()]
        result = self.run_import(av)
        self.assertEqual(result["source"]["sample_rate_hz"], 48000)
        self.assertEqual(result["source"]["channels"], 2)
        self.assertEqual(result["source"]["sample_format"], "fltp")
        self.assertTrue(result["processing"]["resampled"])
        self.assertTrue(result["processing"]["downmixed"])
        self.assertAlmostEqual(result["processing"]["decoded_duration_s"], 0.2)
        self.assertEqual(result["processing"]["decoded_bytes"], 9600 * 2 * 4)
        self.assertEqual(result["output"]["samples"], 3200)

    def test_container_offset_rebased_and_internal_gap_retained(self):
        av, _, _ = self.decoder([frame(1600, pts=16000), frame(1600, pts=20800)])
        result = self.run_import(av)
        self.assertEqual(result["processing"]["timeline_origin_s"], 1.0)
        self.assertEqual(result["processing"]["timestamp_gap_samples"], 3200)
        self.assertEqual(result["output"]["samples"], 6400)
        with wave.open(str(self.output), "rb") as reader:
            raw = reader.readframes(6400)
        self.assertEqual(raw[3200:9600], b"\x00" * 6400)

    def test_overlapping_timestamps_are_rejected_without_sample_removal(self):
        av, _, _ = self.decoder([frame(), frame(pts=1600)])
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.run_import(av)
        self.assert_no_output()

    def test_missing_timestamps_remain_contiguous(self):
        av, _, _ = self.decoder([frame(pts=None), frame(pts=None)])
        result = self.run_import(av)
        self.assertEqual(result["output"]["samples"], 6400)
        self.assertIsNone(result["processing"]["timeline_origin_s"])

    def test_existing_output_and_same_path_are_never_overwritten(self):
        with self.assertRaisesRegex(ValueError, "overwrite"):
            import_audio(self.source, self.source)
        self.output.write_bytes(b"keep me")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            import_audio(self.source, self.output)
        self.assertEqual(self.output.read_bytes(), b"keep me")

    def test_missing_optional_dependency_has_install_hint(self):
        with patch.dict("sys.modules", {"av": None}):
            with self.assertRaisesRegex(ValueError, r"pip install.*audio"):
                import_audio(self.source, self.output)
        self.assert_no_output()

    def test_no_audio_stream_empty_audio_and_too_short_input(self):
        for frames, audio in [([], False), ([], True), ([frame(100)], True)]:
            with self.subTest(audio=audio, frames=len(frames)):
                av, _, _ = self.decoder(frames, audio=audio)
                with self.assertRaises(ValueError):
                    self.run_import(av)
                self.assert_no_output()

    def test_decode_error_after_partial_data_removes_output(self):
        def corrupt():
            yield frame()
            raise RuntimeError("corrupt packet")
        av, _, _ = self.decoder(corrupt())
        with self.assertRaisesRegex(ValueError, "corrupt"):
            self.run_import(av)
        self.assert_no_output()

    def test_flagged_corrupt_frame_is_rejected(self):
        corrupt = frame()
        corrupt.is_corrupt = True
        av, _, _ = self.decoder([corrupt])
        with self.assertRaisesRegex(ValueError, "corrupt"):
            self.run_import(av)
        self.assert_no_output()

    def test_duration_byte_and_timestamp_gap_limits(self):
        for constant, limit, decoded in [
            ("MAX_DURATION_S", 0.15, [frame()]),
            ("MAX_DECODED_BYTES", 100, [frame()]),
            ("MAX_DURATION_S", 0.3, [frame(1600), frame(1600, pts=6400)]),
        ]:
            with self.subTest(constant=constant, limit=limit):
                av, _, _ = self.decoder(decoded)
                with patch("speakingt.import_audio." + constant, limit):
                    with self.assertRaises(ValueError):
                        self.run_import(av)
                self.assert_no_output()

    def test_source_change_is_detected(self):
        def changed():
            yield frame()
            self.source.write_bytes(b"changed original")
        av, _, _ = self.decoder(changed())
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.run_import(av)
        self.assert_no_output()

    def test_publish_failure_removes_partial_final_file(self):
        def interrupted_copy(reader, writer, length):
            writer.write(b"partial")
            raise OSError("disk full")
        av, _, _ = self.decoder()
        with patch("speakingt.import_audio.shutil.copyfileobj", side_effect=interrupted_copy):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.run_import(av)
        self.assert_no_output()


if __name__ == "__main__":
    unittest.main()

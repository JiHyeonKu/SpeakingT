"""Offline preparation checks: synthetic fixtures test plumbing, not validity."""

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np

from scripts.prepare_speechocean import (DatasetError, _download, _fetch, _safe_path,
                                       _table, prepare_dataset, select_recordings)


REVISION = "a" * 40


def fixture_tables():
    tables = {"train": {"wav": {}, "speaker": {}}, "test": {"wav": {}, "speaker": {}}}
    for official, speakers in (("train", range(1, 7)), ("test", range(7, 10))):
        for speaker in speakers:
            speaker_id = f"{speaker:04d}"
            for number in range(1, 4):
                utterance = f"{speaker:05d}{number:04d}"
                tables[official]["wav"][utterance] = f"WAVE/SPEAKER{speaker_id}/{utterance}.WAV"
                tables[official]["speaker"][utterance] = speaker_id
    return tables


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.output = Path(self.directory.name)
        self.tables = fixture_tables()
        self.options = dict(seed=7, train_speakers=3, validation_speakers=2,
                            test_speakers=2, utterances_per_speaker=2)

    def tearDown(self):
        self.directory.cleanup()

    def test_deterministic_selection_and_official_split_preservation(self):
        rows = select_recordings(self.tables, **self.options)
        reordered = {key: {table: dict(reversed(list(values.items()))) for table, values in value.items()}
                     for key, value in self.tables.items()}
        self.assertEqual(rows, select_recordings(reordered, **self.options))
        self.assertEqual(len(rows), 14)
        groups = {split: {row["speaker_id"] for row in rows if row["split"] == split}
                  for split in ("train", "validation", "test")}
        self.assertFalse(groups["train"] & groups["validation"])
        self.assertFalse((groups["train"] | groups["validation"]) & groups["test"])
        for row in rows:
            self.assertEqual(row["official_split"], "test" if row["split"] == "test" else "train")
        self.assertNotEqual(rows, select_recordings(self.tables, **{**self.options, "seed": 8}))

    def test_overlap_and_insufficient_speakers_fail(self):
        with self.assertRaises(DatasetError):
            select_recordings(self.tables, **{**self.options, "train_speakers": 10})
        self.tables["test"] = self.tables["train"]
        with self.assertRaisesRegex(DatasetError, "overlap"):
            select_recordings(self.tables, **self.options)

    def test_metadata_rejects_paths_commands_and_duplicates(self):
        for value in ("000010001 ../../secret", "000010001 command |", "000010001 WAVE/SPEAKER0001/000010002.WAV",
                      "000010001 WAVE/SPEAKER0001/000010001.WAV\n000010001 WAVE/SPEAKER0001/000010001.WAV"):
            with self.subTest(value=value), self.assertRaises(DatasetError):
                _table(value, paths=True)
        for value in ("../secret", "C:/secret", "/secret", "a\\b"):
            with self.subTest(value=value), self.assertRaises(DatasetError):
                _safe_path(self.output, value)

    def test_download_restricted_to_official_host_before_network(self):
        with patch("scripts.prepare_speechocean.urllib.request.build_opener") as opener:
            for url in ("http://raw.githubusercontent.com/jimbozhang/speechocean762/a", "https://evil.example/a",
                        "https://raw.githubusercontent.com/other/repo/a"):
                with self.assertRaises(DatasetError):
                    _fetch(url, 10)
            opener.assert_not_called()

    def test_cache_checksum_detects_corruption_without_redownload(self):
        with patch("scripts.prepare_speechocean._fetch", return_value=b"abc") as fetch:
            path, metadata = _download(self.output, REVISION, "train/wav.scp", 100)
            self.assertEqual(metadata["bytes"], 3)
            _download(self.output, REVISION, "train/wav.scp", 100)
            fetch.assert_called_once()
            path.write_bytes(b"tampered")
            with self.assertRaisesRegex(DatasetError, "checksum"):
                _download(self.output, REVISION, "train/wav.scp", 100)
            fetch.assert_called_once()

    def test_real_pipeline_contract_with_offline_audio_fixtures(self):
        selected = select_recordings(self.tables, **self.options)
        dropped_id = selected[0]["recording_id"]
        scores = {utterance: {"fluency": 7} for table in self.tables.values() for utterance in table["wav"]}
        downloads = []
        def fake_fetch(url, limit):
            relative = url.split(f"/{REVISION}/", 1)[1]
            downloads.append(relative)
            if relative == "resource/scores.json":
                self.assertTrue((self.output / "selection.json").exists())
                return json.dumps(scores).encode()
            if relative.startswith(("train/", "test/")):
                official, name = relative.split("/")
                table = self.tables[official]["wav" if name == "wav.scp" else "speaker"]
                return "\n".join(f"{key} {value}" for key, value in table.items()).encode()
            utterance = Path(relative).stem
            data = np.zeros(32000) if utterance == dropped_id else 0.2 * np.sin(2 * np.pi * 190 * np.arange(32000) / 16000)
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as target:
                target.setnchannels(1)
                target.setsampwidth(2)
                target.setframerate(16000)
                target.writeframes((data * 32767).astype("<i2").tobytes())
            return buffer.getvalue()
        with patch("scripts.prepare_speechocean._fetch", side_effect=fake_fetch), patch("builtins.print"):
            provenance = prepare_dataset(self.output, REVISION, **self.options)
        manifest = [json.loads(line) for line in (self.output / "manifest.jsonl").read_text().splitlines()]
        excluded = json.loads((self.output / "exclusions.json").read_text())
        self.assertEqual(len(manifest), 13)
        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded[0]["reason"], "unreliable_pitch")
        self.assertEqual(provenance["selected"]["train"]["utterances"], 6)
        self.assertEqual(provenance["retained"]["train"]["utterances"], 5)
        self.assertEqual(provenance["license"], "CC BY 4.0")
        self.assertTrue(all(row["target"] == 70 and row["label_source"] == "human" for row in manifest))
        self.assertEqual({path for path in downloads if path.startswith("WAVE/")}, {row["source_path"] for row in selected})
        self.assertTrue(all(len(row["audio_sha256"]) == 64 for row in manifest))
        self.assertTrue(all(len(source["sha256"]) == 64 for source in provenance["sources"]))

    def test_revision_required_before_network(self):
        with patch("scripts.prepare_speechocean._fetch") as fetch:
            with self.assertRaises(DatasetError):
                prepare_dataset(self.output, "main", **self.options)
            fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()

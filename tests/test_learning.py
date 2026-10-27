"""Synthetic numerical fixtures test software only, not real speech validity."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from speakingt.learning import predict, train_model


class LearningTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "human_ratings.jsonl"
        self.model = self.root / "model.json"

    def rows(self):
        rows = []
        for split, values in (
            ("train", np.linspace(-2, 2, 30)),
            ("validation", np.linspace(-1.8, 1.8, 12)),
            ("test", np.linspace(-1.7, 1.7, 11)),
        ):
            for index, value in enumerate(values):
                rows.append({
                    "split": split,
                    "speaker_id": f"{split}-speaker-{index // 2}",
                    "recording_id": f"{split}-recording-{index}",
                    "features": {"synthetic_x": float(value), "constant": 4.0},
                    "target": float(50 + 15 * value),
                    "label_source": "human",
                })
        return rows

    def write(self, rows):
        self.manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    def train(self, rows=None, **kwargs):
        self.write(self.rows() if rows is None else rows)
        return train_model(self.manifest, self.model, **kwargs)

    def test_fits_synthetic_linear_relation_and_reports_baseline(self):
        result = self.train(max_epochs=300, patience=25, seed=4)
        self.assertEqual(result["feature_names"], ["constant", "synthetic_x"])
        self.assertLess(result["metrics"]["test"]["mae"], 1e-5)
        self.assertLess(result["metrics"]["test"]["mae"], result["metrics"]["test"]["baseline_mae"])
        self.assertEqual(result["data"]["source_sha256"], hashlib.sha256(self.manifest.read_bytes()).hexdigest())
        self.assertEqual(json.loads(self.model.read_text(encoding="utf-8")), result)
        estimate = predict(self.model, {"synthetic_x": 1.0, "constant": 4})
        self.assertAlmostEqual(estimate["score"], 65.0, places=4)

    def test_reproducible_seed(self):
        first = self.train(max_epochs=50, patience=10, seed=7)
        second = train_model(self.manifest, self.root / "second.json", max_epochs=50, patience=10, seed=7)
        self.assertEqual(first, second)

    def test_uses_training_statistics_only_and_does_not_select_on_test(self):
        rows = self.rows()
        first = self.train(rows, max_epochs=50, patience=10)
        changed = copy.deepcopy(rows)
        for row in changed:
            if row["split"] == "test":
                row["features"]["synthetic_x"] += 10000
                row["target"] = 0
        second = self.train(changed, max_epochs=50, patience=10)
        self.assertEqual(first["normalization"], second["normalization"])
        self.assertEqual(first["parameters"], second["parameters"])
        self.assertEqual(first["training"], second["training"])
        self.assertEqual(first["metrics"]["validation"], second["metrics"]["validation"])
        self.assertNotEqual(first["metrics"]["test"], second["metrics"]["test"])
        expected = np.array([row["features"]["synthetic_x"] for row in rows if row["split"] == "train"])
        self.assertAlmostEqual(first["normalization"]["mean"][1], float(expected.mean()))
        self.assertAlmostEqual(first["normalization"]["scale"][1], float(expected.std()))
        self.assertEqual(first["normalization"]["scale"][0], 1.0)

    def test_early_stopping_restores_baseline_when_validation_gets_worse(self):
        rows = self.rows()
        for row in rows:
            if row["split"] == "validation":
                row["target"] = 100 - row["target"]
        result = self.train(rows, max_epochs=100, patience=4)
        self.assertEqual(result["training"]["best_epoch"], 0)
        self.assertEqual(result["training"]["epochs_run"], 4)
        self.assertEqual(result["parameters"]["weights"], [0.0, 0.0])
        self.assertEqual(result["metrics"]["validation"]["mae"], result["metrics"]["validation"]["baseline_mae"])

    def test_constant_training_target_is_supported(self):
        rows = self.rows()
        for row in rows:
            if row["split"] == "train":
                row["target"] = 65.0
        result = self.train(rows)
        self.assertTrue(result["training"]["constant_training_target"])
        self.assertEqual(result["training"]["epochs_run"], 0)
        self.assertEqual(predict(self.model, {"synthetic_x": 1000.0, "constant": 4})["score"], 65.0)
        self.assertTrue(any("baseline" in warning for warning in result["warnings"]))

    def test_restores_best_nonzero_epoch(self):
        rows = self.rows()
        for row in rows:
            if row["split"] == "validation":
                row["target"] = 50 + 12 * row["features"]["synthetic_x"]
        single_epoch = self.train(rows, max_epochs=1, patience=4, seed=3)
        result = self.train(rows, max_epochs=100, patience=4, seed=3)
        self.assertEqual(result["training"]["best_epoch"], 1)
        self.assertEqual(result["training"]["epochs_run"], 5)
        self.assertEqual(result["parameters"], single_epoch["parameters"])
        self.assertEqual(result["metrics"], single_epoch["metrics"])

    def test_warns_when_heldout_predictions_are_worse_than_mean(self):
        rows = self.rows()
        for row in rows:
            if row["split"] == "test":
                row["target"] = 100 - row["target"]
        result = self.train(rows)
        self.assertGreater(result["metrics"]["test"]["mae"], result["metrics"]["test"]["baseline_mae"])
        self.assertTrue(any("baseline" in warning for warning in result["warnings"]))
        estimate = predict(self.model, {"synthetic_x": 1, "constant": 4})
        self.assertEqual(estimate["warnings"], result["warnings"])

    def test_rejects_speaker_leakage(self):
        rows = self.rows()
        rows[-1]["speaker_id"] = rows[0]["speaker_id"]
        with self.assertRaisesRegex(ValueError, "speaker leakage"):
            self.train(rows)

    def test_rejects_duplicate_recording_across_splits(self):
        rows = self.rows()
        rows[-1]["recording_id"] = rows[0]["recording_id"]
        with self.assertRaisesRegex(ValueError, "duplicate recording"):
            self.train(rows)

    def test_rejects_duplicate_recording_within_split(self):
        rows = self.rows()
        rows[1]["recording_id"] = rows[0]["recording_id"]
        with self.assertRaisesRegex(ValueError, "duplicate recording"):
            self.train(rows)

    def test_rejects_cloned_audio_with_different_recording_ids(self):
        for other_index in (1, -1):
            rows = self.rows()
            rows[0]["audio_sha256"] = "a" * 64
            rows[other_index]["audio_sha256"] = "A" * 64
            with self.subTest(index=other_index), self.assertRaisesRegex(ValueError, "duplicate audio_sha256"):
                self.train(rows)

    def test_checks_optional_audio_hashes_and_reports_coverage(self):
        rows = self.rows()
        rows[0]["audio_sha256"] = "a" * 64
        result = self.train(rows)
        self.assertEqual(result["data"]["splits"]["train"]["audio_hashes_checked"], 1)
        self.assertEqual(result["data"]["splits"]["test"]["audio_hashes_checked"], 0)
        for bad in (None, True, 1234, "bad", "g" * 64):
            rows[0]["audio_sha256"] = bad
            with self.subTest(audio_hash=bad), self.assertRaisesRegex(ValueError, "audio_sha256"):
                self.train(rows)

    def test_rejects_empty_or_missing_split(self):
        for rows in ([], [row for row in self.rows() if row["split"] != "test"]):
            with self.subTest(size=len(rows)), self.assertRaises(ValueError):
                self.train(rows)

    def test_requires_human_label_provenance(self):
        rows = self.rows()
        rows[0].pop("label_source")
        with self.assertRaisesRegex(ValueError, "label_source"):
            self.train(rows)

    def test_rejects_invalid_labels(self):
        for bad in (-1, 101, None, True, "50", 10**400, float("inf"), float("nan")):
            rows = self.rows()
            rows[0]["target"] = bad
            with self.subTest(target=bad), self.assertRaises(ValueError):
                self.train(rows)

    def test_rejects_missing_additional_and_nonfinite_features(self):
        for bad in (
            {"constant": 4},
            {"constant": 4, "synthetic_x": 1, "extra": 0},
            {"constant": 4, "synthetic_x": None},
            {"constant": 4, "synthetic_x": True},
            {"constant": 4, "synthetic_x": float("nan")},
            {"constant": 4, "synthetic_x": float("inf")},
            {"constant": 4, "synthetic_x": 10**400},
        ):
            rows = self.rows()
            rows[-1]["features"] = bad
            with self.subTest(features=bad), self.assertRaises(ValueError):
                self.train(rows)

    def test_numerical_overflow_rejects_input_without_writing_artifact(self):
        rows = self.rows()
        rows[0]["features"]["synthetic_x"] = 1e300
        with self.assertRaisesRegex(ValueError, "statistics overflowed"):
            self.train(rows)
        self.assertFalse(self.model.exists())

    def test_prediction_overflow_is_a_clear_value_error(self):
        result = self.train()
        result["parameters"]["weights"][1] = 1e300
        self.model.write_text(json.dumps(result), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Prediction overflowed"):
            predict(self.model, {"constant": 4, "synthetic_x": 1e300})

    def test_infers_features_from_training_even_if_validation_is_first(self):
        rows = self.rows()
        rows = [row for row in rows if row["split"] == "validation"] + [row for row in rows if row["split"] != "validation"]
        result = self.train(rows)
        self.assertEqual(result["feature_names"], ["constant", "synthetic_x"])

    def test_prediction_is_clipped_and_rejects_missing_features(self):
        self.train()
        self.assertEqual(predict(self.model, {"constant": 4, "synthetic_x": 100})["score"], 100.0)
        self.assertEqual(predict(self.model, {"constant": 4, "synthetic_x": -100})["score"], 0.0)
        with self.assertRaises(ValueError):
            predict(self.model, {"synthetic_x": 1})

    def test_rejects_malformed_artifact(self):
        result = self.train()
        result["normalization"]["scale"][0] = 0
        self.model.write_text(json.dumps(result), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "scales must be positive"):
            predict(self.model, {"constant": 4, "synthetic_x": 1})

    def test_rejects_invalid_training_configuration(self):
        self.write(self.rows())
        for settings in ({"max_epochs": 0}, {"max_epochs": True}, {"patience": 0}, {"seed": -1}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                train_model(self.manifest, self.model, **settings)
        with self.assertRaisesRegex(ValueError, "output_path"):
            train_model(self.manifest, self.manifest)


if __name__ == "__main__":
    unittest.main()

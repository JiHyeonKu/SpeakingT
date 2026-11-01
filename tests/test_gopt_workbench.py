"""Small synthetic regression fixtures; these are NOT performance evidence."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

HAS_RESEARCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("tensorboard") is not None
if HAS_RESEARCH:
    import torch
    from speakingt import gopt_workbench as g


@unittest.skipUnless(HAS_RESEARCH, "optional research dependencies not installed")
class GoptWorkbenchTests(unittest.TestCase):
    def test_padding_never_contributes_to_loss(self):
        p = torch.tensor([[1., 900.]])
        u = torch.zeros((1, 5))
        w = torch.tensor([[[1., 1., 1.], [800., 800., 800.]]])
        labels = torch.tensor([[[1., 1., 1., 0.], [-1., -1., -1., -1.]]])
        total, _ = g.losses((p, u, w), torch.tensor([[1., -1.]]), u, labels)
        self.assertEqual(float(total), 0.)

    def test_full_length_final_word_is_not_lost(self):
        predictions = np.arange(150).reshape(1, 50, 3)
        labels = np.zeros((1, 50, 4))
        labels[0, -2:, 3] = 1
        p, _ = g.aggregate_words(predictions, labels)
        self.assertEqual(p.shape, (2, 3))
        np.testing.assert_allclose(p[-1], predictions[0, -2:].mean(0))

    def test_constant_correlation_is_missing_not_nan(self):
        self.assertIsNone(g.metric([1, 1], [2, 3])["pcc"])

    def test_checkpoint_train_split_and_weight_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(4)
            n = 12
            x = np.zeros((n, 50, 84), dtype=np.float32)
            x[:, :3] = rng.normal(size=(n, 3, 84))
            phn = np.full((n, 50, 2), -1., dtype=np.float32)
            phn[:, :3, 0] = [0, 1, 2]
            phn[:, :3, 1] = rng.uniform(0, 2, (n, 3))
            word = np.full((n, 50, 4), -1., dtype=np.float32)
            word[:, :3, :3] = 7
            word[:, :3, 3] = 0
            data = dict(feat=x, label_phn=phn, label_word=word,
                        label_utt=rng.uniform(4, 10, (n, 5)).astype(np.float32),
                        ids=np.array([str(i) for i in range(n)]),
                        speakers=np.array([f"s{i//2}" for i in range(n)]))
            np.savez(root/"train.npz", **data)
            digest = hashlib.sha256((root/"train.npz").read_bytes()).hexdigest()
            (root/"manifest.json").write_text(json.dumps({"datasets": {"train": {"sha256": digest}}}))
            config = g.TrainConfig(epochs=2, validation_speakers=2, batch_size=4,
                                   embed_dim=6, depth=1, threads=1)
            run = root/"run"
            history = g.train(root, run, config)
            self.assertGreater(history[-1]["weight_change_l2"], 0)
            split = json.loads((run/"split.json").read_text())
            self.assertFalse(set(split["train_speakers"]) & set(split["validation_speakers"]))
            with (run/"metrics.csv").open(encoding="utf-8") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 3)
            self.assertFalse(json.loads((run/"summary.json").read_text())["test_evaluated"])
            # No test.npz exists: training must never try to read it.
            a, norm = g.load_checkpoint(run/"best.pt")
            b, _ = g.load_checkpoint(run/"best.pt")
            a.eval(); b.eval()
            batch = next(iter(torch.utils.data.DataLoader(g.tensors(data, norm), batch_size=2)))
            torch.testing.assert_close(g.outputs(a, batch[0], batch[1])[0], g.outputs(b, batch[0], batch[1])[0])
            with self.assertRaises(ValueError):
                g.train(root, run, config)

    def test_changed_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"train.npz").write_bytes(b"tampered")
            (root/"manifest.json").write_text(json.dumps({"datasets": {"train": {"sha256": "0"*64}}}))
            with self.assertRaisesRegex(ValueError, "changed"):
                g.load_data(root)

    def test_invalid_training_configuration(self):
        for cfg in (g.TrainConfig(epochs=0), g.TrainConfig(learning_rate=float("nan")),
                    g.TrainConfig(embed_dim=7, heads=2)):
            with self.assertRaises(ValueError):
                cfg.validate()


class GoptDataAuditTests(unittest.TestCase):
    def test_official_uppercase_wav_path_accepts_cached_example(self):
        from scripts.prepare_gopt import sample_audio
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"assets").mkdir(); (root/"examples").mkdir()
            (root/"assets/train_wav.scp").write_text("000010011 WAVE/SPEAKER0001/000010011.WAV\n")
            (root/"assets/test_wav.scp").write_text("")
            audio = root/"examples/000010011.wav"
            audio.write_bytes(b"RIFF")
            self.assertEqual(sample_audio("000010011", root), audio)

    def test_wrong_canonical_phone_and_human_score_are_rejected(self):
        from scripts.prepare_gopt import audit_partition
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"tr_keys_utt.csv").write_text("u1\n")
            feat = np.zeros((1, 50, 84)); feat[:, 0] = 1
            phn = np.full((1, 50, 2), -1.); phn[0, 0] = [0, 0]
            word = np.full((1, 50, 4), -1.); word[0, 0] = [1, 2, 3, 0]
            arrays = dict(feat=feat, label_phn=phn, label_word=word, label_utt=np.ones((1, 5)))
            for key, value in arrays.items(): np.save(root/f"tr_{key}.npy", value)
            scores = {"u1": dict(accuracy=1, completeness=1, fluency=1, prosodic=1, total=1,
                      words=[dict(phones=["AH0"], **{"phones-accuracy": [0]}, accuracy=1, stress=2, total=3)])}
            audit_partition(root, "tr", scores, {"AH": 0}, {"u1": "s1"})
            with self.assertRaisesRegex(ValueError, "Phone identity"):
                audit_partition(root, "tr", scores, {"AH": 1}, {"u1": "s1"})
            scores["u1"]["accuracy"] = 5
            with self.assertRaisesRegex(ValueError, "Human label"):
                audit_partition(root, "tr", scores, {"AH": 0}, {"u1": "s1"})

    def test_download_cache_rejects_tampering_and_path_escape(self):
        from scripts.prepare_gopt import fetch_asset
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/"file.npy").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Cached asset changed"):
                fetch_asset({"file": "file.npy", "bytes": 7, "sha256": "0"*64}, root)
            with self.assertRaisesRegex(ValueError, "filename"):
                fetch_asset({"file": "../outside.npy"}, root)


if __name__ == "__main__":
    unittest.main()

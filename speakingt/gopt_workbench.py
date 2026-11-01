"""Inspectable training/evaluation around the unchanged upstream GOPT model.

This module accepts prepared GOP features, NOT WAVs or ASR confidence scores.
The original LibriSpeech acoustic model is frozen outside this training loop.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .vendor.gopt import GOPT

UTT_NAMES = ("accuracy", "completeness", "fluency", "prosody", "total")
WORD_NAMES = ("accuracy", "stress", "total")
PRETRAINED_SHA256 = "ab07451e51648f9d2455505a51055b20ac4ad7921d771ccc5170ff486a826259"


def _sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def load_data(directory="data/gopt", split="train"):
    if split not in ("train", "test"):
        raise ValueError("split must be train or test")
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    path = directory / f"{split}.npz"
    if _sha(path) != manifest["datasets"][split]["sha256"]:
        raise ValueError("Prepared dataset changed; rerun the label audit")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def subset(data, indices):
    return {key: value[indices] for key, value in data.items()}


def split_training(data, seed=7, validation_speakers=25):
    """Deterministic speaker split fixed without consulting any score."""
    speakers = sorted(set(data["speakers"].tolist()), key=lambda s: hashlib.sha256(f"{seed}:{s}".encode()).digest())
    if not 0 < validation_speakers < len(speakers):
        raise ValueError("Need nonempty, disjoint train and validation speakers")
    held = set(speakers[:validation_speakers])
    val = np.array([s in held for s in data["speakers"]])
    return subset(data, ~val), subset(data, val)


def training_normalization(data):
    valid = data["label_phn"][:, :, 1] >= 0
    values = data["feat"][valid].astype(np.float64)
    mean, std = float(values.mean()), float(values.std())
    if not np.isfinite([mean, std]).all() or std <= 0:
        raise ValueError("Degenerate training features")
    return {"mean": mean, "std": std, "source": "training speakers only; valid phones only"}


def tensors(data, normalization):
    phn = data["label_phn"]
    valid = phn[:, :, 1] >= 0
    if (not np.array_equal(valid, phn[:, :, 0] >= 0)
            or np.any(phn[:, :, 0] < -1) or np.any(phn[:, :, 0] > 38)):
        raise ValueError("GOPT requires its fixed 39-phone vocabulary and -1 padding")
    x = np.zeros_like(data["feat"], dtype=np.float32)
    x[valid] = (data["feat"][valid] - normalization["mean"]) / normalization["std"]
    word = data["label_word"].copy().astype(np.float32)
    word[:, :, :3] /= 5
    return TensorDataset(torch.from_numpy(x), torch.tensor(phn[:, :, 0], dtype=torch.long),
                         torch.tensor(phn[:, :, 1], dtype=torch.float32),
                         torch.tensor(data["label_utt"] / 5, dtype=torch.float32), torch.from_numpy(word))


def outputs(model, x, phones):
    raw = model(x, phones)
    return raw[5].squeeze(-1), torch.cat(raw[:5], dim=1), torch.cat(raw[6:], dim=2)


def losses(predictions, phone, utterance, word):
    p, u, w = predictions
    pmask = phone >= 0
    wmask = word[:, :, :3] >= 0
    if not pmask.any() or not wmask.any():
        raise ValueError("A batch must contain human-scored phones and words")
    lp = torch.mean((p[pmask] - phone[pmask]) ** 2)
    lu = torch.mean((u - utterance) ** 2)
    lw = torch.mean((w[wmask] - word[:, :, :3][wmask]) ** 2)
    return lp + lu + lw, {"phone": lp, "utterance": lu, "word": lw}


def metric(prediction, target):
    prediction, target = np.asarray(prediction, dtype=float), np.asarray(target, dtype=float)
    if not prediction.size or not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Metrics require finite, nonempty observations")
    corr = None
    if prediction.size > 1 and np.std(prediction) > 0 and np.std(target) > 0:
        corr = float(np.corrcoef(prediction, target)[0, 1])
    return {"mse": float(np.mean((prediction-target)**2)),
            "mae": float(np.mean(np.abs(prediction-target))), "pcc": corr, "n": int(target.size)}


def aggregate_words(predictions, labels):
    """Average phone outputs per word, including a last word at sequence capacity."""
    pp, yy = [], []
    for pred, label in zip(predictions, labels):
        for word_id in np.unique(label[:, 3][label[:, 3] >= 0]):
            mask = label[:, 3] == word_id
            pp.append(pred[mask].mean(axis=0))
            yy.append(label[mask, :3].mean(axis=0))
    return np.asarray(pp), np.asarray(yy)


def evaluate(model, data, normalization, device="cpu", batch_size=128, return_predictions=False):
    model.to(device).eval()
    pred = [[], [], []]
    with torch.no_grad():
        for x, phones, _, _, _ in DataLoader(tensors(data, normalization), batch_size=batch_size):
            current = outputs(model, x.to(device), phones.to(device))
            for group, tensor in zip(pred, current):
                group.append(tensor.cpu().numpy())
    p, u, w = [np.concatenate(group) for group in pred]
    mask = data["label_phn"][:, :, 1] >= 0
    wp, wy = aggregate_words(w * 5, data["label_word"])
    report = {"phone": metric(p[mask], data["label_phn"][:, :, 1][mask]),
              "utterance": {name: metric(u[:, i] * 5, data["label_utt"][:, i]) for i, name in enumerate(UTT_NAMES)},
              "word": {name: metric(wp[:, i], wy[:, i]) for i, name in enumerate(WORD_NAMES)},
              "units": {"phone": "0–2 target scale", "utterance_and_word": "0–10 target scale"}}
    if return_predictions:
        return report, {"phone": p, "utterance": u * 5, "word_per_phone": w * 5}
    return report


def load_pretrained(directory="data/gopt"):
    path = Path(directory) / "assets/pretrained.pth"
    if _sha(path) != PRETRAINED_SHA256:
        raise ValueError("Upstream checkpoint hash mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = GOPT(embed_dim=24, num_heads=1, depth=3, input_dim=84)
    model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)
    return model, {"mean": 3.203, "std": 4.045, "source": "upstream published normalization"}


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 10
    learning_rate: float = 1e-3
    batch_size: int = 25
    seed: int = 7
    validation_speakers: int = 25
    patience: int = 5
    embed_dim: int = 24
    heads: int = 1
    depth: int = 3
    device: str = "cpu"
    threads: int = 4

    def validate(self):
        for name in ("epochs", "batch_size", "validation_speakers", "patience", "embed_dim", "heads", "depth", "threads"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if self.embed_dim % self.heads:
            raise ValueError("embed_dim must be divisible by heads")
        if self.device not in ("cpu", "cuda") or (self.device == "cuda" and not torch.cuda.is_available()):
            raise ValueError("Requested device unavailable")


def seed_everything(config):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(True)


def train(directory, run_dir, config=TrainConfig(), on_epoch=None):
    """A fresh GOPT evaluation model, with held-out speakers and visible epochs.

    The upstream pretrained model has already seen all official train speakers;
    it is deliberately NOT used as initialization for this validation experiment.
    """
    config.validate()
    run_dir = Path(run_dir)
    if run_dir.exists():
        raise ValueError("Use a new run directory; existing experiments are never overwritten")
    data = load_data(directory, "train")
    tr, val = split_training(data, config.seed, config.validation_speakers)
    norm = training_normalization(tr)
    seed_everything(config)
    model = GOPT(config.embed_dim, config.heads, config.depth, 84).to(config.device)
    initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 weight_decay=5e-7, betas=(.95, .999))
    # Same architecture/loss/Adam as upstream; constant LR and independent
    # validation replace the original test-selected warm-up/scheduler recipe.
    loader = DataLoader(tensors(tr, norm), batch_size=config.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(config.seed))
    run_dir.mkdir(parents=True)
    split = {"train_ids": tr["ids"].tolist(), "validation_ids": val["ids"].tolist(),
             "train_speakers": sorted(set(tr["speakers"].tolist())),
             "validation_speakers": sorted(set(val["speakers"].tolist()))}
    _json(run_dir / "split.json", split)
    _json(run_dir / "config.json", {**asdict(config), "initialization": "random evaluation model; acoustic features fixed",
          "normalization": norm, "dataset_manifest_sha256": _sha(Path(directory) / "manifest.json"),
          "model_source_sha256": _sha(Path(__file__).parent / "vendor/gopt.py"),
          "versions": {p: importlib.metadata.version(p) for p in ("torch", "numpy")}})
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(run_dir / "tensorboard"))
    history, best, best_epoch, stale, status = [], float("inf"), 0, 0, "completed"
    started = time.perf_counter()

    def record(epoch, batch_loss):
        nonlocal best, best_epoch, stale
        train_metrics = evaluate(model, tr, norm, config.device)
        validation_metrics = evaluate(model, val, norm, config.device)
        change = float(sum(torch.sum((value.detach().cpu()-initial[key])**2).item()
                           for key, value in model.state_dict().items()) ** .5)
        row = {"epoch": epoch, "batch_loss": batch_loss,
               "train_phone_mse": train_metrics["phone"]["mse"], "val_phone_mse": validation_metrics["phone"]["mse"],
               "val_phone_pcc": validation_metrics["phone"]["pcc"],
               "val_utterance_total_pcc": validation_metrics["utterance"]["total"]["pcc"],
               "val_prosody_pcc": validation_metrics["utterance"]["prosody"]["pcc"],
               "weight_change_l2": change, "elapsed_seconds": time.perf_counter()-started}
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "epoch": epoch, "config": asdict(config), "normalization": norm}
        torch.save(checkpoint, run_dir / f"epoch-{epoch:03d}.pt")
        # Epoch zero is a baseline, not a trained checkpoint selection candidate.
        if epoch > 0 and row["val_phone_mse"] < best:
            best, best_epoch, stale = row["val_phone_mse"], epoch, 0
            torch.save(checkpoint, run_dir / "best.pt")
        elif epoch > 0:
            stale += 1
        history.append(row)
        with (run_dir / "metrics.csv").open("a", newline="", encoding="utf-8") as f:
            csv_writer = csv.DictWriter(f, fieldnames=list(row))
            if epoch == 0:
                csv_writer.writeheader()
            csv_writer.writerow(row)
        for key, value in row.items():
            if key != "epoch" and value is not None:
                writer.add_scalar(key, value, epoch)
        writer.flush()
        _json(run_dir / f"metrics-{epoch:03d}.json", {"train": train_metrics, "validation": validation_metrics})
        print(f"epoch {epoch:3d}/{config.epochs}  train phone MSE={row['train_phone_mse']:.4f}  "
              f"val MSE={row['val_phone_mse']:.4f}  val PCC={row['val_phone_pcc']}  "
              f"weight change={change:.4f}", flush=True)
        if on_epoch:
            on_epoch(history)

    try:
        record(0, None)
        for epoch in range(1, config.epochs+1):
            model.train()
            loss_sum, count = 0., 0
            for batch in loader:
                x, phones, p, u, w = [tensor.to(config.device) for tensor in batch]
                loss, _ = losses(outputs(model, x, phones), p, u, w)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss; inspect inputs and learning rate")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * x.shape[0]
                count += x.shape[0]
            record(epoch, loss_sum/count)
            if stale >= config.patience:
                status = "early_stopped"
                break
    except KeyboardInterrupt:
        status = "interrupted"
        print("Stopped. Completed epoch logs and checkpoints are preserved.", flush=True)
    except Exception:
        status = "failed"
        raise
    finally:
        writer.close()
        _json(run_dir / "summary.json", {"status": status, "epochs_completed": history[-1]["epoch"] if history else 0,
              "best_epoch": best_epoch if best_epoch else None,
              "best_validation_phone_mse": best if np.isfinite(best) else None,
              "selection_metric": "validation phone MSE", "test_evaluated": False,
              "elapsed_seconds": time.perf_counter()-started})
    return history


def load_checkpoint(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    config = saved["config"]
    model = GOPT(config["embed_dim"], config["heads"], config["depth"], 84)
    model.load_state_dict(saved["model"], strict=True)
    return model, saved["normalization"]


def plot_history(history):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    epochs = [row["epoch"] for row in history]
    for key, label in (("train_phone_mse", "Train"), ("val_phone_mse", "Validation")):
        axes[0].plot(epochs, [row[key] for row in history], marker=".", label=label)
    axes[0].set(xlabel="Epoch", ylabel="Phone MSE (lower is better)")
    for key, label in (("val_phone_pcc", "Phone"), ("val_prosody_pcc", "Prosody")):
        axes[1].plot(epochs, [np.nan if row[key] is None else row[key] for row in history], marker=".", label=label)
    axes[1].set(xlabel="Epoch", ylabel="Validation Pearson r (higher is better)", ylim=(-1, 1))
    for axis in axes:
        axis.legend()
        axis.grid(alpha=.2)
    fig.tight_layout()
    return fig

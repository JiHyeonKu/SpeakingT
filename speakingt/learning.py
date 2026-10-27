"""Small, supervised fluency calibration experiment using human ratings.

This is not a pronunciation model and cannot learn a correct accent from native
speech alone. The JSONL manifest requires speaker-disjoint train, validation,
and test splits, numeric feature dictionaries, and human-rated targets. Feature
names are inferred from the first training row and saved in the model artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import contextmanager
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

import numpy as np


MODEL_FORMAT = "speakingt.fluency_calibration"
MODEL_VERSION = 1
SPLITS = ("train", "validation", "test")
LIMITATIONS = [
    "This model estimates human-rated fluency from the supplied acoustic features; "
    "it does not assess phoneme accuracy, native accent, grammar, or meaning.",
    "Human ratings and representative speaker-disjoint evaluation data are required. "
    "Native speech recordings alone are not supervised labels.",
    "Synthetic-data tests verify software behavior only, not speech-assessment validity.",
    "The held-out test set must not be reused to choose features or training settings.",
]


def _number(value: Any, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{description} must be a finite numeric value")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{description} must be a finite numeric value") from exc
    if not math.isfinite(result):
        raise ValueError(f"{description} must be a finite numeric value")
    return result


@contextmanager
def _finite_math(description: str):
    """Turn numerical overflow into a recoverable, descriptive input error."""
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            yield
    except (FloatingPointError, np.linalg.LinAlgError, OverflowError) as exc:
        raise ValueError(f"{description}; check feature magnitudes") from exc


def _feature_vector(
    features: Any, names: list[str], description: str
) -> list[float]:
    if not isinstance(features, Mapping) or set(features) != set(names):
        raise ValueError(
            f"{description} must contain exactly these feature names: {', '.join(names)}"
        )
    return [_number(features[name], f"{description}.{name}") for name in names]


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _read_manifest(path: Path) -> tuple[dict[str, list[dict]], list[str], str]:
    raw = path.read_bytes()
    rows: list[tuple[int, dict]] = []
    for line_number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Manifest line {line_number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Manifest line {line_number} must be an object")
        rows.append((line_number, row))
    if not rows:
        raise ValueError("Manifest is empty; human-rated examples are required")

    first_train = next((row for _, row in rows if row.get("split") == "train"), None)
    if first_train is None:
        raise ValueError("Manifest requires nonempty train, validation, and test splits")
    feature_dict = first_train.get("features")
    if not isinstance(feature_dict, dict) or not feature_dict or any(
        not isinstance(name, str) or not name.strip() for name in feature_dict
    ):
        raise ValueError("Training features must be a nonempty dictionary of named values")
    names = sorted(feature_dict)
    grouped: dict[str, list[dict]] = {split: [] for split in SPLITS}
    speaker_splits: dict[str, str] = {}
    recording_splits: dict[str, str] = {}
    audio_splits: dict[str, str] = {}
    for line_number, row in rows:
        context = f"Manifest line {line_number}"
        split = row.get("split")
        if not isinstance(split, str) or split not in SPLITS:
            raise ValueError(f"{context}: split must be train, validation, or test")
        identifiers = {}
        for key in ("speaker_id", "recording_id"):
            value = row.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{context}: {key} must be a nonempty string")
            identifiers[key] = value.strip()
        speaker = identifiers["speaker_id"]
        recording = identifiers["recording_id"]
        if recording in recording_splits:
            raise ValueError(
                f"{context}: duplicate recording_id {recording!r} in "
                f"{recording_splits[recording]} and {split}; recordings must be unique"
            )
        if speaker in speaker_splits and speaker_splits[speaker] != split:
            raise ValueError(
                f"{context}: speaker leakage for {speaker!r} across "
                f"{speaker_splits[speaker]} and {split}"
            )
        audio_hash = row.get("audio_sha256")
        if "audio_sha256" in row:
            if not isinstance(audio_hash, str) or len(audio_hash) != 64 or any(
                character not in "0123456789abcdefABCDEF" for character in audio_hash
            ):
                raise ValueError(f"{context}: audio_sha256 must be a 64-character hexadecimal digest")
            audio_hash = audio_hash.lower()
            if audio_hash in audio_splits:
                raise ValueError(
                    f"{context}: duplicate audio_sha256 in {audio_splits[audio_hash]} "
                    f"and {split}; cloned audio must not be counted as another recording"
                )
            audio_splits[audio_hash] = split
        if row.get("label_source") != "human":
            raise ValueError(f"{context}: label_source must be 'human'")
        target = _number(row.get("target"), f"{context}: target")
        if not 0 <= target <= 100:
            raise ValueError(f"{context}: target must be a human fluency rating from 0 to 100")
        vector = _feature_vector(row.get("features"), names, f"{context}: features")
        grouped[split].append({"features": vector, "target": target, "audio_sha256": audio_hash, **identifiers})
        speaker_splits[speaker] = split
        recording_splits[recording] = split
    if any(not grouped[split] for split in SPLITS):
        raise ValueError("Manifest requires nonempty train, validation, and test splits")
    return grouped, names, hashlib.sha256(raw).hexdigest()


def train_model(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    max_epochs: int = 2000,
    patience: int = 100,
    seed: int = 0,
) -> dict:
    """Fit a reproducible linear calibration and write a JSON model artifact.

    Each manifest line must contain ``split``, ``speaker_id``, ``recording_id``,
    ``features``, ``target`` (human-rated fluency, 0–100), and
    ``label_source: "human"``. All three splits are required. The caller is
    responsible for the truth of the human-label provenance and rating quality.
    Optional ``audio_sha256`` values must be valid unique SHA256 hex digests.
    They detect byte-identical copies even if recording identifiers differ;
    edited or re-encoded duplicate audio still needs a separate dataset audit.

    Normalization uses training data only. Validation MAE selects the best epoch;
    the test split is evaluated only after restoring that epoch. Maximum epochs
    are a limit, not a quality objective. A constant training target is supported
    as a constant predictor instead of dividing by its zero standard deviation.
    """
    max_epochs = _positive_integer(max_epochs, "max_epochs")
    patience = _positive_integer(patience, "patience")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    source = Path(manifest_path)
    destination = Path(output_path)
    if source.resolve() == destination.resolve():
        raise ValueError("output_path must differ from manifest_path")
    grouped, names, source_hash = _read_manifest(source)
    arrays = {
        split: (
            np.asarray([row["features"] for row in rows], dtype=np.float64),
            np.asarray([row["target"] for row in rows], dtype=np.float64),
        )
        for split, rows in grouped.items()
    }
    train_x, train_y = arrays["train"]
    with _finite_math("Feature statistics overflowed"):
        mean = train_x.mean(axis=0)
        scale = train_x.std(axis=0)
        scale = np.where(scale > 0, scale, 1.0)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
        raise ValueError("Feature statistics overflowed; check feature magnitudes")
    target_mean = float(train_y.mean())
    target_median = float(np.median(train_y))
    target_std = float(train_y.std())
    target_scale = target_std if target_std > 0 else 1.0
    normalized = {}
    for split, (x, _) in arrays.items():
        with _finite_math("Feature normalization overflowed"):
            design = np.column_stack(((x - mean) / scale, np.ones(len(x))))
        if not np.all(np.isfinite(design)):
            raise ValueError("Feature normalization overflowed; check feature magnitudes")
        normalized[split] = design

    train_design = normalized["train"]
    scaled_targets = (train_y - target_mean) / target_scale
    rng = np.random.default_rng(seed)
    parameters = np.zeros(len(names) + 1, dtype=np.float64)
    best_parameters = parameters.copy()

    def scores(split: str, coefficients: np.ndarray) -> np.ndarray:
        with _finite_math("Regression produced nonfinite scores"):
            raw_scores = normalized[split] @ coefficients * target_scale + target_mean
        if not np.all(np.isfinite(raw_scores)):
            raise ValueError("Regression produced nonfinite scores; check feature magnitudes")
        return np.clip(raw_scores, 0.0, 100.0)

    def validation_mae(coefficients: np.ndarray) -> float:
        return float(np.mean(np.abs(scores("validation", coefficients) - arrays["validation"][1])))

    best_mae = validation_mae(best_parameters)
    best_epoch = 0
    epochs_run = 0
    stale_epochs = 0
    # For loss 0.5 * mean(error**2), this spectral step is below 1/L.
    with _finite_math("Invalid training matrix"):
        curvature = float(np.linalg.norm(train_design, ord=2) ** 2 / len(train_y))
    if not math.isfinite(curvature) or curvature <= 0:
        raise ValueError("Invalid training matrix; check feature magnitudes")
    learning_rate = 0.9 / curvature
    if target_std > 0:
        parameters[:-1] = rng.normal(0.0, 0.001, len(names))
        for epoch in range(1, max_epochs + 1):
            with _finite_math("Training produced nonfinite parameters"):
                error = train_design @ parameters - scaled_targets
                gradient = train_design.T @ error / len(train_y)
                parameters -= learning_rate * gradient
            if not np.all(np.isfinite(parameters)):
                raise ValueError("Training produced nonfinite parameters; check feature magnitudes")
            current_mae = validation_mae(parameters)
            epochs_run = epoch
            if current_mae < best_mae - 1e-8:
                best_mae = current_mae
                best_parameters = parameters.copy()
                best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= patience:
                break

    # Test scores are computed here, after all model selection has finished.
    metrics = {}
    for split, (_, labels) in arrays.items():
        residuals = scores(split, best_parameters) - labels
        metrics[split] = {
            "mae": float(np.mean(np.abs(residuals))),
            "rmse": float(np.sqrt(np.mean(residuals**2))),
            "baseline_mae": float(np.mean(np.abs(labels - target_mean))),
            "median_baseline_mae": float(np.mean(np.abs(labels - target_median))),
            "count": len(labels),
        }
    warnings = [
        "Experimental fluency estimate: reliability requires representative human "
        "ratings and separate real-speech evaluation. A score is not pronunciation accuracy."
    ]
    if metrics["test"]["mae"] >= metrics["test"]["baseline_mae"]:
        warnings.append(
            "Held-out test MAE does not improve on the training-mean baseline; "
            "this model has not demonstrated useful predictive value on this test set."
        )
    if metrics["test"]["mae"] >= metrics["test"]["median_baseline_mae"]:
        warnings.append("Held-out MAE does not improve on the training-median baseline, which is a relevant constant predictor for absolute error.")
    artifact = {
        "format": MODEL_FORMAT,
        "version": MODEL_VERSION,
        "model_type": "linear_fluency_calibration",
        "target": {"name": "human_rated_fluency", "minimum": 0, "maximum": 100},
        "feature_names": names,
        "normalization": {"mean": mean.tolist(), "scale": scale.tolist()},
        "parameters": {
            "weights": best_parameters[:-1].tolist(),
            "bias": float(best_parameters[-1]),
            "target_mean": target_mean,
            "target_scale": target_scale,
        },
        "metrics": metrics,
        "training": {
            "seed": seed,
            "max_epochs": max_epochs,
            "patience": patience,
            "epochs_run": epochs_run,
            "best_epoch": best_epoch,
            "selection_metric": "validation_mae",
            "learning_rate": learning_rate,
            "constant_training_target": target_std == 0,
            "numpy_version": np.__version__,
        },
        "data": {
            "source_sha256": source_hash,
            "label_source": "human",
            "splits": {
                split: {
                    "recordings": len(rows),
                    "speakers": len({row["speaker_id"] for row in rows}),
                    "audio_hashes_checked": sum(row["audio_sha256"] is not None for row in rows),
                }
                for split, rows in grouped.items()
            },
        },
        "limitations": LIMITATIONS.copy(),
        "warnings": warnings,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return artifact


def predict(model_path: str | Path, features: dict) -> dict:
    """Estimate a human fluency rating; missing or mismatched features are rejected."""
    artifact = json.loads(Path(model_path).read_text(encoding="utf-8"))
    if not isinstance(artifact, dict) or artifact.get("format") != MODEL_FORMAT or artifact.get("version") != MODEL_VERSION:
        raise ValueError("Unsupported fluency calibration model artifact")
    names = artifact.get("feature_names")
    if not isinstance(names, list) or not names or any(
        not isinstance(name, str) or not name.strip() for name in names
    ) or len(set(names)) != len(names):
        raise ValueError("Model feature_names must be unique nonempty strings")
    vector = np.asarray(_feature_vector(features, names, "features"), dtype=np.float64)
    try:
        normalizer = artifact["normalization"]
        parameters = artifact["parameters"]

        def model_vector(values: Any, name: str) -> np.ndarray:
            if not isinstance(values, list) or len(values) != len(names):
                raise ValueError(f"Model {name} has the wrong number of values")
            return np.asarray([_number(value, f"Model {name}") for value in values])

        mean = model_vector(normalizer["mean"], "normalization mean")
        scale = model_vector(normalizer["scale"], "normalization scale")
        weights = model_vector(parameters["weights"], "weights")
        bias = _number(parameters["bias"], "Model bias")
        target_mean = _number(parameters["target_mean"], "Model target_mean")
        target_scale = _number(parameters["target_scale"], "Model target_scale")
    except (KeyError, TypeError) as exc:
        raise ValueError("Incomplete fluency calibration model artifact") from exc
    if np.any(scale <= 0) or target_scale <= 0:
        raise ValueError("Model normalization and target scales must be positive")
    with _finite_math("Prediction overflowed"):
        result = float((((vector - mean) / scale) @ weights + bias) * target_scale + target_mean)
    if not math.isfinite(result):
        raise ValueError("Prediction overflowed; check feature magnitudes")
    return {
        "score": float(np.clip(result, 0.0, 100.0)),
        "score_name": "predicted_human_fluency_0_100",
        "model_type": "linear_fluency_calibration",
        "disclaimer": LIMITATIONS[0],
        "warnings": artifact.get("warnings", [LIMITATIONS[1]]),
    }

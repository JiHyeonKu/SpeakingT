"""Fetch pinned GOPT artifacts and audit every row against SpeechOcean762.

No pickle deserialization, archive extraction, or personal audio upload.
The mirror's dev/new_te partitions are RECOMBINED into the official test set.
They are never used as our validation split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "experiments/gopt/assets.json"
UTT_FIELDS = ("accuracy", "completeness", "fluency", "prosodic", "total")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch_asset(asset, directory):
    root = Path(directory).resolve()
    name = asset["file"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError("Invalid asset filename")
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Asset escapes output directory")
    if path.exists() and path.stat().st_size == asset["bytes"] and sha256(path) == asset["sha256"]:
        return path
    if path.exists():
        raise ValueError(f"Cached asset changed: {path}; inspect it before replacing it")
    if not asset["url"].startswith("https://raw.githubusercontent.com/"):
        raise ValueError("Expected pinned public raw GitHub asset")
    root.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    try:
        request = urllib.request.Request(asset["url"], headers={"User-Agent": "SpeakingT-GOPT/1"})
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > asset["bytes"]:
                    raise ValueError(f"Unexpected download size: {name}")
                output.write(chunk)
        if total != asset["bytes"] or sha256(partial) != asset["sha256"]:
            raise ValueError(f"Download integrity mismatch: {name}")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    print(f"Downloaded {name}: {total:,} bytes", flush=True)
    return path


def read_mapping(path):
    result = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        key, value = line.split(maxsplit=1)
        if key in result:
            raise ValueError(f"Duplicate ID: {key}")
        result[key] = value
    return result


def audit_partition(directory, prefix, scores, phone_map, speakers):
    """Verify row identity, phone order and ALL three levels of human labels."""
    directory = Path(directory)
    ids = (directory / f"{prefix}_keys_utt.csv").read_text().splitlines()
    arrays = {name: np.load(directory / f"{prefix}_{name}.npy", allow_pickle=False)
              for name in ("feat", "label_phn", "label_word", "label_utt")}
    count = len(ids)
    expected_shapes = {"feat": (count, 50, 84), "label_phn": (count, 50, 2),
                       "label_word": (count, 50, 4), "label_utt": (count, 5)}
    if len(set(ids)) != count:
        raise ValueError("Duplicate utterance IDs")
    for name, array in arrays.items():
        if array.shape != expected_shapes[name] or not np.isfinite(array).all():
            raise ValueError(f"Invalid {prefix}/{name} shape or values")
    for i, key in enumerate(ids):
        if key not in scores or key not in speakers:
            raise ValueError(f"Unknown utterance {key}")
        score = scores[key]
        phone_ids = [phone_map[re.sub(r"\d", "", p)]
                     for word in score["words"] for p in word["phones"]]
        phone_scores = [p for word in score["words"] for p in word["phones-accuracy"]]
        word_scores = [[word["accuracy"], word["stress"], word["total"], j]
                       for j, word in enumerate(score["words"]) for _ in word["phones"]]
        n = len(phone_ids)
        phn, word, feat = (arrays[k][i] for k in ("label_phn", "label_word", "feat"))
        if not 0 < n <= 50 or not np.array_equal(phn[:n, 0], phone_ids):
            raise ValueError(f"Phone identity mismatch: {key}")
        if (not np.allclose(phn[:n, 1], phone_scores) or not np.allclose(word[:n], word_scores)
                or not np.allclose(arrays["label_utt"][i], [score[f] for f in UTT_FIELDS])):
            raise ValueError(f"Human label mismatch: {key}")
        if not (np.all(phn[n:] == -1) and np.all(word[n:] == -1) and np.all(feat[n:] == 0)):
            raise ValueError(f"Padding mismatch: {key}")
    arrays["ids"] = np.asarray(ids)
    arrays["speakers"] = np.asarray([speakers[key] for key in ids])
    return arrays


def prepare(output=ROOT / "data/gopt"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    assets = output / "assets"
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    for asset in lock["files"]:
        fetch_asset(asset, assets)
    scores = json.loads((assets / "scores.json").read_text(encoding="utf-8"))
    phone_map = json.loads((assets / "phn_dict.json").read_text())
    if sorted(phone_map.values()) != list(range(39)):
        raise ValueError("Wrong canonical phone inventory")
    train_speakers = read_mapping(assets / "train_utt2spk")
    test_speakers = read_mapping(assets / "test_utt2spk")
    if set(train_speakers.values()) & set(test_speakers.values()):
        raise ValueError("Official speaker splits overlap")
    train = audit_partition(assets, "tr", scores, phone_map, train_speakers)
    parts = [audit_partition(assets, prefix, scores, phone_map, test_speakers)
             for prefix in ("dev", "new_te")]
    test = {key: np.concatenate([part[key] for part in parts]) for key in train}
    order = np.argsort(test["ids"])
    test = {key: value[order] for key, value in test.items()}
    if (set(train["ids"]) != set(train_speakers) or set(test["ids"]) != set(test_speakers)
            or len(train["ids"]) != 2500 or len(test["ids"]) != 2500):
        raise ValueError("Expected the full official 2500/2500 utterance partition")
    metadata = {"data_source": lock["feature_source"], "asset_lock_sha256": sha256(LOCK),
                "label_audit": "all 5000 rows match pinned official phone/word/utterance labels",
                "test_reconstruction": "union of mirror dev/new_te; sorted by ID; no mirror split selection retained",
                "phone_score_floor": 0.0, "datasets": {}}
    for name, arrays in (("train", train), ("test", test)):
        arrays = {k: v.astype(np.float32) if v.dtype.kind == "f" else v for k, v in arrays.items()}
        target = output / f"{name}.npz"
        np.savez(target, **arrays)
        metadata["datasets"][name] = {"utterances": len(arrays["ids"]),
            "speakers": len(set(arrays["speakers"])), "sha256": sha256(target)}
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return metadata


def sample_audio(utterance_id, output=ROOT / "data/gopt"):
    """Fetch one PUBLIC dataset example, never a personal recording."""
    output = Path(output)
    mapping = read_mapping(output / "assets/train_wav.scp")
    mapping.update(read_mapping(output / "assets/test_wav.scp"))
    relative = mapping[utterance_id]
    if not re.fullmatch(r"WAVE/SPEAKER[0-9]{4}/[0-9]{9}\.WAV", relative):
        raise ValueError("Unexpected official WAV path")
    path = output / "examples" / f"{utterance_id}.wav"
    if not path.exists():
        url = "https://raw.githubusercontent.com/jimbozhang/speechocean762/613968e3b0b789fc33936fb5eba1973176ba7d11/" + relative
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read(16 * 1024 * 1024 + 1)
        if len(data) > 16 * 1024 * 1024 or not data.startswith(b"RIFF"):
            raise ValueError("Invalid public audio download")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/gopt")
    prepare(parser.parse_args().output)

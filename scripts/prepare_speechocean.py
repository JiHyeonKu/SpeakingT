"""Prepare a small, reproducible REAL SpeechOcean762 fluency pilot.

Downloads only selected recordings from a pinned official repository revision.
Selection precedes labels/features. This is a pilot subset, not a benchmark.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import sys
import urllib.error
import urllib.parse
import urllib.request


# Support running directly from a repository checkout without an installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from speakingt.audio import analyze_audio
from speakingt.report import feature_vector


REPOSITORY = "https://github.com/jimbozhang/speechocean762"
RAW_ROOT = "https://raw.githubusercontent.com/jimbozhang/speechocean762"
SPLITS = ("train", "validation", "test")
METADATA_PATHS = ("train/wav.scp", "train/utt2spk", "test/wav.scp", "test/utt2spk")
MAX_METADATA_BYTES = 64 * 1024 * 1024
MAX_WAV_BYTES = 16 * 1024 * 1024


class DatasetError(ValueError):
    pass


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise DatasetError("Dataset download redirected unexpectedly; use the pinned raw GitHub source.")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_path(root: Path, relative: str) -> Path:
    if not relative or "\\" in relative or ":" in relative:
        raise DatasetError("Unsafe dataset path.")
    if any(part in ("", ".", "..") for part in relative.split("/")):
        raise DatasetError("Unsafe dataset path.")
    root = root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise DatasetError("Dataset path escapes the output directory.")
    return path


def _fetch(url: str, limit: int) -> bytes:
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != "https" or parsed.netloc != "raw.githubusercontent.com"
            or parsed.query or parsed.fragment
            or not parsed.path.startswith("/jimbozhang/speechocean762/")):
        raise DatasetError("Only the official pinned raw GitHub dataset source is allowed.")
    request = urllib.request.Request(url, headers={"User-Agent": "SpeakingT-research-pilot/0.1"})
    opener = urllib.request.build_opener(_NoRedirect())
    try:
        with opener.open(request, timeout=45) as response:
            if response.status != 200:
                raise DatasetError("Dataset download did not return HTTP 200.")
            data = response.read(limit + 1)
    except (urllib.error.URLError, OSError):
        raise DatasetError("Dataset download failed; check network access and the pinned revision, then rerun to resume.") from None
    if len(data) > limit:
        raise DatasetError("Dataset file exceeds the configured download size limit.")
    return data


def _download(cache: Path, revision: str, relative: str, limit: int) -> tuple[Path, dict]:
    path = _safe_path(cache, relative)
    checksum_path = _safe_path(cache, relative + ".sha256")
    url = f"{RAW_ROOT}/{revision}/{relative}"
    if path.exists() and checksum_path.exists():
        if path.stat().st_size > limit:
            raise DatasetError("Cached dataset file exceeds its size limit.")
        data = path.read_bytes()
        if _sha(data) != checksum_path.read_text(encoding="ascii").strip():
            raise DatasetError("Cached dataset checksum differs; inspect the cache before continuing.")
    else:
        data = _fetch(url, limit)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = _safe_path(cache, relative + ".part")
        temporary.write_bytes(data)
        temporary.replace(path)
        checksum_path.write_text(_sha(data) + "\n", encoding="ascii")
    return path, {"source_path": relative, "url": url, "sha256": _sha(data), "bytes": len(data)}


def _table(data: str, *, paths: bool) -> dict[str, str]:
    result = {}
    for line in data.splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 2 or not re.fullmatch(r"[0-9]{9}", fields[0]):
            raise DatasetError("Unexpected official Kaldi metadata format.")
        key, value = fields
        valid = re.fullmatch(r"WAVE/SPEAKER[0-9]{4}/[0-9]{9}\.WAV", value) if paths else re.fullmatch(r"[0-9]{4}", value)
        if not valid or key in result:
            raise DatasetError("Invalid or duplicate official metadata entry.")
        if paths and value.rsplit("/", 1)[1] != key + ".WAV":
            raise DatasetError("WAV basename does not match its recording ID.")
        result[key] = value
    if not result:
        raise DatasetError("Official metadata table is empty.")
    return result


def select_recordings(tables: dict, *, seed: int, train_speakers: int,
                      validation_speakers: int, test_speakers: int,
                      utterances_per_speaker: int) -> list[dict]:
    """Deterministic hash ranking depends only on seed and official IDs, not labels."""
    counts = {"train": train_speakers, "validation": validation_speakers, "test": test_speakers}
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise DatasetError("Seed must be a nonnegative integer.")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in (*counts.values(), utterances_per_speaker)):
        raise DatasetError("Speaker and utterance counts must be positive integers.")
    grouped = {}
    for official in ("train", "test"):
        waves, speakers = tables[official]["wav"], tables[official]["speaker"]
        if set(waves) != set(speakers):
            raise DatasetError("wav.scp and utt2spk IDs differ.")
        grouped[official] = {}
        for utterance, speaker in speakers.items():
            if waves[utterance].split("/")[1] != "SPEAKER" + speaker:
                raise DatasetError("WAV directory and speaker metadata disagree.")
            grouped[official].setdefault(speaker, []).append(utterance)
    if set(grouped["train"]) & set(grouped["test"]):
        raise DatasetError("Official train/test speaker overlap detected.")
    if set(tables["train"]["wav"]) & set(tables["test"]["wav"]):
        raise DatasetError("Official train/test recording overlap detected.")

    def rank(values, scope):
        return sorted(values, key=lambda value: (_sha(f"{seed}:{scope}:{value}".encode()), value))

    train_pool = rank(grouped["train"], "speaker:train")
    test_pool = rank(grouped["test"], "speaker:test")
    if len(train_pool) < train_speakers + validation_speakers or len(test_pool) < test_speakers:
        raise DatasetError("Not enough official speakers for the requested split sizes.")
    selected_speakers = {
        "train": train_pool[:train_speakers],
        "validation": train_pool[train_speakers:train_speakers + validation_speakers],
        "test": test_pool[:test_speakers],
    }
    rows = []
    for split in SPLITS:
        official = "test" if split == "test" else "train"
        for speaker in selected_speakers[split]:
            utterances = rank(grouped[official][speaker], "utterance:" + speaker)[:utterances_per_speaker]
            for utterance in utterances:
                rows.append({"split": split, "official_split": official,
                             "speaker_id": speaker, "recording_id": utterance,
                             "source_path": tables[official]["wav"][utterance]})
    return rows


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _counts(rows: list[dict]) -> dict:
    return {split: {"utterances": sum(row["split"] == split for row in rows),
                    "speakers": len({row["speaker_id"] for row in rows if row["split"] == split})}
            for split in SPLITS}


def prepare_dataset(output_dir: str | Path, revision: str, *, seed: int = 20261027,
                    train_speakers: int = 40, validation_speakers: int = 10,
                    test_speakers: int = 10, utterances_per_speaker: int = 5) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise DatasetError("--revision must be a full lowercase 40-character Git commit SHA.")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = _safe_path(output, "cache/" + revision)
    sources, tables = [], {"train": {}, "test": {}}
    print("Fetching pinned official split metadata...", flush=True)
    for relative in METADATA_PATHS:
        path, source = _download(cache, revision, relative, MAX_METADATA_BYTES)
        sources.append(source)
        official, filename = relative.split("/")
        tables[official]["wav" if filename == "wav.scp" else "speaker"] = _table(path.read_text(encoding="utf-8"), paths=filename == "wav.scp")
    options = {"seed": seed, "train_speakers": train_speakers, "validation_speakers": validation_speakers,
               "test_speakers": test_speakers, "utterances_per_speaker": utterances_per_speaker}
    selection = select_recordings(tables, **options)
    selection_path = _safe_path(output, "selection.json")
    _json(selection_path, {"revision": revision, "algorithm": "sha256_rank_v1", "options": options,
                           "recordings": selection, "counts": _counts(selection),
                           "selection_before_labels_and_features": True})
    print(f"Fixed selection: {len(selection)} recordings; now fetching human labels and chosen WAVs.", flush=True)
    score_path, score_source = _download(cache, revision, "resource/scores.json", MAX_METADATA_BYTES)
    sources.append(score_source)
    scores = json.loads(score_path.read_text(encoding="utf-8"))
    if not isinstance(scores, dict):
        raise DatasetError("Official score data must be an utterance-keyed object.")
    accepted, excluded = [], []
    for index, row in enumerate(selection, 1):
        path, source = _download(cache, revision, row["source_path"], MAX_WAV_BYTES)
        sources.append(source)
        reason = None
        try:
            audio = analyze_audio(path)
        except (ValueError, OSError):
            reason = "invalid_audio"
        else:
            if not audio["pitch"]["reliable"]:
                reason = "unreliable_pitch"
            elif audio["activity"]["speech_span_s"] <= 0:
                reason = "no_activity"
            else:
                features = feature_vector(audio)
        label = scores.get(row["recording_id"], {})
        fluency = label.get("fluency") if isinstance(label, dict) else None
        if reason is None and (isinstance(fluency, bool) or not isinstance(fluency, (int, float))
                               or not math.isfinite(fluency) or not 0 <= fluency <= 10):
            reason = "missing_or_invalid_human_fluency"
        if reason:
            excluded.append({**row, "reason": reason, "sha256": source["sha256"]})
        else:
            accepted.append({**row, "audio_path": path.relative_to(output).as_posix(),
                             "audio_sha256": source["sha256"], "features": features,
                             "target": float(fluency) * 10, "label_source": "human",
                             "label_detail": "Official aggregate sentence fluency (0-10), multiplied by 10",
                             "duration_s": audio["duration_s"]})
        if index % 20 == 0 or index == len(selection):
            print(f"Prepared {index}/{len(selection)}; retained={len(accepted)}, excluded={len(excluded)}", flush=True)
    manifest = _safe_path(output, "manifest.jsonl")
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in accepted), encoding="utf-8")
    _json(_safe_path(output, "exclusions.json"), excluded)
    provenance = {
        "dataset": "speechocean762", "source_repository": REPOSITORY, "revision": revision,
        "created_utc": datetime.now(timezone.utc).isoformat(), "python_version": platform.python_version(),
        "license": "CC BY 4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "license_source": "https://www.openslr.org/101/",
        "citation": "Zhang et al. (2021), speechocean762: An Open-Source Non-native English Speech Corpus For Pronunciation Assessment, Interspeech.",
        "changes": "Seeded speaker/utterance subset; local acoustic feature extraction; aggregate fluency scores scaled from 0-10 to 0-100.",
        "selection_options": options, "selection_algorithm": "sha256_rank_v1",
        "selection_sha256": _sha(selection_path.read_bytes()), "manifest_sha256": _sha(manifest.read_bytes()),
        "feature_code_sha256": {name: _sha((Path(__file__).resolve().parents[1] / name).read_bytes())
                                for name in ("speakingt/audio.py", "speakingt/report.py", "scripts/prepare_speechocean.py")},
        "selected": _counts(selection), "retained": _counts(accepted), "excluded": _counts(excluded),
        "exclusion_reasons": {split: dict(Counter(row["reason"] for row in excluded if row["split"] == split)) for split in SPLITS},
        "sources": sources,
        "limitations": [
            "Small Mandarin-L1 read-speech pilot subset, not the full official benchmark or a Korean-user validation.",
            "Train and validation speakers come only from official train; test speakers come only from official test.",
            "Speaker-disjoint splits do not guarantee sentence-disjoint prompts.",
            "Selection was fixed before labels/features; exclusions were not replaced and may introduce attrition bias.",
            "Reliable-pitch filtering can exclude difficult voices or recording conditions; report all exclusions.",
            "Human fluency target is the published aggregate of expert ratings, not a pronunciation or native-accent label.",
        ],
    }
    _json(_safe_path(output, "provenance.json"), provenance)
    if any(provenance["retained"][split]["utterances"] == 0 for split in SPLITS):
        raise DatasetError("A split has no usable examples. Provenance and exclusions were saved; do not train this manifest.")
    print(f"Saved manifest and provenance to {output}", flush=True)
    return provenance


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="Full 40-character official repository commit SHA")
    parser.add_argument("--output-dir", default="data/speechocean762-pilot")
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--train-speakers", type=int, default=40)
    parser.add_argument("--validation-speakers", type=int, default=10)
    parser.add_argument("--test-speakers", type=int, default=10)
    parser.add_argument("--utterances-per-speaker", type=int, default=5)
    args = vars(parser.parse_args(argv))
    try:
        prepare_dataset(**args)
    except (DatasetError, OSError, json.JSONDecodeError, UnicodeError) as exc:
        parser.exit(1, f"Preparation failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

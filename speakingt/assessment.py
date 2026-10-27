"""Explicit, optional Azure assessment of short en-US read-aloud recordings.

Calling assess_azure uploads audio and the reference text and may incur charges.
There is no network activity at import time. Provider scores are not human labels.
REST contract: https://learn.microsoft.com/azure/ai-services/speech-service/rest-speech-to-text-short
"""

from __future__ import annotations

import base64
import http.client
import io
import json
import math
import os
from pathlib import Path
import re
import wave


MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_RESPONSE_BYTES = 1024 * 1024
MAX_REFERENCE_CHARS = 4000
PROVIDER = "azure_pronunciation_assessment"
_SCORE_FIELDS = {
    "AccuracyScore": "accuracy",
    "FluencyScore": "fluency",
    "CompletenessScore": "completeness",
    "PronScore": "pronunciation",
    "ProsodyScore": "prosody",
}


class AssessmentError(ValueError):
    """A user-actionable assessment failure without credentials or raw responses."""


def _wav_payload(path: str | Path) -> bytes:
    try:
        with Path(path).open("rb") as source:
            data = source.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES:
            raise AssessmentError("Azure input WAV must be at most 2 MiB.")
        with wave.open(io.BytesIO(data), "rb") as source:
            if (source.getcomptype() != "NONE" or source.getsampwidth() != 2
                    or source.getnchannels() != 1 or source.getframerate() != 16000):
                raise AssessmentError("Azure assessment requires PCM16 mono WAV at 16000 Hz.")
            frames = source.getnframes()
            if not 1600 <= frames <= 30 * 16000:
                raise AssessmentError("Azure assessment requires 0.1 to 30 seconds of audio.")
            samples = source.readframes(frames)
            if len(samples) != frames * 2:
                raise AssessmentError("The WAV audio is truncated.")
    except (OSError, wave.Error, EOFError):
        raise AssessmentError("Cannot read the assessment WAV; export PCM16 mono at 16000 Hz.") from None
    # Transmit a canonical WAV containing audio only, without ancillary metadata.
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16000)
        target.writeframes(samples)
    return output.getvalue()


def _number(value, *, maximum: float | None = None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssessmentError("Azure returned an invalid numeric assessment field.")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (maximum is not None and number > maximum):
        raise AssessmentError("Azure returned an invalid numeric assessment field.")
    return number


def _scores(item: dict) -> dict:
    # REST currently uses flat fields; support the SDK-shaped nested equivalent too.
    assessment = item.get("PronunciationAssessment", item)
    if not isinstance(assessment, dict):
        raise AssessmentError("Azure returned an invalid assessment object.")
    return {
        normalized: _number(assessment[field], maximum=100)
        for field, normalized in _SCORE_FIELDS.items()
        if assessment.get(field) is not None
    }


def _timing(item: dict) -> dict:
    offset = _number(item.get("Offset"))
    duration = _number(item.get("Duration"))
    return {
        "start_s": offset / 10_000_000 if offset is not None else None,
        "end_s": (offset + duration) / 10_000_000 if offset is not None and duration is not None else None,
    }


def _items(value, name: str) -> list:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise AssessmentError(f"Azure returned invalid {name} data.")
    return value


def _text(value) -> str:
    if not isinstance(value, str):
        raise AssessmentError("Azure returned an invalid text field.")
    return value


def _normalize(result: dict) -> dict:
    if not isinstance(result, dict):
        raise AssessmentError("Azure returned an invalid response object.")
    output = {
        "provider": PROVIDER,
        "mode": "scripted",
        "locale": "en-US",
        "status": "no_match",
        "scores": {},
        "words": [],
        "recognized_text": "",
        "caveats": [
            "Provider estimates, not human ground truth or a locally trained pronunciation model.",
            "ASR and alignment errors can affect scores; review audio before correcting pronunciation.",
        ],
    }
    status = result.get("RecognitionStatus")
    if status in ("NoMatch", "InitialSilenceTimeout", "BabbleTimeout", "EndOfDictation"):
        output["caveats"].append("No usable speech assessment was returned; this is not a score of zero.")
        return output
    if status != "Success":
        raise AssessmentError("Azure could not assess this recording; check audio and service configuration.")
    candidates = _items(result.get("NBest"), "candidate")
    if not candidates:
        raise AssessmentError("Azure returned success without an assessment candidate.")
    best = candidates[0]
    scores = _scores(best)
    if not scores:
        raise AssessmentError("Azure returned transcription without pronunciation scores.")
    words = []
    for item in _items(best.get("Words", []), "word"):
        assessment = item.get("PronunciationAssessment", item)
        word_scores = _scores(item)
        phonemes = []
        for phoneme in _items(item.get("Phonemes", []), "phoneme"):
            phonemes.append({
                "phoneme": _text(phoneme.get("Phoneme", "")),
                "accuracy": _scores(phoneme).get("accuracy"),
                **_timing(phoneme),
            })
        words.append({
            "word": _text(item.get("Word", "")),
            "accuracy": word_scores.get("accuracy"),
            "error_type": _text(assessment.get("ErrorType", "Unknown")),
            "phonemes": phonemes,
            **_timing(item),
        })
    output.update(status="success", scores=scores, words=words,
                  recognized_text=_text(best.get("Display", result.get("DisplayText", ""))))
    return output


def assess_azure(path: str | Path, reference_text: str) -> dict:
    """Upload <=30s PCM16 mono 16kHz WAV for en-US phoneme/prosody assessment.

    Requires SPEAKINGT_AZURE_KEY and SPEAKINGT_AZURE_REGION. The caller must make
    this upload explicit to the user. No retries, redirects, or SDK downloads.
    """
    if not isinstance(reference_text, str) or not reference_text.strip():
        raise AssessmentError("Azure read-aloud assessment requires non-empty reference text.")
    if len(reference_text) > MAX_REFERENCE_CHARS:
        raise AssessmentError("Reference text must be at most 4000 characters for short assessment.")
    region = os.environ.get("SPEAKINGT_AZURE_REGION", "").strip().lower()
    key = os.environ.get("SPEAKINGT_AZURE_KEY", "").strip()
    if not re.fullmatch(r"[a-z][a-z0-9]{1,31}", region):
        raise AssessmentError("Set SPEAKINGT_AZURE_REGION to an Azure region name, such as eastus.")
    if not key or len(key) > 256 or not key.isascii() or any(character.isspace() or ord(character) < 32 or ord(character) == 127 for character in key):
        raise AssessmentError("Set SPEAKINGT_AZURE_KEY to a valid Speech resource key.")
    data = _wav_payload(path)
    parameters = {
        "ReferenceText": reference_text,
        "GradingSystem": "HundredMark",
        "Granularity": "Phoneme",
        "Dimension": "Comprehensive",
        "EnableMiscue": True,
        "EnableProsodyAssessment": True,
    }
    header = base64.b64encode(json.dumps(parameters, ensure_ascii=False).encode("utf-8")).decode("ascii")
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000",
        "Accept": "application/json",
        "Pronunciation-Assessment": header,
    }
    # Restrict the host to Microsoft's documented regional Speech endpoint.
    # HTTPSConnection verifies TLS and does not follow any HTTP redirects.
    connection = http.client.HTTPSConnection(f"{region}.stt.speech.microsoft.com", timeout=45)
    try:
        connection.request("POST", "/speech/recognition/conversation/cognitiveservices/v1?language=en-US&format=detailed", body=data, headers=headers)
        response = connection.getresponse()
        if response.status != 200:
            if response.status in (401, 403):
                raise AssessmentError("Azure authentication failed; check the Speech key and region.")
            if response.status == 429:
                raise AssessmentError("Azure rate or quota limit reached; try again later.")
            raise AssessmentError("Azure assessment request failed; check service availability and configuration.")
        raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise AssessmentError("Azure returned an oversized assessment response.")
    except (OSError, http.client.HTTPException):
        raise AssessmentError("Azure connection failed; check your network and Speech region.") from None
    finally:
        connection.close()
    try:
        result = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        raise AssessmentError("Azure returned an invalid JSON assessment response.") from None
    return _normalize(result)

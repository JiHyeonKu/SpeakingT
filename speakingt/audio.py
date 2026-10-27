"""Bounded, local acoustic measurements for uncompressed PCM WAV recordings.

These measurements are not pronunciation judgements. In particular, energy
activity is not speech recognition and periodicity is not proof of human voicing.
"""

from __future__ import annotations

import math
from pathlib import Path
import wave

import numpy as np


MAX_DURATION_S = 300.0
MAX_PCM_BYTES = 128 * 1024 * 1024
MIN_DURATION_S = 0.1
MIN_SAMPLE_RATE = 8000
MAX_SAMPLE_RATE = 48000
ACTIVITY_FRAME_S = 0.02
BRIDGE_GAP_S = 0.12
MIN_PAUSE_S = 0.25
PITCH_WINDOW_S = 0.06
PITCH_HOP_S = 0.02
MIN_PITCH_HZ = 65.0
MAX_PITCH_HZ = 500.0
MIN_PERIODICITY = 0.70


def _dbfs(value: float) -> float:
    """Use a finite -120 dBFS floor so JSON has no infinity values."""
    return float(max(-120.0, 20.0 * math.log10(max(float(value), 1e-6))))


def _read_pcm(path: str | Path) -> tuple[np.ndarray, int, int, float]:
    try:
        with wave.open(str(Path(path)), "rb") as source:
            channels = source.getnchannels()
            rate = source.getframerate()
            width = source.getsampwidth()
            count = source.getnframes()
            if source.getcomptype() != "NONE" or width not in (1, 2, 3, 4):
                raise ValueError("Use an uncompressed 8-, 16-, 24-, or 32-bit PCM WAV file.")
            if not 1 <= channels <= 8:
                raise ValueError("WAV recordings must have between 1 and 8 channels.")
            if not MIN_SAMPLE_RATE <= rate <= MAX_SAMPLE_RATE:
                raise ValueError("WAV sample rate must be between 8000 and 48000 Hz.")
            duration = count / rate
            if not MIN_DURATION_S <= duration <= MAX_DURATION_S:
                raise ValueError("Recording duration must be between 0.1 and 300 seconds.")
            expected = count * channels * width
            if expected > MAX_PCM_BYTES:
                raise ValueError("Decoded PCM audio must not exceed 128 MiB.")
            raw = source.readframes(count)
            if len(raw) != expected:
                raise ValueError("The WAV file is truncated or its frame count is invalid.")
    except (wave.Error, EOFError) as exc:
        raise ValueError("Cannot read WAV audio. Export uncompressed integer PCM WAV.") from exc

    if width == 1:
        values = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        values = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        octets = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        integer = (octets[:, 0].astype(np.int32)
                   | (octets[:, 1].astype(np.int32) << 8)
                   | (octets[:, 2].astype(np.int32) << 16))
        integer = (integer ^ 0x800000) - 0x800000
        values = integer.astype(np.float32) / 8388608.0
    else:
        # Float64 retains the positive full-scale 32-bit integer before division.
        values = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    full_scale_threshold = 1.0 - 1.0 / (2 ** (8 * width - 1))
    clipping = float(np.count_nonzero(np.abs(values) >= full_scale_threshold) / values.size)
    mono = values.reshape(-1, channels).mean(axis=1, dtype=np.float64).astype(np.float32)
    return mono, rate, channels, clipping


def _runs(mask: np.ndarray, value: bool) -> list[tuple[int, int]]:
    """Return half-open runs of the requested Boolean value."""
    selected = mask == value
    boundaries = np.flatnonzero(np.diff(np.r_[False, selected, False].astype(np.int8)))
    return [(int(start), int(end)) for start, end in boundaries.reshape(-1, 2)]


def _activity(samples: np.ndarray, rate: int) -> tuple[dict, float]:
    frame_size = round(rate * ACTIVITY_FRAME_S)
    full_count = len(samples) // frame_size
    full = samples[:full_count * frame_size].reshape(-1, frame_size)
    energy = np.einsum("ij,ij->i", full, full, dtype=np.float64) / frame_size
    if len(samples) % frame_size:
        tail = samples[full_count * frame_size:].astype(np.float64)
        energy = np.r_[energy, np.dot(tail, tail) / len(tail)]
    frame_rms = np.sqrt(energy)
    frame_db = 20.0 * np.log10(np.maximum(frame_rms, 1e-6))
    threshold_db = float(max(-50.0, np.percentile(frame_db, 95) - 25.0))
    threshold = 10.0 ** (threshold_db / 20.0)
    active = frame_rms >= threshold
    for start, end in _runs(active, False):
        if start > 0 and end < len(active) and (end - start) * ACTIVITY_FRAME_S <= BRIDGE_GAP_S + 1e-9:
            active[start:end] = True
    duration = len(samples) / rate
    edges = np.minimum(np.arange(len(active) + 1) * frame_size / rate, duration)
    active_duration = float(np.sum(np.diff(edges)[active]))
    active_indexes = np.flatnonzero(active)
    if active_indexes.size:
        first = int(active_indexes[0])
        last = int(active_indexes[-1])
        start_s = float(edges[first])
        end_s = float(edges[last + 1])
    else:
        first, last = 0, -1
        start_s = end_s = None
    pauses = []
    for start, end in _runs(active, False):
        length = float(edges[end] - edges[start])
        if start > first and end <= last and length >= MIN_PAUSE_S:
            pauses.append({"start_s": float(edges[start]), "end_s": float(edges[end]), "duration_s": length})
    span = float(end_s - start_s) if start_s is not None else 0.0
    return {
        "method": "energy_threshold",
        "frame_duration_s": frame_size / rate,
        "threshold_dbfs": threshold_db,
        "bridged_gap_max_s": BRIDGE_GAP_S,
        "minimum_pause_s": MIN_PAUSE_S,
        "active_duration_s": active_duration,
        "active_fraction": active_duration / duration,
        "start_s": start_s,
        "end_s": end_s,
        "speech_span_s": span,
        "internal_pauses": pauses,
        "internal_pause_ratio": sum(pause["duration_s"] for pause in pauses) / span if span else 0.0,
        "limitations": [
            "Energy activity is not a neural voice activity detector and can include noise, music, and non-speech sounds.",
            "An internal acoustic pause is not necessarily hesitation or a speaking error.",
        ],
    }, threshold


def _pitch(samples: np.ndarray, rate: int, threshold: float, clipping: float) -> dict:
    # Analyze at the original sample rate; avoiding decimation prevents aliasing.
    frame_size = round(PITCH_WINDOW_S * rate)
    hop = round(PITCH_HOP_S * rate)
    min_lag = math.ceil(rate / MAX_PITCH_HZ)
    max_lag = math.floor(rate / MIN_PITCH_HZ)
    fft_size = 1 << (2 * frame_size - 1).bit_length()
    lags = np.arange(min_lag, max_lag + 1)
    contour = []
    active_frames = 0
    total_frames = 0
    for start in range(0, len(samples) - frame_size + 1, hop):
        total_frames += 1
        frame = samples[start:start + frame_size].astype(np.float64)
        frame -= np.mean(frame)
        rms = math.sqrt(float(np.dot(frame, frame)) / frame_size)
        if rms < max(threshold, 1e-5):
            continue
        active_frames += 1
        spectrum = np.fft.rfft(frame, n=fft_size)
        correlation = np.fft.irfft(spectrum.real ** 2 + spectrum.imag ** 2, n=fft_size)
        cumulative = np.r_[0.0, np.cumsum(frame * frame)]
        # Normalize by the energy in each overlapping pair, not by lag-zero energy.
        all_lags = np.arange(min_lag - 1, max_lag + 2)
        denominator = np.sqrt(cumulative[frame_size - all_lags] * (cumulative[frame_size] - cumulative[all_lags]))
        normalized = correlation[all_lags] / np.maximum(denominator, 1e-15)
        normalized = np.clip(normalized, -1.0, 1.0)
        interior = normalized[1:-1]
        peak_indexes = np.flatnonzero((interior >= normalized[:-2]) & (interior > normalized[2:]))
        if not peak_indexes.size:
            continue
        best = float(np.max(interior[peak_indexes]))
        if best < MIN_PERIODICITY:
            continue
        # First sufficiently strong peak avoids choosing integer multiples of a
        # periodic waveform's fundamental period. Octave errors remain possible.
        candidate_indexes = peak_indexes[interior[peak_indexes] >= max(MIN_PERIODICITY, 0.95 * best)]
        index = int(candidate_indexes[0])
        left, center, right = normalized[index:index + 3]
        curvature = float(left - 2.0 * center + right)
        offset = float(0.5 * (left - right) / curvature) if abs(curvature) > 1e-12 else 0.0
        offset = float(np.clip(offset, -0.5, 0.5))
        f0 = float(rate / (float(lags[index]) + offset))
        if MIN_PITCH_HZ <= f0 <= MAX_PITCH_HZ:
            contour.append({"time_s": float((start + frame_size / 2) / rate), "f0_hz": f0, "confidence": float(center)})
    median = float(np.median([frame["f0_hz"] for frame in contour])) if contour else None
    for frame in contour:
        frame["semitones_from_median"] = float(12.0 * math.log2(frame["f0_hz"] / median))
    voiced_fraction = len(contour) / total_frames if total_frames else 0.0
    voiced_active_fraction = len(contour) / active_frames if active_frames else 0.0
    reliable = (len(samples) / rate >= 1.0 and len(contour) * hop / rate >= 0.5
                and voiced_active_fraction >= 0.4 and clipping < 0.05)
    return {
        "method": "autocorrelation",
        "analysis_sample_rate_hz": rate,
        "window_duration_s": frame_size / rate,
        "hop_duration_s": hop / rate,
        "frequency_range_hz": [MIN_PITCH_HZ, MAX_PITCH_HZ],
        "minimum_periodicity": MIN_PERIODICITY,
        "median_hz": median,
        "voiced_fraction": voiced_fraction,
        "voiced_active_fraction": voiced_active_fraction,
        "voiced_frame_count": len(contour),
        "analyzed_frame_count": total_frames,
        "contour": contour,
        "reliable": bool(reliable),
        "reliability_criteria": "At least 1 s of audio, 0.5 s of periodic frame coverage, 40% periodic active frames, and under 5% clipping.",
        "limitations": [
            "Confidence is normalized autocorrelation periodicity, not a calibrated probability of correct pitch.",
            "Pitch may contain octave errors; tonal noise or music can pass these periodicity checks.",
            "The 65-500 Hz range can exclude some voices, vocal fry, and extreme expressive speech.",
            "Reliable means basic acoustic checks passed; it does not establish correct pronunciation or intonation.",
            "Unvoiced consonants have no fundamental frequency and are absent from this contour.",
            "Median-relative semitones remove habitual pitch height but do not align words or compare sentence meaning.",
        ],
    }


def analyze_audio(path: str | Path) -> dict:
    """Return JSON-safe acoustic features for a 0.1-300 s integer PCM WAV.

    Supports 8/16/24/32-bit PCM, 8-48 kHz, 1-8 channels, and at most 128 MiB
    of PCM data. Channels are averaged. Pitch work is capped by duration, rate,
    and a fixed 20 ms hop. No network, transcription, or trained model is used.
    """
    samples, rate, channels, clipping = _read_pcm(path)
    rms = math.sqrt(float(np.einsum("i,i->", samples, samples, dtype=np.float64)) / len(samples))
    activity, threshold = _activity(samples, rate)
    pitch = _pitch(samples, rate, threshold, clipping)
    warnings = []
    if channels > 1:
        warnings.append("Channels were averaged to mono; opposite-phase channels can cancel.")
    if rms < 10 ** (-45 / 20):
        warnings.append("The recording is silent or very quiet; record closer to the microphone.")
    if clipping > 0.01:
        warnings.append("More than 1% of input samples reach full scale; lower microphone gain.")
    if not pitch["reliable"]:
        warnings.append("There is insufficient reliable periodic audio for interpreting the pitch contour.")
    return {
        "duration_s": len(samples) / rate,
        "sample_rate_hz": rate,
        "channels": channels,
        "signal": {
            "rms": rms,
            "rms_dbfs": _dbfs(rms),
            "peak": float(np.max(np.abs(samples))),
            "clipping_fraction": clipping,
        },
        "activity": activity,
        "pitch": pitch,
        "warnings": warnings,
    }

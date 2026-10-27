"""Bounded local decoding to analysis WAV, with the original left untouched.

PyAV is an optional dependency (``speakingt[audio]`` or ``speakingt[asr]``). Decoding
does not perform VAD, silence trimming, loudness normalization, or denoising.
Container timestamps are rebased to the first decoded sample; internal gaps
are retained as silence and overlapping/discontinuous timestamps are rejected.
"""

from __future__ import annotations

from array import array
import hashlib
import math
from pathlib import Path
import shutil
import sys
import tempfile
import wave


OUTPUT_RATE = 16000
MAX_DURATION_S = 300.0
MAX_DECODED_BYTES = 128 * 1024 * 1024
MAX_INPUT_BYTES = 128 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for block in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _channels(frame) -> int:
    return len(frame.layout.channels)


def import_audio(source: str | Path, output: str | Path) -> dict:
    """Decode the first audio stream to mono, 16 kHz, PCM16 WAV.

    Return JSON-safe provenance including SHA256 digests. Refuse overwrites and
    inputs exceeding 300 decoded seconds or 128 MiB of decoded sample data.
    Missing PyAV and malformed/unsupported audio raise ValueError.
    Ordinary filesystem errors retain their original exception.
    A failed conversion removes its temporary file and any newly created output.
    """
    origin = Path(source).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    if origin == destination or destination.exists():
        raise ValueError("Cannot overwrite the original or an existing output file.")
    if not origin.is_file():
        raise FileNotFoundError(origin)
    if origin.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError("Input audio must not exceed 128 MiB.")
    try:
        import av
    except ImportError as exc:
        raise ValueError("Audio import requires PyAV: run pip install -e '.[audio]'.") from exc

    original_digest = _sha256(origin)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    output_created = False
    source_info: dict = {}
    written_samples = 0
    decoded_bytes = 0
    decoded_duration = 0.0
    gap_samples = 0
    timestamp_origin: float | None = None
    max_samples = int(MAX_DURATION_S * OUTPUT_RATE)
    try:
        with tempfile.NamedTemporaryFile(prefix=".speakingt-import-", suffix=".wav", dir=destination.parent, delete=False) as staging:
            temporary = Path(staging.name)
        with wave.open(str(temporary), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(OUTPUT_RATE)

            def write_frame(frame) -> None:
                nonlocal written_samples, gap_samples, timestamp_origin
                count = int(frame.samples)
                if count <= 0:
                    return
                position = written_samples
                if frame.pts is not None and frame.time_base is not None:
                    timestamp = float(frame.pts * frame.time_base)
                    if not math.isfinite(timestamp):
                        raise ValueError("Audio has invalid timestamps.")
                    if timestamp_origin is None:
                        timestamp_origin = timestamp - written_samples / OUTPUT_RATE
                    position = round((timestamp - timestamp_origin) * OUTPUT_RATE)
                    # Resampling timestamp rounding may differ by one sample.
                    if abs(position - written_samples) <= 1:
                        position = written_samples
                    if position < written_samples:
                        raise ValueError("Audio timestamps overlap; refusing to discard samples.")
                if position + count > max_samples:
                    raise ValueError("Decoded audio duration must not exceed 300 seconds.")
                gap = position - written_samples
                gap_samples += gap
                while gap:
                    block = min(gap, OUTPUT_RATE)
                    writer.writeframesraw(b"\x00\x00" * block)
                    written_samples += block
                    gap -= block
                raw = bytes(frame.planes[0])[:count * 2]
                if len(raw) != count * 2:
                    raise ValueError("Decoder returned truncated sample data.")
                if sys.byteorder != "little":
                    samples = array("h")
                    samples.frombytes(raw)
                    samples.byteswap()
                    raw = samples.tobytes()
                writer.writeframesraw(raw)
                written_samples += count

            try:
                with av.open(str(origin), mode="r") as container:
                    if not container.streams.audio:
                        raise ValueError("The input contains no audio stream.")
                    stream = container.streams.audio[0]
                    source_info = {
                        "codec": stream.codec_context.name,
                        "stream_index": int(stream.index),
                        "sample_rate_hz": None,
                        "channels": None,
                        "sample_format": None,
                    }
                    resampler = av.AudioResampler(format="s16", layout="mono", rate=OUTPUT_RATE)
                    for frame in container.decode(stream):
                        if getattr(frame, "is_corrupt", False):
                            raise ValueError("The input contains corrupt audio frames.")
                        rate = int(frame.sample_rate)
                        channels = _channels(frame)
                        count = int(frame.samples)
                        width = int(frame.format.bytes)
                        if rate <= 0 or channels <= 0 or count < 0 or width <= 0:
                            raise ValueError("Decoder returned invalid audio parameters.")
                        if source_info["sample_rate_hz"] is None:
                            source_info.update(sample_rate_hz=rate, channels=channels, sample_format=frame.format.name)
                        elif rate != source_info["sample_rate_hz"] or channels != source_info["channels"]:
                            raise ValueError("Changing sample rate or channel count is unsupported.")
                        decoded_duration += count / rate
                        decoded_bytes += count * channels * width
                        if decoded_duration > MAX_DURATION_S + 1e-9:
                            raise ValueError("Decoded audio duration must not exceed 300 seconds.")
                        if decoded_bytes > MAX_DECODED_BYTES:
                            raise ValueError("Decoded audio must not exceed 128 MiB.")
                        for converted in resampler.resample(frame):
                            write_frame(converted)
                    for converted in resampler.resample(None):
                        write_frame(converted)
            except ValueError:
                raise
            except Exception as exc:
                raise ValueError("Cannot decode the input audio; it may be corrupt or unsupported.") from exc
            if written_samples < math.ceil(0.1 * OUTPUT_RATE):
                raise ValueError("Decoded audio must contain at least 0.1 seconds of samples.")

        if _sha256(origin) != original_digest:
            raise ValueError("The source changed during import; retry with a stable file.")
        output_digest = _sha256(temporary)
        # Exclusive creation prevents a race from overwriting a newly added file.
        with destination.open("xb") as final_writer:
            output_created = True
            with temporary.open("rb") as final_reader:
                shutil.copyfileobj(final_reader, final_writer, length=1024 * 1024)
        return {
            "source": {"path": str(origin), "sha256": original_digest, **source_info},
            "output": {
                "path": str(destination), "sha256": output_digest,
                "codec": "pcm_s16le", "sample_rate_hz": OUTPUT_RATE, "channels": 1,
                "sample_width_bytes": 2, "samples": written_samples,
                "duration_s": written_samples / OUTPUT_RATE,
            },
            "processing": {
                "decoder": "PyAV", "decoder_version": str(getattr(av, "__version__", "unknown")),
                "resampled": source_info["sample_rate_hz"] != OUTPUT_RATE,
                "downmixed": source_info["channels"] != 1,
                "normalized": False, "trimmed": False, "vad_applied": False,
                "timeline_origin_s": timestamp_origin, "timestamp_gap_samples": gap_samples,
                "decoded_duration_s": decoded_duration, "decoded_bytes": decoded_bytes,
                "note": "Container offset rebased; decoded silence and internal timestamp gaps retained. Lossy-source details cannot be recovered by conversion.",
            },
        }
    except BaseException:
        if output_created:
            destination.unlink(missing_ok=True)
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

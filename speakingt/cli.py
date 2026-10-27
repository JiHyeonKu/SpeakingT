from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import wave

import numpy as np

from .report import build_report, feature_vector, save_report


def write_json(path: str | Path, value: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_wav(path: Path, samples: np.ndarray, rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())


def demo(output_dir: str) -> tuple[Path, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rate = 16000
    time = np.arange(2 * rate) / rate
    phase = 2 * np.pi * (150 * time + 15 * time ** 2)
    tone = .2 * np.sin(phase) + .07 * np.sin(2 * phase)
    samples = np.concatenate([np.zeros(rate // 3), tone, np.zeros(rate // 2), tone, np.zeros(rate // 3)])
    wav_path = directory / "synthetic_signal.wav"
    write_wav(wav_path, samples)
    report = build_report(wav_path)
    report["stage"] = "synthetic_pipeline_demo_not_speech"
    report["limitations"].insert(0, "이 녹음은 합성 신호입니다. 실제 사람의 말소리나 발음 평가 결과가 아닙니다.")
    return save_report(report, directory / "synthetic_report.json")


def clip_wav(source: str, output: str, start: float, end: float) -> None:
    from .audio import analyze_audio
    info = analyze_audio(source)
    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= info["duration_s"]):
        raise ValueError("잘라낼 구간은 0 <= start < end <= 녹음 길이여야 합니다.")
    target = Path(output)
    if target.resolve() == Path(source).resolve() or target.exists():
        raise ValueError("원본/기존 파일을 덮어쓸 수 없습니다. 새 출력 경로를 사용하세요.")
    with wave.open(source, "rb") as reader:
        params = reader.getparams()
        start_frame, end_frame = int(start * params.framerate), int(end * params.framerate)
        if end_frame - start_frame < math.ceil(.1 * params.framerate):
            raise ValueError("분석할 수 있도록 최소 0.1초 구간을 선택하세요.")
        reader.setpos(start_frame)
        frames = reader.readframes(end_frame - start_frame)
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(frames)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SpeakingT: 원본 음성 관찰 및 발음 평가 연구 도구")
    commands = parser.add_subparsers(dest="command", required=True)
    importer = commands.add_parser("import-audio", help="M4A 등 음성을 분석용 PCM WAV로 변환; 원본 보존")
    importer.add_argument("audio")
    importer.add_argument("--output", required=True)
    analyze = commands.add_parser("analyze", help="PCM WAV 분석과 HTML/JSON 보고서")
    analyze.add_argument("audio")
    analyze.add_argument("--output", default="reports/analysis.json")
    text = analyze.add_mutually_exclusive_group()
    text.add_argument("--transcript-file", help="실제 발화를 그대로 적은 UTF-8 텍스트")
    text.add_argument("--asr", metavar="MODEL", help="선택: 로컬 faster-whisper 모델, 첫 실행 시 가중치 다운로드")
    analyze.add_argument("--reference-audio")
    analyze.add_argument("--reference-text", help="읽기 연습의 목표 문장; 실제 전사와 별도")
    analyze.add_argument("--azure", action="store_true", help="선택: 원본을 유료 Azure 서비스에 보내 발음 평가 (30초 이내)")
    analyze.add_argument("--fluency-model", help="사람 평가로 학습한 실험용 보정 모델 JSON")
    record = commands.add_parser("record", help="마이크를 사용해 16kHz mono WAV 녹음")
    record.add_argument("--seconds", type=float, default=20)
    record.add_argument("--output", required=True)
    clip = commands.add_parser("clip", help="원본을 보존하면서 지정 구간 추출")
    clip.add_argument("audio")
    clip.add_argument("--start", required=True, type=float)
    clip.add_argument("--end", required=True, type=float)
    clip.add_argument("--output", required=True)
    synthetic = commands.add_parser("demo", help="합성 신호로 분석 파이프라인만 확인")
    synthetic.add_argument("--output-dir", default="reports/demo")
    label = commands.add_parser("prepare-label", help="분석 보고서를 사람 채점용 JSONL 초안으로 변환")
    label.add_argument("report")
    label.add_argument("--speaker-id", required=True)
    label.add_argument("--recording-id", required=True)
    label.add_argument("--split", choices=["train", "validation", "test"], required=True)
    label.add_argument("--output", required=True)
    train = commands.add_parser("train", help="사람이 채점한 JSONL로 작은 유창성 보정 모델 학습")
    train.add_argument("manifest")
    train.add_argument("--output", default="models/fluency.json")
    train.add_argument("--max-epochs", type=int, default=2000)
    train.add_argument("--patience", type=int, default=100)
    train.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = make_parser().parse_args(argv)
    try:
        if args.command == "import-audio":
            from .import_audio import import_audio
            metadata = import_audio(args.audio, args.output)
            write_json(str(Path(args.output)) + ".import.json", metadata)
            print(Path(args.output).resolve())
        elif args.command == "demo":
            paths = demo(args.output_dir)
            print("합성 신호 테스트입니다. 실제 발음 평가가 아닙니다.")
            for path in paths:
                print(path.resolve())
        elif args.command == "analyze":
            if args.azure and not args.reference_text:
                raise ValueError("이 버전의 Azure 평가는 읽기 연습용입니다. --reference-text가 필요합니다.")
            transcript = Path(args.transcript_file).read_text(encoding="utf-8-sig") if args.transcript_file else None
            report = build_report(args.audio, transcript=transcript, reference_audio=args.reference_audio, reference_text=args.reference_text)
            if args.asr:
                from .transcript import inspect_transcript, transcribe_local
                asr = transcribe_local(args.audio, args.asr, ".cache/asr")
                report["asr"] = asr
                report["transcript"] = inspect_transcript(asr["text"])
                report["transcript"]["source"] = "automatic_asr_unverified"
            if args.azure:
                from .assessment import assess_azure
                report["pronunciation_assessment"] = assess_azure(args.audio, args.reference_text)
            if args.fluency_model:
                from .learning import predict
                report["experimental_fluency_prediction"] = predict(args.fluency_model, feature_vector(report["audio"]))
            for path in save_report(report, args.output):
                print(path.resolve())
        elif args.command == "record":
            if not math.isfinite(args.seconds) or not .1 <= args.seconds <= 300:
                raise ValueError("녹음 길이는 0.1초 이상 300초 이하여야 합니다.")
            output = Path(args.output)
            if output.exists():
                raise ValueError("기존 녹음을 덮어쓸 수 없습니다. 새 이름을 사용하세요.")
            try:
                import sounddevice as sd
            except ImportError as exc:
                raise ValueError('녹음 선택 기능을 설치하세요: python -m pip install -e ".[record]"') from exc
            print(f"지금부터 {args.seconds:g}초간 녹음합니다.", flush=True)
            try:
                samples = sd.rec(int(args.seconds * 16000), samplerate=16000, channels=1, dtype="float32")
                sd.wait()
            except sd.PortAudioError as exc:
                raise ValueError("마이크를 열 수 없습니다. 장치 연결과 OS 마이크 권한을 확인하세요.") from exc
            write_wav(output, samples)
            print(output.resolve())
        elif args.command == "clip":
            clip_wav(args.audio, args.output, args.start, args.end)
            print(Path(args.output).resolve())
        elif args.command == "prepare-label":
            report = json.loads(Path(args.report).read_text(encoding="utf-8-sig"))
            if report.get("stage") != "measurement_prototype":
                raise ValueError("합성 데모는 사람의 발음 학습 데이터로 사용할 수 없습니다.")
            row = {
                "speaker_id": args.speaker_id, "recording_id": args.recording_id,
                "split": args.split, "features": feature_vector(report["audio"]),
                "feature_schema_version": 1, "target": None, "label_source": "pending_human",
                "audio_sha256": report["recording"]["sha256"],
            }
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            print("초안 저장. 사람이 실제 녹음을 듣고 target과 label_source를 채워야 학습할 수 있습니다.")
        elif args.command == "train":
            from .learning import train_model
            artifact = train_model(args.manifest, args.output, max_epochs=args.max_epochs, patience=args.patience, seed=args.seed)
            print(json.dumps(artifact["metrics"], ensure_ascii=False, indent=2))
            print(Path(args.output).resolve())
        return 0
    except (ValueError, OSError, KeyError, wave.Error) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

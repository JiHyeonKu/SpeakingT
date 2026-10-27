"""Evidence-based Korean feedback and portable offline reports."""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

import numpy as np

from .audio import analyze_audio
from .transcript import inspect_transcript


def feature_vector(audio: dict) -> dict:
    activity, pitch = audio["activity"], audio["pitch"]
    if not pitch["reliable"] or activity["speech_span_s"] <= 0:
        raise ValueError("신뢰 가능한 음높이/활동 구간이 없어 학습용 특징을 만들 수 없습니다.")
    contour = [p["semitones_from_median"] for p in pitch["contour"]]
    pauses = activity["internal_pauses"]
    return {
        "active_fraction": activity["active_fraction"],
        "internal_pause_ratio": sum(p["duration_s"] for p in pauses) / activity["speech_span_s"],
        "internal_pause_count_per_min": len(pauses) * 60 / activity["speech_span_s"],
        "pitch_span_semitones": float(np.percentile(contour, 90) - np.percentile(contour, 10)),
    }


def build_report(path: str | Path, *, transcript: str | None = None,
                 reference_audio: str | Path | None = None, reference_text: str | None = None) -> dict:
    audio = analyze_audio(path)
    feedback = []
    for pause in audio["activity"]["internal_pauses"]:
        feedback.append({
            "kind": "pause_candidate", "start_s": pause["start_s"], "end_s": pause["end_s"],
            "message": f'{pause["start_s"]:.2f}–{pause["end_s"]:.2f}초에 낮은 에너지 구간이 있습니다. 문장 경계의 자연스러운 쉼인지 원본을 들어 확인하세요.',
        })
    if audio["signal"]["clipping_fraction"] > 0.001:
        feedback.insert(0, {"kind": "recording_quality", "message": "신호가 녹음 한계에 닿았습니다. 마이크 입력 볼륨을 낮춰 다시 녹음해 보세요."})
    report = {
        "schema_version": 1, "stage": "measurement_prototype",
        "recording": {"name": Path(path).name, "sha256": audio_hash(path)},
        "audio": audio, "reference_text": reference_text,
        "pronunciation_assessment": {"status": "not_assessed", "reason": "로컬 음향 지표만으로 음소 발음의 정확도를 판정할 수 없습니다."},
        "feedback": feedback,
        "limitations": [
            "이 버전은 학습 완료된 발음 교정 모델이 아닙니다. 휴지/음높이 측정값과 확인할 후보를 제공합니다.",
            "음높이 추정은 잡음·성대 특성에 영향을 받으며, 에너지 활동 검출은 말소리와 잡음을 구분하지 못합니다.",
            "미국 영어에도 다양한 정상 발음과 억양이 있습니다. 원어민과의 차이가 곧 오류는 아닙니다.",
        ],
    }
    if transcript is not None:
        report["transcript"] = inspect_transcript(transcript)
        report["transcript"]["source"] = "user_supplied"
    if reference_audio is not None:
        if not reference_text or not reference_text.strip():
            raise ValueError("참고 음성과 비교하려면 두 녹음에 공통인 --reference-text를 제공하세요.")
        reference = analyze_audio(reference_audio)
        comparison = {
            "status": "descriptive_only", "same_words": "user_asserted_not_verified",
            "alignment": "none", "reference_name": Path(reference_audio).name,
            "reference_audio": reference,
            "limitation": "같은 문장을 읽었다는 사용자 입력에 근거한 전체 구간 통계입니다. 음소/단어 정렬이나 정답 판정이 아닙니다.",
        }
        try:
            learner_features, reference_features = feature_vector(audio), feature_vector(reference)
            comparison["feature_differences_learner_minus_reference"] = {
                key: learner_features[key] - reference_features[key] for key in learner_features
            }
        except ValueError:
            comparison["status"] = "insufficient_periodic_audio"
        report["reference_comparison"] = comparison
    return report


def audio_hash(path: str | Path) -> str:
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def render_html(report: dict) -> str:
    esc = html.escape
    audio = report["audio"]
    contour = audio["pitch"]["contour"]
    duration = max(audio["duration_s"], .001)
    # Draw only adjacent reliable frames, never connect across an unvoiced gap.
    lines = []
    for a, b in zip(contour, contour[1:]):
        if b["time_s"] - a["time_s"] > .08:
            continue
        coords = []
        for p in (a, b):
            coords.extend([30 + 740 * p["time_s"] / duration,
                           110 - max(-18, min(18, p["semitones_from_median"])) * 4])
        lines.append(f'<line x1="{coords[0]:.2f}" y1="{coords[1]:.2f}" x2="{coords[2]:.2f}" y2="{coords[3]:.2f}"/>')
    pauses = "".join(f'<li>{esc(f["message"])}</li>' for f in report["feedback"])
    warnings = "".join(f'<li>{esc(x)}</li>' for x in report["limitations"] + audio["warnings"])
    observed = report.get("transcript")
    transcript_html = ""
    if observed:
        events = "".join(f'<li>{esc(e["kind"])}: {esc(e["text"])} (문자 {e["char_start"]}–{e["char_end"]})</li>' for e in observed["events"])
        transcript_html = f'<h2>전사에서 관찰한 후보</h2><p>{esc(observed["text"])}</p><ul>{events}</ul><p>{esc(observed["limitations"][0])}</p>'
    raw = esc(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SpeakingT 분석</title><style>body{{font:16px/1.7 system-ui,sans-serif;max-width:900px;margin:40px auto;padding:0 24px;background:#f5f5f0;color:#182f38}}h1{{font-size:36px}}section{{background:white;padding:24px;border-radius:16px;margin:20px 0}}svg{{width:100%;height:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}.tag{{color:#576870}}</style>
<p class="tag">SpeakingT · 측정 프로토타입</p><h1>내 말하기 관찰 기록</h1><p>{esc(report["recording"]["name"])} · {audio["duration_s"]:.2f}초</p>
<section><h2>음높이의 상대 변화</h2><p>화자 중앙값 기준 반음 단위, 세로 범위 ±18. 선이 없는 곳은 신뢰할 음높이를 추정하지 못한 구간입니다.</p>
<svg viewBox="0 0 800 240" role="img" aria-label="녹음 시간에 따른 상대 음높이"><path d="M30 25V200H770 M30 110H770" stroke="#ccd7d7" fill="none"/><g stroke="#187e79" stroke-width="2">{"".join(lines)}</g><text x="30" y="225">0초</text><text x="700" y="225">{duration:.1f}초</text><text x="0" y="114">0</text></svg></section>
<section><h2>다시 들어볼 구간</h2><ul>{pauses or '<li>설정한 기준의 내부 휴지 후보가 없습니다. 유창함을 보장하는 결과는 아닙니다.</li>'}</ul>{transcript_html}</section>
<section><h2>해석 범위</h2><ul>{warnings}</ul><p>음소 발음 평가: {esc(report["pronunciation_assessment"]["status"])}</p></section>
<details><summary>전체 측정값 · 참고 음성 비교 · 선택 평가 결과</summary><pre>{raw}</pre></details></html>'''


def save_report(report: dict, output: str | Path) -> tuple[Path, Path]:
    json_path = Path(output).with_suffix(".json")
    html_path = json_path.with_suffix(".html")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    html_path.write_text(render_html(report), encoding="utf-8")
    return json_path, html_path

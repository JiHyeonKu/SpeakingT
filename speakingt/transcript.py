"""Observed transcript candidates, never a complete acoustic disfluency detector."""

from __future__ import annotations

import re


def inspect_transcript(text: str) -> dict:
    tokens = list(re.finditer(r"[A-Za-z]+(?:['’][A-Za-z]+)?|[가-힣ㄱ-ㅎㅏ-ㅣ]+", text))
    events = []
    previous = None
    for token in tokens:
        word = token.group().lower()
        event = {"text": token.group(), "char_start": token.start(), "char_end": token.end()}
        if word in {"um", "uh", "uhm", "erm", "er", "hmm", "음", "어", "으음"}:
            events.append({**event, "kind": "filler_candidate"})
        if re.search(r"[가-힣ㄱ-ㅎㅏ-ㅣ]", word):
            events.append({**event, "kind": "korean_text_observed"})
        if previous and previous.group().lower() == word:
            events.append({**event, "kind": "adjacent_repetition_candidate"})
        previous = token
    return {
        "text": text,
        "english_token_count": sum(bool(re.match(r"[A-Za-z]", t.group())) for t in tokens),
        "events": events,
        "limitations": [
            "전사에서 관찰된 후보만 표시합니다. 전사 모델이 빠뜨린 음.., 반복, 한국어를 검출하지 못합니다.",
            "반복은 강조 또는 자연스러운 문법일 수 있습니다. 자동으로 오류로 판정하지 않습니다.",
            "문자 위치는 음성 타임스탬프가 아닙니다. 원본을 들어 확인하세요.",
        ],
    }


def transcribe_local(path: str, model: str, cache_dir: str) -> dict:
    """Opt-in model download; no audio is sent to a speech service."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ValueError('ASR 선택 기능을 설치하세요: python -m pip install -e ".[asr]"') from exc
    recognizer = WhisperModel(model, device="cpu", compute_type="int8", download_root=cache_dir)
    segments, info = recognizer.transcribe(
        path, task="transcribe", language=None, word_timestamps=True,
        vad_filter=False, condition_on_previous_text=False,
    )
    items = []
    for segment in segments:
        items.append({
            "start_s": segment.start, "end_s": segment.end, "text": segment.text,
            "words": [{"text": w.word, "start_s": w.start, "end_s": w.end,
                       "probability": w.probability} for w in (segment.words or [])],
        })
    return {
        "provider": "faster-whisper-local", "model": model,
        "language": info.language, "language_probability": info.language_probability,
        "text": "".join(s["text"] for s in items).strip(), "segments": items,
        "limitation": "ASR 단어 시점은 추정치입니다. 발음 채점/음소 정렬이 아니며 언어 혼용과 머뭇거림을 누락할 수 있습니다.",
    }

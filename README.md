# SpeakingT

원본 음성을 보존하면서 영어 말하기의 휴지·음높이·전사 후보를 관찰하는 개인 튜터 연구 프로젝트.

**현재는 측정 프로토타입입니다. 검증된 개인 발음 교정 모델은 아직 아닙니다.** 목표는 선택한 미국 영어 발음에 가까워지면서 이해하기 쉬운 말하기를 연습하는 것입니다. 하나의 원어민 억양을 유일한 정답으로 간주하지 않습니다.

## 구현한 것

- 정수 PCM WAV 입력, 녹음 품질 점검, 내부 휴지의 시간 위치.
- 자기상관 기반 기본주파수(F0) 추정과 화자 중앙값 기준 상대 음높이 그래프.
- 제공한 전사 또는 선택적 로컬 ASR에서 `um`, `uh`, 반복, 한국어 **관찰 후보** 표시.
- 같은 문장의 참고 음성과 전체 통계 비교. 단어/음소 정렬과 발음 정답 판정은 아직 없음.
- 원본을 덮어쓰지 않는 구간 추출, 선택적 마이크 녹음, 한국어 HTML/JSON 보고서.
- 선택적 Azure 읽기 발음 평가: 외부 서비스의 단어·음소 점수와 시간 정보. 실제 호출 검증 전.
- 사람의 유창성 평가를 학습하는 작은 선형 보정 모델: 화자 분리, 중복 차단, 조기 종료, 별도 시험 데이터와 평균 예측 기준선 비교.

파형 지표는 발음 정확도 점수가 아닙니다. 전사에 없는 머뭇거림은 정규식으로 복구할 수 없습니다. ASR의 단어 인식 확률도 발음 정확도가 아닙니다.

## 실행

Python 3.11 이상. 기본 기능은 NumPy만 필요합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m speakingt demo
```

데모 결과는 `reports/demo/synthetic_report.html`입니다. **합성 신호**로 소프트웨어 연결만 검증하며, 사람 음성 평가나 모델 학습 결과가 아닙니다.

```powershell
python -m speakingt analyze recordings/day01.wav --output reports/day01.json
python -m speakingt analyze recordings/day01.wav --transcript-file data/day01-verbatim.txt --output reports/day01.json
python -m speakingt clip recordings/day01.wav --start 2 --end 12 --output recordings/day01-sentence.wav
```

가상환경을 활성화하지 않았다면 위 `python` 대신 `.\.venv\Scripts\python.exe`를 사용합니다. 입력은 8–48kHz, 1–8채널, 8/16/24/32bit 정수 PCM WAV, 0.1–300초입니다. MP3/M4A/WebM/float WAV는 먼저 PCM WAV로 내보내야 합니다. 내부 휴지는 0.25초 이상, 짧은 에너지 공백은 0.12초까지 연결합니다. 잡음도 활동 구간에 포함될 수 있습니다.

같은 문장의 참고 녹음이 있을 때:

```powershell
python -m speakingt analyze recordings/mine.wav --reference-audio recordings/reference.wav --reference-text "I get that things come up." --output reports/comparison.json
```

두 녹음이 같은 문장인지는 사용자가 확인해야 합니다. 현재 비교는 전체 통계이고 음소별 정렬은 아닙니다.

## 선택 기능

마이크 입력은 명령을 실행한 시점부터 기록됩니다.

```powershell
python -m pip install -e ".[record]"
python -m speakingt record --seconds 20 --output recordings/day01-take01.wav
```

로컬 전사 모델은 첫 실행에 가중치를 다운로드합니다. 음성은 이 기능에서 외부 음성 서비스로 전송하지 않습니다. CPU에서 느릴 수 있습니다. 모델 크기는 필요에 따라 바꿉니다.

```powershell
python -m pip install -e ".[asr]"
python -m speakingt analyze recordings/day01.wav --asr small --output reports/day01.json
```

언어를 영어로 강제하거나 번역하도록 요청하지 않습니다. 그래도 ASR은 한국어 혼용, filler, 반복을 누락할 수 있습니다. 실제 녹음을 듣고 확인한 축어 전사가 평가용 기준입니다.

Azure 평가는 **`--azure`를 지정할 때만 원본 음성을 외부 유료 서비스에 전송**합니다. 별도 Azure Speech 리소스가 필요합니다. 키를 소스나 GitHub에 기록하지 말고 실행 환경의 `SPEAKINGT_AZURE_KEY`, `SPEAKINGT_AZURE_REGION`으로 제공합니다. `.env` 자동 로딩은 하지 않습니다.

```powershell
python -m speakingt analyze recordings/read-aloud.wav --reference-text "I get that things come up." --azure --output reports/read-aloud.json
```

이 어댑터는 en-US 읽기, PCM16/16kHz/mono, 30초 이내만 지원합니다. 자유 대화의 답안을 정답 원고로 강제 정렬하지 않습니다. 서비스 점수는 사람 평가의 정답이 아닙니다. 모의 요청 테스트만 완료했으며 실제 서비스 응답과 개인 음성의 타당성은 별도 검증이 필요합니다.

## 학습

큰 음성 모델을 처음부터 학습하지 않습니다. 현재 구현은 음향 특징 네 개로 **사람의 유창성 평가**를 예측하는 저비용 연구 기준선입니다. 음소 발음·억양 적절성·문법 평가는 수행하지 않습니다.

```powershell
python -m speakingt prepare-label reports/day01.json --speaker-id learner01 --recording-id day01-take01 --split train --output data/label-draft.jsonl
```

초안의 `target: null`, `label_source: "pending_human"`을 실제 녹음을 들은 사람이 채점한 0–100 점수와 `"human"`으로 바꿉니다. 모델은 그 진위를 자동 검증하지 못합니다. 여러 줄을 `data/labels.jsonl`에 합치고, **train/validation/test에 같은 화자를 중복 배치하지 않습니다.** 한 사람만으로는 이 일반화 검증을 할 수 없습니다. 개인의 날짜별 변화는 별도의 시간 순 시험으로 평가해야 합니다.

```powershell
python -m speakingt train data/labels.jsonl --output models/fluency.json --max-epochs 2000 --patience 100
python -m speakingt analyze recordings/new.wav --fluency-model models/fluency.json --output reports/new.json
```

입력 특징: 전체 길이 대비 에너지 활동 비율, 발화 구간 내 긴 휴지 비율, 발화 구간 분당 긴 휴지 수, 상대 음높이 90–10 백분위 범위. 신뢰할 음높이가 없는 녹음은 특징 생성에서 제외합니다. 이 제외율도 실제 실험 결과와 함께 확인해야 합니다.

학습 횟수는 상한이고, 검증 오차가 좋아지지 않으면 중단합니다. 시험 데이터는 특징/epoch 선택에 재사용하지 않습니다. 예측이 단순 평균보다 못할 수 있으며, 그 경우 유용한 모델을 얻었다고 주장하지 않습니다.

## 개발 및 현황

```powershell
python -m unittest discover -s tests -v
```

기본 테스트는 네트워크·API 키·마이크·모델 다운로드 없이 실행됩니다. 선택 ASR/마이크/외부 평가는 각 환경에서 추가 검증이 필요합니다.

- [개발 로드맵과 50달러 예산](docs/ROADMAP.md)
- [공식 근거·데이터·설계](docs/RESEARCH.md)
- [기존 훈련 템플릿에 연결하기](docs/DAILY_TRAINING.md)
- [현재 검증 상태](docs/STATUS.md)

`data/`, `recordings/`, `reports/`, `models/`, 키 파일 및 오디오 확장자는 Git에서 제외합니다. 이 저장소는 공개이므로 개인 녹음·전체 대화·키는 커밋하지 않습니다. 데이터와 모델의 라이선스는 각각 확인합니다.

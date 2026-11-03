# SpeakingT

**Phase 3:** WavLM Base+의 상위 Transformer 층을 실제 WAV로 fine-tuning합니다.
[JupyterLab 실행·복원·GPU 준비 안내](docs/PHASE3_GUIDE.md), [실제 CPU 검증 결과](experiments/phase3/README.md).
Windows에서 `start-phase3.cmd`를 실행하면 기본 20개 WAV / CPU / 1 epoch 검증을 시작할 수 있습니다.
새 커널 복원·중단 후 재개·103개 테스트를 검증했습니다. GPU 전체 학습과 성능 향상 검증은 아직 남아 있습니다.

기존 GOPT와 [Phase 2 Frozen WavLM](docs/PHASE2_GUIDE.md), `start-phase2.cmd`, 기존 checkpoint는 보존했습니다.
이전 기능 전체 사용법은 [START_HERE.md](START_HERE.md)에 있습니다.

영어 발음·유창성·운율 평가 모델을 직접 학습하고 동일한 화자 분할에서 성능을 검증하는 개인 연구 프로젝트입니다.
현재는 JupyterLab 기반 공개 데이터 실험을 우선하며 앱 제작은 범위에 포함하지 않습니다.

**공개 데이터의 평가 모델과 개인 음성 진단은 별도 과제입니다.** 개인 음성의 검증된 오류 진단은 아직 제공하지 않습니다.
문장 점수의 PCC를 음소 정확도나 개인 교정 효과로 환산하지 않습니다.

## 공개 발음 모델 직접 실행하기

[GOPT 학습 노트북](notebooks/SpeakingT_GOPT.ipynb)에서 실제 공개 음성·사람 점수를 확인하고,
공식 가중치로 예측한 뒤 epoch, learning rate 등을 바꿔 직접 학습할 수 있습니다.
Windows 프로젝트 폴더의 `start-gopt.cmd`를 실행하고 셀마다 Shift+Enter를 누릅니다.
설치·데이터 출처·재현 범위는 [실행 안내](docs/GOPT_WORKBENCH.md)에 있습니다.

[Colab에서 열기](https://colab.research.google.com/github/JiHyeonKu/SpeakingT/blob/main/notebooks/SpeakingT_GOPT.ipynb)

학습 그래프, CSV, 가중치 변화량과 epoch별 체크포인트를 로컬에 보존합니다.
이 단계는 공개 GOP 특징을 사용하는 평가 모델 실행입니다. 개인 WAV/M4A의 GOP 추출 연결은 아직 구현하지 않았습니다.

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

가상환경을 활성화하지 않았다면 위 `python` 대신 `.\.venv\Scripts\python.exe`를 사용합니다. 분석 입력은 8–48kHz, 1–8채널, 8/16/24/32bit 정수 PCM WAV, 0.1–300초입니다. 내부 휴지는 0.25초 이상, 짧은 에너지 공백은 0.12초까지 연결합니다. 잡음도 활동 구간에 포함될 수 있습니다.

M4A 등 다른 형식은 선택 기능으로 로컬 변환할 수 있습니다. ASR 선택 기능을 설치했다면 변환 라이브러리도 함께 설치됩니다.

```powershell
python -m pip install -e ".[audio]"
python -m speakingt import-audio "recording.m4a" --output recordings/imported.wav
python -m speakingt analyze recordings/imported.wav --output reports/imported.json
```

원본을 보존하고 mono/16kHz/PCM16 WAV와 `.wav.import.json` 변환 기록을 만듭니다. 원본과 출력의 SHA256, 입력 코덱, 변환 조건을 기록하며, 기존 출력 파일을 덮어쓰지 않습니다. 음량 정규화·무음 삭제·잡음 제거는 하지 않습니다. 시간 좌표는 변환 WAV의 첫 샘플을 기준으로 하며, 컨테이너의 시작 오프셋은 변환 기록에 남깁니다. 손실 압축에서 이미 사라진 음향 정보를 복원하는 기능은 아닙니다.

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

## 이전 선형 유창성 기준선 학습

GOPT 학습은 위의 노트북에서 실행합니다. 아래 CLI는 별도로 구현한 이전 기준선이며, 음향 특징 네 개로 **사람의 유창성 평가**를 예측합니다. 음소 발음·억양 적절성·문법 평가는 수행하지 않습니다.

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

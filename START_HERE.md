# SpeakingT 사용 설명서 — 이 문서부터 시작

확인 날짜: **2026-11-02**. 목표는 공개 발음 평가 모델을 활용해 개인 영어 말하기 도구를 만드는 것이다.
현재 공개 모델 실행·학습 환경과 원음 관찰 도구가 준비돼 있다. **개인 녹음을 GOPT에 넣어 자동 채점하는 연결은 아직 미완료**다.

다시 시작할 때는 **2번 → 3번**, 예전 결과만 볼 때는 **6번**, 개인 녹음을 관찰할 때는 **7번**으로 간다.

## 목차

1. [지금 할 수 있는 일과 설치 상태](#1-지금-할-수-있는-일과-설치-상태)
2. [PC를 다시 켰을 때 시작하기](#2-pc를-다시-켰을-때-시작하기)
3. [GOPT 노트북을 처음부터 실행하기](#3-gopt-노트북을-처음부터-실행하기)
4. [학습 설정과 결과를 읽는 법](#4-학습-설정과-결과를-읽는-법)
5. [실험을 비교하고 기록하기](#5-실험을-비교하고-기록하기)
6. [종료·재시작·예전 실험 복원](#6-종료재시작예전-실험-복원)
7. [내 음성 파일 관찰하기](#7-내-음성-파일-관찰하기)
8. [추가 기능과 이전 유창성 모델](#8-추가-기능과-이전-유창성-모델)
9. [파일 보관·백업·환경 재설치](#9-파일-보관백업환경-재설치)
10. [문제가 생겼을 때](#10-문제가-생겼을-때)
11. [지금부터 할 일과 완료 기준](#11-지금부터-할-일과-완료-기준)

## 1. 지금 할 수 있는 일과 설치 상태

### 기능 전체 목록

| 기능 | 현재 상태 / 사용 위치 | 해석 범위 |
|---|---|---|
| 공개 GOPT 가중치 불러오기 | 사용 가능. 노트북 4번 | 이미 학습된 모델의 음소·단어·문장 평가 출력 |
| 공개 음성 듣기·사람 점수 확인 | 사용 가능. 노트북 3번 | SpeechOcean762 학습자 음성. 원어민 정답 녹음이 아님 |
| 공개 특징/라벨 5,000개 검증 | 사용 가능. 노트북 2번 | 파일 해시, 발화 ID, 음소 순서, 사람 점수를 검사 |
| GOPT 직접 재학습 | 사용 가능. 노트북 5–7번 | 기존 구조를 사용해 작은 평가 모델의 가중치를 학습. 기반 음향 모델은 고정 |
| 실시간 그래프·CSV·체크포인트 | 사용 가능. 노트북 7–8번 | epoch별 오차와 실제 가중치 변경 확인 |
| TensorBoard·별도 최종 시험 | 선택 실행. 노트북 9번 | 기본값은 꺼짐. 최종 시험은 설정 선택을 마친 뒤 실행 |
| M4A 등 음성 가져오기 | 사용 가능. `import-audio` | 원본 보존, mono/16kHz/PCM16 WAV 변환, 해시/변환 기록 |
| 원음 품질·휴지·음높이 분석 | 사용 가능. `analyze` | 클리핑, 에너지 활동, 내부 휴지, F0 및 상대 음높이. 발음 정확도 점수는 아님 |
| HTML/JSON 보고서 | 사용 가능. `analyze` | HTML은 관찰용, JSON은 전체 측정값/선택 평가 저장용 |
| 로컬 ASR 전사 | 설치·실행 확인. `--asr small` | faster-whisper, CPU int8. 자동 전사는 사람이 확인해야 함 |
| filler·반복·한국어 표시 | 사용 가능. 전사 입력 시 | 텍스트에 남은 후보만 검출. 문자 위치와 음성 시점은 다름 |
| 참고 음성과 비교 | 사용 가능. `--reference-audio` | 같은 문장의 전체 통계 비교. 음소 정렬/오류 판정은 없음 |
| 음성 구간 추출 | 사용 가능. `clip` | 원본 보존, 기존 출력 덮어쓰기 거부 |
| 마이크 직접 녹음 | 코드 있음, **sounddevice 미설치·장치 미검증** | 설치 후 `record`. 휴대폰 녹음 파일 가져오기는 지금 가능 |
| 작은 선형 유창성 모델 | 실행 가능. `prepare-label`, `train`, `--fluency-model` | 과거 비교 기준선. GOPT와 별개이며 발음 교정 모델이 아님 |
| 공개 음성 300개 파일럿 준비 | 스크립트/기록 있음 | 과거 선형 모델 재현용. 현재 GOPT 실험에 필수 아님 |
| 영문 청취 평가 양식 | 기존 세션의 로컬 HTML 있음 | 수동 작성·JSON 다운로드. 자동 저장/자동 학습 반영 없음 |
| Azure 읽기 발음 평가 | 어댑터·모의 테스트 있음, **실제 호출 미검증** | 계정/키와 유료 외부 전송을 별도로 설정해야 함 |
| 내 WAV/M4A → GOPT 채점 | **미구현** | 원본과 동일한 Kaldi 정렬/GOP 특징 추출을 연결해야 함 |
| 음소 오류 설명·억양 교정 지시·음성 기반 filler 완전 검출 | **미완료** | 점수 예측과 실제 교정 근거는 추가 검증 필요 |
| 오후 8시 알림 자동 연결·누적 개인 대시보드 | **미구현** | 현재는 파일과 노트북으로 실행 |

### 현재 PC의 개발 환경

2026-11-02에 프로젝트 `.venv`에서 직접 확인했다. **현재 PC에서 공개 모델 실험을 시작하기 위한 재설치는 필요 없다.**

| 구성 | 설치 확인 값 |
|---|---|
| Python | 3.11.4, Windows 64bit |
| NumPy / PyTorch | 2.4.6 / 2.15.0+cpu |
| JupyterLab / ipykernel | 4.6.5 / 7.4.0 |
| Matplotlib / TensorBoard | 3.11.2 / 2.21.0 |
| nbclient | 0.11.0, 노트북 전체 실행 검증용 |
| PyAV / faster-whisper | 18.1.0 / 1.2.1 |
| GPU | 이 가상환경의 CUDA 사용 불가. CPU로 실행 |
| 데이터·공식 가중치 | `data/gopt/`에 준비됨 |
| 의존성 | `python -m pip check` → `No broken requirements found.` |

검증 기록: 실제 공개 자료 2 epoch 학습, 노트북 전체 실행, 로컬 테스트 84개, GitHub 자동 검사 통과.
Colab/GPU와 실제 마이크 장치는 아직 검증하지 않았다. 설치 완료와 개인 발음 평가 도구 완성은 별도 단계다.

## 2. PC를 다시 켰을 때 시작하기

1. 탐색기에서 **SpeakingT 프로젝트 폴더**를 연다.
2. `start-gopt.cmd`를 더블클릭한다. 실행 창은 켜둔다.
3. 브라우저의 JupyterLab에서 `notebooks/SpeakingT_GOPT.ipynb`가 열린다.
4. 오른쪽 위 커널이 `Python 3 (ipykernel)`인지 확인한다.
5. 맨 위부터 셀을 선택하고 **Shift+Enter**로 진행한다. 설명 셀은 다음으로 넘어가고, 코드 셀은 실행된다.

브라우저가 자동으로 열리지 않으면 실행 창에 표시된 `http://localhost:.../lab?token=...` 주소를 연다.
포트 번호와 token은 실행 때 달라질 수 있으므로 예전 주소를 고정해서 쓰지 않는다. token 주소는 공유하지 않는다.
파일을 GitHub에서 보는 것만으로는 실행되지 않는다. 실행 창을 닫거나 PC를 끄면 서버/커널도 종료될 수 있다.

명령을 사용하려면 탐색기의 **프로젝트 폴더 주소창에 `powershell`을 입력**한다.
아래 PowerShell 명령은 모두 프로젝트 폴더에서 실행한다. 가상환경 활성화는 필요 없다.

```powershell
.\start-gopt.cmd
```

이후 PowerShell 예제는 `.\.venv\Scripts\python.exe`를 명시하므로 시스템 Python과 혼동하지 않는다.
Python 노트북 예제는 Jupyter의 코드 셀에 넣는다. 두 종류의 코드를 서로 바꾸어 붙여 넣지 않는다.

## 3. GOPT 노트북을 처음부터 실행하기

먼저 **공식 모델로 예측만 해봐도 된다.** 재학습은 모델 동작을 이해하고 실험하기 위한 선택 경로다.
완성될 개인 도구를 쓸 때마다 재학습할 필요는 없다.

| 노트북 구역 | 내가 할 일 | 정상 실행 시 보이는 것 |
|---|---|---|
| 1. 실행 환경 | 코드 실행 | Python/Torch 버전, Workspace 경로, `CUDA: False`도 정상 |
| 2. 공개 자료 | 코드 실행 | 해시/라벨 검증 결과, `Features: (2500, 50, 84)` |
| 3. 음성·사람 점수 | 재생 버튼으로 듣고 출력 확인 | 발화 ID, 텍스트, 사람 문장·음소 점수 |
| 4. 학습된 모델 | 코드 실행 | GOPT 구조, 파라미터 26,577개, 모델 예측 |
| 5. 학습 설정 | 처음에는 기본값 유지 | 학습 2,000개, 검증 500개, 화자 중복 `set()` |
| 6. 역전파 데모 | 코드 실행 | loss, gradient norm, weight change. 데모 모델만 변경됨 |
| 7. 실제 학습 | 코드 실행 | epoch 0부터 그래프, 실험 폴더명, 마지막 summary |
| 8. 체크포인트 비교 | 코드 실행 | 동일한 검증 샘플의 학습 전/후 예측과 사람 점수 |
| 9. 선택 기능 | 첫 실행은 둘 다 `False` 유지 | TensorBoard/최종 시험을 실행하지 않음 |

공개 자료 최초 다운로드는 약 190 MB의 고정 파일과 선택한 예제 WAV다.
전체 5,000개 원음이 모두 다운로드된다는 뜻은 아니다. 기존 파일이 있으면 검사 후 재사용한다.

`SAMPLE_INDEX = 0`을 0–2499 범위의 다른 값으로 바꾸면 다른 공개 음성을 확인할 수 있다.
3번과 4번의 코드를 순서대로 다시 실행한다. 이 샘플은 공식 모델의 학습에 포함됐으므로 일반화 성능의 근거로 쓰지 않는다.

실제 계산 흐름:

```text
이미 추출된 음향 특징 + 고정 음소 ID
                 ↓
         기존 GOPT Transformer
                 ↓
       음소 / 단어 / 문장 점수 예측
                 ↓ 재학습할 때만
   사람 점수와의 차이(MSE) → 역전파 → 가중치 변경
```

노트북의 재학습은 **기존 GOPT 구조를 새 가중치로 학습**한다. 공식 가중치는 공식 train 전체를 이미 봤으므로
그 가중치에서 시작하면 이번에 떼어낸 validation이 독립적이지 않다. 원본 음향 모델까지 처음부터 학습하지는 않는다.

## 4. 학습 설정과 결과를 읽는 법

### 기본 설정

| 설정 | 기본값 | 의미 / 처음 바꿀 때 |
|---|---:|---|
| `MAX_EPOCHS` | 10 | 학습 자료를 반복해서 사용하는 최대 횟수. 반드시 10회까지 간다는 뜻은 아님 |
| `LEARNING_RATE` | 0.001 | 한 번의 가중치 갱신 크기에 영향을 줌. 비교 실험은 0.0003 한 가지만 추가 |
| `BATCH_SIZE` | 25 | 한 번에 처리하는 발화 수. 첫 비교에서는 유지 |
| `SEED` | 7 | 초기 가중치·학습 순서·검증 화자 선택에 영향. 같은 분할로 비교하려면 유지 |
| `PATIENCE` | 5 | 검증 음소 MSE가 5회 연속 개선되지 않으면 조기 종료 |
| `DEVICE` | 자동 선택 | 현재 PC는 `cpu`. `cuda`를 강제로 쓰지 않음 |

5번 설정을 고쳤다면 **5번 코드를 다시 실행**해야 적용된다. 이어 6–8번을 실행한다.
7번만 다시 실행하면 현재 커널에 남아 있는 설정으로 **새 실험**이 시작된다.
epoch 수를 늘리는 것만으로 모델의 유용성이 보장되지는 않는다.

### 그래프와 숫자

| 항목 | 읽는 방법 |
|---|---|
| Train phone MSE | 학습 자료의 음소 점수 제곱 오차. 낮을수록 좋음 |
| Validation phone MSE | 학습에 사용하지 않은 화자의 음소 오차. **best.pt 선택 기준** |
| Validation Pearson r / PCC | 사람 점수와 예측의 상관, -1~1. 높을수록 같은 방향으로 변하는 경향. 정답률/퍼센트가 아님 |
| Prosody PCC | 문장 운율 점수와의 상관. 어느 음절을 어떻게 교정할지는 알려주지 않음 |
| Weight change L2 | 처음 대비 가중치 변화량. 0보다 크면 실제 변경됨. 클수록 좋은 모델이라는 뜻은 아님 |
| Epoch 0 | 무작위 가중치의 학습 전 기준선 |
| `null` / 빈 PCC | 표본이 하나이거나 점수에 변화가 없어 상관을 계산할 수 없는 경우 |

학습 오차는 내려가는데 검증 오차가 올라가면 과적합 가능성을 먼저 살펴본다.
단일 샘플의 예측 점수가 올라갔다고 성능이 좋아진 것은 아니다. 사람 점수에 가까워졌는지와 전체 검증 지표를 함께 본다.

사람 라벨은 음소 0–2, 단어/문장 0–10이다. 모델의 원시 회귀 출력은 범위를 벗어날 수 있으며 현재 임의로 잘라 숨기지 않는다.
특히 학습 전 예측은 부정확할 수 있다. 점수는 미국인 유사도, CEFR 등급, 실제 영어 실력 향상률이 아니다.

공식 가중치의 공개 test 결과는 음소 PCC 약 0.6183, 운율 PCC 약 0.7648이다.
완전성 PCC는 약 0.0869로 낮다. 모든 평가 항목이 같은 수준으로 검증된 것은 아니다.
이 수치는 중국어 모국어 학습자의 읽기 자료에서 나온 것이며 개인 한국어 화자·자유 발화에 대한 검증은 남아 있다.
[전체 결과와 제약](experiments/gopt/reproduction.json)을 참고한다.

## 5. 실험을 비교하고 기록하기

처음에는 다음 두 번이면 충분하다. **최종 test는 두 실험 모두 꺼둔다.**

| 실험 | epoch 상한 | learning rate | seed / batch / patience | 목적 |
|---|---:|---:|---|---|
| A | 10 | 0.001 | 7 / 25 / 5 | 기본 실행 확인 |
| B, 선택 | 10 | 0.0003 | 7 / 25 / 5 | 한 조건만 바꿨을 때 검증 결과 비교 |

각 학습은 `runs/gopt/run-날짜-시간-고유값/`에 따로 저장된다. 화면에 출력된 경로를 기록한다.
새 실행은 이어 학습이 아니라 새 가중치에서 시작한다.

`reports/experiment-log.md`를 만들어 다음 양식을 실험마다 복사한다. 이 파일은 로컬 보관 대상이다.

```markdown
## 실험 A
- 날짜 / Git commit:
- 목적과 바꾼 조건:
- run 폴더:
- epoch 상한 / learning rate / seed / batch / patience:
- 완료 상태 / 실제 완료 epoch / best epoch:
- best epoch의 validation phone MSE / phone PCC / prosody PCC:
- test 실행 여부: 아니오
- 관찰한 점 / 다음 결정:
```

`summary.json`의 best epoch와 같은 번호의 `metrics-NNN.json`을 보아야 **선택된 모델**의 결과를 기록할 수 있다.
마지막 epoch와 best epoch는 다를 수 있다. GOPT와 예전 선형 모델은 데이터·점수 단위가 다르므로 숫자를 직접 비교하지 않는다.

여러 실험을 한 화면에서 보려면 노트북 9번의 `OPEN_TENSORBOARD = True`로 바꾸고 그 셀을 실행한다.
또는 프로젝트 폴더의 별도 PowerShell에서 아래 명령을 실행하고 출력되는 로컬 주소를 연다.

```powershell
.\.venv\Scripts\python.exe -m tensorboard.main --logdir runs/gopt --host 127.0.0.1
```

모델과 설정 선택을 끝낸 뒤에만 원하는 run을 대상으로 `RUN_FINAL_TEST = True`를 실행한다.
`best.pt`의 최종 결과가 `final-test.json`에 기록되고 `summary.json`의 `test_evaluated`가 바뀐다.
이 결과를 보고 같은 test에 맞춰 설정을 계속 고르면 독립 시험으로 해석할 수 없다.
공식 공개 가중치의 기존 공개 결과와, 이번에 직접 학습한 모델의 최종 시험은 서로 다른 기록이다.

## 6. 종료·재시작·예전 실험 복원

### 안전하게 끝내기

1. 학습 중이면 ■ 버튼으로 중단한다. 완료된 epoch의 기록은 보존된다.
2. 마지막 출력의 `RUN` 경로를 실험 일지에 복사한다.
3. 노트북을 저장한다. 출력에 개인 정보가 있으면 GitHub에는 올리지 않는다.
4. JupyterLab **File → Shut Down** 또는 실행 창에서 Ctrl+C로 서버를 종료한다.

브라우저 탭을 닫는 것만으로 계산이 종료되는 것은 아니다. 강제 종료나 PC 절전은 진행 중 epoch를 잃게 할 수 있다.
체크포인트에는 optimizer 상태도 저장하지만 **현재 이어 학습 기능은 구현하지 않았다**.
다시 실행한 학습은 새 run이며, 과거 모델은 평가·비교용으로 다시 불러올 수 있다.

### 저장 파일

| 파일 | 용도 |
|---|---|
| `config.json` | 학습 조건, 표준화, 코드/데이터 해시, Torch/NumPy 버전 |
| `split.json` | 학습/검증 화자와 발화 ID |
| `metrics.csv` | epoch별 요약. Excel이나 텍스트 편집기로 열 수 있음 |
| `metrics-000.json`, `metrics-001.json`, … | 해당 epoch의 음소·단어·문장별 상세 결과 |
| `epoch-000.pt`, `epoch-001.pt`, … | 각 epoch의 모델 상태 |
| `best.pt` | 검증 음소 MSE가 가장 좋은 학습 epoch의 모델 |
| `summary.json` | 완료 상태, 실제 epoch 수, 선택된 epoch |
| `tensorboard/` | 그래프 로그 |
| `final-test.json` | 최종 시험을 직접 실행했을 때만 생성 |

### 학습 없이 이전 결과 다시 보기

Jupyter를 다시 열고 **1번 실행 환경 코드만 실행**한다. 새 코드 셀을 추가해 다음 Python 코드를 실행한다.

```python
import csv, json
from pathlib import Path
from IPython.display import display

available_runs = sorted((ROOT / 'runs/gopt').glob('run-*'))
print('\n'.join(p.name for p in available_runs))
# 위 목록에서 확인한 이름으로 바꾼다. 최근 run이 항상 원하는 실험은 아니다.
RUN = ROOT / 'runs/gopt' / 'run-여기에-실제-이름'
print((RUN / 'config.json').read_text())
print((RUN / 'summary.json').read_text())
with (RUN / 'metrics.csv').open(encoding='utf-8') as f:
    saved_history = [
        {key: (None if value == '' else float(value)) for key, value in row.items()}
        for row in csv.DictReader(f)
    ]
display(workbench.plot_history(saved_history))
```

같은 run의 검증 샘플과 가중치를 다시 평가하려면 이어서 실행한다. **새 학습은 하지 않는다.**

```python
cfg = json.loads((RUN / 'config.json').read_text())
all_train = workbench.load_data(DATA, 'train')
train_data, validation_data = workbench.split_training(
    all_train, cfg['seed'], cfg['validation_speakers'])
model, norm = workbench.load_checkpoint(RUN / 'best.pt')
result = workbench.evaluate(model, validation_data, norm)
print(json.dumps(result, indent=2))
```

첫 epoch 전에 중단한 run에는 `best.pt`가 없을 수 있다. 이때는 존재하는 `epoch-000.pt`로 학습 전 상태만 확인한다.
복원한 run에 최종 시험을 할 경우 9번 셀의 `DEVICE`도 `cpu`로 정의해야 한다.
복원 뒤 **7번 학습 셀을 실행하면 `RUN`이 새 실험으로 바뀐다**는 점에 주의한다.

## 7. 내 음성 파일 관찰하기

이 경로는 지금 사용할 수 있다. GOPT 채점과는 아직 연결되지 않았으며, 원음의 휴지·음높이·전사 후보를 확인한다.

### 7-1. 파일 가져오기 → 측정 → 보고서 열기

프로젝트 폴더에서 PowerShell을 열고 다음 블록을 실행한다.
질문이 나오면 원본 M4A/WAV 파일의 **전체 경로를 따옴표 없이** 입력한다.

```powershell
$sourceAudio = Read-Host '분석할 원본 음성 파일의 전체 경로'
$sessionId = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$audioOut = "recordings/$sessionId/input.wav"
$reportOut = "reports/$sessionId/analysis.json"
.\.venv\Scripts\python.exe -m speakingt import-audio "$sourceAudio" --output "$audioOut"
if ($LASTEXITCODE -ne 0) { throw '가져오기 실패: 오류를 확인하세요.' }
.\.venv\Scripts\python.exe -m speakingt analyze "$audioOut" --output "$reportOut"
if ($LASTEXITCODE -ne 0) { throw '분석 실패: 오류를 확인하세요.' }
Start-Process "reports/$sessionId/analysis.html"
```

생성 파일: 분석용 WAV, `.wav.import.json`, `analysis.json`, `analysis.html`.
가져오기는 원본·기존 출력을 덮어쓰지 않는다. 반면 **analyze 보고서는 같은 출력 경로를 쓰면 덮어쓴다**.
위 예제는 새 세션 이름을 사용해 결과를 구분한다.

분석 입력은 정수 PCM WAV, 8–48kHz, 1–8채널, 0.1–300초다. 가져오기는 16kHz mono PCM16으로 통일한다.
음량 정규화·무음 삭제·잡음 제거는 수행하지 않는다. 긴 녹음은 적절한 구간을 먼저 준비한다.

### 7-2. 자동 전사도 함께 보기

같은 PowerShell 세션에서 앞의 `$audioOut`, `$sessionId` 변수를 그대로 사용한다.

```powershell
.\.venv\Scripts\python.exe -m speakingt analyze "$audioOut" --asr small --output "reports/$sessionId/with-asr.json"
Start-Process "reports/$sessionId/with-asr.html"
```

첫 사용 시 선택 모델을 다운로드하며 캐시는 `.cache/asr/`에 저장한다. 현재 `small`로 개인 음성 전사 실행 이력이 있다.
추론은 로컬 CPU에서 이루어진다. JSON의 `asr.segments`에는 추정 단어/구간 시점이 있고,
`transcript.events`는 텍스트에 실제 남은 filler·연속 반복·한국어 후보다. ASR은 이를 누락할 수 있다.

직접 듣고 확인한 전사를 UTF-8 텍스트 파일로 저장했다면 다음과 같이 사용한다.
`data/verbatim.txt`는 먼저 작성해야 하는 예시 경로다. 모범 답안 대신 실제로 말한 내용과 머뭇거림을 적는다.

```powershell
.\.venv\Scripts\python.exe -m speakingt analyze "$audioOut" --transcript-file data/verbatim.txt --output "reports/$sessionId/verified-transcript.json"
```

`--asr`와 `--transcript-file`은 동시에 사용할 수 없다. 자동 전사 파일이 별도 TXT로 자동 생성되지는 않으며 전체 결과는 JSON에 있다.

### 7-3. 그래프 확인·구간 재생

HTML에서 낮은 에너지 구간의 초 단위 시점과 상대 음높이를 확인한다. 원본을 다시 듣고 자연스러운 쉼인지 판단한다.
F0 선이 없는 구간은 신뢰할 음높이가 추정되지 않은 곳이며 발음 오류를 뜻하지 않는다.
일반 `analyze` HTML에는 오디오 플레이어가 없으므로 WAV를 재생기로 열거나 구간을 추출한다.

```powershell
Start-Process "$audioOut"
# 녹음 길이가 12초 이상일 때만 아래 2~12초 예제를 사용한다.
.\.venv\Scripts\python.exe -m speakingt clip "$audioOut" --start 2 --end 12 --output "recordings/$sessionId/segment-02-12.wav"
Start-Process "recordings/$sessionId/segment-02-12.wav"
```

### 7-4. 같은 문장의 참고 녹음과 비교

`recordings/mine.wav`, `recordings/reference.wav`는 실제로 준비한 두 WAV의 경로로 바꾼다.
`--reference-text`에는 **두 사람이 공통으로 읽은 문장**을 넣는다. 다른 내용을 말한 자유 답변과 모범 답안 녹음은 이 비교에 쓰지 않는다.

```powershell
.\.venv\Scripts\python.exe -m speakingt analyze recordings/mine.wav --reference-audio recordings/reference.wav --reference-text "I get that things come up." --output "reports/$sessionId/comparison.json"
```

전체 통계의 차이는 HTML 맨 아래 상세 결과 또는 JSON `reference_comparison`에서 확인한다.
단어 시점 정렬, 발음 정오 판정, 자동 교정 문장 생성은 수행하지 않는다.

### 7-5. 이미 받은 청취 의견

기존 세션의 `listener-review.html`은 음성이 포함된 별도 청취 양식이다.
작성 내용은 자동 저장되지 않으므로 닫기 전에 **Download notes**로 JSON을 저장한다.
기존에 받은 의견은 지시 해석/오류 여부가 미확인인 참고 관찰이며 정답 라벨로 자동 전환하지 않는다.
이 양식 생성이나 새 JSON 자동 가져오기를 위한 일반 CLI는 아직 없다. 기존 세션의 개별 산출물이다.
개인 파일 위치는 로컬 전용 [내 파일 바로가기](reports/START_HERE_LOCAL.md)에 정리한다. 이 링크는 GitHub에는 존재하지 않는다.

## 8. 추가 기능과 이전 유창성 모델

### 마이크 직접 녹음 — 선택 설치 후 장치 확인 필요

현재 `sounddevice`는 설치하지 않았다. 필요할 때만 아래 순서로 사용한다.
두 번째 명령을 실행하는 순간부터 20초간 기본 마이크를 녹음한다.

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[record]"
$takeId = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
.\.venv\Scripts\python.exe -m speakingt record --seconds 20 --output "recordings/$takeId/microphone.wav"
```

### 합성 데모 — 설치 연결 확인용

```powershell
.\.venv\Scripts\python.exe -m speakingt demo --output-dir reports/demo
Start-Process reports/demo/synthetic_report.html
```

이 데모의 소리는 합성 신호다. 실제 사람 음성 성능이나 모델 학습 결과로 기록하지 않는다.

### 이전 선형 유창성 모델 — 선택적 비교 기준선

`python -m speakingt train`은 **GOPT 학습 명령이 아니다**. 네 가지 음향 지표로 사람의 유창성 점수를 예측하는 선형 모델이다.
현재 주 실험은 GOPT 노트북이다. 이전 파일럿은 시험 MAE 10.097/100, 중앙값 기준선 10.217로 개선이 작았다.

준비된 과거 모델을 적용하려면 실제 WAV 경로로 바꾸어 실행한다.

```powershell
.\.venv\Scripts\python.exe -m speakingt analyze recordings/mine.wav --fluency-model experiments/fluency-pilot/model.json --output reports/baseline-observation.json
```

새 사람 평가 자료로 이 선형 모델을 실험하는 절차:

1. 실제 WAV를 `analyze`로 분석한다.
2. 아래 명령으로 JSONL 초안을 만든다. 출력은 기존 파일과 다른 이름을 쓴다.
3. 실제 녹음을 평가한 사람이 `target`에 유창성 0–100, `label_source`에 `human`을 기록한다. 단순 의견을 임의 점수로 바꾸지 않는다.
4. train/validation/test를 화자가 겹치지 않게 준비한다. 한 사람의 녹음 두 개로 이 검증을 대체할 수 없다.
5. JSONL을 합친 파일로 학습한다. 신뢰할 pitch가 없는 입력은 제외/거부될 수 있다.

```powershell
.\.venv\Scripts\python.exe -m speakingt prepare-label reports/example.json --speaker-id speaker01 --recording-id example01 --split train --output data/label-draft.jsonl
.\.venv\Scripts\python.exe -m speakingt train data/labels.jsonl --output models/fluency-new.json --max-epochs 2000 --patience 100
```

위 경로의 분석 보고서/완성된 labels 파일은 먼저 준비해야 한다.
공개 300개 음성 파일럿의 다운로드·선택·재학습 명령은 [파일럿 재현 절차](experiments/fluency-pilot/README.md#재현)에 있다.
GOPT 준비 스크립트는 `scripts/prepare_gopt.py`, 과거 파일럿 준비 스크립트는 `scripts/prepare_speechocean.py`다.

### Azure 읽기 평가 — 아직 실사용 검증 전

현재 필수 경로가 아니다. 별도 Azure Speech 리소스, 지역, 키를 준비하고 해당 실행 세션의
`SPEAKINGT_AZURE_KEY`, `SPEAKINGT_AZURE_REGION` 환경변수에 설정해야 한다. `.env` 자동 로딩은 없다.
키는 노트북·문서·Git에 적지 않는다. 이 저장소의 코드/테스트 존재가 계정 설정 완료를 의미하지 않는다.

환경 설정을 마친 뒤의 호출 형식:

```powershell
.\.venv\Scripts\python.exe -m speakingt analyze recordings/read-aloud.wav --reference-text "I get that things come up." --azure --output reports/azure-read-aloud.json
```

`--azure` 실행은 음성과 원고를 외부 서비스로 보내며 비용이 발생할 수 있다.
어댑터 입력은 en-US 읽기, PCM16/16kHz/mono, 0.1–30초, 최대 2 MiB다.
실제 응답·요금·개인 음성 타당성은 사용 전에 확인해야 하며 현재 모의 요청 테스트만 통과했다.
결과는 JSON `pronunciation_assessment`와 HTML 상세 결과에 나타난다. 모든 응답에 운율 점수가 제공된다고 가정하지 않는다.

## 9. 파일 보관·백업·환경 재설치

### 폴더 지도

| 경로 | 내용 | GitHub 저장 여부 |
|---|---|---|
| `START_HERE.md`, `README.md`, `docs/` | 사용법·설계·상태 | 공개 |
| `notebooks/SpeakingT_GOPT.ipynb` | 기본 학습 노트북 | 공개. 개인 출력 없이 유지 |
| `speakingt/`, `scripts/`, `tests/` | 코드·준비 스크립트·테스트 | 공개 |
| `experiments/` | 공개 자료 실험 요약·출처·해시·과거 작은 모델 | 공개 |
| `data/gopt/` | 현재 사용할 GOPT 특징·라벨·공식 가중치 | 제외 |
| `data/speechocean762-pilot/` | 과거 선형 모델용 데이터 | 제외 |
| `recordings/`, `reports/`, `models/` | 개인 음성·보고서·로컬 모델 | 제외 |
| `runs/gopt/` | 직접 수행한 실험 로그와 체크포인트 | 제외 |
| `.cache/`, `.venv/`, `.jupyter/` | 캐시·설치 환경·서버 설정 | 제외 |

현재 작업 데이터는 **`data/gopt/`**다. 다른 `gopt-*` 폴더가 남아 있다면 준비 과정에서 검토한 자료다.
특징 차원만 같다고 서로 섞지 않는다.

GitHub에는 개인 기록과 run이 자동 백업되지 않는다. 필요한 `recordings`, `reports`, `runs`, `models`와
원본 녹음은 별도의 개인 백업에 보관한다. 현재 프로젝트가 OneDrive 경로에 있으므로 클라우드 동기화 여부는
사용자의 OneDrive 설정에 따른다. Git 제외는 OneDrive 동기화 제외를 뜻하지 않는다.

노트북을 수정·저장하면 실행 출력도 파일에 들어갈 수 있다. 개인 출력은 로컬 `reports/` 아래 사본에 보관한다.
코드를 공유할 때는 `git status --short`로 범위를 확인하고, 노트북 출력 제거 후 필요한 파일만 선택한다.
서버 token·API 키·개인 피드백은 공개 커밋에 넣지 않는다.

실험 재현 기록을 더 남기고 싶으면 프로젝트 PowerShell에서:

```powershell
git rev-parse HEAD
.\.venv\Scripts\python.exe -m pip freeze
```

출력은 개인 실험 일지와 함께 보관한다. pip freeze에 로컬 경로가 포함될 수 있으므로 그대로 공개하지 않는다.

### 새 PC 또는 가상환경이 사라졌을 때만 재설치

Git과 Python 3.11을 설치한 환경에서 저장소를 clone하고 프로젝트 폴더에서 실행한다.
현재 PC에서는 아래 재설치를 반복할 필요가 없다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[research,audio,asr]"
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/prepare_gopt.py
.\start-gopt.cmd
```

새 설치의 패키지 버전은 시점에 따라 달라질 수 있다. 기존 run의 버전 기록과 비교하고 재현 수치 차이를 기록한다.
가상환경을 다른 PC에서 복사해 사용하기보다 그 PC에서 다시 설치한다.
개인 녹음·과거 run은 Git clone으로 돌아오지 않으므로 별도 백업에서 복원해야 한다.

[Colab 노트북](https://colab.research.google.com/github/JiHyeonKu/SpeakingT/blob/main/notebooks/SpeakingT_GOPT.ipynb)은 대안이다.
현재 검증된 기본 경로는 로컬 Windows CPU다. Colab의 `/content` 파일은 런타임 종료로 사라질 수 있어 별도 저장이 필요하다.
유료 GPU 가입이나 Drive 연결은 이 프로젝트의 필수 조건이 아니다.

## 10. 문제가 생겼을 때

| 증상 | 확인할 것 / 조치 |
|---|---|
| 브라우저에 접속할 수 없음 | `start-gopt.cmd`로 서버를 시작하고 이번 실행 창의 주소를 사용 |
| `python`/패키지를 못 찾음 | 프로젝트 폴더인지 확인. `.\.venv\Scripts\python.exe`를 사용. 사라진 환경만 재설치 |
| 노트북 `NameError: ROOT/DATA/config/RUN` | 커널 재시작으로 변수가 사라진 상태일 수 있음. 앞의 필요한 셀부터 순서대로 실행. 예전 run은 6번 복원 절차 사용 |
| `ModuleNotFoundError` | 첫 셀에서 `print(sys.executable)`로 프로젝트 `.venv`인지 확인. 해당 환경에 research/audio/asr 선택 기능 설치 |
| 왼쪽 실행 번호가 `[*]` | 계산 중. 처음 다운로드/ASR은 기다리거나 ■로 중단. 같은 셀을 연달아 누르지 않음 |
| `CUDA: False` | 현재 설치에서는 정상. CPU 사용 |
| 10 epoch 전에 종료 | `summary.json`의 `status` 확인. `early_stopped`이면 patience에 따른 정상 종료 |
| `Cached asset changed` / `hash mismatch` | 어떤 파일인지 기록하고 원본 잠금 목록과 대조. 검사 우회를 하지 말고 해당 캐시를 별도로 보존한 뒤 원본 재다운로드 |
| 다운로드 실패 | 인터넷·오류 URL 확인. 고정 출처를 임의 다른 모델로 교체하지 않음. 캐시가 있는 자료는 재사용 |
| `best.pt` 없음 | 첫 학습 epoch 완료 전 중단했는지 확인. `epoch-000.pt`는 학습 전 모델 |
| 보고서에서 음소 발음 `not_assessed` | 일반 `analyze`의 정상 상태. GOPT 자동 연결이 아직 없음 |
| 전사에 `um`/한국어가 없음 | 모델이 누락했을 수 있음. 원음 청취와 확인된 축어 전사 사용 |
| 음높이 데이터가 부족함 | 무성음·잡음·조용한 신호에서 추정 실패 가능. 오류 발음이라고 단정하지 않음 |
| 원본/기존 파일 덮어쓰기 오류 | `import-audio`/`clip`/`record`의 출력 이름을 새로 지정 |
| 마이크 오류 | sounddevice 설치, 기본 입력 장치, Windows 마이크 권한 확인 |
| 결과가 다르게 나옴 | seed·분할·learning rate·정규화·가중치·코드 commit·환경 버전을 비교 |

환경 점검 명령:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m speakingt --help
```

코드를 수정했거나 동작 이상을 확인할 때만 필요한 테스트를 실행한다. 매 실험마다 전체 테스트를 반복할 필요는 없다.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

문제 기록에는 실패한 명령/셀, 오류 메시지, run 폴더, `config.json`, 변경한 조건을 남긴다. 키/token은 제외한다.

## 11. 지금부터 할 일과 완료 기준

날짜를 고정하기보다 작업 회차별로 진행한다. 아래 시간은 사람이 읽고 확인할 시간을 포함한 계획용 추정이다.
GPU/API 비용을 쓰기 위해 실험을 늘릴 필요는 없다. 현재 지출은 $0, 외부 지출 상한은 $50다.

| 순서 | 할 일 | 담당 / 예상 시간 | 완료 기준 |
|---|---|---|---|
| 1. 첫 사용 | 노트북 1–4번 실행, 공개 음성 2개 듣기, 사람 점수/예측 확인 | 사용자 / 15–20분 | 입력·정답·예측이 어디에 있는지 찾을 수 있음 |
| 2. 학습 한 번 | 기본 10 epoch 상한으로 5–8번 실행, run 경로 기록 | 사용자 / 15–20분 | 그래프와 best epoch, 저장된 가중치를 확인 |
| 3. 비교와 복원, 선택 | LR만 0.0003으로 바꾼 두 번째 run 비교, 종료 후 이전 run 복원 | 사용자 / 20–30분 | 두 실험 조건/검증 결과를 일지에 기록. 최종 test는 보류 |
| 4. 개인 입력 연결 | 원본 Kaldi 음향 모델·정렬·음소 사전·GOP 특징을 연결하고 공개 예제로 대조 | 개발 작업 / 환경 조사 후 산정 | 같은 공개 음성에서 호환되는 특징과 예측을 얻음 |
| 5. 기존 녹음 적용 | 이미 받은 두 녹음의 원고/확인 전사로 연결 결과와 실패 구간 점검 | 개발 + 사용자 청취 / 연결 후 | 원음 시점과 모델 출력을 함께 확인. 결과를 근거 없이 정답으로 확정하지 않음 |
| 6. 실제 훈련에 사용 | 교정 후보 1–2개를 골라 다시 말하고, 사람 의견과 비교 | 사용자 / 회당 10–15분 | 이해 용이성·재현성·오탐을 기록하며 개인 도구의 유용성 판단 |

**지금 당장 할 일은 1번과 2번이다.** 비교 실험은 한 번으로도 충분하며, 여러 모델 구조를 새로 설계할 필요는 없다.
개인 입력 연결은 사용자가 음성 모델을 처음부터 설계해야 하는 과제가 아니라 다음 개발 작업이다.
새 녹음을 반복 제출하거나 기존 청취 의견을 숫자 라벨로 만드는 것을 선행 조건으로 두지 않는다.

예산 상한: GPU 실험 $30, 평가 API $10, 예비비 $10, 무료 청취 평가 $0.
필요한 유료 실행은 먼저 작은 한도로 검증하며, 현재 코드에 계정 수준 자동 과금 차단 기능이 있는 것은 아니다.

더 자세한 원리·출처: [GOPT 기술 안내](docs/GOPT_WORKBENCH.md) · [현재 상태](docs/STATUS.md) ·
[개발 로드맵/예산](docs/ROADMAP.md) · [훈련 템플릿 연결](docs/DAILY_TRAINING.md).

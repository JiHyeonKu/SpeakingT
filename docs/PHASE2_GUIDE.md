# Phase 2 JupyterLab 실행 안내

2026-11-02. 이 문서는 **새 SSL 연구 노트북** 사용법이다. 기존 GOPT 노트북과 Baseline 01은 그대로 보존한다.

## 1. 실행 전 알아둘 것

현재 PC에는 CPU용 PyTorch와 Phase 2용 Transformers 4.57.6, safetensors, psutil이 설치돼 있다.
공식 Train 원음 2,500개와 고정 WavLM Base+ 가중치를 준비했다. 실제 실행 결과는
[실험 기록](../experiments/phase2/README.md)의 Stage 상태를 확인한다.

**현재 Stage A–D는 실제 실행 완료 상태다.** 먼저 완료 결과를 읽고 필요한 실험만 실행하면 된다.
출력이 담긴 전체 실행 노트북은 로컬 `reports/phase2/notebook-full-verified.ipynb`다.
Jupyter 왼쪽 파일 목록에서 `reports` → `phase2`로 이동해 열면 재학습 없이 그래프/지표를 볼 수 있다.
새 실험은 아래의 깨끗한 연구 노트북에서 진행한다.

노트북은 [Phase2_Frozen_WavLM.ipynb](../notebooks/Phase2_Frozen_WavLM.ipynb)다.
이 파일은 프로그램이다. GitHub 미리보기에서 보는 것만으로 실행되지 않는다.
개인 녹음이나 API 키를 넣지 않고 공개 데이터만으로 시작할 수 있다.

## 2. Windows에서 여는 순서

1. 탐색기에서 SpeakingT 폴더를 연다.
2. **`start-phase2.cmd`를 더블클릭**한다. 실행 창은 켜둔다.
3. JupyterLab이 열리면 `Phase2_Frozen_WavLM.ipynb` 탭을 확인한다.
4. 자동으로 안 열리면 실행 창에 나온 이번 `localhost` 주소를 연다. token 주소는 공유하지 않는다.
5. 오른쪽 위 커널이 `Python 3 (ipykernel)`인지 확인한다. 2번 코드 셀의 `Python:`이 프로젝트 `.venv`를 가리켜야 한다.
6. 맨 위 셀을 클릭하고 **Shift+Enter**를 눌러 한 셀씩 진행한다.

설명(Markdown) 셀은 설명을 표시하고 다음으로 이동한다. 코드 셀은 계산한다.
왼쪽 `[*]`는 실행 중, 숫자 `[1]` 등은 실행 완료를 뜻한다. `[*]`일 때 같은 셀을 여러 번 누르지 않는다.
중단은 상단 ■ 버튼, 변수까지 초기화하는 재시작은 Kernel 메뉴의 Restart Kernel이다.

## 3. 첫 실행 — Stage A/B만 확인

1–8번을 순서대로 실행하면서 다음을 확인한다.

| 구역 | 확인 사항 |
|---|---|
| 1 | 목표: 문장 점수 4개, 음소 점수 아님 |
| 2 | 실제 Python 경로, CPU/GPU, RAM |
| 3 | Baseline 해시·2,000/500 분할·WAV 검사 |
| 4 | 실제 공개 음성 재생, 파형/스펙트로그램, 사람 점수 |
| 5 | WavLM 94,381,936개 파라미터, 학습 가능한 인코더 파라미터 0 |
| 6 | 입력 `[1, samples]`, hidden `[1, frames, 768]`, pooled `[12, 1536]` |
| 7 | 20개 특징 `[20, 12, 1536]`와 사람 점수 `[20, 4]` |
| 8 | 작은 평가 헤드 구조 |

9번 설정은 처음에는 그대로 둔다.

```python
RUN_FULL_TRAINING = False
RUN_MULTI_SEED = False
SEEDS = [7, 19, 42]
```

10번을 실행하면 한 배치의 실제 loss·gradient·weight change를 보여주고,
Training 16개/Validation 4개로 3 epoch 기능 검증을 한다. 새 run이 생긴다.
11번 그래프, 12번 지표, 13번 저장 모델 복원까지 실행한다.
작은 Stage B의 낮거나 불안정한 점수를 성능 실패/성공으로 판단하지 않는다.
14번은 전체 Validation이 아닌 경우 GOPT 비교를 건너뛴다.

10번 실행 후 메모리를 줄이기 위해 인코더 변수를 해제한다.
다시 6–10번을 실행하려면 먼저 **5번 모델 로딩 셀**을 다시 실행해야 한다.

## 4. 전체 학습 — Stage C

9번 설정을 아래처럼 바꾸고 **9번 셀 자체를 다시 실행**한다.

```python
RUN_FULL_TRAINING = True
RUN_MULTI_SEED = False
SEEDS = [7, 19, 42]
CONFIG = ssl.TrainConfig(
    epochs=150, patience=20, learning_rate=1e-3, weight_decay=1e-3,
    batch_size=64, hidden=128, dropout=.2, seed=7, threads=4, device='cpu')
```

첫 실행은 학습 조건을 바꾸지 않고 기준 실험을 확인한다. 인코더를 이미 해제했다면 5번을 다시 실행한 뒤 10번으로 간다.
10번에서는 원음 특징 캐시를 완성/검증하고 Training 2,000개로 헤드를 학습한다.
Validation 500개는 학습 loss/정규화 추정에 사용하지 않는다. `patience=20` 때문에 150회 전에 정상 종료될 수 있다.
캐시가 없다면 이 단계의 원음 계산은 CPU에서 수십 분 걸릴 수 있다. 완성된 캐시는 해시 확인 후 재사용한다.

학습 중 볼 것:

- Train/Validation MSE가 함께 줄어드는지.
- Train만 좋아지고 Validation이 악화되는지.
- 네 목표별 MSE/MAE/PCC가 어떻게 다른지.
- 단순 평균 예측과 비교해서 오차를 줄이는지.
- `best epoch`와 마지막 epoch가 다른지.

MSE/MAE는 낮을수록, PCC는 높을수록 좋다. PCC 0.8은 정확도 80%가 아니다.
문장 점수는 0–10이며 원시 회귀 예측을 임의로 clipping하지 않는다.
그래프의 reported MSE는 0–10 단위다. 내부 batch loss는 점수를 10으로 나눈 척도라 값이 다르다.

## 5. 세 seed 확인 — Stage D

`RUN_FULL_TRAINING=True`, `RUN_MULTI_SEED=True`로 설정하면 seed 7, 19, 42를 각각 **새 run**으로 실행한다.
기존 seed 7 실행에 이어서 학습하는 기능이 아니므로 같은 seed를 다시 실행할 수도 있다.
이미 수행한 전체 실험은 아래 복원 절차로 먼저 확인하고 불필요하게 반복하지 않는다.

seed를 바꿔도 Train/Validation 화자 분할은 바뀌지 않는다.
14번에서 GOPT와 동일한 500개 Validation 문장에 대해 비교하고 평균·표본 표준편차,
주 seed 7의 화자 단위 paired bootstrap 구간을 저장한다. 최고 seed만 골라 결과로 쓰지 않는다.
이 노트북에는 공식 Test를 실행하는 버튼/변수가 없다.

## 6. 무엇이 어디에 저장되는가

| 경로 | 내용 |
|---|---|
| `runs/gopt/run-20261102-014018-444335/` | 보존된 사용자 Baseline 01 |
| `models/baseline01-preserved/` | Baseline 핵심 파일의 해시 검증 사본 |
| `data/phase2/manifest.json` | 고정 ID/화자/라벨/원본 경로 |
| `data/phase2/audio/` | 공식 Train 원음 2,500개 |
| `data/phase2/audio-audit.json` | 파일별 원본/PCM 해시와 길이 |
| `data/phase2/features/` | 발화별 WavLM 통계 특징 |
| `data/phase2/feature-manifest.json` | 전체 특징 식별자/해시. 캐시 재확인으로 바꾸지 않음 |
| `data/phase2/extraction-events/` | 재확인/추출마다 별도의 처리 시간 기록 |
| `.cache/ssl-models/` | 고정 공식 인코더·전처리기 |
| `runs/ssl/<run>/` | 이번에 직접 실행한 평가 헤드 실험 |
| `reports/phase2/comparison-.../` | 같은 Validation의 GOPT/SSL 표, 그래프, seed 요약·불확실성 |

각 run의 주요 파일:

- `config.json`, `split.json`: 하이퍼파라미터, 코드/프로토콜/데이터 해시, ID 순서, 버전.
- `initial.pt`, `last.pt`, `best.pt`: 초기 상태, 마지막 완료 epoch, 최적 Validation epoch.
- 체크포인트 안의 `normalization_mean/std`: Training에서만 계산한 정규화.
- `metrics.csv`, `metrics-NNN.json`: 각 epoch의 요약/목표별 상세 지표.
- `best-metrics.json`, `best-validation-predictions.npz`: 선택된 모델 지표와 ID별 예측/정답.
- `mean-baseline.json`: Training 평균만 예측하는 비교 기준선.
- `summary.json`: 완료/조기종료/중단 상태, best epoch. `test_evaluated=false`.
- `learning-curves.png`, `resources.json`: 그래프와 실제 process RSS/CPU/경과 시간.

기존 GOPT처럼 매 epoch 전체 `.pt`를 모두 저장하지는 않는다. 초기/마지막/최적 checkpoint와 모든 epoch 지표를 저장한다.
체크포인트는 원본 WavLM 전체 가중치를 중복 저장하지 않는다. 인코더의 고정 버전과 캐시를 함께 보관한다.

## 7. 며칠 뒤 다시 열어 복원하기

1. `start-phase2.cmd`로 JupyterLab을 연다.
2. **2번 실행 환경 코드**를 실행한다.
3. 13번 셀의 `RESTORE_RUN`을 저장해 둔 run 경로로 바꾼다.

```python
RESTORE_RUN = 'runs/ssl/notebook-full-seed7-20261102-095132-822399'
```

4. 13번을 실행하면 필요한 캐시, 같은 분할, 저장된 정규화, `best.pt`를 복원하고 기록된 Validation MSE와 대조한다.
5. 학습을 다시 시작할 필요는 없다. 가중치 복원과 이어 학습은 다르며 **optimizer 상태에서 이어 학습은 아직 지원하지 않는다**.

위 경로는 이번 주 seed 7의 실제 run이며 best epoch는 53이다. 이 절차를 새 Jupyter 커널에서 실행해
기록된 Validation 500개 예측과 비트 단위 일치를 확인했다.

run 이름을 잊었으면 2번 실행 뒤 새 코드 셀에서:

```python
for p in sorted((ROOT/'runs/ssl').glob('*')):
    if (p/'summary.json').exists():
        print(p.relative_to(ROOT), (p/'summary.json').read_text(encoding='utf-8'))
```

기존 비교 결과는 `reports/phase2/comparison-.../comparison.csv`와 `summary.json`을 읽으면 된다.
GitHub에는 원음/개인 자료/실험 checkpoint가 올라가지 않는다. 새 PC에서는 이 폴더를 개인 백업에서 복원한다.

## 8. 실험을 바꾸는 방법

현재 프로토콜을 먼저 완료한 뒤 한 조건씩 바꾼다. 새 run 이름은 자동 생성된다.
실험 일지에는 목적, 변경 항목, Git commit, run 경로, best epoch, 네 목표의 Validation 지표, 자원 사용량을 남긴다.
Validation을 보며 조정한 횟수도 기록한다. 더 나쁜 결과도 보관한다.

현재 9번에서 바로 바꿀 수 있는 것은 epochs/patience/lr/weight decay/batch/hidden/dropout/seed다.
다른 인코더, 마지막 층만 사용, 시간 attention, 음소 평가 등은 **현재 토글 기능이 아니라 후속 구현 과제**다.
그런 실험을 기존 cache의 차원만 맞춰 흉내 내지 않는다.

## 9. 오류와 종료

| 증상 | 조치 |
|---|---|
| `NameError` | 커널 재시작으로 변수 소실 가능. 2번부터 필요한 셀 실행. `encoder`만 없으면 5번 재실행 |
| `Protected baseline changed` | 원본이 바뀌었는지 해시/백업 확인. 자동으로 새 hash를 덮어쓰지 않음 |
| 원본 run 또는 `data/gopt` 없음 | Git clone만으로 원본 실험을 복구할 수 없음. 개인 백업에서 복원 |
| 캐시 provenance/hash mismatch | 원음/encoder/장치/전처리 변경 여부 확인. 기존 캐시를 덮어쓰지 않고 새 실험 공간 준비 |
| `best.pt` 없음 | 첫 학습 epoch 전에 중단했는지 확인. `initial.pt`는 학습 전 상태 |
| 검증 성능이 GOPT보다 낮음 | 오류로 숨기지 말고 비교 표에 기록. 입력 정보·시간 pooling·과적합 등을 분석 |
| RAM 부족 | 다른 무거운 작업 종료, CPU 인코더/데이터 변수 해제 또는 kernel 재시작. 결과는 run에 남음 |
| CUDA unavailable | 현재 CPU/Intel 환경에서는 정상. 별도 NVIDIA 환경은 자원 계획 참고 |
| 오래 걸림 | 최초 원음 추출은 긴 단계. 출력된 완료 파일 수와 경과 시간을 확인. 이후 캐시 사용 |

중단은 ■, 노트북 저장은 Ctrl+S, 서버 종료는 File → Shut Down 또는 실행 창 Ctrl+C다.
브라우저만 닫으면 계산이 계속될 수 있다. PC 강제 종료는 현재 epoch를 잃을 수 있다.
개인 출력이 포함된 노트북은 공개 커밋하지 않는다. 사용자가 수정한 기존 GOPT 노트북은 그대로 둔다.

## 10. 권장 진행 계획

| 회차 | 할 일 | 완료 기준 |
|---|---|---|
| 1 | 1–8번을 실행하고 원음·라벨·텐서 크기 확인 | 원음→특징→점수의 연결을 설명할 수 있음 |
| 2 | Stage B 3 epoch, 그래프와 저장 모델 복원 | 실제 gradient/weight change와 복원 일치 확인 |
| 3 | 이미 수행된 Stage C/D 결과부터 읽고, 필요할 때 같은 설정 재현 | 평균 기준선/GOPT/모든 seed의 같은 문장 지표를 비교 |
| 4 | 한 가지 ablation을 정하고 새 run | 사전 가설·변경 조건·검증 결과·자원 기록 |
| 5 | 후속 encoder 또는 일부 층 fine-tune 판단 | 기존 결과와 비용 근거가 있을 때 진행. 유료 자원은 승인 후 |

상세 선정 근거·데이터 감사·GPU 설치/요금은 [연구 설계](PHASE2_RESEARCH.md)를 참고한다.

이번 결과에서 큰 Train/Validation 차이를 확인했으므로 **다음 실험은 `hidden=128`만 `hidden=32`로 변경**하는 것이다.
같은 분할·seed 7/19/42와 나머지 설정을 유지한다. 먼저 실행 일지에 가설을 기록한 뒤
9번에서 전체/다중 seed 실행을 켜고 새 run으로 실행한다. 기존 실험을 덮어쓰지 않는다.
후속 실험은 아직 실행하지 않았으며, 현재 결론을 확인한 뒤 사용자가 직접 진행할 수 있는 첫 과제다.

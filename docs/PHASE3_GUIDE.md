# Phase 3 실행 안내 — WavLM 상위 층 Fine-tuning

이 단계는 실제 WAV에서 WavLM 상위 Transformer 층까지 역전파하는 문장 점수 회귀 실험이다. 현재 PC에서 **실제 20개 WAV 학습, 새 커널 복원, 중단 후 재개**를 검증했다. 전체 2,000/500 GPU 실험은 아직 실행하지 않았다. 현재 결과는 [실험 기록](../experiments/phase3/README.md)에 있다.

## 1. 지금 바로 실행하기

1. Windows 탐색기에서 `C:\Users\domin\OneDrive\Desktop\SpeakingT`를 연다.
2. `start-phase3.cmd`를 더블클릭한다. 검은 실행 창은 JupyterLab 서버이므로 사용하는 동안 열어 둔다. 기존 `start-phase2.cmd`도 그대로 남아 있다.
3. `notebooks/Phase3_Partial_WavLM.ipynb`가 열리면 **2번 환경** 코드 셀을 클릭하고 `Shift+Enter`를 누른다. 출력 Python 경로가 프로젝트의 `.venv\Scripts\python.exe`인지 확인한다.
4. **3번 설정**을 확인한다. 기본값 `DEVICE='cpu'`, `RUN_GPU_PILOT=False`, `RUN_FULL_GPU=False`는 무료 소규모 실행이다. 수정 없이 첫 실행이 가능하다.
5. 위에서 아래로 `Shift+Enter`를 반복한다. `[*]`는 실행 중이다. **5번**에서 실제 WAV를 듣고 **6번**에서 모델 크기와 입력/출력 차원을 확인한다.
6. **7번**에서 실제 학습한다. epoch마다 그래프와 오차가 바뀌고, `Saved run:`에 이번 저장 경로가 표시된다. 이 경로를 기록한다.
7. **8–9번**에서 자원, 목표별 지표, gradient와 가중치 변화량을 본다. **10번**은 저장된 모델로 예측만 다시 만든다. **11번** 재학습은 기본적으로 꺼져 있다.

인터넷 연결 없이도 기존 PC에서 실행할 수 있도록 원음·사전학습 가중치가 이미 로컬에 준비돼 있다. GitHub clone만 한 새 PC에는 이 ignored 파일들이 없으므로 8번의 데이터 이전 절차가 필요하다.

**CPU 전체 fine-tuning은 실행하지 않는다.** 코드가 CPU full 모드와 smoke 2 epoch 초과를 거부한다. 기본 한 번의 CPU run은 이전 측정에서 약 1분이었으나, Jupyter 시작·라이브러리 import·모델 미리보기 시간을 합치면 더 오래 걸린다.

## 2. 어떤 모델을 학습하는가

```text
16 kHz mono WAV (실제 길이)
  → 고정 CNN + feature projection (발화별 처리)
  → latent frame 동적 padding + 유효 frame attention mask
  → WavLM Transformer 12층 (기본: 0–9층 고정, 10–11층 학습)
  → 12층 각각의 유효 frame 평균 + 표준편차 [12, 1536]
  → 고정된 Training-only 정규화
  → 학습 가능한 층 혼합 + hidden 128 / dropout .5 MLP
  → Accuracy / Fluency / Prosody / Total
```

모델은 `microsoft/wavlm-base-plus`, revision `4c66d4806a428f2e922ccfa1a962776e232d487b`로 고정했다. 가중치와 processor 파일의 hash를 로딩 전에 검사한다. [공식 WavLM 설명](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/wavlm)은 원시 파형 입력과 모델 구조를 설명한다. 이 저장소의 세부 동작은 설치된 Transformers **4.57.6** 구현을 직접 확인했다.

공식 processor의 `do_normalize=False`를 따른다. CNN의 GroupNorm은 시간축 padding 영향을 받을 수 있으므로 **raw padding을 제거한 뒤 CNN에 통과**시킨다. 이후 latent frame을 padding하며 attention과 pooling에서 유효 frame만 사용한다. 공식 단일 forward와 수치 대조를 완료했다. 12층 모두의 출력이 head에 연결되며, 상위 층 gradient가 끊기지 않는다. 고정 특징 `.npy/.npz`는 학습 입력으로 사용하지 않는다.

고정 층은 `eval` 모드, 선택된 상위 층과 head는 `train` 모드다. 이번 통제 실험에서는 LayerDrop/SpecAugment를 쓰지 않는다. 최상위 2층은 14,176,808개, head는 197,264개 파라미터가 학습 대상이다. 별도 phoneme 정렬이나 목표 문장 입력은 아직 없으며, 문장 점수를 음소 오류 위치나 미국인 유사도로 해석하지 않는다.

## 3. 한곳에서 바꾸는 설정

노트북 **3번 CONFIG** 또는 CLI 인자를 수정한다. 여러 설정을 동시에 바꾸기보다 먼저 `top_layers=2`와 `0`을 같은 조건으로 비교한다.

| 설정 | 초기값 | 의미 |
|---|---:|---|
| `top_layers` | 2 | 0: 전체 인코더 고정, 1: 마지막 층, 2: 마지막 두 층 |
| `hidden`, `dropout` | 128, .5 | 기존 Phase 2 평가 head와 동일 |
| `encoder_lr`, `head_lr` | 1e-5, 1e-4 | 사전학습 인코더에 더 작은 업데이트 적용 |
| `batch_size`, `accumulation` | 2, 2 | 한 번에 2개, 보통 4개마다 optimizer step |
| `clip_norm` | 1.0 | gradient 폭주 제한 |
| `weight_decay` | .01 | AdamW 규제 |
| `epochs`, `patience` | 1, 3 | smoke 기본 1 epoch; full 초기 상한 10 |
| `seed` | 7 | 초기화·dropout·셔플 재현 설정 |
| `precision` | fp32 | CUDA에서만 fp16 또는 지원되는 bf16 선택 가능 |

기본 warm-start는 Phase 2 seed 7/dropout=.5의 epoch 8 head이다. 원래 Training 2,000개에서만 구한 정규화 통계를 이어받아 고정한다. `warm_start=None` 또는 CLI `--fresh`는 head를 새로 초기화하고 **선택된 Training WAV만** 사용해 통계를 계산한다. 초기화 방식과 원본 checkpoint SHA는 run 설정에 저장한다. 두 초기화 방식의 결과를 섞어 평균 내지 않는다.

학습 loss는 정답을 `/10`한 척도이고, 평가 MSE/MAE는 **원래 0–10 척도**다. 따라서 같은 예측에서 두 MSE는 100배 차이가 난다. 예측을 0–10으로 강제 clipping하지 않는다. 모델 선정 기준은 **Validation macro MSE**이며 네 목표에 동일 가중치를 둔다. 최고 PCC를 보고 checkpoint를 바꾸지 않는다.

## 4. 그래프와 저장 파일 읽기

각 run은 `runs/phase3/<새 시간 ID>/`에 저장된다. 기존 run 경로를 지정하면 덮어쓰지 않고 실패한다.

| 파일 | 내용 |
|---|---|
| `best.pt` | Validation macro MSE 최소 모델. 초기 epoch 0도 후보 |
| `last.pt` | 마지막 완료 epoch의 모델·optimizer·scheduler·scaler·난수 상태 |
| `*.sha256.json` | checkpoint 무결성 검사 |
| `config.json`, `split.json` | 설정·환경·초기화 출처·데이터 식별자·정확한 분할 |
| `metrics.csv`, `metrics-NNN.json` | epoch별 오차, 목표별 MSE/MAE/PCC, gradient, frozen hash, encoder 변화량 |
| `best-metrics.json` | 선택된 모델의 Train/Validation 평가 |
| `learning-curves.png` | Train/Validation macro MSE 및 PCC 그래프 |
| `resources.json`, `summary.json` | 시간·메모리·종료 상태·선택 epoch |
| `*-validation-predictions.npz` | 로컬 개별 예측; 공개 Git에 올리지 않음 |

**best epoch=0은 정상적인 결과다.** 학습 후 개발셋 성능이 나빠졌다면 기존 모델이 선택된다. 실제 갱신 여부는 `last.pt`와 `encoder_delta_l2`, `encoder_grad_norm`으로 확인한다. gradient 값은 epoch의 첫 optimizer step에서 clipping 전 측정한 값이다. encoder 변화량은 해당 run 시작점 대비 L2 거리라서 재개 run에서는 0부터 다시 잰다. 매 epoch 모든 frozen 파라미터 SHA가 그대로인지 검사한다.

현재 smoke의 Validation은 4개뿐이므로 PCC가 불안정하다. 이를 500개 Validation의 GOPT/Phase 2 점수와 직접 비교하지 않는다.

## 5. 커널을 재시작한 뒤 예측만 복원하기

Jupyter 메뉴 `Kernel → Restart Kernel`은 Python 메모리를 비운다. 디스크 checkpoint는 남아 있다.

1. 노트북 **2번 환경** 셀만 실행한다.
2. **10번**에서 다음 값을 지정한다.

   ```python
   RESTORE_RUN = 'runs/phase3/20261102-172648-542194'
   CHECKPOINT = 'last'
   ```

3. **10번** 셀을 실행한다. 앞선 학습 셀은 실행할 필요가 없다. 초기 검증 run의 epoch 1 모델과 4개 Validation 예측을 복원한다.

독립 커널 검증을 CLI로 반복하려면 출력 이름을 새로 지정한다.

```powershell
.venv\Scripts\python.exe scripts/execute_phase3_notebook.py --restore-run runs/phase3/20261102-172648-542194 --output reports/phase3/my-fresh-restore.ipynb
```

같은 CPU 환경에서는 예측의 비트 단위 일치를 검증했다. GPU↔CPU나 라이브러리 버전 변경 시 같은 수치 일치를 보장하지 않는다. 출처·분할·학습 코드 hash·Transformers 버전 불일치를 조용히 무시하지 않는다. 코드가 달라졌다면 checkpoint를 만든 commit을 별도 checkout에 복원하거나 별도의 명시적 migration을 검토한다. hash 확인을 삭제해서 통과시키지 않는다. GPU checkpoint를 CPU에서 읽는 기능은 구현되어 있으나 **실제 GPU checkpoint로 검증한 것은 아니다**.

## 6. 중단 후 학습 재개

Jupyter의 ■ Interrupt 버튼으로 중단한다. 마지막 **완료된 epoch**의 `last.pt`가 기준이며, 미완료 epoch의 batch들은 다시 계산한다. `best.pt`는 재학습보다 선택 모델의 평가용으로 사용한다.

노트북 **10번 복원**을 실행한 뒤 **11번**의 `RESUME_TRAINING=True`와 epoch 상한을 설정한다. CPU smoke는 총 2 epoch까지만 허용한다. GPU의 10 epoch 이후 재개라면 해당 상한도 더 크게 지정해야 한다. early stopping을 이미 충족했으면 재개를 거부한다.

```powershell
.venv\Scripts\python.exe scripts/run_phase3.py smoke --epochs 2 --resume runs/phase3/20261102-172648-542194/last.pt
```

기존 run은 보존하고 새 run에 저장한다. 설정·분할·장치는 유지하며 epoch 상한만 늘릴 수 있다. batch나 LR 변경은 동일 학습의 정확한 재개가 아니다. 재개 시 optimizer/scheduler/scaler와 Python/Torch 난수 상태까지 읽는다. 이번 CPU 실험에서는 연속 2 epoch와 1+1 epoch의 **모델 255개 tensor, optimizer, scheduler, 난수 상태, 예측이 모두 비트 단위로 일치**했다.

전원 종료 중 checkpoint/sidecar 저장이 끊기면 checksum 불일치로 실패할 수 있다. 이 경우 이전 run이나 별도 보존한 checkpoint를 확인한다. 임대 GPU의 임시 디스크는 종료 시 사라질 수 있으므로 종료 전에 run 폴더를 PC나 승인된 영구 저장소로 복사한다.

## 7. GPU 환경 준비 — 아직 실행하지 않은 단계

현재 PC는 CPU PyTorch이며 CUDA가 없다. Intel GPU에 CUDA용 PyTorch를 설치해도 NVIDIA CUDA 장치가 생기지 않는다. 지금의 `.venv`는 보존하고, NVIDIA GPU가 있는 PC 또는 승인받은 클라우드에서 별도 환경을 만든다.

초기 계획은 **24 GB급 NVIDIA GPU**(예: RTX 3090/4090 또는 A10)를 권장한다. batch 1–2에서 **12–16 GB도 후보**지만 실제 최소 VRAM은 측정 전 확정할 수 없다. WavLM 가중치 약 0.38 GB 외에 optimizer 상태·gradient·긴 발화의 attention/activation·CUDA workspace가 추가된다. 따라서 parameter 수만으로 필요한 VRAM을 판단하지 않는다. host RAM은 16 GB 이상, 여유 디스크는 run당 약 1 GB에 여러 실험·데이터 공간을 더해 준비한다. 이 숫자는 장비 선택용 추정이며 GPU 실측 결과가 아니다.

Windows 예시:

```powershell
py -3.11 -m venv .venv-gpu
.venv-gpu\Scripts\python.exe -m pip install --upgrade pip
```

[PyTorch 공식 설치 선택기](https://pytorch.org/get-started/locally/)에서 해당 OS/Pip/Python/CUDA 조합의 명령을 확인한다. 표시된 명령의 `pip3`를 `.venv-gpu\Scripts\python.exe -m pip`로 바꾸어 실행한다. NVIDIA 드라이버와 제공 wheel의 호환 조합을 선택하고, CPU-only index를 사용하지 않는다. 이 프로젝트에는 torchvision/torchaudio 기능은 필요 없다. GPU PyTorch 설치 후:

```powershell
.venv-gpu\Scripts\python.exe -m pip install -e ".[research,ssl]"
.venv-gpu\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
.venv-gpu\Scripts\python.exe -m pip freeze > reports/phase3/gpu-environment.txt
.venv-gpu\Scripts\python.exe -m jupyterlab --ServerApp.ip=127.0.0.1 notebooks/Phase3_Partial_WavLM.ipynb
```

마지막 명령으로 연 Lab은 GPU 환경의 커널을 사용한다. **노트북 2번의 Python 경로와 CUDA True를 다시 확인한다.** 기존 `start-phase3.cmd`는 기존 CPU `.venv`를 사용한다. Linux에서는 `python3.11 -m venv .venv-gpu`와 `.venv-gpu/bin/python`으로 바꾸고 OS에 맞는 공식 CUDA wheel을 설치한다. CPU 검증은 torch 2.15.0+cpu / Transformers 4.57.6에서 수행했다. GPU 환경은 아직 감사하지 않았으므로 버전을 기록하고 Stage A/B를 다시 확인한다.

## 8. 다른 GPU PC로 필요한 파일 옮기기

공개 Git 코드를 같은 commit으로 가져오고, 아래 **공개 데이터에서 나온 로컬 파일만** 같은 상대 경로로 따로 복사한다. 개인 녹음과 기존 공식 Test 자료는 필요 없다. 라이선스/출처는 [Phase 2 연구 기록](PHASE2_RESEARCH.md)을 따른다.

- `data/phase2/audio/`, `data/phase2/manifest.json`, `data/phase2/feature-manifest.json`.
- `.cache/ssl-models/wavlm-base-plus/4c66d4806a428f2e922ccfa1a962776e232d487b/`의 잠금 파일에 나열된 모델 assets.
- `runs/gopt/run-20261102-014018-444335/split.json`.
- warm-start 기본 run `runs/ssl/notebook-full-seed7-20261102-103405-853366/`의 `config.json`, `best.pt`. 다른 seed의 warm-start 실험이면 해당 run도 복사한다.

`feature-manifest.json`은 기존 head의 출처 검사용이며, 고정 특징 배열인 `data/phase2/features/`는 이 학습에 필요 없다. 기존 체크포인트 복원이 목적이면 해당 Phase 3 run 전체도 복사한다. 원음과 모델 파일은 GitHub에 커밋하지 않는다. 복사 후 notebook **4번 데이터**, **6번 모델 로딩**에서 hash/분할 검사를 통과해야 한다.

## 9. GPU pilot → 비용 계산 → 전체 학습

유료 GPU 임대나 실행은 **예상 비용과 상한을 사용자에게 알리고 사전 승인받은 뒤** 수행한다. 이번 작업의 유료 지출은 $0이다. 기존 총 예산은 $50이며, 이전 배분상 GPU $30/평가 API $10/예비비 $10이다. 초기 GPU 실험 상한 $10 계획은 승인 자체를 의미하지 않는다.

승인된 GPU에서 먼저 64개 실제 WAV(Train 48/Validation 16, 가장 긴 발화 포함)를 사용한다.

```powershell
.venv-gpu\Scripts\python.exe scripts/run_phase3.py pilot --device cuda --epochs 1 --batch-size 2 --precision fp32
```

`summary.json`이 completed인지 확인하고 `resources.json`의 `cuda_peak_allocated_bytes`, `cuda_peak_reserved_bytes`, CPU RSS와 시간을 읽는다. 별도 창에서 `nvidia-smi -l 1`로 다른 프로세스를 포함한 GPU 메모리도 확인한다. sampler의 RSS는 관측 최대치이며 순간 peak를 완벽히 포착하지 않을 수 있다.

지원 장치에서 fp16/bf16을 사용할 경우 **같은 작은 pilot을 새 run으로** 먼저 검증한다. AMP는 autocast와 fp16 GradScaler를 쓰고, accumulation 전체에 걸쳐 scale을 유지한 뒤 unscale/gradient clipping/step한다. 이 순서는 [PyTorch AMP 예제](https://docs.pytorch.org/docs/2.15/notes/amp_examples.html)를 따른다. GPU 경로는 아직 실측하지 않았다.

pilot 경로와 실제 요금표의 시간당 금액을 넣어 비용을 추정한다. 아래 `0.50`은 **계산법을 보여주는 가정값이며 실제 가격이 아니다**.

```powershell
.venv-gpu\Scripts\python.exe scripts/phase3_cost_estimate.py runs/phase3/PILOT_RUN --hourly-usd 0.50 --epochs 10
```

도구는 실제 완료 CUDA pilot만 받는다. `epoch 1 경과시간 − epoch 0 경과시간`을 train pass + train 평가 + validation의 발화 수 비율 `4500/112`로 확장하고 기본 2배 여유 및 15분 준비 시간을 더한다. 길이 분포·GPU·batch·precision·저장 속도가 다르면 오차가 크다. 저장공간/전송/세금은 별도다. **GPU 시간과 금액을 CPU 시간으로 대신 계산하지 않는다.**

파일럿 결과와 비용 승인 후 전체 실행:

```powershell
.venv-gpu\Scripts\python.exe scripts/run_phase3.py full --device cuda --allow-full --epochs 10 --batch-size 2 --accumulation 2 --precision fp32 --top-layers 2 --seed 7
```

노트북에서는 `RUN_GPU_PILOT=False`, `RUN_FULL_GPU=True`, `DEVICE='cuda'`로 설정한다. 기본 10 epoch 상한이며 early stopping이 더 일찍 종료할 수 있다. GPU 사용료는 학습 완료 후에도 인스턴스를 켜 두면 발생할 수 있으므로 결과를 복사하고 대여 자원을 종료한다.

OOM 대응: 기존 run을 보존하고 `batch_size=1`/`accumulation=4`로 작은 pilot을 다시 실행한다. 필요하면 지원 AMP 또는 상위 1층 실험을 검토한다. 발화를 조용히 자르거나 검증 자료를 제외하지 않는다. `resources.json`과 `summary.json`을 함께 확인한다. batch/precision 변경 시 새로운 실험으로 시작하며 동일 설정 resume로 가장하지 않는다. 미세조정 전에 fp32 통과 여부를 확보한다.

## 10. 공정한 비교와 다음 연구 순서

1. GPU pilot에서 forward/backward, finite gradient, frozen hash, 저장/복원을 확인한다.
2. seed 7에서 동일 warm-start의 `top_layers=2`와 `top_layers=0`을 같은 2,000/500 분할·학습 상한·평가 단위로 비교한다. 한 번에 LR과 head 크기를 함께 바꾸지 않는다.
3. 결과가 유망하면 사전에 정한 seed 7/19/42를 반복한다. seed별 초기화 출처를 고정한다. seed 19/42를 각자의 기존 Phase 2 head에서 시작하는 설계라면 두 조건 모두 해당 head를 사용한다.
4. Train/Validation 차이, 네 목표의 MSE/MAE/PCC, seed 분산과 화자 bootstrap 구간을 보고한다. 같은 정렬의 500개 예측으로 기존 GOPT/Phase 2와 비교한다. 단일 fine-tuned 모델과 3-seed ensemble은 별도로 표기한다.
5. 상위 1층 또는 encoder LR 하나를 바꾸는 ablation은 앞 결과에 근거해 정한다. 그 다음에 시간 정보 보존 pooling, 목표 문장 결합을 독립 실험으로 검토한다.

전체 Training 평균 점수를 모든 Validation에 예측하는 단순 baseline도 함께 계산한다. 기존 공식 Test는 이미 관찰했으며 **새 모델 선택·튜닝·확증에 재사용하지 않는다**. 현재 Validation도 여러 차례 사용했으므로 선택 편향이 있다. 유의한 개발셋 차이가 나오더라도 독립 일반화 성능이나 개인 발음 교정 효과가 입증됐다고 표현하지 않는다.

## 11. 오류와 검증 명령

- `Encoder asset changed`, `provenance mismatch`: 잠금 revision, 복사한 파일, manifest/split을 확인한다. 새 파일을 몰래 섞거나 hash를 덮어쓰지 않는다.
- `Training code differs`: checkpoint 당시 소스가 필요하다. 별도 checkout에서 복원한다.
- `CPU verification uses fp32` / `CUDA unavailable`: CPU는 fp32 smoke를 사용한다. GPU 설치만으로 실제 NVIDIA GPU를 대체할 수 없다.
- `Run already exists`: 다른 저장 경로를 사용한다.
- mask dtype upstream 경고: 현재 Transformers/PyTorch 조합에서 발생한다. 단일/배치 수치 검사는 통과했다. 경고를 없애기 위해 검증 없이 dependency를 교체하지 않는다.
- Jupyter의 `ModuleNotFoundError`: 다른 Python 커널인지 먼저 확인하고 해당 환경에서 `pip install -e ".[research,ssl]"`을 실행한다.

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe scripts/audit_phase3.py --verify
```

두 번째 명령은 로컬 사전 보존 snapshot이 있는 기존 PC에서 사용한다. 옵션 없는 감사 명령은 이미 존재하는 사전 snapshot을 덮어쓰지 못하도록 되어 있다. 테스트에는 작은 무작위 backbone을 쓰지만, 이 문서의 실제 학습 검증은 공개 WAV와 고정된 사전학습 WavLM으로 수행했다.

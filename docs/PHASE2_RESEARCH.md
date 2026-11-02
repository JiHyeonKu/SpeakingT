# Phase 2 — 모델 선정, 데이터 감사와 실험 설계

2026-11-02. 현재 목표는 발음 평가 모델의 성능과 재현성을 연구하는 것이다. 앱 제작·개인 녹음 수집·논문 작성은 이번 범위가 아니다.
결과를 확인하기 전의 첫 전체 실험 설정은 [protocol.json](../experiments/phase2/protocol.json)에 고정했다.
실행 방법은 [PHASE2_GUIDE.md](PHASE2_GUIDE.md), 실행 증거/결과는 [Phase 2 기록](../experiments/phase2/README.md)에 정리한다.

## 1. Baseline 01 보존

사용자가 지정한 `runs/gopt/run-20261102-014018-444335`를 확인했다.
최대 60 epoch, lr 0.001, seed 7, patience 5로 실행되어 35 epoch에서 조기 종료됐고 best는 30이다.
사용자가 제공한 Test Phone MSE/PCC 및 Prosody/Fluency/Total PCC는 `final-test.json`과 일치한다.

보존 조치:

- 원본 run의 모든 파일 SHA256을 [baseline01.json](../experiments/phase2/baseline01.json)에 기록한다.
- `best.pt`, 설정, 분할, summary, final-test, epoch 30 평가를 `models/baseline01-preserved/`에도 복사하고 해시를 확인한다.
- 정규화는 원본 config와 checkpoint에 함께 보존한다. 원본을 덮어쓰거나 파일 속성을 변경하지 않는다.
- 사용자가 수정한 `notebooks/SpeakingT_GOPT.ipynb`는 수정·재생성·커밋하지 않는다.
- Phase 2 데이터 준비/비교 시 원본 run 해시를 다시 검사한다. 기존 파일이 달라지면 자동으로 새 기준선을 만들지 않고 중단한다.

공식 Test는 이미 관찰했으므로 이번 SSL 개발의 설정/체크포인트/seed 선택에 사용하지 않는다.
과거 Test 수치는 기록일 뿐 새 모델의 목표값으로 튜닝하지 않는다. 현재 Validation도 반복 사용한 개발 자료이며 독립 최종 시험이 아니다.

## 2. 데이터 인프라 감사

원본 배포자는 SpeechOcean762 공식 저장소와 OpenSLR SLR101이다. 음성 라이선스는 **CC BY 4.0**이며,
5,000개의 비원어민 영어 발화와 전문가 평가를 제공한다. 중국어 모국어 화자이며 아동/성인이 포함된다.
[공식 배포·라이선스](https://www.openslr.org/101/), [공식 저장소](https://github.com/jimbozhang/speechocean762).

| 항목 | 작업 시작 때 | 이번 준비/정책 |
|---|---|---|
| 공식 Train 라벨/GOP | 2,500개 준비됨 | 전체 원본 라벨과 재대조 |
| 기존 Training | 2,000개 / 100명 | Baseline 01 `split.json`의 ID 순서 그대로 사용 |
| 기존 Validation | 500개 / 25명 | 같은 파일에서 그대로 사용. seed 변경으로 재분할하지 않음 |
| 공식 Test 라벨/GOP | 2,500개 / 125명 준비됨 | 구조/출처만 감사, 새 모델 평가에는 사용하지 않음 |
| 원본 Train WAV | 고유 251개만 존재 | 공식 Train 2,500개 전체 확보·검증 |
| 원본 Test WAV | 과거 파일럿의 50개 존재 | 추가 다운로드/SSL 특징 추출/평가 안 함 |
| 개인 WAV | 이전 두 개 존재 | 이번 공개 데이터 학습/평가에 사용 안 함 |

처음에는 `data/gopt/examples/`에 WAV 1개, 이전 파일럿에 300개(공식 Train 250 + Test 50)가 있었다.
Windows의 대소문자 무시 glob을 두 번 합산하지 않고 파일을 한 번씩 세었다.
이번 Train 원음은 약 331 MB, 총 10,354.11초(약 2.876시간), 길이 1.59–20.408초다.

**확인한 무결성:** 공식 Train/Test 화자 분리, Training/Validation 화자·발화 ID 분리,
원본 5,000개와 GOP 음소/단어/문장 라벨 일치, 공식 Git blob 해시 중복 0,
학습/검증 2,500개 디코딩 PCM의 정확한 중복 0.
Training/Validation 문장 텍스트는 소문자·공백 정규화 기준 공유 문장 0이었다.
이는 근접 중복 음성이나 의미가 비슷한 문장을 완전히 검출했다는 주장은 아니다.

출처 고정:

- Corpus commit: `613968e3b0b789fc33936fb5eba1973176ba7d11`.
- 공식 Git tree의 파일 크기·Git blob SHA1: [corpus-files.json](../experiments/phase2/corpus-files.json).
  트리에는 공식 분할 외 파일도 있지만, 다운로드 대상을 공식 Train ID로 한정한다.
- 다운로드 시 크기와 Git blob SHA1을 검사하고, 로컬 SHA256·디코딩 PCM SHA256도 기록한다.
- `data/phase2/manifest.json`: ID, 화자, 원문, 목표 점수, 고정 분할, 소스 경로.
- `data/phase2/audit-initial.json`, `audio-audit.json`: 준비 전 재고·감사·파일별 해시·길이.
- 기존 GOP 특징 보관 출처는 이전에 고정한 JazminVidal/gopt-w다. 이번 SSL 입력은 그 GOP가 아니라 공식 원본 WAV다.

라벨 구조:

| 단위 | 필드 | 사람 점수 척도 | 이번 사용 |
|---|---|---:|---|
| 음소 | `words[].phones`, `phones-accuracy` | 0–2 | 구조 감사만, SSL 음소 평가 미구현 |
| 단어 | `words[].accuracy`, `stress`, `total` | 0–10 | 구조 감사만 |
| 문장 | `accuracy`, `completeness`, `fluency`, `prosodic`, `total` | 0–10 | 완전성 제외 4개 회귀 |

`prosodic`을 코드 출력 이름 `prosody`로 매핑한다. 음소 라벨 0점을 0.1로 올리지 않는다.
문장 라벨은 원본 점수를 유지하고 학습 loss 계산에서만 `/10`, 평가 시 `×10`을 적용한다.

## 3. 후보 모델 비교

아래 GPU/시간 범위는 모델 저자의 보장 사양이 아니라 **이번 16kHz 짧은 발화 실험을 위한 계획 추정**이다.
정확한 성능/메모리는 동일 입력·배치·정밀도에서 측정해야 한다. ASR/SUPERB 성능이 발음 점수 성능을 보장하지 않는다.

| 후보 | 구조/사전학습 | 평가 가능성·SpeechOcean762 적용 | 코드·가중치 | 첫 실험 자원 판단 |
|---|---|---|---|---|
| **WavLM Base+** | CNN + 12층 Transformer, 768차원, 약 94M. masked prediction와 denoising/혼합 음성 학습, gated relative position bias | 문장 회귀 헤드 연결 가능. 음소/단어에는 정렬·별도 헤드 필요 | Microsoft 공식 PyTorch/Transformers 가중치, MIT 링크 확인 | Frozen batch 1은 CPU 가능. GPU 계획 하한 6–8GB, 권장 12GB 이상 |
| wav2vec 2.0 Base | CNN + 12층 Transformer, 약 95M. 잠재 표현 마스킹, 학습된 양자화 목표와 contrastive objective | 같은 문장 헤드로 공정한 encoder 교체 실험 가능. 정렬 없이 음소 점수를 얻는 모델은 아님 | Meta 공식 `facebook/wav2vec2-base`, Apache-2.0 표시 | Base 크기에서 WavLM과 비슷한 자원급. 동일 20개로 실제 속도 비교 필요 |
| HuBERT Base | CNN + 12층 Transformer, 약 95M. offline clustering으로 얻은 hidden-unit 목표의 masked prediction | 문장 평가 및 관련 발음 연구 근거 있음. 단어/음소는 정렬 또는 별도 학습 필요 | Meta 공식 `facebook/hubert-base-ls960`, Apache-2.0 표시 | Base 크기에서 유사. frozen cache 방식은 CPU 실험 가능 |
| MultiPA | HuBERT 기반 다중 과제, Whisper/Charsiu/RoBERTa 등 보조 처리 | 문장 accuracy/fluency/prosody 및 단어 평가. 공개 SpeechOcean762 학습 코드 | Interspeech 2024 저자 코드(MIT), 저자가 링크한 학습 가중치 | 여러 모델·정렬 의존성, Windows 환경 이식 부담. 16–24GB GPU 검토 후 실제 측정 필요 |
| E2E-R | SSL 표현을 발음 점수 회귀에 fine-tune하는 SpeechBrain 구현 | SpeechOcean762 실험 코드, 일부 LSTM 경로는 Kaldi 사용 | IEEE Access 2023 저자 코드(MIT). 학습 완료 평가 가중치의 이용 가능성은 이번에 확인하지 않음 | end-to-end fine-tune은 frozen보다 메모리/시간 증가. 16–24GB 이상을 초기 검토 |
| APSIPA SSL-FT-PRON | 12개 wav2vec 2.0/HuBERT/WavLM 변형, CTC·CNN 고정·일반 회귀 전략 비교 | SpeechOcean762의 문장 accuracy/fluency/prosody/total PCC. CNN만 고정하는 전략은 이번 인코더 전체 고정과 다름 | APSIPA ASC 2024 저자 구현(MIT 표기), HF 가중치 모음 공개. 모델별 가중치 라이선스는 도입 전 별도 확인 | Base부터 XLarge까지 규모가 다름. 전체 fine-tune VRAM은 변형별 측정 필요 |
| GOPT Baseline 01 | 고정 GOP·목표 음소 + 작은 Transformer 평가기 | 음소/단어/문장 평가, 이미 수행한 동일 Validation 비교 기준 | 현재 코드/가중치/실험 보존 | 평가기 학습은 CPU로 충분. GOP 생성기 비용은 별도 |

근거 원문:

- [WavLM 논문](https://arxiv.org/abs/2110.13900), [공식 코드](https://github.com/microsoft/unilm/tree/master/wavlm),
  [Base+ 카드](https://huggingface.co/microsoft/wavlm-base-plus).
  Base+ 사전학습 자료는 Libri-Light 60k + GigaSpeech 10k + VoxPopuli 24k, 합계 약 94k시간이다.
- [wav2vec 2.0 논문](https://arxiv.org/abs/2006.11477), [Base 카드](https://huggingface.co/facebook/wav2vec2-base).
- [HuBERT 논문](https://arxiv.org/abs/2106.07447), [Base 카드](https://huggingface.co/facebook/hubert-base-ls960).
- Kim et al., Interspeech 2022: [Automatic Pronunciation Assessment using Self-Supervised Speech Representation Learning](https://www.isca-archive.org/interspeech_2022/kim22k_interspeech.pdf).
  층별 SSL 표현과 발음 회귀를 연구했다. 논문의 CTC 적응/BLSTM/원고 사용과 이번 frozen 통계 pooling은 다르다.
- Chen et al., Interspeech 2024: [MultiPA 논문](https://www.isca-archive.org/interspeech_2024/chen24c_interspeech.pdf),
  [저자 구현](https://github.com/yuwchen/MultiPA), [저자 가중치](https://huggingface.co/yuwchen/multipa/tree/main).
- Zahran et al., IEEE Access 2023: [논문](https://ieeexplore.ieee.org/document/10255657), [E2E-R 저자 구현](https://github.com/ai-zahran/E2E-R).
- Gong et al., ICASSP 2022: [GOPT 저자 구현](https://github.com/YuanGongND/gopt).
- Lee et al., APSIPA ASC 2024: [논문](https://ieeexplore.ieee.org/abstract/document/10848954),
  [저자 코드·비교 표](https://github.com/hy310/ssl_finetuning), [모델 카드](https://huggingface.co/haeylee/ssl_ft_pron).
  저자의 비교에서도 WavLM이 모든 목표에서 우세하지 않다. 공개 평가용 가중치는 SpeechOcean762 학습 이력이 있으므로
  현재 Validation 오염 가능성을 배제하기 전에는 이번 인코더를 대체하지 않는다. 저자 표의 Test 수치와 우리의 개발용 Validation은 직접 순위 비교하지 않는다.

학습 완료된 발음 평가 checkpoint는 공식 Train 전체나 Test 선택 절차를 이미 봤을 수 있다.
그 가중치에서 시작한 모델을 현재 Validation의 독립 실험처럼 비교하지 않는다. 첫 인코더는 SpeechOcean762 평가용으로 fine-tune하지 않은 일반 SSL checkpoint를 사용한다.

## 4. 첫 후속 모델 선정

**Frozen WavLM Base+ + 학습 가능한 층 혼합/통계 pooling 회귀 헤드**를 선택했다.

선정 근거는 공식 가중치 접근성, 여러 음성 특성을 학습하려는 사전학습 설계, Base 규모의 실행 가능성,
그리고 한 번 계산한 특징을 재사용하며 seed/헤드 실험을 저렴하게 반복할 수 있다는 점이다.
이 근거는 WavLM이 HuBERT/wav2vec 2.0이나 GOPT보다 우수하다는 실험 결과가 아니다.

구체적인 계산:

1. 원본 mono PCM16/16kHz를 float32로 변환. 전체 길이/무음 유지, 자르기·잡음 제거·증강 없음.
2. 고정 버전의 공식 feature extractor 설정 `do_normalize=False`를 그대로 사용. 추가 발화별 진폭 표준화 없음.
3. eval/no-grad 인코더에서 12개 Transformer 층의 프레임 표현을 추출.
4. 각 층의 시간 평균과 모집단 표준편차를 연결해 `12 × 1536` 특징 생성.
5. Training 2,000개로만 각 특징의 평균/표준편차를 계산. Validation에는 같은 값 적용.
6. 학습 가능한 softmax 층 가중치로 통계 특징을 혼합.
7. `1536 → 128 → 4`, GELU, dropout 0.2의 MLP로 문장 점수를 예측.

시간 통계의 층 혼합은 프레임 표현을 혼합한 뒤 표준편차를 구하는 것과 같지 않다.
또한 시간 순서를 버리므로 억양의 정확한 위치나 순서 오류를 직접 모델링하지 못한다.
이번 구조의 한계이자 후속 ablation/시간 모델링의 근거다.

인코더의 실제 파라미터 수는 94,381,936이며 모두 고정한다.
학습 대상은 작은 평가 헤드와 12개 층 혼합 계수다. 원음→특징은 실제 모델 추론이며 ASR 전사 확률로 대체하지 않는다.
체크포인트/모델 파일은 `weights_only=True`로 읽고 원격 사용자 코드를 실행하지 않는다.
버전/모델 파일 해시는 [encoder-lock.json](../experiments/phase2/encoder-lock.json)에 있다.

## 5. 비교·학습·신뢰성 규칙

- Training 2,000 / Validation 500, 기존 화자 분할 고정. seed는 가중치 초기화·shuffle·dropout만 바꾼다.
- AdamW, lr 0.001, weight decay 0.001, batch 64, 최대 150 epoch, patience 20.
- 체크포인트 선택은 **Validation의 네 문장 목표 MSE 평균** 최소. 각 목표 MSE/MAE/PCC와 macro 값 모두 기록.
- seed 7/19/42를 보고하고 최고 seed만 선택하지 않는다. 평균과 표본 표준편차를 보고한다.
- Training 라벨 평균을 모든 Validation에 예측하는 기준선도 평가한다. 상수 예측 PCC는 `null`로 보고한다.
- GOPT best epoch 30으로 **같은 순서의 Validation 500개와 같은 네 문장 목표**를 다시 평가한다.
- GOPT에는 목표 음소 정보가 있고 SSL은 원음만 받는다. 동일 과제·평가 단위 비교이며 입력 정보까지 통제한 ablation은 아니다.
- 기존 GOPT의 epoch 30은 Validation **음소 MSE**로 선택된 고정 checkpoint다. 이번 SSL은 네 문장 MSE로 선택한다.
  GOPT의 다중 단위 학습 목표·선택 기준까지 동일하게 맞춘 비교는 아니며, 보존한 기준 모델과의 실용적 성능 비교로 해석한다.
- 주 seed 7의 SSL–GOPT 차이는 Validation 화자를 재표집하는 paired bootstrap 1,000회로 기술한다.
  화자 내 발화를 독립 표본으로 오인하지 않는다. 다중 목표 보정·개발 자료 선택 편향은 이 구간이 해결하지 않는다.
- Trial 설정 변경은 새 run/프로토콜로 기록한다. 현재 Test를 새 실험의 독립 최종 시험이라고 부르지 않는다.

## 6. 자원·비용 계획

실제 PC: Intel i7-1360P(12 physical/16 logical), RAM 약 15.69 GiB,
Intel Iris Xe와 Arc A350M. 현재 PyTorch는 `2.15.0+cpu`, CUDA 사용 불가다.
WMI의 AdapterRAM 값을 실제 VRAM 용량으로 확정하지 않았다. Intel GPU를 CUDA GPU로 취급하지 않는다.
이번 실험은 CPU thread 4개로 원음 특징을 계산하고 저장된 특징으로 헤드를 학습한다.

실제 시간/RSS/CPU 시간은 추출 이벤트와 각 run의 `resources.json`에 저장한다.
CPU 사용량은 process CPU seconds / wall seconds로 코어 수 환산하며, RSS는 50ms 간격 관측 최대다.
시스템 전체 RAM·전력 소비를 측정한 값은 아니다. CUDA 사용 시 peak allocated/reserved도 기록한다.

| 작업 | 현재 선택 | GPU 계획 추정 |
|---|---|---|
| Frozen Base 특징 추출 | 로컬 CPU, 1개 발화씩, 캐시 재사용 | NVIDIA 6–8GB로 소배치부터 확인, 12GB 이상 권장. 24GB A5000/3090/4090이면 여유 확보 |
| 저장된 특징의 작은 헤드 | 로컬 CPU | GPU 구매/임대가 필수 아님 |
| 상위 층 미세조정 | 아직 실행 안 함 | 16GB 최소 검토, 24GB 권장. 배치·20초 길이·AMP·checkpointing에 따라 달라짐 |
| Large 인코더 전체 fine-tune | 이번에 실행 안 함 | 24–48GB 이상부터 별도 측정. 최초 실험으로 선택하지 않음 |

GPU 속도는 이번 PC에서 측정하지 않았다. Base frozen 전체 추출의 임시 계획 범위는 GPU 5–20분,
소규모 head 학습은 수분 이내를 목표로 하되, 실제 임대 전 20개 벤치마크와 I/O를 포함해 다시 산정해야 한다.
현재 CPU는 Stage B 20개 추출 약 16.7초였으며 전체 소요 시간은 완료 기록을 우선한다.

2026-11-02 확인한 [Runpod 공식 Pods 요금](https://www.runpod.io/pricing)은 A5000 24GB $0.27/h,
RTX 3090 24GB $0.50/h, RTX 4090 24GB $0.89/h였다. 1시간 연산만 단순 계산하면 해당 금액이며
가용성·상품 종류·저장·유휴·세금 등의 추가 비용은 별도다. 결제 시점의 견적이 우선한다.
이 프로젝트 예산에서는 GPU 구매보다 필요할 때 짧은 클라우드 실험이 적합하다.
**이번 유료 사용은 $0. 유료 계정 생성/결제/임대/호출은 비용을 알리고 사용자의 승인을 받은 뒤에만 진행한다.**

### 향후 NVIDIA 환경에서 CUDA를 설치하는 방법

현재 CPU 가상환경과 Baseline 01 환경을 교체하지 않고 별도 `.venv-gpu`를 만든다.
아래는 **NVIDIA 드라이버가 준비된 Windows PC**의 예시이며 현재 Intel GPU에 CUDA를 활성화하는 명령이 아니다.
드라이버 호환성과 wheel index는 [PyTorch 공식 설치 선택기](https://pytorch.org/get-started/locally/)로 확인한다.

```powershell
py -3.11 -m venv .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv-gpu\Scripts\python.exe -m pip install -e ".[research,ssl]"
.\.venv-gpu\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
.\.venv-gpu\Scripts\python.exe -m ipykernel install --user --name speakingt-ssl-gpu --display-name "SpeakingT SSL GPU"
```

Jupyter에서 해당 커널을 선택한다. 현재 노트북의 인코더 장치는 CPU로 명시돼 있으므로 GPU 실험에서는
5번 `FrozenEncoder(device=...)`와 9번 `CONFIG.device`를 함께 확인하고 새 실험으로 기록한다.
GPU와 CPU 특징의 수치 차이를 숨기지 않도록 캐시 provenance가 장치를 검사한다. GPU 추출 실험은
기존 CPU 캐시를 덮어쓰지 말고 독립 작업 복사본/데이터 경로에서 실행한다. 환경 변경 자체는 이번에 수행하지 않았다.

## 7. 후속 실험 결정 순서

1. Stage C/D 결과로 평균 기준선·GOPT 대비 차이, Train/Validation gap, seed 변동을 확인한다.
2. 시간 통계의 정보 손실 가설: 평균 vs 평균+표준편차, 마지막 층 vs 층 혼합, 선형 vs MLP를 한 조건씩 비교한다.
3. 같은 헤드/분할/예산으로 HuBERT Base, wav2vec 2.0 Base를 교체한다. encoder 우열은 결과로 판단한다.
4. 성능 부족과 비용 근거가 확보되면 상위 1–2층 fine-tune 또는 시간 attention을 검토한다.
5. 원고/음소 정보가 accuracy에 필요한지 별도 text-conditioned ablation으로 평가한다.
6. 음소 평가는 정렬 품질 검증과 음소별 라벨 연결부터 시작한다. 문장 PCC 개선을 음소 성능으로 환산하지 않는다.
7. 한국어 모국어/개인/자유 발화 평가는 별도의 데이터·평가 설계로 진행한다. 추가 개인 데이터는 현재 필수 조건이 아니다.

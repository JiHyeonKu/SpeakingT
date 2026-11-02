"""Generate only the independent Phase 3 notebook."""
from pathlib import Path
import nbformat as n
import textwrap

cells=[]
def md(s):cells.append(n.v4.new_markdown_cell(textwrap.dedent(s).strip()))
def code(s):cells.append(n.v4.new_code_cell(textwrap.dedent(s).strip()))
md('''
# SpeakingT Phase 3 — WavLM 상위 층 Fine-tuning
## 1. 목표와 현재 Stage
WAV → 발화별 CNN → 마스크가 있는 Transformer → 12층 통계 pooling → 층 혼합/MLP → 4개 문장 점수.
기본값은 **실제 음성 20개, CPU, 1 epoch 기능 검증**입니다. 이 점수로 성능 향상을 주장하지 않습니다.
전체 2,000/500 학습은 CUDA와 명시적 설정이 필요합니다. 유료 GPU는 비용 승인 후 사용하세요.
처음에는 위에서부터 Shift+Enter로 실행합니다. `[*]`는 계산 중입니다. ■ 버튼으로 중단할 수 있습니다.
Windows 진입점은 `start-phase3.cmd`, 전체 안내는 `docs/PHASE3_GUIDE.md`입니다.
''')
md('''
## 2. 커널과 환경
`Python` 경로가 프로젝트 `.venv`인지 확인하세요. CPU-only 설치에서 CUDA False는 정상입니다.
커널 재시작 후 저장 모델만 복원할 때는 **이 코드 셀과 10번 복원 셀만** 실행하면 됩니다.
''')
code('''
from pathlib import Path
import sys, os, json, csv, gc
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p/'speakingt/partial_wavlm.py').exists())
os.chdir(ROOT); sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import matplotlib.pyplot as plt
from IPython.display import display, Audio, clear_output
from dataclasses import replace
from speakingt import partial_wavlm as p3
torch.set_num_threads(4)
print('Python:',sys.executable)
print(json.dumps(p3.environment(),indent=2))
''')
md('''
## 3. 수정할 설정 — 한곳에서 관리
`top_layers=2`: 맨 위 2개 층(0부터 세면 10,11)을 학습합니다. 1 또는 0도 가능합니다.
`warm_start`는 검증된 dropout .5 seed 7 헤드입니다. `None`이면 새 헤드와 Train-only 정규화를 만듭니다.
encoder LR과 head LR을 분리합니다. 기본 유효 batch는 `2 × accumulation 2 = 4`입니다.
마지막 불완전 batch도 실제 샘플 수로 loss를 나눕니다. 정규화 통계는 학습 중 고정합니다.
GPU를 사용하려면 승인된 환경에서 `DEVICE='cuda'`로 바꾸고 아래 실행 범위를 선택하세요.
GPU pilot은 최대 64개(가장 긴 발화 포함), 1 epoch입니다. 전체 학습 전 반드시 이 측정을 먼저 확인합니다.
''')
code('''
RUN_TRAINING = True
RUN_GPU_PILOT = False
RUN_FULL_GPU = False
DEVICE = 'cpu'
PRECISION = 'fp32'  # CUDA에서만 'fp16' 또는 지원되는 'bf16'
CONFIG = p3.Config(
    top_layers=2, hidden=128, dropout=.5, encoder_lr=1e-5, head_lr=1e-4,
    weight_decay=.01, batch_size=2, accumulation=2, clip_norm=1.,
    epochs=1, patience=3, seed=7, threads=4, device=DEVICE, precision=PRECISION,
    warm_start=p3.WARM_RUN)
if RUN_FULL_GPU and RUN_GPU_PILOT:
    raise ValueError('pilot과 full 중 하나만 선택하세요.')
if RUN_FULL_GPU:
    CONFIG = replace(CONFIG,stage='full',allow_full=True,epochs=10)
elif RUN_GPU_PILOT:
    CONFIG = replace(CONFIG,stage='pilot',epochs=1)
CONFIG.validate()
print(CONFIG)
''')
md('''
## 4. 원음과 화자 분할 확인
고정된 기존 Training/Validation에 속한 ID만 허용합니다. 공식 Test를 선택하는 기능은 없습니다.
Phase 2 파일은 읽기만 합니다. 기존 특징 캐시는 Fine-tuning 입력으로 사용하지 않습니다.
''')
code('''
rows = p3.select_rows(CONFIG.stage)
groups = p3.validate_rows(rows,CONFIG)
print({g:len(v) for g,v in groups.items()})
print(p3.data_identity())
''')
md('''
## 5. 실제 입력 음성 확인
첫 공개 WAV를 재생하고 파형을 봅니다. 무음이나 길이를 임의로 자르지 않습니다.
''')
code('''
sample,target,uid = p3.WaveDataset(rows)[0]
display(Audio(sample.numpy(),rate=16000))
print('ID:',uid,'human scores:',target.tolist())
fig,ax=plt.subplots(figsize=(10,2)); ax.plot(np.arange(len(sample))/16000,sample.numpy(),linewidth=.5)
ax.set(xlabel='Time (s)',ylabel='Amplitude'); display(fig);plt.close(fig)
''')
md('''
## 6. 사전학습 모델과 trainable 파라미터
CNN의 GroupNorm이 padding에 영향을 받지 않도록 CNN은 실제 길이로 처리합니다.
이후 특징을 동적으로 padding하고 attention과 평균/표준편차에서 유효 프레임만 사용합니다.
상위 층만 train 모드로 두고 나머지는 eval/freeze합니다. SpecAugment/LayerDrop은 이번 실험에서 사용하지 않습니다.
아래 모델은 구조 확인용이며, 7번의 학습은 같은 seed/설정으로 별도 생성합니다.
''')
code('''
model,initialization = p3.initialize_model(CONFIG,rows)
print(initialization)
print('Trainable encoder:',sum(p.numel() for p in model.wavlm.parameters() if p.requires_grad))
print('Trainable head:',sum(p.numel() for p in model.head.parameters()))
print(model.head)
wave,mask,y,ids = p3.collate([p3.WaveDataset(rows)[0],p3.WaveDataset(rows)[1]])
model.eval()
with torch.no_grad():
    features,frame_mask=model.features(wave.to(DEVICE),mask.to(DEVICE))
print('Wave:',wave.shape,'features:',features.shape,'valid frames:',frame_mask.sum(1).tolist())
del model,features;gc.collect()
if DEVICE=='cuda': torch.cuda.empty_cache()
''')
md('''
## 7. 실제 학습 — Forward/Backward/가중치 저장
각 epoch에 Train/Validation MSE와 PCC, 첫 optimizer step의 인코더 gradient, 누적 가중치 변화를 기록합니다.
모든 frozen 파라미터의 hash를 비교해 변경되면 중단합니다. 기본 CPU 실행에는 수 분이 걸릴 수 있습니다.
초기 epoch 0도 best 후보입니다. Fine-tuning이 나빠지면 best가 0일 수 있으며 `last.pt`는 마지막 학습 상태입니다.
■로 중단하면 마지막 **완료된 epoch**의 checkpoint에서 새 run으로 재개할 수 있습니다.
''')
code('''
if RUN_TRAINING:
    def update(history):
        clear_output(wait=True)
        print('Stage:',CONFIG.stage,'latest:',history[-1])
        fig=p3.plot_history(history);display(fig);plt.close(fig)
    LAST_RUN=p3.run(CONFIG,on_epoch=update)
    print('Saved run:',LAST_RUN)
    print(p3.read_json(LAST_RUN/'summary.json'))
else:
    print('학습을 건너뛰었습니다. 10번에서 이전 checkpoint를 복원할 수 있습니다.')
''')
md('''
## 8. 저장 그래프와 실제 자원
MSE는 원래 0–10 점수 단위입니다. 내부 학습 loss는 `/10` 척도이므로 값이 다릅니다.
CPU RSS와 GPU allocated/reserved는 다른 측정값입니다. 결제 금액은 이 모니터가 측정하지 않습니다.
''')
code('''
if 'LAST_RUN' in globals():
    with (LAST_RUN/'metrics.csv').open(encoding='utf-8') as f: history=list(csv.DictReader(f))
    fig=p3.plot_history(history);display(fig);plt.close(fig)
    print(p3.read_json(LAST_RUN/'resources.json'))
''')
md('''
## 9. 네 목표의 MSE/MAE/PCC와 gradient
작은 표본의 PCC는 불안정하거나 정의되지 않을 수 있습니다. 이번 Stage B는 기능 확인입니다.
전체 검증 결과도 반복 사용한 개발 자료의 수치이며 독립 Test 일반화 성능은 아닙니다.
''')
code('''
if 'LAST_RUN' in globals():
    print(json.dumps(p3.read_json(LAST_RUN/'best-metrics.json'),indent=2))
    for f in sorted(LAST_RUN.glob('metrics-*.json')):
        record=p3.read_json(f)
        print(f.name,'encoder change:',record['encoder_delta_l2'],'gradients:',record['gradients'])
''')
md('''
## 10. 새 커널에서 복원·예측 재생성
2번 환경 셀 실행 후 `RESTORE_RUN`에 이전 run 경로를 넣습니다. 아래는 기본적으로 학습된 `last.pt`를 복원합니다.
`CHECKPOINT='best'`로 바꾸면 선택된 모델을 읽습니다. 출처/코드/hash가 다르면 자동 무시하지 않고 실패합니다.
학습과 달리 예측만 하므로 GPU checkpoint도 CPU로 복원할 수 있습니다. 장치가 다르면 수치 오차가 생길 수 있습니다.
''')
code('''
RESTORE_RUN = ''  # 예: 'runs/phase3/실제-run-이름'
CHECKPOINT = 'last'  # 'best'도 가능
restore_run=ROOT/RESTORE_RUN if RESTORE_RUN else globals().get('LAST_RUN')
if restore_run is None: raise ValueError('RESTORE_RUN에 저장 경로를 넣으세요.')
restored,saved,restored_rows=p3.restore(restore_run/(CHECKPOINT+'.pt'),device='cpu')
metrics,predictions,ids=p3.evaluate(restored,[r for r in restored_rows if r['split']=='validation'],saved['config']['batch_size'])
with np.load(restore_run/(CHECKPOINT+'-validation-predictions.npz')) as prior:
    assert np.array_equal(ids,prior['ids'])
    print('Max saved prediction difference:',float(np.abs(predictions-prior['predictions']).max()))
print('Restored epoch:',saved['epoch'],metrics)
del restored;gc.collect()
''')
md('''
## 11. 중단 후 이어 학습 — 명시적으로 선택
완료된 epoch의 model/optimizer/scheduler/scaler/RNG를 함께 복원합니다. 기존 run 대신 새 run에 저장합니다.
같은 설정/분할/장치를 유지하고 epoch 상한만 늘릴 수 있습니다. CPU smoke는 총 2 epoch까지입니다.
epoch 도중 중단한 미완료 batch는 재개 시 해당 epoch 처음부터 다시 계산합니다.
''')
code('''
RESUME_TRAINING = False
if RESUME_TRAINING:
    resume_cfg=replace(p3.Config(**saved['config']),epochs=2)
    resumed_run=p3.run(resume_cfg,resume=restore_run/'last.pt')
    print('Resumed in a new run:',resumed_run)
''')
md('''
## 12. GPU로 넘어가기
기본 CPU 검증 뒤 승인된 NVIDIA 환경에서 3번의 `RUN_GPU_PILOT=True`, `DEVICE='cuda'`를 설정합니다.
우선 fp32를 확인하고, 지원되는 fp16/bf16으로 같은 pilot을 비교하세요. 64개 pilot은 긴 발화도 포함합니다.
측정한 시간/VRAM으로 비용을 계산한 후 전체 2,000/500 실행을 승인받습니다.
전체 실행은 `RUN_FULL_GPU=True`가 필요하며, `epochs=10`부터 시작하도록 준비되어 있습니다.
OOM이면 마지막 완료 checkpoint를 보존하고 batch를 줄입니다. batch 변경은 정확한 resume가 아니라 새 실험입니다.
자세한 설치·명령·비용 계산·한계는 `docs/PHASE3_GUIDE.md`에 있습니다.
''')
nb=n.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3 (ipykernel)','language':'python','name':'python3'}})
path=Path(__file__).resolve().parents[1]/'notebooks/Phase3_Partial_WavLM.ipynb'
n.write(nb,path);print(path)

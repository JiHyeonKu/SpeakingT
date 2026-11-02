"""Build the independent Korean Phase 2 notebook; never edits the GOPT notebook."""
import json
from pathlib import Path
import textwrap

cells=[]
def cell(kind,source):
    value={'cell_type':kind,'metadata':{},'id':f'ssl-{len(cells):02d}',
           'source':textwrap.dedent(source).strip().splitlines(keepends=True)}
    # splitlines above retains only original line endings; append final newline.
    value['source']=[line if line.endswith('\n') else line+'\n' for line in value['source']]
    if kind=='code': value.update(execution_count=None,outputs=[])
    cells.append(value)

cell('markdown','''
# SpeakingT Phase 2 — Frozen WavLM 문장 평가 연구
## 1. 목표와 모델
WAV → **고정 WavLM Base+** → 12개 층의 시간 평균/표준편차 → 학습 가능한 층 혼합과 MLP → 문장 점수 4개.
목표는 accuracy, fluency, prosody, total이며 사람 라벨은 0–10입니다. **음소 PCC와 비교하지 않습니다.**
이것은 기존 논문의 완전 재현이 아닌, 공개 SSL 인코더를 이용한 첫 후속 실험입니다.

Baseline 01은 `run-20261102-014018-444335`, best epoch 30으로 보존합니다.
같은 2,000/500 화자 분할을 그대로 쓰며 공식 Test는 이 노트북에서 사용하지 않습니다.
GOPT는 목표 음소 정보를 받고, 이번 SSL 헤드는 원음만 받으므로 정보 조건의 차이도 기록합니다.

**처음에는 Shift+Enter로 위에서부터 실행하세요. 기본 실행은 실제 20개 음성의 Stage B 검증입니다.**
전체 학습은 9번 설정에서 직접 켭니다. 유료 서비스 호출은 없습니다. 자세한 안내: `docs/PHASE2_GUIDE.md`.
''')
cell('markdown','''
## 2. Python 환경과 CPU/GPU 확인
프로젝트의 `.venv` 커널을 사용합니다. `CUDA: False`는 현재 CPU 설치에서 정상입니다.
패키지가 없다면 프로젝트 PowerShell에서 `.\\.venv\\Scripts\\python.exe -m pip install -e ".[research,ssl]"`를 실행하세요.
아래 코드는 설치를 자동 변경하지 않습니다. 커널 재시작 뒤에도 이 셀부터 실행합니다.
''')
cell('code','''
from pathlib import Path
import sys, os, json, csv, gc
ROOT = next((p for p in (Path.cwd(), *Path.cwd().parents)
             if (p/'speakingt/ssl_workbench.py').exists()), None)
if ROOT is None:
    raise RuntimeError('SpeakingT 프로젝트 폴더에서 열어주세요.')
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import numpy as np
import torch, psutil
import matplotlib.pyplot as plt
from IPython.display import Audio, display, clear_output
from speakingt import ssl_workbench as ssl
from speakingt import phase2_data as corpus
from speakingt.phase2_data import DATA, TARGETS, write_json
torch.set_num_threads(4)
print('Python:', sys.executable)
print('Torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('RAM GiB:', round(psutil.virtual_memory().total/2**30, 2))
print('Workspace:', ROOT)
''')
cell('markdown','''
## 3. 데이터셋과 보존된 Baseline 검증
원본 GOPT run의 SHA256을 검사하고 정확히 같은 ID/화자 분할을 사용합니다.
공식 Train 2,500개 원음만 준비합니다. 기존 캐시는 검증 후 재사용합니다.
원음 출처: 공식 SpeechOcean762 고정 commit, CC BY 4.0. Test는 메타데이터 감사에만 포함됩니다.
다른 PC에서는 먼저 Baseline 01의 로컬 run 백업과 `data/gopt/`를 복원해야 합니다.
''')
cell('code','''
baseline_audit = corpus.preserve_baseline()
manifest = corpus.audit_metadata()
audio_audit = corpus.prepare_audio(workers=8)
print(json.dumps(manifest['audit'], ensure_ascii=False, indent=2))
print('Verified WAVs:', audio_audit['count'], '| hours:', round(audio_audit['duration_seconds']/3600, 3))
''')
cell('markdown','''
## 4. 실제 WAV 로딩·재생·시각화
원음의 무음과 길이를 보존합니다. 파형/스펙트로그램은 입력 확인용이며 발음 정오 판정이 아닙니다.
`SAMPLE_INDEX`로 공개 발화를 바꿀 수 있습니다. 개인 녹음은 필요하지 않습니다.
''')
cell('code','''
SAMPLE_INDEX = 0
sample_row = manifest['rows'][SAMPLE_INDEX]
sample_path = DATA/sample_row['audio_path']
waveform, pcm_hash = corpus.read_audio(sample_path)
display(Audio(waveform, rate=16000))
print('ID:', sample_row['id'], '| split:', sample_row['split'])
print('Text:', sample_row['text'])
print('Human scores:', dict(zip(TARGETS, sample_row['targets'])))
fig, axes = plt.subplots(2, 1, figsize=(10, 4))
axes[0].plot(np.arange(len(waveform))/16000, waveform, linewidth=.5)
axes[0].set(xlabel='Time (s)', ylabel='Amplitude')
axes[1].specgram(waveform, NFFT=256, Fs=16000, noverlap=192)
axes[1].set(xlabel='Time (s)', ylabel='Frequency (Hz)')
fig.tight_layout(); display(fig); plt.close(fig)
''')
cell('markdown','''
## 5. 사전학습 모델 로딩 — Stage A
공식 고정 가중치(약 378 MB)를 확인합니다. 처음만 다운로드하며 원음 업로드는 없습니다.
아래에서는 **모든 인코더 파라미터의 requires_grad=False**를 확인합니다.
''')
cell('code','''
from scripts.prepare_ssl_encoder import prepare_encoder
model_folder = prepare_encoder()
with ssl.ResourceMonitor() as model_load_resources:
    encoder = ssl.FrozenEncoder(device='cpu', threads=4, folder=model_folder)
print('Encoder parameters:', encoder.parameter_count)
print('Trainable encoder parameters:', sum(p.numel() for p in encoder.model.parameters() if p.requires_grad))
print('Model load resources:', model_load_resources.result)
''')
cell('markdown','''
## 6. 음성 특징 추출
음성 하나의 전체 샘플을 사용하므로 padding/잘림이 없습니다. 고정한 공식 전처리기는 `do_normalize=False`입니다.
따라서 PCM을 float32 진폭으로 바꾼 뒤 추가 발화별 표준화는 하지 않습니다. 헤드 입력의 Train-only 정규화와는 별개입니다.
각 Transformer 층의 프레임 특징에서 평균과 모집단 표준편차를 구해 `12 × 1536`으로 만듭니다.
이렇게 하면 파형의 자세한 시간 순서는 잃습니다. 시간 모델링은 이후 개선 가설로 다룹니다.
''')
cell('code','''
with ssl.ResourceMonitor() as sample_resources:
    sample_features, tensor_shapes = encoder.extract(waveform)
print('Tensor dimensions:', tensor_shapes)
print('Feature sample:', sample_features[0, :8])
print('Extraction resources:', sample_resources.result)
''')
cell('markdown','''
## 7. 특징과 사람 라벨 연결
작은 검증에는 고정된 Training 16개와 Validation 4개를 씁니다. 이 20개 결과를 성능 성과로 해석하지 않습니다.
ID로 연결하며, 파일 순서와 점수를 따로 섞지 않습니다. 특징 캐시는 원음·모델 해시를 검사합니다.
''')
cell('code','''
preview_rows = [r for r in manifest['rows'] if r['split']=='train'][:16] + [r for r in manifest['rows'] if r['split']=='validation'][:4]
preview_report = ssl.extract_features(encoder, preview_rows)
preview_values = [ssl.read_feature(DATA/r['file'], {'id':r['id']})[0] for r in preview_report['rows']]
small_data = {
    'features':np.stack(preview_values),
    'targets':np.asarray([r['targets'] for r in preview_rows], dtype=np.float32),
    'ids':np.asarray([r['id'] for r in preview_rows]),
    'speakers':np.asarray([r['speaker'] for r in preview_rows]),
    'split':np.asarray([r['split'] for r in preview_rows])}
print('X:', small_data['features'].shape, '| y:', small_data['targets'].shape)
print('First ID/targets:', small_data['ids'][0], small_data['targets'][0])
''')
cell('markdown','''
## 8. 평가 헤드 구조
12개 층에 대한 softmax 혼합 계수와 작은 MLP만 학습합니다. 인코더는 고정입니다.
이번 구현은 프레임별 attention이 아니라 **시간 통계 특징의 층 혼합**입니다.
''')
cell('code','''
head_example = ssl.AssessmentHead(hidden=128, dropout=.2)
print(head_example)
print('Trainable head parameters:', sum(p.numel() for p in head_example.parameters()))
del head_example
''')
cell('markdown','''
## 9. 수정할 학습 설정 — 한곳에서 관리
`RUN_FULL_TRAINING=False`: Stage B, 실제 음성 20개로 3 epoch 기능 검증.
`True`: Stage C, 기존 Training 2,000개 / Validation 500개 전체 학습.
`RUN_MULTI_SEED=True`: Stage D, 같은 분할·설정에서 seed 7/19/42를 각각 실행합니다.

전체 실행을 켜면 첫 특징 추출은 CPU에서 수십 분 걸릴 수 있습니다. 캐시가 완성되면 재사용합니다.
각 실행은 새 폴더에 저장하며 기존 run을 덮어쓰지 않습니다. 최종 Test 실행 옵션은 제공하지 않습니다.
''')
cell('code','''
RUN_FULL_TRAINING = False
RUN_MULTI_SEED = False
SEEDS = [7, 19, 42]
CONFIG = ssl.TrainConfig(
    epochs=150, patience=20, learning_rate=1e-3, weight_decay=1e-3,
    batch_size=64, hidden=128, dropout=.2, seed=7, threads=4, device='cpu')
# 자동 통합 검증에서도 위와 같은 공개 설정을 사용하며, 환경변수는 실행 범위만 선택합니다.
if os.environ.get('SPEAKINGT_SSL_FULL') == '1':
    RUN_FULL_TRAINING = True
    RUN_MULTI_SEED = True
print(CONFIG)
print('Full training:', RUN_FULL_TRAINING, '| Multi-seed:', RUN_MULTI_SEED)
''')
cell('markdown','''
## 10. 실제 학습 — Forward / Backward / 저장
먼저 한 배치에서 `loss.backward()`와 `optimizer.step()`의 실제 변화를 보여줍니다.
이 데모는 본 학습과 별도 모델이며 본 학습은 seed를 다시 설정합니다.
■ 버튼으로 중단하면 완료된 epoch 기록과 마지막 완료 체크포인트가 남습니다.
''')
cell('code','''
from dataclasses import replace
from datetime import datetime
tr_mask, _ = ssl.validate_dataset(small_data)
small_x, _, _ = ssl.normalize_features(small_data['features'], tr_mask)
ssl.initialize_seed(CONFIG)
demo_head = ssl.AssessmentHead(CONFIG.hidden, CONFIG.dropout)
demo_optimizer = torch.optim.AdamW(demo_head.parameters(), lr=CONFIG.learning_rate)
before = demo_head.network[0].weight.detach().clone()
loss = torch.nn.functional.mse_loss(demo_head(torch.from_numpy(small_x[:8])), torch.from_numpy(small_data['targets'][:8]/10))
demo_optimizer.zero_grad(); loss.backward()
print('Actual loss:', float(loss.detach()), '| gradient norm:', float(demo_head.network[0].weight.grad.norm()))
backward_demo = {'loss':float(loss.detach()), 'gradient_norm':float(demo_head.network[0].weight.grad.norm())}
demo_optimizer.step()
print('Actual weight change:', float((demo_head.network[0].weight.detach()-before).norm()))
backward_demo['weight_change_l2'] = float((demo_head.network[0].weight.detach()-before).norm())
assert all(p.grad is None and not p.requires_grad for p in encoder.model.parameters())
backward_demo['encoder_frozen_without_gradients'] = True
del demo_head, demo_optimizer

if RUN_FULL_TRAINING:
    feature_report = ssl.extract_features(encoder)
    dataset = ssl.feature_dataset()
    selected_seeds = SEEDS if RUN_MULTI_SEED else SEEDS[:1]
else:
    dataset = small_data
    selected_seeds = SEEDS[:1]
del encoder
gc.collect()
RUN_DIRS = []
for seed in selected_seeds:
    cfg = replace(CONFIG, seed=seed) if RUN_FULL_TRAINING else replace(CONFIG, seed=seed, epochs=3, patience=3, batch_size=8)
    scope = 'full' if RUN_FULL_TRAINING else 'smoke'
    RUN = ROOT/'runs/ssl'/datetime.now().strftime(f'notebook-{scope}-seed{seed}-%Y%m%d-%H%M%S-%f')
    def update_plot(history):
        clear_output(wait=True)
        print('Run:', RUN.name, '| latest:', history[-1])
        fig = ssl.plot_history(history); display(fig); plt.close(fig)
    history = ssl.train(dataset, RUN, cfg, on_epoch=update_plot)
    fig = ssl.plot_history(history); fig.savefig(RUN/'learning-curves.png', dpi=160); plt.close(fig)
    write_json(RUN/'model-load-resources.json', model_load_resources.result)
    write_json(RUN/'backward-demo.json', backward_demo)
    RUN_DIRS.append(RUN)
    print((RUN/'summary.json').read_text(encoding='utf-8'))
print('Saved runs:', [str(p) for p in RUN_DIRS])
print('Actual one-batch backward verification:', backward_demo)
''')
cell('markdown','''
## 11. Training / Validation 그래프
loss는 원래 0–10 점수 단위의 문장 MSE로 표시합니다. 학습 내부의 `/10` 스케일 loss와 혼동하지 마세요.
Train만 좋아지고 Validation이 악화되면 과적합 가능성을 확인합니다.
''')
cell('code','''
for run in RUN_DIRS:
    with (run/'metrics.csv').open(encoding='utf-8') as f:
        saved_history = list(csv.DictReader(f))
    print(run.name)
    fig = ssl.plot_history(saved_history); display(fig); plt.close(fig)
''')
cell('markdown','''
## 12. MSE / MAE / PCC와 평균 예측 기준선
best는 4개 문장 목표의 Validation MSE 평균으로 선택합니다. PCC는 정답률이 아닙니다.
Training의 사람 점수 평균만 예측하는 모델보다 나은지도 확인합니다. 작은 Stage B 수치는 성능 근거가 아닙니다.
''')
cell('code','''
for run in RUN_DIRS:
    best = json.loads((run/'best-metrics.json').read_text(encoding='utf-8'))
    print(run.name, '| best epoch:', best['epoch'])
    for name in TARGETS:
        print(name, best['validation']['targets'][name])
    print('Mean baseline:', json.loads((run/'mean-baseline.json').read_text(encoding='utf-8')))
    print('Measured resources:', json.loads((run/'resources.json').read_text(encoding='utf-8')))
''')
cell('markdown','''
## 13. 커널 재시작 후 체크포인트 복원
다시 열었으면 **2번 환경 셀**을 먼저 실행하고, 아래 `RESTORE_RUN`에 이전 run의 프로젝트 상대 경로를 넣습니다.
이 셀은 이전 셀의 `dataset` 변수 없이도 캐시와 저장된 분할·정규화를 복원합니다. 새 학습은 하지 않습니다.
`best.pt`, `last.pt`, `initial.pt`, 설정/분할/정규화/지표는 run에 보존됩니다. 이어 학습 기능은 아직 제공하지 않습니다.
''')
cell('code','''
RESTORE_RUN = ''  # 예: 'runs/ssl/notebook-full-seed7-...'
if RESTORE_RUN:
    restored_run = ROOT/RESTORE_RUN
elif 'RUN_DIRS' in globals() and RUN_DIRS:
    restored_run = RUN_DIRS[0]
else:
    raise RuntimeError('이전 run 경로를 RESTORE_RUN에 넣어주세요.')
restored_dataset = ssl.restore_run_dataset(restored_run)
restored_model, saved = ssl.restore(restored_run/'best.pt')
restored_metrics, restored_predictions = ssl.evaluate_checkpoint(restored_run/'best.pt', restored_dataset)
recorded = json.loads((restored_run/'best-metrics.json').read_text(encoding='utf-8'))['validation']
assert abs(restored_metrics['macro_mse']-recorded['macro_mse']) < 1e-5
print('Restored epoch:', saved['epoch'], '| Validation MSE:', restored_metrics['macro_mse'])
print('Layer mixture weights:', restored_model.layer_logits.softmax(0).detach().numpy())
''')
cell('markdown','''
## 14. Baseline 01과 공정한 문장 평가 비교 — Stage D
전체 Validation 500개를 사용한 run에서만 비교합니다. 같은 ID·라벨·점수 단위를 검사합니다.
GOPT는 음소 MSE로 선택된 기존 epoch 30을 고정합니다. SSL의 문장 MSE 선택과 학습 목표·입력 조건까지 동일한 ablation은 아닙니다.
seed 7/19/42의 평균·표준편차를 보고하며 최고 seed만 골라 보고하지 않습니다.
주 비교 seed 7은 화자 단위 paired bootstrap 1,000회도 계산합니다. 개발용 Validation의 구간이므로 독립 Test 성과는 아닙니다.
''')
cell('code','''
if RUN_FULL_TRAINING:
    from speakingt.ssl_analysis import summarize
    COMPARISON = ROOT/'reports/phase2'/datetime.now().strftime('comparison-%Y%m%d-%H%M%S-%f')
    gopt_metrics = ssl.gopt_comparison(dataset, COMPARISON)
    comparison = summarize(RUN_DIRS, COMPARISON)
    for name, value in comparison['seed_aggregate'].items():
        print(name, '| GOPT PCC:', value['pcc']['gopt'], '| SSL PCC mean/SD:', value['pcc']['mean'], value['pcc']['sample_std'])
    from IPython.display import Image
    display(Image(filename=str(COMPARISON/'comparison.png')))
    write_json(COMPARISON/'runs.json', {'runs':[p.relative_to(ROOT).as_posix() for p in RUN_DIRS], 'test_evaluated':False})
    print('Comparison saved:', COMPARISON)
else:
    print('20개 기능 검증은 GOPT 전체 Validation과 비교하지 않습니다. 전체 학습 후 비교하세요.')
''')
cell('markdown','''
## 15. 다음 실험과 한계
- 현재 인코더와 시간 통계를 고정한 작은 헤드의 결과부터 검토합니다. 성능 향상은 실제 비교 결과로 판단합니다.
- 결과가 낮으면 시간 순서 손실, 목표 음소/텍스트 정보 차이, 데이터 규모, 화자/나이 영향, 정규화를 먼저 살펴봅니다.
- 다음 ablation 후보는 마지막 층 vs 층 혼합, 평균만 vs 평균+표준편차, 선형 vs MLP입니다. 한 조건씩 바꿉니다.
- 이후 Wav2Vec2/HuBERT를 같은 조건으로 비교하거나 상위 인코더 층만 미세조정할 수 있습니다. 유료 GPU는 별도 승인 후 사용합니다.
- 공식 Test는 이미 관찰한 자료입니다. 새 모델 선택에 재사용하지 않고, 향후 별도의 잠금 평가 자료/프로토콜을 마련합니다.
- 문장 점수 예측은 음소 오류 위치·구체적인 억양 교정·자유 발화 일반화를 입증하지 않습니다.

출처: [WavLM](https://huggingface.co/microsoft/wavlm-base-plus), [SpeechOcean762](https://www.openslr.org/101/).
상세 선정 근거와 자원 계획은 `docs/PHASE2_RESEARCH.md`, 사용법은 `docs/PHASE2_GUIDE.md`에 있습니다.
''')

nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3 (ipykernel)','language':'python','name':'python3'},
                             'language_info':{'name':'python','version':'3.11'}},'nbformat':4,'nbformat_minor':5}
path=Path(__file__).resolve().parents[1]/'notebooks/Phase2_Frozen_WavLM.ipynb'
path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
print(path)

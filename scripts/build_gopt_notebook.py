"""Build the reviewable Korean GOPT notebook; no network or training here."""
import json
from pathlib import Path
import textwrap

cells = []


def cell(kind, source):
    source = textwrap.dedent(source).strip() + "\n"
    value = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True),
             "id": f"gopt-{len(cells):02d}"}
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    cells.append(value)


cell("markdown", """
# SpeakingT — 공개 GOPT 모델을 직접 실행하고 학습하기

목표는 내가 쓸 영어 말하기 도구를 만드는 것입니다. 이 노트북은 그 첫 단계인 **공개 평가 모델 실행과 재학습**입니다.

**지금 되는 것:** 공개 음성 재생 → 음소/사람 점수 확인 → 공개 가중치 예측 → 직접 학습 → epoch별 그래프·체크포인트 비교.

**아직 연결되지 않은 것:** 임의의 내 WAV/M4A → Kaldi GOP 특징 생성. 이 노트북에 내 음성을 올려도 아직 자동 채점하지 않습니다.
학습 중 바뀌는 것은 GOPT 평가 모델의 가중치입니다. 이미 추출된 음향 특징을 사용하므로 기반 ASR 모델은 학습하지 않습니다.

처음에는 위에서부터 `Shift+Enter`로 한 셀씩 실행하세요. **전체 실행은 설정된 학습 epoch까지 실행합니다.**
계정·유료 API 키가 필요하지 않습니다. 로컬 CPU 실행에 유료 GPU 비용은 없습니다. Colab의 유료 런타임은 선택한 경우 별도 과금됩니다.

[GOPT 저자 코드](https://github.com/YuanGongND/gopt) · [SpeechOcean762](https://github.com/jimbozhang/speechocean762)
""")
cell("markdown", """
## 1. 실행 환경
로컬에서는 프로젝트의 `.venv` 커널을 사용합니다. Colab에서는 이 셀이 공개 SpeakingT 저장소와 필요한 패키지를 설치합니다.
다운로드 대상은 공개 코드·데이터뿐이며 개인 녹음은 전송하지 않습니다.
""")
cell("code", """
from pathlib import Path
import sys, os, subprocess, importlib.util

if importlib.util.find_spec('google.colab'):
    ROOT = Path('/content/SpeakingT')
    if not ROOT.exists():
        subprocess.run(['git', 'clone', '--depth', '1',
                        'https://github.com/JiHyeonKu/SpeakingT.git', str(ROOT)], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', str(ROOT)+'[research]'], check=True)
else:
    ROOT = next((p for p in (Path.cwd(), *Path.cwd().parents)
                 if (p/'speakingt/gopt_workbench.py').exists()), None)
    if ROOT is None:
        raise RuntimeError('SpeakingT 프로젝트 안에서 노트북을 열어주세요.')
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import json, torch, numpy as np
from IPython.display import Audio, display, clear_output
from scripts.prepare_gopt import prepare, sample_audio
from speakingt import gopt_workbench as workbench
from speakingt.vendor.gopt import GOPT
torch.set_num_threads(4)
DATA = ROOT/'data/gopt'
print('Python:', sys.version.split()[0], '| Torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('Workspace:', ROOT)
""")
cell("markdown", """
## 2. 공개 자료 다운로드와 검증
약 190 MB의 고정 버전 파일을 캐시합니다. 재실행하면 해시를 검사하고 재사용합니다.
GOPT의 원래 Dropbox 링크가 비활성화되어, GOPT 형식 자료를 보관한 공개 파생 저장소를 사용합니다.
**5,000개 발화의 ID·음소 순서·음소/단어/문장 사람 점수를 공식 SpeechOcean762와 전부 대조**합니다.
파생 저장소의 dev/new_te는 합쳐서 공식 test 2,500개로 복원합니다. 자체 학습의 validation은 공식 train에서 따로 나눕니다.
출처·고정 버전·SHA256은 `experiments/gopt/assets.json`에 있습니다. 데이터 라이선스는 CC BY 4.0입니다.
""")
cell("code", """
manifest = prepare(DATA)
all_train = workbench.load_data(DATA, 'train')
print('Features:', all_train['feat'].shape, '= 발화 × 최대 50음소 × 84차원 GOP 특징')
print('Human phone labels:', all_train['label_phn'].shape)
""")
cell("markdown", """
## 3. 실제 음성 한 개와 모델 입력 확인
아래 재생은 SpeechOcean762의 실제 학습자 녹음입니다. 중국어 모국어 학습자 자료이며 미국 원어민의 정답 음성은 아닙니다.
음소 점수는 0–2, 단어/문장 점수는 0–10입니다. 먼저 소리를 듣고 라벨을 살펴보세요.
""")
cell("code", """
SAMPLE_INDEX = 0   # 0~2499 중 바꿀 수 있습니다.
sample_id = str(all_train['ids'][SAMPLE_INDEX])
audio_path = sample_audio(sample_id, DATA)
display(Audio(filename=str(audio_path)))
human_scores = json.loads((DATA/'assets/scores.json').read_text(encoding='utf-8'))
phone_map = json.loads((DATA/'assets/phn_dict.json').read_text())
phone_names = {number: name for name, number in phone_map.items()}
score = human_scores[sample_id]
print('Utterance ID:', sample_id)
print('Text:', score['text'])
print('Human sentence scores:', {key: score[key] for key in ('accuracy','fluency','prosodic','total')})
phone_labels = all_train['label_phn'][SAMPLE_INDEX]
print('Phone / human score:', [(phone_names[int(p)], float(s)) for p,s in phone_labels if s >= 0])
""")
cell("markdown", """
## 4. 이미 학습된 모델 불러오기
공식 GOPT 구조와 공식 LibriSpeech 체크포인트를 사용합니다. 가중치 해시와 모든 파라미터 이름·크기를 검사합니다.
여기서는 **추론만** 합니다. 위 샘플은 공개 가중치의 학습에 포함됐으므로 아래 예측은 일반화 성능 증거가 아닙니다.
GOPT의 prosody 출력은 사람이 준 운율 점수의 예측입니다. 구체적인 F0 곡선 교정이나 '미국인 유사도'와 같지 않습니다.
""")
cell("code", """
pretrained, pretrained_norm = workbench.load_pretrained(DATA)
print(pretrained)
print('Parameters:', sum(parameter.numel() for parameter in pretrained.parameters()))
sample = workbench.subset(all_train, np.array([SAMPLE_INDEX]))
_, prediction = workbench.evaluate(pretrained, sample, pretrained_norm, return_predictions=True)
print('Model sentence predictions:', dict(zip(workbench.UTT_NAMES, prediction['utterance'][0].round(3).tolist())))
print('Model phone predictions:', prediction['phone'][0, phone_labels[:,1] >= 0].round(3).tolist())
""")
cell("markdown", """
## 5. 네가 바꿀 학습 설정
우선 10 epoch를 제안합니다. 1 epoch는 학습 자료 전체를 한 번 사용하는 것입니다.
비교 실험에서는 learning rate 또는 epoch 등 **한 조건씩** 바꾸고 새 run 이름을 사용하세요.

공식 pretrained 모델은 공식 train 전체를 이미 봤으므로, 공정한 validation을 위해 여기서는 **같은 GOPT 구조를 새 가중치로 재학습**합니다.
음향 모델을 처음부터 새로 만드는 것이 아닙니다. 원본 구조·세 종류의 MSE 손실·Adam을 유지하고, 독립 검증 분할·일정 learning rate·조기 종료를 사용합니다.
따라서 논문의 학습 절차와 완전히 동일한 재현이라고 주장하지 않습니다.
""")
cell("code", """
MAX_EPOCHS = 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 25
SEED = 7
PATIENCE = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 자동 통합 검증에서만 짧게 실행; 사용자가 실행할 때는 위 설정 그대로 사용합니다.
if os.environ.get('SPEAKINGT_NOTEBOOK_SMOKE') == '1':
    MAX_EPOCHS = 2
config = workbench.TrainConfig(epochs=MAX_EPOCHS, learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE, seed=SEED, patience=PATIENCE, device=DEVICE)
train_data, validation_data = workbench.split_training(all_train, SEED, config.validation_speakers)
print('Train:', len(train_data['ids']), 'Validation:', len(validation_data['ids']))
print('Speaker overlap:', set(train_data['speakers']) & set(validation_data['speakers']))
print('Test는 이 학습 셀에서 읽지 않습니다.')
print(config)
""")
cell("markdown", """
## 6. 역전파 한 번을 눈으로 확인
실험용 모델 복사본에서만 한 배치를 학습해 봅니다. `loss.backward()`가 기울기를 계산하고 `optimizer.step()`이 가중치를 바꿉니다.
다음 본 학습은 seed를 다시 설정한 새 모델로 시작하므로 이 데모가 결과에 섞이지 않습니다.
""")
cell("code", """
workbench.seed_everything(config)
demo_model = GOPT(embed_dim=24, num_heads=1, depth=3, input_dim=84)
normalization = workbench.training_normalization(train_data)
demo_loader = torch.utils.data.DataLoader(workbench.tensors(train_data, normalization), batch_size=BATCH_SIZE)
x, phones, phone_targets, utterance_targets, word_targets = next(iter(demo_loader))
optimizer = torch.optim.Adam(demo_model.parameters(), lr=LEARNING_RATE)
before = demo_model.in_proj.weight.detach().clone()
pred = workbench.outputs(demo_model, x, phones)
loss, components = workbench.losses(pred, phone_targets, utterance_targets, word_targets)
optimizer.zero_grad()
loss.backward()
print('Loss:', float(loss.detach()), 'Components:', {k:float(v.detach()) for k,v in components.items()})
print('Input projection gradient norm:', float(demo_model.in_proj.weight.grad.norm()))
optimizer.step()
print('Input projection weight change:', float((demo_model.in_proj.weight.detach()-before).norm()))
del demo_model, optimizer
""")
cell("markdown", """
## 7. 학습 시작 — epoch별 실시간 그래프
이 셀을 실행하면 실제 학습이 시작됩니다. 중단하려면 Jupyter의 **■ 중단** 버튼을 누르세요.
완료한 epoch마다 CSV, 상세 평가, 체크포인트와 TensorBoard 로그를 저장합니다. 이전 실험은 덮어쓰지 않습니다.
그래프의 epoch 0은 학습 전 기준선입니다. best는 validation 음소 MSE로 선택하며, test는 학습 중 사용하지 않습니다.
""")
cell("code", """
from datetime import datetime
import matplotlib.pyplot as plt
RUN = ROOT/'runs/gopt'/datetime.now().strftime('run-%Y%m%d-%H%M%S-%f')
def live_plot(history):
    clear_output(wait=True)
    print('Experiment:', RUN.name, '| Latest:', history[-1])
    fig = workbench.plot_history(history)
    display(fig)
    plt.close(fig)

history = workbench.train(DATA, RUN, config, on_epoch=live_plot)
print((RUN/'summary.json').read_text())
print('Saved:', RUN)
""")
cell("markdown", """
## 8. 서로 다른 epoch의 같은 샘플 예측 비교
이 셀은 validation 샘플을 사용합니다. `COMPARE_EPOCHS`를 바꾸어 학습 전후의 예측과 사람 평가를 확인하세요.
같은 발화의 점수가 올라가는 것만으로 개선이라고 판단하지 말고, 전체 validation 오차와 함께 보세요.
""")
cell("code", """
summary = json.loads((RUN/'summary.json').read_text())
COMPARE_EPOCHS = sorted(set([0, summary['epochs_completed']]))
validation_sample = workbench.subset(validation_data, np.array([0]))
print('Held-out sample:', validation_sample['ids'][0])
print('Human utterance scores:', validation_sample['label_utt'][0])
for epoch in COMPARE_EPOCHS:
    model, norm = workbench.load_checkpoint(RUN/f'epoch-{epoch:03d}.pt')
    result, pred = workbench.evaluate(model, validation_sample, norm, return_predictions=True)
    print('Epoch', epoch, 'Predicted sentence scores:', pred['utterance'][0].round(3),
          'Phone MSE:', round(result['phone']['mse'], 4))
""")
cell("markdown", """
## 9. 선택 사항 — TensorBoard와 최종 시험
기본 그래프와 CSV만으로도 실험할 수 있습니다. 여러 run을 함께 볼 때 아래 TensorBoard 셀을 실행하세요.
""")
cell("code", """
OPEN_TENSORBOARD = False
if OPEN_TENSORBOARD:
    get_ipython().run_line_magic('load_ext', 'tensorboard')
    get_ipython().run_line_magic('tensorboard', f'--logdir "{ROOT / "runs/gopt"}" --host 127.0.0.1')
""")
cell("markdown", """
모델과 설정 선택이 끝난 다음에만 `RUN_FINAL_TEST = True`로 바꾸세요.
여기서 본 test 결과로 설정을 다시 고르면 더 이상 독립된 최종 시험이 아닙니다.
사람 점수와의 상관은 정답률이 아니며, 이 자료에서 좋은 결과가 나와도 한국인 개인 음성·자유 발화 성능은 별도 확인이 필요합니다.
""")
cell("code", """
RUN_FINAL_TEST = False
if RUN_FINAL_TEST:
    test_data = workbench.load_data(DATA, 'test')
    best_model, best_norm = workbench.load_checkpoint(RUN/'best.pt')
    final_metrics = workbench.evaluate(best_model, test_data, best_norm, device=DEVICE)
    (RUN/'final-test.json').write_text(json.dumps(final_metrics, indent=2, allow_nan=False))
    summary = json.loads((RUN/'summary.json').read_text())
    summary['test_evaluated'] = True
    (RUN/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(final_metrics, indent=2))
else:
    print('최종 시험은 아직 실행하지 않았습니다.')
""")
cell("markdown", """
## 다음 연결 작업
실제 내 녹음에 적용하려면 `원음 + 원고 → Kaldi 정렬/GOP 특징 → 고정 음소 사전 → GOPT` 경로가 추가로 필요합니다.
84차원이라는 크기만 같아서는 충분하지 않습니다. 학습 때와 동일한 음향 모델·전처리·음소 순서여야 합니다.
이후 이 연결을 검증하고, 기존 두 녹음으로 실제 결과를 확인합니다. 이를 위해 새 녹음을 반복 제출할 필요는 없습니다.

GOPT: Gong et al., ICASSP 2022, BSD-3-Clause 코드. SpeechOcean762: Zhang et al., Interspeech 2021, CC BY 4.0.
특징 보관 출처: JazminVidal/gopt-w 고정 commit. 재조합·float32 변환·검증 분할 등 변경은 `docs/GOPT_WORKBENCH.md` 참고.
""")

notebook = {"cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"}, "colab": {"name": "SpeakingT_GOPT.ipynb"}},
    "nbformat": 4, "nbformat_minor": 5}
target = Path(__file__).resolve().parents[1] / "notebooks/SpeakingT_GOPT.ipynb"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(target)

# Phase 3 중단·재개 기록

2026-11-02. 사용자가 Wi-Fi 연결 종료 전에 작업 저장과 일시 중단을 요청했다.
**Phase 3 전체 작업은 아직 미완료다. 새 작업은 사용자가 재개를 요청한 뒤 수행한다.**
학습 프로세스가 남아 있지 않음을 확인했다. 유료 GPU/API 사용은 없었다.

## 완료하고 직접 확인한 것

- 기존 Phase 2 코드·노트북·데이터 구조와 사용자 추가 실험을 조사했다. 기존 노트북 수정분은 사용자의 작업이며 그대로 보존한다.
- `scripts/audit_phase3.py`로 기존 run, checkpoint, 데이터, 캐시, 기존 노트북 등 6,781개 파일의 사전 해시를 `reports/phase3/preservation-before.json`에 저장했다. **사후 전체 해시 비교는 아직 남아 있다. 이 사전 파일을 덮어쓰지 않는다.**
- dropout=.5 seed 7/19/42 예측을 재집계했다. 앙상블−GOPT Validation macro MSE 차이 −0.0736295134, 화자 bootstrap 95% CI [−0.4601528451, 0.3362095512]. 집계 결과는 `experiments/phase3/phase2-audit.json`에 있다. 우위가 확정된 것은 아니다.
- 새 `speakingt/partial_wavlm.py` 구현: WavLM Base+ 고정 revision, 상위 0/1/2층 모드, 원음에서 미분 가능한 12층 통계 pooling, 기존 head warm-start 또는 신규 초기화, 별도 학습률, accumulation, clipping, early stopping, checkpoint·복원·재개 경로, 실행 자원 기록.
- 실제 원음의 raw padding을 CNN 전에 제거하고 latent frame을 동적 padding한다. 유효 프레임 attention mask와 masked pooling을 사용한다. 기존 `.npy` 특징은 fine-tuning 학습 입력으로 사용하지 않는다.
- 실제 WavLM/원음 2개 Stage A: 공식 단일 forward 대비 최대 오차 1.19e-7, batch/개별 pooling 오차 4.77e-7, padding 값 변경의 영향 없음. 상위 2층 gradient와 optimizer 후 가중치 변경, frozen 가중치 해시 불변을 확인했다. `experiments/phase3/stage-a.json` 참조.
- 신규 단위 테스트 7개 통과(약 32초). **기존 테스트까지 포함한 전체 재실행은 아직 하지 않았다.**
- 새 노트북과 실행 파일을 만들고 실제 Jupyter kernel에서 CPU 학습/예측/복원 셀을 실행했다. 실행 결과: `reports/phase3/notebook-stage-ab.ipynb` (코드 셀 10개, 저장된 오류 출력 없음).
- 실제 20개 WAV(Train 16, Validation 4), top 2층, seed 7, warm-start, 1 epoch 실행 완료. `runs/phase3/20261102-172648-542194`에 best/last checkpoint, SHA sidecar, 예측, metrics, 그래프, 환경 설정을 저장했다.
- CPU 학습 run 측정: 경과 68.36초, CPU 시간 240.53초, 관측 peak RSS 1,186,910,208 bytes(약 1.11 GiB), CUDA 사용 없음.
- 이 작은 run은 기능 검증이다. Validation macro MSE가 epoch 0의 0.414404에서 epoch 1의 0.443111로 증가하여 best epoch는 **0**이다. 실제 갱신된 epoch 1 모델은 **last.pt**다. 전체 500개 Validation 성능이나 개선 증거로 해석하지 않는다.

## 다시 시작할 때 우선 읽을 파일

1. `docs/STATUS.md`, 이 파일, `AGENTS.md`, 로컬 `.speakingt-context/conversations.md`.
2. `speakingt/partial_wavlm.py`, `tests/test_partial_wavlm.py`.
3. `scripts/run_phase3.py`, `scripts/execute_phase3_notebook.py`, `scripts/build_phase3_notebook.py`.
4. `notebooks/Phase3_Partial_WavLM.ipynb`, 위 실행 노트북과 run의 `summary.json`, `resources.json`, `metrics.csv`.

기존 `docs/PHASE2_GUIDE.md`, `docs/PHASE2_RESEARCH.md`, `experiments/phase2/README.md`, `speakingt/ssl_workbench.py`, `speakingt/phase2_data.py`는 이미 조사했다. Phase 2 원본 노트북과 GOPT 노트북은 덮어쓰지 않는다.

## 다음에 수행할 순서

1. **새 Jupyter kernel에서 재학습 없이 last.pt 예측 재생성 및 bitwise 일치 확인**. 아직 별도 새 커널 검증은 수행하지 않았다.

   ```powershell
   .venv\Scripts\python.exe scripts/execute_phase3_notebook.py --restore-run runs/phase3/20261102-172648-542194 --output reports/phase3/notebook-fresh-restore.ipynb
   ```

   출력 파일이 이미 생겼다면 덮어쓰지 말고 먼저 내용을 확인한다. Windows sandbox에서 kernel 생성이 막히면 적절한 실행 권한으로 재시도한다.

2. 체크포인트에서 optimizer/scheduler/RNG를 복원하는 재개 경로를 실제 검증한다. 필요하면 이 작은 CPU run만 2번째 epoch까지 이어서 **새 run 디렉터리**에 저장한다. CPU 전체 2,000개 fine-tuning은 실행하지 않는다. 정확한 CLI는 `scripts/run_phase3.py --help`와 해당 하위 명령 도움말을 확인한다.
3. 전체 기존+신규 테스트 실행과 필요한 오류 수정. 모델 로더는 모듈 코드 SHA를 엄격히 검사하므로 `partial_wavlm.py` 변경 시 기존 checkpoint 출처 검증을 임의로 우회하지 않는다. 원래 소스 버전을 보존하거나 변경된 코드로 별도의 최소 검증 run을 만든다.
4. `docs/PHASE3_GUIDE.md`, `experiments/phase3/README.md`와 실험 protocol을 완성한다. 한국어 Jupyter 초보자 안내, CPU/GPU 구분, epoch 0 선택 의미, restore/resume 차이, CUDA 별도 환경, OOM 대응, GPU pilot 실측 후 비용 산정, 유료 실행 사전 승인, Validation 반복 튜닝 편향을 포함한다.
5. 실제 결과의 **집계값과 그래프만** 공개 실험 폴더로 정리한다. 모델, 원음, 개별 예측, 개인 자료는 로컬 ignored 경로에 둔다. CPU 작은 검증을 성능 개선으로 주장하지 않는다.
6. `scripts/audit_phase3.py --verify`로 사전 6,781개 파일의 보존 상태를 확인한다. 사전 snapshot은 다시 만들지 않는다.
7. notebook 검증 및 전체 테스트 결과, 실제 자원량, 아직 GPU 미실행이라는 사실을 STATUS/README에 반영한다. 변경 내용을 검토하고 공개 가능한 파일만 commit/push한다. 사용자가 수정한 기존 노트북은 stage하지 않는다.

## 구현상 주의 사항

- 모델 revision: `4c66d4806a428f2e922ccfa1a962776e232d487b`. 공식 processor `do_normalize=False`, 16kHz. WavLM CNN은 group norm이므로 raw zero padding 영향을 회피하도록 설계했다.
- 기본 warm-start: `runs/ssl/notebook-full-seed7-20261102-103405-853366/best.pt`, epoch 8, SHA256 `5a18b488e1c6a96eaf1207e68579ce7117e0f404cf565d136c5147ee0ffced64`. head hidden=128/dropout=.5, 원래 Training 2,000개로 계산한 정규화 통계를 고정한다.
- 상위 2층 학습 파라미터 14,176,808개, head 197,264개. 기본 encoder LR=1e-5, head LR=1e-4, batch=2, accumulation=2.
- 환경: Python 3.11.4, torch 2.15.0+cpu, transformers 4.57.6. CUDA GPU 없음. 기존 환경을 CUDA용으로 무작정 교체하지 않는다.
- HF/PyTorch에서 bool attention mask와 float attention bias dtype 관련 upstream 경고가 발생한다. 현재 수치 비교는 통과했으나 경고 사실과 고정 환경을 기록한다.
- resume는 epoch 경계 기준, 같은 설정에서 수행한다. OOM으로 batch 등 설정을 바꿀 경우 동일 재개로 표현하지 말고 별도 실험으로 취급한다.
- checkpoint와 SHA sidecar 쌍은 중단 시 불일치할 수 있다. 불일치하면 명시적으로 실패하게 되어 있으며 다른 보존 checkpoint를 확인한다.
- 현재 새 Phase 3 노트북은 초기 CPU 기능 검증용이다. GPU pilot/full 버튼에 해당하는 명시 설정과 CLI는 준비됐지만 GPU 실행은 미검증이다.

## 저장·동기화 상태

이번 중단 시점의 소스와 이 메모는 로컬에 저장한다. 대용량 checkpoint와 실행 출력은 `runs/phase3/`, `reports/phase3/`에 남아 있다. **GitHub push와 OneDrive 동기화 완료를 확인한 것은 아니다.** Wi-Fi가 끊겨도 같은 PC의 이 폴더에서 이어갈 수 있다. 로컬 Git 저장 여부와 commit ID는 중단 응답을 참고한다.

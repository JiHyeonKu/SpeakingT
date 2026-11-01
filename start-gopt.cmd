@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Create the project .venv and install SpeakingT research dependencies first.
  pause
  exit /b 1
)
set "IPYTHONDIR=%~dp0.cache\ipython"
set "JUPYTER_CONFIG_DIR=%~dp0.jupyter\config"
set "JUPYTER_RUNTIME_DIR=%~dp0.jupyter\runtime"
set "MPLCONFIGDIR=%~dp0.cache\matplotlib"
".venv\Scripts\python.exe" -m jupyterlab --ServerApp.ip=127.0.0.1 --ServerApp.root_dir="%~dp0." --LabApp.default_url=/lab/tree/notebooks/SpeakingT_GOPT.ipynb
pause

@echo off
setlocal
cd /d "%~dp0"

set "PY=%~dp0.venv\Scripts\pythonw.exe"
if not exist "%PY%" set "PY=%LOCALAPPDATA%\Programs\Python\Python312\pythonw.exe"
if not exist "%PY%" set "PY=%LOCALAPPDATA%\Programs\Python\Python311\pythonw.exe"

if not exist "%PY%" (
  echo Python not found. Install Python 3.12 and: python -m pip install -r requirements.txt
  pause
  exit /b 1
)

start "" "%PY%" -m finetuning.finetune_gui
endlocal

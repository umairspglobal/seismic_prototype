@echo off
setlocal
cd /d "%~dp0"

set "PATH=C:\Program Files\nodejs;%PATH%"
set "NODE_OPTIONS=--use-system-ca"
set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not exist "%PY%" set "PY=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"

if not exist "%PY%" (
  echo Python not found. Install Python 3.12 and: python -m pip install -r requirements.txt
  pause
  exit /b 1
)

if not exist "frontend\node_modules\react" (
  echo Installing frontend dependencies...
  pushd frontend
  call npm install
  popd
)

start "Seismic SAM API" cmd /k ""%PY%" -m uvicorn server.main:app --host 127.0.0.1 --port 8000"
start "Seismic SAM frontend" cmd /k "cd /d "%~dp0frontend" && npm run dev"

timeout /t 3 /nobreak >nul
start "" http://localhost:5173

echo Opened http://localhost:5173
echo Keep the two new command windows running.
endlocal

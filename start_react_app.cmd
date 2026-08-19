@echo off
setlocal
cd /d "%~dp0"

set "PATH=C:\Program Files\nodejs;%PATH%"
set "PY=C:\Users\U\AppData\Local\Programs\Python\Python311\python.exe"

if not exist "%PY%" (
  echo Python 3.11 not found at %PY%
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

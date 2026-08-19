# Launch the FastAPI inference server and the Vite dev server together.
# Frontend: http://localhost:5173   API: http://127.0.0.1:8000

$root = $PSScriptRoot

Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$root'; uvicorn server.main:app --host 127.0.0.1 --port 8000"
)

Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$root\frontend'; npm run dev"
)

Write-Host "Inference API starting on http://127.0.0.1:8000"
Write-Host "Frontend starting on http://localhost:5173"

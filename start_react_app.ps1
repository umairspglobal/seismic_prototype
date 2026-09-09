# Launch the FastAPI inference server and the Vite dev server together.
# Frontend: http://localhost:5173   API: http://127.0.0.1:8000

$root = $PSScriptRoot
$nodeDir = "C:\Program Files\nodejs"

# New shells often miss a just-installed Node.js until PATH is rebuilt.
$env:Path = "$nodeDir;$([Environment]::GetEnvironmentVariable('Path', 'Machine'));$([Environment]::GetEnvironmentVariable('Path', 'User'))"
# Corporate TLS inspection: let Node use the Windows certificate store.
$env:NODE_OPTIONS = "--use-system-ca"

$npmCmd = Join-Path $nodeDir "npm.cmd"
if (-not (Test-Path $npmCmd)) {
    Write-Host "Node.js is not installed (npm not found at $npmCmd)."
    Write-Host "Install Node.js LTS from https://nodejs.org, then close and reopen this window."
    exit 1
}

$pythonCandidates = @(
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
)
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if ($pythonCmd) {
    $pythonCandidates += $pythonCmd.Source
}

$python = $null
foreach ($candidate in $pythonCandidates) {
    if (-not (Test-Path $candidate)) { continue }
    & $candidate -c "import uvicorn" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $python = $candidate
        break
    }
}

if (-not $python) {
    Write-Host "Python with uvicorn was not found."
    Write-Host "Install project deps with: python -m pip install -r requirements.txt"
    exit 1
}

$reactPkg = Join-Path $root "frontend\node_modules\react"
if (-not (Test-Path $reactPkg)) {
    Write-Host "Installing frontend dependencies..."
    Push-Location (Join-Path $root "frontend")
    & $npmCmd install
    $installExit = $LASTEXITCODE
    Pop-Location
    if ($installExit -ne 0) {
        Write-Host "npm install failed."
        exit 1
    }
}

Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$root'; `$env:Path = '$nodeDir;' + `$env:Path; `$env:NODE_OPTIONS = '--use-system-ca'; & '$python' -m uvicorn server.main:app --host 127.0.0.1 --port 8000"
)

Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$root\frontend'; `$env:Path = '$nodeDir;' + `$env:Path; `$env:NODE_OPTIONS = '--use-system-ca'; & '$npmCmd' run dev"
)

Write-Host "Inference API starting on http://127.0.0.1:8000"
Write-Host "Frontend starting on http://localhost:5173"

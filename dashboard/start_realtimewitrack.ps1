$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:DATA_SOURCE = if ($env:DATA_SOURCE) { $env:DATA_SOURCE } else { "mock" }

$BackendPath = Join-Path $ScriptDir "backend"
$FrontendPath = Join-Path $ScriptDir "frontend"
$PythonPath = Join-Path (Split-Path -Parent $ScriptDir) ".venv\Scripts\python.exe"

if (-not (Test-Path $PythonPath)) {
    $PythonPath = Join-Path (Split-Path -Parent $ScriptDir) ".venv\bin\python"
}

$Backend = Start-Process `
    -FilePath $PythonPath `
    -ArgumentList "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000", "--reload" `
    -WorkingDirectory $BackendPath `
    -PassThru

try {
    Push-Location $FrontendPath
    npm run dev -- --host 127.0.0.1 --port 5173
}
finally {
    Pop-Location
    if ($Backend -and -not $Backend.HasExited) {
        Stop-Process -Id $Backend.Id
    }
}

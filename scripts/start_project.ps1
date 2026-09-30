param(
    [ValidateRange(5, 3600)]
    [int]$TimeoutSeconds = 900,
    [switch]$SkipInfrastructure,
    [switch]$SkipWarmup
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "project_runtime.ps1")
Initialize-NaweiLocalEnvironment

$modelRestorePython = Join-Path $script:NaweiProjectRoot ".venv\Scripts\python.exe"
& $modelRestorePython (Join-Path $PSScriptRoot "restore_models.py")
if ($LASTEXITCODE -ne 0) {
    throw "Bundled model restoration failed. Ensure the local model files and required split parts are complete."
}

if (-not $SkipInfrastructure) {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw "Docker is required for PostgreSQL, Redis, Qdrant, and Neo4j. Use -SkipInfrastructure only when external instances are configured."
    }
    docker info --format "{{.ServerVersion}}" *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Docker is installed, but its engine is not running. Start Docker Desktop and rerun this script."
    }
    Write-Host "Starting PostgreSQL, Redis, Qdrant, and Neo4j infrastructure..." -ForegroundColor Cyan
    docker compose -f (Join-Path $script:NaweiProjectRoot "deploy\docker-compose.llm-infra.yml") up -d --wait
    if ($LASTEXITCODE -ne 0) {
        throw "LLM infrastructure startup failed."
    }
}

$services = Get-NaweiServiceDefinitions
$occupied = @($services | Where-Object { Test-NaweiPortListening $_.Port })
if ($occupied.Count -gt 0) {
    $summary = ($occupied | ForEach-Object { "$($_.Name):$($_.Port)" }) -join ", "
    throw "Project ports are already in use: $summary. Run .\status_project.ps1 or .\stop_project.ps1 first."
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$runtimeDirectory = Join-Path $script:NaweiProjectRoot "logs\runtime-$stamp"
New-Item -ItemType Directory -Path $runtimeDirectory -Force | Out-Null

$records = @()
foreach ($service in $services) {
    Write-Host "Starting $($service.Name) on port $($service.Port)..." -ForegroundColor Cyan
    $records += Start-NaweiServiceProcess -Service $service -RuntimeDirectory $runtimeDirectory
    Save-NaweiProcessRecords -Records $records -Path (Join-Path $runtimeDirectory "processes.json")
}

Set-Content -Encoding UTF8 -Path (Join-Path $script:NaweiProjectRoot "logs\latest-runtime.txt") -Value $runtimeDirectory

$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
$pending = @($services)
while ($pending.Count -gt 0 -and (Get-Date) -lt $deadline) {
    $pending = @($pending | Where-Object { -not (Test-NaweiHealth $_.HealthUrl) })
    if ($pending.Count -gt 0) {
        $names = ($pending | ForEach-Object Name) -join ", "
        Write-Host "Waiting for: $names"
        Start-Sleep -Seconds 5
    }
}

if ($pending.Count -gt 0) {
    foreach ($service in $pending) {
        Write-Warning "$($service.Name) did not become healthy. Check $runtimeDirectory\$($service.Name).stderr.log"
    }
    throw "Project startup did not complete within $TimeoutSeconds seconds."
}

if (-not $SkipWarmup) {
    Write-Host "Warming up TTS and ASR inference..." -ForegroundColor Cyan
    & $modelRestorePython (Join-Path $PSScriptRoot "warm_voice.py")
    if ($LASTEXITCODE -ne 0) {
        throw "Speech inference warmup failed. Services are running; check .runtime\voice-warmup.json and runtime logs."
    }
}

Write-Host "All services are ready." -ForegroundColor Green
Write-Host "Gateway: http://127.0.0.1:8000"
Write-Host "LLM memory: Redis + PostgreSQL"
Write-Host "Knowledge GraphRAG: Qdrant + Neo4j"
Write-Host "Knowledge API: http://127.0.0.1:8003/docs"
Write-Host "Robot OTA: http://127.0.0.1:8000/xiaozhi/ota/"
Write-Host "Logs: $runtimeDirectory"

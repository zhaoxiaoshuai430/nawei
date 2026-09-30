param(
    [switch]$KeepInfrastructure,
    [switch]$ForceStop
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "project_runtime.ps1")

$latestRuntime = Join-Path $script:NaweiProjectRoot "logs\latest-runtime.txt"
if (-not (Test-Path $latestRuntime)) {
    Write-Host "No tracked runtime was found."
    if (-not $KeepInfrastructure -and (Get-Command docker -ErrorAction SilentlyContinue)) {
        docker compose -f (Join-Path $script:NaweiProjectRoot "deploy\docker-compose.llm-infra.yml") down
    }
    exit 0
}

$runtimeDirectory = (Get-Content $latestRuntime -Raw).Trim()
$processFile = Join-Path $runtimeDirectory "processes.json"
if (-not (Test-Path $processFile)) {
    throw "Process metadata was not found: $processFile"
}

$records = Get-Content $processFile -Raw | ConvertFrom-Json
$definitions = Get-NaweiServiceDefinitions
foreach ($serviceName in @('llm', 'gateway', 'knowledge', 'asr', 'tts')) {
    $service = $definitions | Where-Object Name -eq $serviceName
    $record = $records | Where-Object name -eq $serviceName | Select-Object -First 1
    Write-Host "Stopping $serviceName..." -ForegroundColor Cyan
    Stop-NaweiServiceProcess -Service $service -Record $record -ForceStop:$ForceStop
}

Start-Sleep -Seconds 2
$remaining = @(Get-NaweiServiceDefinitions | Where-Object { Test-NaweiPortListening $_.Port })
if ($remaining.Count -gt 0) {
    $summary = ($remaining | ForEach-Object { "$($_.Name):$($_.Port)" }) -join ", "
    Write-Warning "Some project ports are still listening: $summary"
    exit 1
}

Write-Host "All tracked services stopped." -ForegroundColor Green

if (-not $KeepInfrastructure -and (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "Stopping PostgreSQL, Redis, Qdrant, and Neo4j infrastructure..." -ForegroundColor Cyan
    docker compose -f (Join-Path $script:NaweiProjectRoot "deploy\docker-compose.llm-infra.yml") down
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "LLM infrastructure did not stop cleanly."
    }
}

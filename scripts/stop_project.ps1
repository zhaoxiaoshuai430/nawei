param(
    [switch]$KeepInfrastructure
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
foreach ($record in $records) {
    $process = Get-Process -Id ([int]$record.pid) -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        continue
    }

    Write-Host "Stopping $($record.name)..." -ForegroundColor Cyan
    & taskkill.exe /PID ([string]$record.pid) /T /F | Out-Null
    if ($LASTEXITCODE -ne 0 -and (Get-Process -Id ([int]$record.pid) -ErrorAction SilentlyContinue)) {
        Write-Warning "Unable to stop $($record.name) process tree (PID $($record.pid))."
    }
}

# A Windows virtual-environment launcher can exit while its base Python
# process keeps serving. Clean up any remaining listeners on project ports.
foreach ($service in Get-NaweiServiceDefinitions) {
    $listenerPids = @(
        Get-NetTCPConnection -State Listen -LocalPort $service.Port -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty OwningProcess -Unique
    )
    foreach ($listenerPid in $listenerPids) {
        Write-Host "Stopping $($service.Name) listener (PID $listenerPid)..." -ForegroundColor Cyan
        & taskkill.exe /PID ([string]$listenerPid) /T /F | Out-Null
        if ($LASTEXITCODE -ne 0 -and (Get-Process -Id ([int]$listenerPid) -ErrorAction SilentlyContinue)) {
            Write-Warning "Unable to stop $($service.Name) listener (PID $listenerPid)."
        }
    }
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
    Write-Host "Stopping LLM Redis/PostgreSQL infrastructure..." -ForegroundColor Cyan
    docker compose -f (Join-Path $script:NaweiProjectRoot "deploy\docker-compose.llm-infra.yml") down
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "LLM infrastructure did not stop cleanly."
    }
}

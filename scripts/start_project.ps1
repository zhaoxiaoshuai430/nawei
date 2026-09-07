param(
    [int]$TimeoutSeconds = 420,
    [switch]$SkipInfrastructure
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "project_runtime.ps1")

$modelRestorePython = Join-Path $script:NaweiProjectRoot ".venv\Scripts\python.exe"
& $modelRestorePython (Join-Path $PSScriptRoot "restore_models.py")
if ($LASTEXITCODE -ne 0) {
    throw "Bundled model restoration failed. Ensure Git LFS objects were downloaded."
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

if (-not $env:DATABASE_URL) { $env:DATABASE_URL = "postgresql://nawei:nawei@127.0.0.1:5432/nawei" }
if (-not $env:REDIS_URL) { $env:REDIS_URL = "redis://127.0.0.1:6379/0" }
if (-not $env:QDRANT_URL) { $env:QDRANT_URL = "http://127.0.0.1:6333" }
if (-not $env:QDRANT_API_KEY) { $env:QDRANT_API_KEY = "nawei-qdrant-local" }
if (-not $env:NEO4J_URI) { $env:NEO4J_URI = "neo4j://127.0.0.1:7687" }
if (-not $env:NEO4J_USER) { $env:NEO4J_USER = "neo4j" }
if (-not $env:NEO4J_PASSWORD) { $env:NEO4J_PASSWORD = "nawei-neo4j-local" }
if (-not $env:KNOWLEDGE_ENABLED) { $env:KNOWLEDGE_ENABLED = "true" }
if (-not $env:KNOWLEDGE_REQUIRED) { $env:KNOWLEDGE_REQUIRED = "true" }
if (-not $env:KNOWLEDGE_RETRIEVE_URL) { $env:KNOWLEDGE_RETRIEVE_URL = "http://127.0.0.1:8003/api/v1/knowledge/retrieve" }
if (-not $env:KNOWLEDGE_HEALTH_URL) { $env:KNOWLEDGE_HEALTH_URL = "http://127.0.0.1:8003/health" }
if (-not $env:MEMORY_ENABLED) { $env:MEMORY_ENABLED = "true" }
if (-not $env:LLM_MEMORY_REQUIRED) { $env:LLM_MEMORY_REQUIRED = "true" }

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
}

$records | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $runtimeDirectory "processes.json")
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

Write-Host "All services are ready." -ForegroundColor Green
Write-Host "Gateway: http://127.0.0.1:8000"
Write-Host "LLM memory: Redis + PostgreSQL"
Write-Host "Knowledge GraphRAG: Qdrant + Neo4j"
Write-Host "Knowledge API: http://127.0.0.1:8003/docs"
Write-Host "Gateway API docs: http://127.0.0.1:8000/docs"
Write-Host "Logs: $runtimeDirectory"

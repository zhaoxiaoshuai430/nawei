param(
    [switch]$Live
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$MainPython = Join-Path $Root ".venv\Scripts\python.exe"
$AsrPython = Join-Path $Root "services\asr\.venv\Scripts\python.exe"
$TtsPython = Join-Path $Root "services\tts\cuda_env\Scripts\python.exe"

function Invoke-Checked {
    param(
        [string]$Name,
        [scriptblock]$Command
    )
    Write-Host "`n[$Name]" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

Push-Location $Root
try {
    Invoke-Checked "Main tests" { & $MainPython -m pytest -q }
    Invoke-Checked "Ruff" { & $MainPython -m ruff check app_protocols services\gateway services\llm tests deploy\scripts }
    Invoke-Checked "Mypy" { & $MainPython -m mypy app_protocols services\gateway services\llm }
    if ($Live) {
        $env:NAWEI_RUN_LIVE_TESTS = "1"
        Invoke-Checked "Live service connectivity" {
            & $MainPython -m pytest -q tests\test_service_connectivity.py -k LiveService
        }
    }
}
finally {
    Pop-Location
}

Push-Location (Join-Path $Root "services\asr")
try {
    Invoke-Checked "ASR unit tests" { & $AsrPython -m unittest discover -s tests -v }
    Invoke-Checked "ASR compile" { & $AsrPython -m compileall -q src tests }
}
finally {
    Pop-Location
}

Push-Location (Join-Path $Root "services\tts")
try {
    Invoke-Checked "TTS tests" { & $TtsPython -m pytest -q -W error tests }
    Invoke-Checked "TTS compile" { & $TtsPython -m compileall -q app tests }
}
finally {
    Pop-Location
}

Invoke-Checked "PowerShell syntax" {
    $scriptPaths = @(
        "scripts\start_project.ps1",
        "scripts\stop_project.ps1",
        "scripts\status_project.ps1",
        "scripts\check_project.ps1",
        "scripts\project_runtime.ps1"
    )
    foreach ($relativePath in $scriptPaths) {
        $tokens = $null
        $errors = $null
        [System.Management.Automation.Language.Parser]::ParseFile(
            (Join-Path $Root $relativePath),
            [ref]$tokens,
            [ref]$errors
        ) | Out-Null
        if ($errors.Count -gt 0) {
            $messages = ($errors | ForEach-Object Message) -join "; "
            throw "$relativePath has PowerShell syntax errors: $messages"
        }
    }
    $global:LASTEXITCODE = 0
}

Invoke-Checked "Text quality" {
    & $MainPython (Join-Path $Root "scripts\check_text_quality.py")
}

Invoke-Checked "Deployment scripts compile" {
    & $MainPython -m compileall -q (Join-Path $Root "deploy\scripts")
}

if (Get-Command docker -ErrorAction SilentlyContinue) {
    Invoke-Checked "LLM Infrastructure Compose" {
        docker compose -f (Join-Path $Root "deploy\docker-compose.llm-infra.yml") config --quiet
    }
    Invoke-Checked "Production Compose" {
        $previousEnvFile = $env:NAWEI_ENV_FILE
        try {
            $env:NAWEI_ENV_FILE = ".env.production.example"
            docker compose `
                --env-file (Join-Path $Root "deploy\.env.production.example") `
                -f (Join-Path $Root "deploy\docker-compose.production.yml") `
                config --quiet
        }
        finally {
            if ($null -eq $previousEnvFile) {
                Remove-Item Env:NAWEI_ENV_FILE -ErrorAction SilentlyContinue
            }
            else {
                $env:NAWEI_ENV_FILE = $previousEnvFile
            }
        }
    }
}
else {
    Write-Warning "Docker is not installed; runtime Compose startup was not checked. YAML structure was covered by pytest."
}

Write-Host "`nAll available project checks passed." -ForegroundColor Green

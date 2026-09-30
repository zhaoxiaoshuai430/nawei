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

if (-not (Test-Path -LiteralPath $MainPython)) {
    throw "Main Python runtime was not found: $MainPython"
}

Push-Location $Root
try {
    Invoke-Checked "Compile Python sources" {
        & $MainPython -m compileall -q `
            app_protocols `
            services\gateway `
            services\llm `
            services\knowledge `
            services\asr\src `
            services\asr\sensevoice_runtime `
            services\asr\tests `
            services\tts\app `
            services\tts\tests `
            deploy\scripts `
            tests
    }
    Invoke-Checked "Ruff" {
        & $MainPython -m ruff check `
            --force-exclude `
            app_protocols `
            services `
            deploy\scripts `
            tests
    }
    Invoke-Checked "Mypy" {
        & $MainPython -m mypy app_protocols services\gateway services\llm --ignore-missing-imports
    }
    Invoke-Checked "Additional Mypy" {
        & $MainPython -m mypy `
            services\knowledge `
            services\asr\src `
            services\asr\sensevoice_runtime `
            deploy\scripts `
            --ignore-missing-imports `
            --follow-imports=skip
    }
    Invoke-Checked "Main tests" {
        & $MainPython -m pytest -q
    }
    Invoke-Checked "Service boundary tests" {
        & $MainPython -m pytest -q services\asr\tests services\tts\tests
    }

    Write-Host "`n[Source secret scan]" -ForegroundColor Cyan
    $scanRoots = @(
        "app_protocols", "services\gateway", "services\llm", "services\knowledge",
        "services\asr\src", "services\asr\sensevoice_runtime", "services\tts\app",
        "scripts", "deploy", "tests"
    )
    $sourceExtensions = @(".py", ".ps1", ".js", ".css", ".html", ".json", ".toml", ".yml", ".yaml", ".txt", ".ini", ".conf", ".sql", ".sh", ".bat")
    $secretMatches = @(
        foreach ($relativePath in $scanRoots) {
            Get-ChildItem -LiteralPath (Join-Path $Root $relativePath) -File -Recurse |
                Where-Object {
                    $_.Name -notlike ".env*" -and
                    ($_.Extension -in $sourceExtensions -or $_.Name -like "Dockerfile*")
                } |
                ForEach-Object {
                    if (Select-String -LiteralPath $_.FullName -Quiet -Pattern 'sk-[A-Za-z0-9_-]{16,}|BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY') {
                        $_.FullName
                    }
                }
        }
    )
    if ($secretMatches.Count -gt 0) {
        $secretMatches | Write-Host
        throw "Source secret scan found a possible private credential."
    }
    Write-Host "No private credentials found in application source files."

    if ($Live) {
        Invoke-Checked "Live service health" {
            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\status_project.ps1
        }
    }
}
finally {
    Pop-Location
}

if (Test-Path -LiteralPath $AsrPython) {
    Push-Location (Join-Path $Root "services\asr")
    try {
        $asrTestFiles = @(Get-ChildItem -Path tests -Filter "test_*.py" -File -Recurse)
        if ($asrTestFiles.Count -gt 0) {
            Invoke-Checked "ASR unit tests" {
                & $AsrPython -m unittest discover -s tests -v
            }
        }
        else {
            Write-Warning "No ASR test source files were found; ASR unit tests were skipped."
        }
        Invoke-Checked "ASR compile" {
            & $AsrPython -m compileall -q src tests
        }
    }
    finally {
        Pop-Location
    }
}
else {
    Write-Warning "ASR runtime was not found; ASR-specific checks were skipped: $AsrPython"
}

if (Test-Path -LiteralPath $TtsPython) {
    Push-Location (Join-Path $Root "services\tts")
    try {
        $ttsTestFiles = @(Get-ChildItem -Path tests -Filter "test_*.py" -File -Recurse)
        if ($ttsTestFiles.Count -gt 0) {
            Invoke-Checked "TTS tests" {
                & $TtsPython -m pytest -q tests
            }
        }
        else {
            Write-Warning "No TTS test source files were found; TTS tests were skipped."
        }
        Invoke-Checked "TTS compile" {
            & $TtsPython -m compileall -q app tests
        }
    }
    finally {
        Pop-Location
    }
}
else {
    Write-Warning "TTS runtime was not found; TTS-specific checks were skipped: $TtsPython"
}

Write-Host "`n[PowerShell syntax]" -ForegroundColor Cyan
$scriptPaths = @(
    "scripts\start_project.ps1",
    "scripts\stop_project.ps1",
    "scripts\status_project.ps1",
    "scripts\check_project.ps1",
    "scripts\project_runtime.ps1"
    "scripts\restart_robot_services.ps1"
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
Write-Host "PowerShell scripts parsed successfully."

if (Get-Command docker -ErrorAction SilentlyContinue) {
    Invoke-Checked "LLM infrastructure Compose" {
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
    Write-Warning "Docker is not installed; Compose validation was skipped."
}

Write-Host "`nAll available project checks passed." -ForegroundColor Green

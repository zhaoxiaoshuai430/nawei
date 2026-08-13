$script:NaweiProjectRoot = Split-Path -Parent $PSScriptRoot
$script:NaweiModelRoot = Join-Path $script:NaweiProjectRoot "models"

function Get-NaweiServiceDefinitions {
    $mainPython = Join-Path $script:NaweiProjectRoot ".venv\Scripts\python.exe"
    $mainDirectory = $script:NaweiProjectRoot

    return @(
        [pscustomobject]@{
            Name = "asr"
            Port = 8001
            HealthUrl = "http://127.0.0.1:8001/health"
            FilePath = Join-Path $script:NaweiProjectRoot "services\asr\.venv\Scripts\python.exe"
            Arguments = @("src\main.py")
            WorkingDirectory = Join-Path $script:NaweiProjectRoot "services\asr"
            Environment = @{
                PYTHONUNBUFFERED = "1"
                ASR_HOST = "0.0.0.0"
                ASR_PORT = "8001"
                ASR_DEVICE = "cpu"
                ASR_MODEL_ID = "iic/SenseVoiceSmall"
                ASR_MODEL_PATH = Join-Path $script:NaweiModelRoot "asr\SenseVoiceSmall"
                ASR_VAD_MODEL_PATH = Join-Path $script:NaweiModelRoot "asr\fsmn-vad"
                MODELSCOPE_CACHE = Join-Path $script:NaweiProjectRoot "services\asr\hf_download\modelscope"
                MODELSCOPE_HOME = Join-Path $script:NaweiProjectRoot "services\asr\hf_download\modelscope"
                HF_HOME = Join-Path $script:NaweiProjectRoot "services\asr\hf_download\huggingface"
            }
        }
        [pscustomobject]@{
            Name = "tts"
            Port = 19000
            HealthUrl = "http://127.0.0.1:19000/api/v1/status"
            FilePath = Join-Path $script:NaweiProjectRoot "services\tts\cuda_env\Scripts\python.exe"
            Arguments = @("-m", "app.services.tts.server", "--model", "voice_lab/nana", "--port", "19000")
            WorkingDirectory = Join-Path $script:NaweiProjectRoot "services\tts"
            Environment = @{
                PYTHONUNBUFFERED = "1"
                TTS_SKIP_PRESET_FETCH = "1"
                NAWEI_MODEL_ROOT = $script:NaweiModelRoot
            }
        }
        [pscustomobject]@{
            Name = "llm"
            Port = 8002
            HealthUrl = "http://127.0.0.1:8002/health"
            FilePath = $mainPython
            Arguments = @("-m", "services.llm.run")
            WorkingDirectory = $mainDirectory
            Environment = @{ PYTHONUNBUFFERED = "1" }
        }
        [pscustomobject]@{
            Name = "gateway"
            Port = 8000
            HealthUrl = "http://127.0.0.1:8000/api/v1/status"
            FilePath = $mainPython
            Arguments = @("-m", "services.gateway.run")
            WorkingDirectory = $mainDirectory
            Environment = @{ PYTHONUNBUFFERED = "1" }
        }
    )
}

function Test-NaweiPortListening {
    param([int]$Port)

    return $null -ne (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)
}

function Test-NaweiHealth {
    param(
        [string]$Url,
        [int]$TimeoutSeconds = 5
    )

    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec $TimeoutSeconds
        return $response.StatusCode -eq 200
    }
    catch {
        return $false
    }
}

function Repair-NaweiProcessEnvironment {
    $pathKeys = @(
        [Environment]::GetEnvironmentVariables("Process").Keys |
            Where-Object { [string]$_ -ieq "Path" }
    )

    # Windows PowerShell Start-Process rejects inherited Path/PATH duplicates.
    if ($pathKeys -ccontains "Path" -and $pathKeys -ccontains "PATH") {
        Remove-Item -LiteralPath "Env:PATH"
    }
}

function Start-NaweiServiceProcess {
    param(
        [pscustomobject]$Service,
        [string]$RuntimeDirectory
    )

    if (-not (Test-Path $Service.FilePath)) {
        throw "Runtime not found for $($Service.Name): $($Service.FilePath)"
    }

    $previousValues = @{}
    foreach ($entry in $Service.Environment.GetEnumerator()) {
        $previousValues[$entry.Key] = [Environment]::GetEnvironmentVariable($entry.Key, "Process")
        [Environment]::SetEnvironmentVariable($entry.Key, [string]$entry.Value, "Process")
    }

    try {
        Repair-NaweiProcessEnvironment
        $stdout = Join-Path $RuntimeDirectory "$($Service.Name).stdout.log"
        $stderr = Join-Path $RuntimeDirectory "$($Service.Name).stderr.log"
        $process = Start-Process `
            -FilePath $Service.FilePath `
            -ArgumentList $Service.Arguments `
            -WorkingDirectory $Service.WorkingDirectory `
            -RedirectStandardOutput $stdout `
            -RedirectStandardError $stderr `
            -WindowStyle Hidden `
            -PassThru

        return [pscustomobject]@{
            name = $Service.Name
            port = $Service.Port
            health_url = $Service.HealthUrl
            pid = $process.Id
            stdout = $stdout
            stderr = $stderr
        }
    }
    finally {
        foreach ($entry in $previousValues.GetEnumerator()) {
            [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, "Process")
        }
    }
}

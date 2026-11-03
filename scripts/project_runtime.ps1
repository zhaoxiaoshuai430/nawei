$script:NaweiProjectRoot = Split-Path -Parent $PSScriptRoot
$script:NaweiModelRoot = Join-Path $script:NaweiProjectRoot "models"

function Initialize-NaweiLocalEnvironment {
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
}

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
                ASR_HOST = "127.0.0.1"
                ASR_PORT = "8001"
                ASR_DEVICE = "cpu"
                ASR_MODEL_ID = "iic/SenseVoiceSmall"
                ASR_MODEL_PATH = Join-Path $script:NaweiModelRoot "asr\SenseVoiceSmall"
                ASR_VAD_MODEL_PATH = Join-Path $script:NaweiModelRoot "asr\fsmn-vad"
                ASR_REALTIME_END_SILENCE_MS = "420"
                ASR_REALTIME_BACKEND = $(if ($env:ASR_REALTIME_BACKEND) { $env:ASR_REALTIME_BACKEND } else { "paraformer" })
                ASR_STREAMING_MODEL_PATH = Join-Path $script:NaweiModelRoot "asr\paraformer-online"
                ASR_STREAMING_DEVICE = "cpu"
                ASR_STREAMING_THREADS = "2"
                ASR_STREAMING_WORKERS = $(if ($env:ASR_STREAMING_WORKERS) { $env:ASR_STREAMING_WORKERS } else { "2" })
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
            Arguments = @("-m", "app.services.tts.server", "--port", "19000", "--host", "127.0.0.1")
            WorkingDirectory = Join-Path $script:NaweiProjectRoot "services\tts"
            Environment = @{
                PYTHONUNBUFFERED = "1"
                TTS_SKIP_PRESET_FETCH = "1"
                NAWEI_MODEL_ROOT = $script:NaweiModelRoot
            }
        }
        [pscustomobject]@{
            Name = "knowledge"
            Port = 8003
            HealthUrl = "http://127.0.0.1:8003/health"
            FilePath = $mainPython
            Arguments = @("-m", "services.knowledge.run")
            WorkingDirectory = $mainDirectory
            Environment = @{ PYTHONUNBUFFERED = "1" }
        }
        [pscustomobject]@{
            Name = "llm"
            Port = 8002
            HealthUrl = "http://127.0.0.1:8002/health"
            FilePath = $mainPython
            Arguments = @("-m", "services.llm.run")
            WorkingDirectory = $mainDirectory
            Environment = @{
                PYTHONUNBUFFERED = "1"
                LLM_TIMEOUT_SECONDS = "25"
            }
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

    try {
        if ($null -ne (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction Stop)) {
            return $true
        }
        return $false
    }
    catch {
        # Some Windows sessions cannot query the TCP table without elevation.
        # A loopback connect still provides the needed occupied-port check.
        $client = [System.Net.Sockets.TcpClient]::new()
        try {
            $connect = $client.ConnectAsync("127.0.0.1", $Port)
            if (-not $connect.Wait(250)) {
                return $false
            }
            return $client.Connected
        }
        catch {
            return $false
        }
        finally {
            $client.Dispose()
        }
    }
}

function Test-NaweiHealth {
    param(
        [string]$Url,
        [int]$TimeoutSeconds = 15
    )

    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec $TimeoutSeconds
        if ($response.StatusCode -ne 200) { return $false }
        $body = $response.Content | ConvertFrom-Json
        if ($null -ne $body.loaded) { return $body.loaded -eq $true }
        if ($body.status -notin @('ok', 'ready')) { return $false }
        if ($null -ne $body.ready -and $body.ready -ne $true) { return $false }
        return $true
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

    $drainDirectory = $null
    $workerCount = 0
    if ($Service.Name -eq 'llm') {
        $drainDirectory = Join-Path $RuntimeDirectory ('llm-drain-' + [guid]::NewGuid().ToString('N'))
        New-Item -ItemType Directory -Path $drainDirectory | Out-Null
        $workerCount = if ($env:LLM_WORKERS) { [int]$env:LLM_WORKERS } else { 4 }
        $Service.Environment['LLM_DRAIN_DIRECTORY'] = $drainDirectory
        $Service.Environment['LLM_DRAIN_TOKEN'] = [guid]::NewGuid().ToString('N')
        $Service.Environment['LLM_WORKERS'] = [string]$workerCount
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
            drain_directory = $drainDirectory
            workers = $workerCount
        }
    }
    finally {
        foreach ($entry in $previousValues.GetEnumerator()) {
            [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, "Process")
        }
    }
}

function Save-NaweiProcessRecords {
    param([object[]]$Records, [string]$Path)
    $target = [IO.Path]::GetFullPath($Path)
    $temporary = $target + '.' + [guid]::NewGuid().ToString('N') + '.tmp'
    try {
        $json = ConvertTo-Json -InputObject @($Records) -Depth 8
        [IO.File]::WriteAllText($temporary, $json, [Text.UTF8Encoding]::new($false))
        for ($attempt = 0; $attempt -lt 20; $attempt++) {
            try {
                if ([IO.File]::Exists($target)) { [IO.File]::Replace($temporary, $target, $target + '.bak') }
                else { [IO.File]::Move($temporary, $target) }
                return
            } catch [IO.IOException] {
                if ($attempt -eq 19) { throw }
                Start-Sleep -Milliseconds 100
            }
        }
    } finally {
        if ([IO.File]::Exists($temporary)) { [IO.File]::Delete($temporary) }
    }
}

function Assert-NaweiProcessOwner {
    param($Service, [int]$ProcessId)
    $candidate = Get-CimInstance Win32_Process -Filter "ProcessId=$ProcessId" -ErrorAction Stop
    if (-not $candidate) { return }
    $entryPoint = if ($Service.Arguments[0] -eq '-m') { $Service.Arguments[1] } else { $Service.Arguments[0] }
    $entryPattern = if ($Service.Arguments[0] -eq '-m') {
        '(?i)(?:^|\s)-m\s+"?' + [regex]::Escape($entryPoint) + '"?(?:\s|$)'
    } else {
        '(?i)(?:^|[\s"])' + [regex]::Escape($entryPoint) + '(?:[\s"]|$)'
    }
    for ($depth = 0; $depth -lt 8 -and $candidate -and $candidate.Name -eq 'python.exe'; $depth++) {
        if ($candidate.ExecutablePath -eq $Service.FilePath -and $candidate.CommandLine -match $entryPattern) { return }
        if ($candidate.ParentProcessId -le 0) { break }
        $candidate = Get-CimInstance Win32_Process -Filter "ProcessId=$($candidate.ParentProcessId)" -ErrorAction Stop
    }
    throw "Refusing to stop PID ${ProcessId}: it is not the tracked $($Service.Name) application."
}

function Wait-NaweiLlmDrain {
    param($Record, [int]$TimeoutSeconds = 330, [switch]$ForceStop)
    if ($ForceStop) {
        Write-Warning 'Forcing LLM stop; unfinished memory writes may be lost.'
        return
    }
    if (-not $Record.drain_directory -or $Record.workers -lt 1) {
        throw 'This LLM process predates drain support. Use -ForceStop only after verifying pending work is complete.'
    }
    $requestFile = Join-Path $Record.drain_directory 'drain.request'
    Set-Content -LiteralPath $requestFile -Value 'drain' -Encoding ASCII
    $drained = $false
    try {
        $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
        while ((Get-Date) -lt $deadline) {
            $states = @(Get-ChildItem -LiteralPath $Record.drain_directory -Filter '*.json' | ForEach-Object {
                try {
                    $state = Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json
                    if ((Get-Process -Id $state.pid -ErrorAction SilentlyContinue) -and
                        [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() / 1000 - $state.updated_at -lt 3) { $state }
                } catch { }
            })
            if ($states.Count -eq $Record.workers -and @($states | Where-Object { $_.drained -ne $true }).Count -eq 0) {
                $drained = $true
                Write-Host "Drained all $($Record.workers) LLM workers."
                return
            }
            Start-Sleep -Milliseconds 250
        }
        throw "LLM did not drain within $TimeoutSeconds seconds; processes were left running. Check worker status files."
    }
    finally {
        if (-not $drained) { Remove-Item -LiteralPath $requestFile -ErrorAction SilentlyContinue }
    }
}

function Stop-NaweiServiceProcess {
    param($Service, $Record, [switch]$ForceStop)
    $processIds = @()
    if ($Record -and (Get-Process -Id $Record.pid -ErrorAction SilentlyContinue)) { $processIds += [int]$Record.pid }
    $processIds += @(Get-NetTCPConnection -State Listen -ErrorAction Stop |
        Where-Object LocalPort -eq $Service.Port | Select-Object -ExpandProperty OwningProcess)
    $processIds = @($processIds | Select-Object -Unique)
    foreach ($processId in $processIds) { Assert-NaweiProcessOwner -Service $Service -ProcessId $processId }
    if ($processIds.Count -gt 0 -and $Service.Name -eq 'llm') {
        Wait-NaweiLlmDrain -Record $Record -ForceStop:$ForceStop
    }
    foreach ($processId in $processIds) {
        if (-not (Get-Process -Id $processId -ErrorAction SilentlyContinue)) { continue }
        Assert-NaweiProcessOwner -Service $Service -ProcessId $processId
        & taskkill.exe /PID ([string]$processId) /T /F | Out-Null
        if ($LASTEXITCODE -ne 0 -and (Get-Process -Id $processId -ErrorAction SilentlyContinue)) {
            throw "Unable to stop $($Service.Name) PID $processId."
        }
    }
}

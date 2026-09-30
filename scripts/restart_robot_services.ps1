param(
    [ValidateSet('gateway', 'tts', 'llm', 'asr', 'knowledge')]
    [string[]]$Names = @('gateway'),
    [ValidateRange(5, 3600)]
    [int]$TimeoutSeconds = 900,
    [switch]$ForceStop
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'project_runtime.ps1')
Initialize-NaweiLocalEnvironment
$runtime = (Get-Content (Join-Path $script:NaweiProjectRoot 'logs\latest-runtime.txt') -Raw).Trim()
$recordsPath = Join-Path $runtime 'processes.json'
$records = Get-Content $recordsPath -Raw | ConvertFrom-Json
if (@($records | Where-Object { -not $_.name -or -not $_.pid }).Count -gt 0) {
    throw 'Invalid process metadata; restore service records before restarting.'
}
foreach ($serviceName in $Names) {
    $definition = Get-NaweiServiceDefinitions | Where-Object Name -eq $serviceName
    $record = $records | Where-Object name -eq $serviceName | Select-Object -First 1
    Stop-NaweiServiceProcess -Service $definition -Record $record -ForceStop:$ForceStop
    $replacement = Start-NaweiServiceProcess -Service $definition -RuntimeDirectory $runtime
    $records = @($records | Where-Object name -ne $serviceName) + @($replacement)
    Save-NaweiProcessRecords -Records $records -Path $recordsPath
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $ready = $false
    while ((Get-Date) -lt $deadline) {
        if (-not (Get-Process -Id $replacement.pid -ErrorAction SilentlyContinue)) {
            throw "$serviceName exited before becoming ready. Check $($replacement.stderr)."
        }
        try {
            $health = Invoke-RestMethod -Uri $definition.HealthUrl -TimeoutSec 15
            $ready = $health.status -in @('ok', 'ready')
            if ($serviceName -eq 'tts') { $ready = $health.loaded -eq $true }
            if ($serviceName -eq 'knowledge') { $ready = $ready -and $health.ready -eq $true }
            if ($serviceName -eq 'llm' -and $env:LLM_MEMORY_REQUIRED -eq 'true') {
                $ready = $ready -and $health.runtime.memory.postgres -eq 'ok' -and $health.runtime.memory.redis -eq 'ok'
            }
            if ($ready) {
                $owners = @(Get-NetTCPConnection -State Listen -ErrorAction Stop | Where-Object LocalPort -eq $definition.Port)
                $belongsToReplacement = $false
                foreach ($owner in $owners) {
                    $ancestor = [int]$owner.OwningProcess
                    for ($depth = 0; $depth -lt 8 -and $ancestor -gt 0; $depth++) {
                        if ($ancestor -eq $replacement.pid) { $belongsToReplacement = $true; break }
                        $ancestorProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$ancestor" -ErrorAction Stop
                        if (-not $ancestorProcess) { break }
                        $ancestor = [int]$ancestorProcess.ParentProcessId
                    }
                }
                $ready = $belongsToReplacement
                if ($ready) { break }
            }
        } catch { }
        Start-Sleep -Milliseconds 500
    }
    if (-not $ready) { throw "$serviceName did not become ready within $TimeoutSeconds seconds. Check runtime logs." }
    Write-Host "Restarted and verified $serviceName on port $($definition.Port)"
}

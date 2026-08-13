$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "project_runtime.ps1")

$rows = foreach ($service in Get-NaweiServiceDefinitions) {
    $listening = Test-NaweiPortListening $service.Port
    [pscustomobject]@{
        Service = $service.Name
        Port = $service.Port
        Listening = $listening
        Healthy = $listening -and (Test-NaweiHealth $service.HealthUrl)
    }
}

$rows | Format-Table -AutoSize

$latestRuntime = Join-Path $script:NaweiProjectRoot "logs\latest-runtime.txt"
if (Test-Path $latestRuntime) {
    Write-Host "Logs: $((Get-Content $latestRuntime -Raw).Trim())"
}

if (@($rows | Where-Object { -not $_.Healthy }).Count -gt 0) {
    exit 1
}

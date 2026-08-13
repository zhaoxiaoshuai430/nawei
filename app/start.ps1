param([int]$Port = 5173)

$appDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $appDirectory
Write-Host "Nawei Voice Lab: http://127.0.0.1:$Port" -ForegroundColor Cyan
python -m http.server $Port --bind 127.0.0.1

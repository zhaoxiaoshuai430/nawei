param(
    [switch]$Flash,
    [string]$Port = "COM1"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot

Push-Location $projectRoot
try {
    python scripts/build.py bread-compact-wifi --config config.nawei.json --name bread-compact-wifi-nawei
    if ($LASTEXITCODE -ne 0) {
        throw "Nawei firmware build failed."
    }

    if ($Flash) {
        idf.py -p $Port flash
        if ($LASTEXITCODE -ne 0) {
            throw "Nawei firmware flash failed."
        }
    }
} finally {
    Pop-Location
}

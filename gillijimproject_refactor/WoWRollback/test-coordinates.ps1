#!/usr/bin/env pwsh
# Test coordinate extraction from converted LK ADT files

$ErrorActionPreference = "Stop"

Write-Host "=== Testing Coordinate Extraction ===" -ForegroundColor Cyan
Write-Host ""

# Paths
$wdtPath = "I:\wow_alpha\05x\0.5.3.3368\World\Maps\Kalimdor\Kalimdor.wdt"
$convertedAdtDir = "i:\parp-tools\pm4next-branch\parp-tools\gillijimproject_refactor\output_dirfart2\World\Maps\Kalimdor"
$exePath = ".\WoWRollback.Cli\bin\Release\net9.0\WoWRollback.Cli.exe"

# Verify inputs exist
if (-not (Test-Path $wdtPath)) {
    Write-Host "ERROR: Alpha WDT not found: $wdtPath" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $convertedAdtDir)) {
    Write-Host "ERROR: Converted ADT directory not found: $convertedAdtDir" -ForegroundColor Red
    exit 1
}

Write-Host "[1/2] Analyzing Alpha WDT WITH converted LK ADT coordinates..." -ForegroundColor Yellow
& $exePath analyze-alpha-wdt `
    --wdt-file $wdtPath `
    --converted-adt-dir $convertedAdtDir `
    --out "rollback_outputs"

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Analysis failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[2/2] Checking generated asset ledger for coordinates..." -ForegroundColor Yellow

# Find the asset ledger
$ledgerPath = Get-ChildItem -Path "rollback_outputs\0.5.3.3368\Kalimdor" -Filter "*_assetledger.csv" | 
    Sort-Object LastWriteTime -Descending | 
    Select-Object -First 1 -ExpandProperty FullName

if (-not $ledgerPath) {
    Write-Host "ERROR: Asset ledger not found!" -ForegroundColor Red
    exit 1
}

Write-Host "Asset ledger: $ledgerPath" -ForegroundColor Gray
Write-Host ""

# Read and check first few entries
$csv = Import-Csv $ledgerPath
$sampleSize = 5
$samples = $csv | Select-Object -First $sampleSize

Write-Host "Sample entries from asset ledger:" -ForegroundColor Cyan
foreach ($entry in $samples) {
    $coords = "($($entry.WorldX), $($entry.WorldY), $($entry.WorldZ))"
    
    if ($entry.WorldX -eq 0 -and $entry.WorldY -eq 0 -and $entry.WorldZ -eq 0) {
        Write-Host "  UID $($entry.UniqueID): $coords" -ForegroundColor Red -NoNewline
        Write-Host " ← ZERO!" -ForegroundColor Red
    } else {
        Write-Host "  UID $($entry.UniqueID): $coords" -ForegroundColor Green -NoNewline
        Write-Host " ← OK" -ForegroundColor Green
    }
}

# Count zeros vs real coordinates
$zeroCount = ($csv | Where-Object { $_.WorldX -eq 0 -and $_.WorldY -eq 0 -and $_.WorldZ -eq 0 }).Count
$totalCount = $csv.Count
$realCount = $totalCount - $zeroCount

Write-Host ""
Write-Host "=== Results ===" -ForegroundColor Cyan
Write-Host "Total objects: $totalCount" -ForegroundColor White
Write-Host "With coordinates: $realCount" -ForegroundColor Green
Write-Host "Still (0,0,0): $zeroCount" -ForegroundColor $(if ($zeroCount -gt 0) { "Yellow" } else { "Green" })

if ($realCount -gt 0) {
    Write-Host ""
    Write-Host "SUCCESS: Coordinate extraction working!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "FAILURE: No coordinates found" -ForegroundColor Red
    exit 1
}

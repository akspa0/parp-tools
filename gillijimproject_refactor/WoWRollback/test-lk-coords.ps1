#!/usr/bin/env pwsh
# Quick test to check if converted LK ADT files have placement coordinates

$testAdt = "i:\parp-tools\pm4next-branch\parp-tools\gillijimproject_refactor\output_dirfart2\World\Maps\Azeroth\Azeroth_30_30.adt"

Write-Host "Testing LK ADT: $testAdt" -ForegroundColor Cyan

# Read the file as binary
$bytes = [System.IO.File]::ReadAllBytes($testAdt)

# Find MDDF chunk (stored as "FDDM" on disk - reversed byte order)
$mddfPattern = [byte[]]@(0x46, 0x44, 0x44, 0x4D) # "FDDM"
$mddfIndex = -1

for ($i = 0; $i -lt $bytes.Length - 4; $i++) {
    if ($bytes[$i] -eq $mddfPattern[0] -and
        $bytes[$i+1] -eq $mddfPattern[1] -and
        $bytes[$i+2] -eq $mddfPattern[2] -and
        $bytes[$i+3] -eq $mddfPattern[3]) {
        $mddfIndex = $i
        break
    }
}

if ($mddfIndex -eq -1) {
    Write-Host "MDDF chunk not found" -ForegroundColor Red
    exit 1
}

Write-Host "Found MDDF at offset: $mddfIndex" -ForegroundColor Green

# Read chunk size (4 bytes after FourCC)
$chunkSize = [BitConverter]::ToInt32($bytes, $mddfIndex + 4)
Write-Host "MDDF chunk size: $chunkSize bytes" -ForegroundColor Yellow

# Calculate entry count (36 bytes per entry)
$entryCount = $chunkSize / 36
Write-Host "MDDF entries: $entryCount" -ForegroundColor Yellow

# Read first entry coordinates (offset 8, 12, 16 from data start)
$dataStart = $mddfIndex + 8
$x = [BitConverter]::ToSingle($bytes, $dataStart + 8)
$z = [BitConverter]::ToSingle($bytes, $dataStart + 12)
$y = [BitConverter]::ToSingle($bytes, $dataStart + 16)

Write-Host ""
Write-Host "First MDDF entry coordinates:" -ForegroundColor Cyan
Write-Host "  X: $x" -ForegroundColor $(if ($x -eq 0) { "Red" } else { "Green" })
Write-Host "  Y: $y" -ForegroundColor $(if ($y -eq 0) { "Red" } else { "Green" })
Write-Host "  Z: $z" -ForegroundColor $(if ($z -eq 0) { "Red" } else { "Green" })

if ($x -eq 0 -and $y -eq 0 -and $z -eq 0) {
    Write-Host ""
    Write-Host "RESULT: Coordinates are (0,0,0) - Alpha data was copied directly!" -ForegroundColor Red
    Write-Host "The converted LK ADT files don't have placement coordinates either." -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "RESULT: Coordinates are populated!" -ForegroundColor Green
    Write-Host "We can use these LK ADT files for placement data." -ForegroundColor Green
}

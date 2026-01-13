#!/usr/bin/env pwsh
# Test script for MCLY/MCAL/MCSH format fix
# Tests the corrected chunk writing to verify texture placement

Write-Host "=== Testing MCLY/MCAL/MCSH Format Fix ===" -ForegroundColor Cyan
Write-Host ""

$ErrorActionPreference = "Stop"

# Build the project
Write-Host "[1/4] Building LkToAlphaModule..." -ForegroundColor Yellow
dotnet build WoWRollback.LkToAlphaModule\WoWRollback.LkToAlphaModule.csproj
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}
Write-Host "Build succeeded!" -ForegroundColor Green
Write-Host ""

# Pack Kalidar with verbose logging
Write-Host "[2/4] Packing Kalidar with verbose logging..." -ForegroundColor Yellow
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
dotnet run --project WoWRollback.AdtConverter -- pack-monolithic `
    --lk-dir ..\test_data\0.6.0\tree\World\Maps\Kalidar\ `
    --lk-wdt ..\test_data\0.6.0\tree\World\Maps\Kalidar\Kalidar.wdt `
    --map Kalidar

if ($LASTEXITCODE -ne 0) {
    Write-Host "Pack failed!" -ForegroundColor Red
    exit 1
}
Write-Host "Pack succeeded!" -ForegroundColor Green
Write-Host ""

# Find the output file
Write-Host "[3/4] Checking output file..." -ForegroundColor Yellow
$outputFiles = Get-ChildItem "project_output\Kalidar_*\Kalidar.wdt" | Sort-Object LastWriteTime -Descending
if ($outputFiles.Count -eq 0) {
    Write-Host "No output file found!" -ForegroundColor Red
    exit 1
}

$outputFile = $outputFiles[0]
$sizeMB = [math]::Round($outputFile.Length / 1MB, 2)

Write-Host "Output file: $($outputFile.Name)" -ForegroundColor Cyan
Write-Host "File size: $sizeMB MB" -ForegroundColor Cyan

# Verify file size is in expected range (40-41 MB)
if ($sizeMB -lt 35) {
    Write-Host "WARNING: File size too small ($sizeMB MB)! Expected ~40-41 MB" -ForegroundColor Red
    Write-Host "This suggests texture data may still be missing." -ForegroundColor Red
} elseif ($sizeMB -gt 45) {
    Write-Host "WARNING: File size too large ($sizeMB MB)! Expected ~40-41 MB" -ForegroundColor Yellow
} else {
    Write-Host "File size looks good! ($sizeMB MB)" -ForegroundColor Green
}
Write-Host ""

# Inspect structure
Write-Host "[4/4] Inspecting structure (first 3 tiles)..." -ForegroundColor Yellow
dotnet run --project WoWRollback.AdtConverter -- inspect-alpha `
    --wdt $outputFile.FullName --tiles 3

Write-Host ""
Write-Host "=== Test Complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Copy output file to Alpha 0.5.3 client: Data\World\Maps\Kalidar\" -ForegroundColor White
Write-Host "2. Launch client and test in-game" -ForegroundColor White
Write-Host "3. Verify textures appear correctly (roads follow road bed)" -ForegroundColor White
Write-Host ""
Write-Host "Output location: $($outputFile.DirectoryName)" -ForegroundColor Cyan

#!/usr/bin/env pwsh
# Test with verbose logging to debug terrain issue

Write-Host "=== Testing with Verbose Logging ===" -ForegroundColor Cyan
Write-Host ""

$ErrorActionPreference = "Stop"

# Pack with verbose logging - limit output to first few chunks
Write-Host "Packing Shadowfang with verbose logging (first tile only)..." -ForegroundColor Yellow
dotnet run --project WoWRollback.AdtConverter -- pack-monolithic `
    --lk-dir ..\lk_shadowfang\Shadowfang\ `
    --lk-wdt ..\lk_shadowfang\Shadowfang.wdt `
    --map Shadowfang `
    --verbose-logging 2>&1 | Select-Object -First 100

Write-Host ""
Write-Host "Check the output above for:" -ForegroundColor Yellow
Write-Host "1. [MCVT] extraction messages - are they found?" -ForegroundColor White
Write-Host "2. [MCNR] extraction messages - are they found?" -ForegroundColor White
Write-Host "3. [WRITE] messages - what sizes are being written?" -ForegroundColor White
Write-Host "4. Look for 'McvtOffset=0' which means MCVT is missing" -ForegroundColor White

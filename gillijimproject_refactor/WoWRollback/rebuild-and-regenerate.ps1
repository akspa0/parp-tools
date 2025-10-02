#!/usr/bin/env pwsh
# Rebuild WoWRollback and invoke the consolidated regeneration script with viewer linting

[CmdletBinding(PositionalBinding=$false)]
param(
    [string[]]$Maps = @("DeadminesInstance"),
    [string[]]$Versions = @("0.5.3.3368","0.5.5.3494"),
    [Alias("AlphaRoot")][string]$TestDataRoot = "..\test_data",
    [string]$OutputRoot = "rollback_outputs",
    [string]$ConvertedAdtDir,
    [string]$ImageFormat = "webp",
    [int]$ImageQuality = 85,
    [switch]$Clean,
    [switch]$Serve,
    [switch]$SkipViewerPrompt
)

Set-Location -Path $PSScriptRoot

Write-Host "=== WoWRollback Rebuild & Regenerate ===" -ForegroundColor Cyan
Write-Host ""

if ($Clean) {
    Write-Host "[0/2] Cleaning previous outputs..." -ForegroundColor Yellow
    if (Test-Path $OutputRoot) {
        Remove-Item $OutputRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
    Write-Host "✓ Clean complete" -ForegroundColor Green
    Write-Host ""
}

function Get-MapsFromTestData([string]$root) {
    if (-not $root -or -not (Test-Path $root)) { return @() }
    $results = New-Object System.Collections.Generic.HashSet[string]([System.StringComparer]::OrdinalIgnoreCase)

    try {
        $wdtFiles = Get-ChildItem -Path $root -Recurse -File -Filter '*.wdt' -ErrorAction SilentlyContinue
        foreach ($f in $wdtFiles) {
            $full = $f.FullName
            if ($full -match '(?i)[\\/](World)[\\/]Maps[\\/]') {
                [void]$results.Add($f.Directory.Name)
            }
        }

        $minimapDirs = Get-ChildItem -Path $root -Recurse -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Parent -and $_.Parent.Name -match '^(?i)Minimaps$' }
        foreach ($d in $minimapDirs) {
            [void]$results.Add($d.Name)
        }
    } catch {
        Write-Host "Warning: failed to auto-discover maps under $root ($_ )" -ForegroundColor Yellow
    }

    return (@($results) | Sort-Object)
}

if ($Maps.Count -eq 1 -and $Maps[0].Equals('auto', [StringComparison]::OrdinalIgnoreCase)) {
    $discoveredMaps = Get-MapsFromTestData -root $TestDataRoot
    if ($discoveredMaps.Count -gt 0) {
        Write-Host "Auto-discovered maps: $([string]::Join(', ', $discoveredMaps))" -ForegroundColor Gray
        $Maps = $discoveredMaps
    } else {
        Write-Host "No maps discovered under $TestDataRoot; defaulting to Azeroth,Kalimdor" -ForegroundColor Yellow
        $Maps = @('Azeroth','Kalimdor')
    }
}

Write-Host "[1/2] Building solution..." -ForegroundColor Yellow
dotnet build WoWRollback.sln --configuration Release
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Build complete" -ForegroundColor Green
Write-Host ""

$regenScript = Join-Path $PSScriptRoot 'regenerate-with-coordinates.ps1'
if (-not (Test-Path $regenScript)) {
    Write-Host "Unable to locate regenerate-with-coordinates.ps1" -ForegroundColor Red
    exit 1
}

$regenParams = @{}
if ($ConvertedAdtDir) { $regenParams.ConvertedAdtDir = $ConvertedAdtDir }
if ($Versions) { $regenParams.Versions = $Versions }
if ($Maps) { $regenParams.Maps = $Maps }
if ($TestDataRoot) { $regenParams.TestDataRoot = $TestDataRoot }
if ($OutputRoot) { $regenParams.OutputRoot = $OutputRoot }
if ($ImageFormat) { $regenParams.ImageFormat = $ImageFormat }
$regenParams.ImageQuality = $ImageQuality
if ($Serve -or $SkipViewerPrompt) { $regenParams.SkipViewerPrompt = $true }

Write-Host "[2/2] Regenerating comparison data via regenerate-with-coordinates.ps1" -ForegroundColor Yellow
& $regenScript @regenParams
if ($LASTEXITCODE -ne 0) {
    Write-Host "Regeneration failed" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Regeneration complete" -ForegroundColor Green
Write-Host ""

if ($Serve) {
    $comparisonsRoot = Join-Path $OutputRoot 'comparisons'
    if (-not (Test-Path $comparisonsRoot)) {
        Write-Host "Viewer outputs not found under $comparisonsRoot; skipping server start" -ForegroundColor Yellow
        return
    }

    $latestViewer = Get-ChildItem -Path $comparisonsRoot -Directory |
        Sort-Object LastWriteTime -Descending |
        Where-Object { Test-Path (Join-Path $_.FullName 'viewer\index.html') } |
        Select-Object -First 1

    if ($latestViewer) {
        $viewerDir = Join-Path $latestViewer.FullName 'viewer'
        Write-Host "Starting server on http://localhost:8080 from $viewerDir" -ForegroundColor Cyan
        Set-Location $viewerDir
        python -m http.server 8080
    } else {
        Write-Host "Viewer outputs not found; skipping server start" -ForegroundColor Yellow
    }
}

Write-Host "Viewer outputs are under $OutputRoot/comparisons" -ForegroundColor Cyan

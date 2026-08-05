#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Run the full archaeology pipeline: harvest MPQ -> NPZ -> V50 Zarr -> archaeology.

.DESCRIPTION
    A single command: point at a game client, grabs terrain from MPQ archives,
    builds the V50-format Zarr store, and runs the full archaeology suite.

    Pipeline:
      1. harvest-map-mpq: C# tool reads ADT tiles from MPQ archives, writes NPZ shards
      2. build_v50_store_from_npz: Python script builds V50 Zarr store + runs archaeology

.PARAMETER ClientRoot
    Path to the game client root (e.g. H:\CLIENTS\3_3_5_12340\World of Warcraft)

.PARAMETER MapName
    Map directory name (e.g. Expansion01, Northrend, Azeroth, Kalimdor)

.PARAMETER BuildId
    Build identifier for output paths (e.g. 3_3_5_12340, 3_0_1_8303)

.PARAMETER Limit
    Max tiles to harvest (default: all)

.PARAMETER ConfirmRun
    Actually run (without this, only prints commands)

.EXAMPLE
    .\run-archaeology.ps1 -ClientRoot "H:\CLIENTS\2.X_Pre-Release_Windows_enUS_2.0.0.5610\World of Warcraft" -MapName Expansion01 -BuildId 2_0_0_5610 -ConfirmRun
    .\run-archaeology.ps1 -ClientRoot "H:\CLIENTS\3.X_Pre-Release_Windows_enUS_3.0.1.8303\World of Warcraft" -MapName Northrend -BuildId 3_0_1_8303 -ConfirmRun
#>

param(
    [Parameter(Mandatory = $true)] [string]$ClientRoot,
    [Parameter(Mandatory = $true)] [string]$MapName,
    [Parameter(Mandatory = $true)] [string]$BuildId,
    [int]$Limit = 0,
    [switch]$ConfirmRun
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path "$PSScriptRoot/.."
$DataHarvester = "$RepoRoot/data-harvester"
$HarvestTool = "$RepoRoot/tools/harvest/WowViewer.Tool.Harvest"
$OutputRoot = "$RepoRoot/output/archaeology/$BuildId"
$NpzDir = "$OutputRoot/npz/$MapName"
$StoreDir = "$OutputRoot/store"
$StorePath = "$StoreDir/$MapName.zarr"
$ArchaeoDir = "$OutputRoot/archaeo"

# Build harvest tool if needed
$HarvestDll = "$HarvestTool/bin/Debug/net10.0/WowViewer.Tool.Harvest.dll"
if (-not (Test-Path $HarvestDll)) {
    Write-Host "Building harvest tool..." -ForegroundColor Yellow
    Push-Location $RepoRoot
    try { dotnet build "wow-viewer/tools/harvest/WowViewer.Tool.Harvest/WowViewer.Tool.Harvest.csproj" -c Debug -nologo
          if ($LASTEXITCODE -ne 0) { throw "Build failed ($LASTEXITCODE)" } }
    finally { Pop-Location }
}

Write-Host "Client: $ClientRoot`nMap: $MapName`nBuild: $BuildId`nOutput: $OutputRoot" -ForegroundColor Cyan

# Step 1: Harvest from MPQ
Write-Host "`n=== Step 1: Harvest NPZ from MPQ ===" -ForegroundColor Green
New-Item -ItemType Directory -Force -Path $NpzDir | Out-Null
$HarvestArgs = @("harvest-map-mpq", "--client-root", $ClientRoot, "--map", $MapName, "--output-dir", $NpzDir)
if ($Limit -gt 0) { $HarvestArgs += "--limit"; $HarvestArgs += $Limit }
Write-Host "dotnet $HarvestDll $($HarvestArgs -join ' ')" -ForegroundColor Gray
if ($ConfirmRun) {
    Write-Host "Running harvest..." -ForegroundColor Yellow
    & dotnet $HarvestDll $HarvestArgs
    if ($LASTEXITCODE -ne 0) { throw "Harvest failed" }
    Write-Host "NPZ -> $NpzDir" -ForegroundColor Green
} else { Write-Host "[DRY-RUN] Pass -ConfirmRun" -ForegroundColor Yellow; exit }

# Step 2: Build V50 Zarr store + run archaeology
Write-Host "`n=== Step 2: Build V50 Zarr store + archaeology ===" -ForegroundColor Green
New-Item -ItemType Directory -Force -Path $StoreDir | Out-Null
New-Item -ItemType Directory -Force -Path $ArchaeoDir | Out-Null

Write-Host "Building V50 store and running archaeology..." -ForegroundColor Yellow
Push-Location $DataHarvester
try {
    uv run python scripts/build_v50_store_from_npz.py `
        --npz-dir "$NpzDir" `
        --store "$StorePath" `
        --output "$ArchaeoDir" `
        --map $MapName `
        --near-zero-band inf
    if ($LASTEXITCODE -ne 0) { throw "Build/archaeology failed" }
} finally { Pop-Location }

Write-Host "`n=== COMPLETE ===" -ForegroundColor Green
Write-Host "NPZ:  $NpzDir" -ForegroundColor Gray
Write-Host "Zarr: $StorePath" -ForegroundColor Gray
Write-Host "Arch: $ArchaeoDir" -ForegroundColor Gray
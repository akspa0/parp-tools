#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Run archaeology on all maps in all 1.x clients in H:\CLIENTS.

.DESCRIPTION
    Discovers all 1.x (Vanilla/Classic) client builds in H:\CLIENTS, finds all
    terrain-trainable maps in each, and runs the full archaeology pipeline on each
    map: harvest MPQ -> V50 Zarr store -> tile inventory -> synthesis -> composites.

.PARAMETER ClientRootsDir
    Directory containing client builds (default: H:\CLIENTS)

.PARAMETER ClientFilter
    Regex filter for client names (default: "1\.X_Retail_Windows")

.PARAMETER Limit
    Max tiles per-map (default: 0 = all)

.PARAMETER MaxMapsPerClient
    Max maps to process per client (default: 0 = all)

.PARAMETER OutputRoot
    Root output directory (default: wow-viewer/output/archaeology)

.PARAMETER ConfirmRun
    Actually run (without this, only prints the plan)

.EXAMPLE
    # Dry-run to see the plan:
    .\run-batch-archaeology.ps1

    # Full run on all 1.x Windows clients:
    .\run-batch-archaeology.ps1 -ConfirmRun

    # Limited test (2 maps, 10 tiles each):
    .\run-batch-archaeology.ps1 -MaxMapsPerClient 2 -Limit 10 -ConfirmRun
#>

param(
    [string]$ClientRootsDir = "H:\CLIENTS",
    [string]$ClientFilter = "1\.X_Retail_Windows",
    [int]$Limit = 0,
    [int]$MaxMapsPerClient = 0,
    [string]$OutputRoot = "",
    [switch]$ConfirmRun
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$DataHarvester = "$RepoRoot/data-harvester"
$HarvestTool = "$RepoRoot/tools/harvest/WowViewer.Tool.Harvest"
if (-not $OutputRoot) { $OutputRoot = "$RepoRoot/output/archaeology" }

# Build harvest tool if needed
$HarvestDll = "$HarvestTool/bin/Debug/net10.0/WowViewer.Tool.Harvest.dll"
if (-not (Test-Path $HarvestDll)) {
    Write-Host "Building harvest tool..." -ForegroundColor Yellow
    Push-Location $RepoRoot
    try { dotnet build "wow-viewer/tools/harvest/WowViewer.Tool.Harvest/WowViewer.Tool.Harvest.csproj" -c Debug -nologo
          if ($LASTEXITCODE -ne 0) { throw "Build failed ($LASTEXITCODE)" } }
    finally { Pop-Location }
}

# Discover all 1.x clients
$clients = Get-ChildItem -Path $ClientRootsDir -Directory | Where-Object { $_.Name -match $ClientFilter } | Sort-Object Name
if ($clients.Count -eq 0) {
    Write-Error "No clients matching '$ClientFilter' found in $ClientRootsDir"
    exit 1
}

Write-Host "Found $($clients.Count) 1.x clients:" -ForegroundColor Cyan
foreach ($c in $clients) { Write-Host "  $($c.Name)" -ForegroundColor Gray }

$totalClients = 0
$totalMaps = 0
$totalTiles = 0

foreach ($client in $clients) {
    $clientRoot = $client.FullName
    $clientName = $client.Name
    $buildId = if ($clientName -match '(\d+\.\d+\.\d+\.\d+)') { $matches[1].Replace(".", "_") } else { $clientName.Replace(".", "_") }

    Write-Host "`n========================================================" -ForegroundColor Magenta
    Write-Host "CLIENT: $clientName" -ForegroundColor Magenta
    Write-Host "BUILD:  $buildId" -ForegroundColor Magenta
    Write-Host "========================================================" -ForegroundColor Magenta

    # Step 1: Discover maps - capture stdout only
    Write-Host "`n--- Discovering maps..." -ForegroundColor Yellow
    $stdout = @()
    $stderr = @()
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "dotnet"
    $psi.Arguments = "`"$HarvestDll`" discover-maps --client-root `"$clientRoot`""
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $p = [System.Diagnostics.Process]::Start($psi)
    $stdout = $p.StandardOutput.ReadToEnd()
    $stderr = $p.StandardError.ReadToEnd()
    $p.WaitForExit()
    if ($p.ExitCode -ne 0) {
        Write-Host "  SKIP: discover-maps failed for $clientName" -ForegroundColor Red
        if ($stderr) { Write-Host "  $($stderr -split "`n")[0]" -ForegroundColor DarkGray }
        continue
    }

    # Parse discovered maps from JSON output
    $maps = @()
    try {
        $discovered = $stdout | ConvertFrom-Json
        $maps = $discovered | Where-Object { $_.Include -and $_.HasReadableTile } | ForEach-Object { $_.Map }
        Write-Host "  Found $($maps.Count) maps: $($maps -join ', ')" -ForegroundColor Green
    } catch {
        Write-Host "  WARNING: could not parse discover-maps output" -ForegroundColor Yellow
        Write-Host "  First 200 chars: $($stdout.Substring(0, [Math]::Min(200, $stdout.Length)))" -ForegroundColor DarkGray
    }

    if ($maps.Count -eq 0) {
        Write-Host "  No maps discovered for $clientName" -ForegroundColor Yellow
        continue
    }

    Write-Host "  Found $($maps.Count) maps: $($maps -join ', ')" -ForegroundColor Green

    $mapsToProcess = if ($MaxMapsPerClient -gt 0) { $maps | Select-Object -First $MaxMapsPerClient } else { $maps }

    foreach ($map in $mapsToProcess) {
        $buildOutput = "$OutputRoot/$buildId"
        $NpzDir = "$buildOutput/npz/$map"
        $StorePath = "$buildOutput/store/$map.zarr"
        $ArchaeoDir = "$buildOutput/archaeo/$map"

        Write-Host "`n--- Map: $map ---" -ForegroundColor Cyan

        # Step 2: Harvest from MPQ
        Write-Host "  Harvesting..." -ForegroundColor Yellow
        New-Item -ItemType Directory -Force -Path $NpzDir | Out-Null
        $HarvestArgs = @("harvest-map-mpq", "--client-root", $clientRoot, "--map", $map, "--output-dir", $NpzDir)
        if ($Limit -gt 0) { $HarvestArgs += "--limit"; $HarvestArgs += $Limit }

        if ($ConfirmRun) {
            & dotnet $HarvestDll $HarvestArgs
            if ($LASTEXITCODE -ne 0) { Write-Host "  SKIP: harvest failed for $map" -ForegroundColor Red; continue }
        } else {
            Write-Host "  [DRY-RUN] dotnet $HarvestDll $($HarvestArgs -join ' ')" -ForegroundColor Gray
        }

        # Step 3: Build V50 store + archaeology
        Write-Host "  Building V50 store + archaeology..." -ForegroundColor Yellow
        New-Item -ItemType Directory -Force -Path (Split-Path $StorePath -Parent) | Out-Null
        New-Item -ItemType Directory -Force -Path $ArchaeoDir | Out-Null

        if ($ConfirmRun) {
            Push-Location $DataHarvester
            try {
                uv run python scripts/build_v50_store_from_npz.py `
                    --npz-dir "$NpzDir" `
                    --store "$StorePath" `
                    --output "$ArchaeoDir" `
                    --map $map `
                    --near-zero-band inf
                if ($LASTEXITCODE -eq 0) {
                    $totalTiles++
                    Write-Host "  DONE: $map" -ForegroundColor Green
                } else {
                    Write-Host "  FAILED: $map" -ForegroundColor Red
                }
            } finally { Pop-Location }
        } else {
            Write-Host "  [DRY-RUN] uv run python scripts/build_v50_store_from_npz.py ..." -ForegroundColor Gray
        }

        $totalMaps++
    }
    $totalClients++
}

# Summary
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "BATCH ARCHAEOLOGY COMPLETE" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Clients processed: $totalClients" -ForegroundColor White
Write-Host "Maps processed:    $totalMaps" -ForegroundColor White
if ($ConfirmRun) {
    Write-Host "Output: $OutputRoot" -ForegroundColor White
} else {
    Write-Host "STATUS: DRY-RUN (pass -ConfirmRun to execute)" -ForegroundColor Yellow
}
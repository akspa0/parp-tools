#!/usr/bin/env pwsh

param(
    [string]$OutputDir = "",
    [int]$ValidationPerBucket = 5,
    [switch]$Resume,
    [switch]$SkipValidation,
    [switch]$SkipVisualization
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step([string]$Message) { Write-Host "`n==> $Message" -ForegroundColor Cyan }
function Write-Ok([string]$Message) { Write-Host "[OK]  $Message" -ForegroundColor Green }

function Invoke-Converter([string[]]$CommandArgs) {
    & dotnet $script:ConverterDll @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Converter command failed: dotnet $script:ConverterDll $($CommandArgs -join ' ')"
    }
}

function Invoke-Harvest([string[]]$CommandArgs) {
    & dotnet $script:HarvestDll @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Harvest command failed: dotnet $script:HarvestDll $($CommandArgs -join ' ')"
    }
}

function Get-ClientBuilds() {
    return @(
        @{ Build = "0_5_3_3368"; Root = "I:\parp\parp-tools\output\tmp\wowarchive-clients\0_5_3_3368\World of Warcraft" },
        @{ Build = "0_5_5_3494"; Root = "I:\parp\parp-tools\output\tmp\wowarchive-clients\0_5_5_3494\World of Warcraft" },
        @{ Build = "0_7_0_3694"; Root = "I:\parp\parp-tools\output\tmp\wowarchive-clients\0_7_0_3694\World of Warcraft" },
        @{ Build = "3_0_1_8303"; Root = "I:\parp\parp-tools\output\tmp\wowarchive-clients\3_0_1_8303\World of Warcraft" },
        @{ Build = "3_3_5_12340"; Root = "I:\parp\parp-tools\output\tmp\wowarchive-clients\3_3_5_12340\World of Warcraft" },
        @{ Build = "4_0_0_11927"; Root = "I:\parp\parp-tools\output\tmp\wowarchive-clients\4_0_0_11927\World of Warcraft" }
    )
}

function Write-Json($Path, $Object) {
    $dir = Split-Path -Parent $Path
    if ($dir) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    $Object | ConvertTo-Json -Depth 16 | Set-Content -Path $Path -Encoding UTF8
}

$WowViewerRoot = Split-Path -Parent $PSScriptRoot
$script:ConverterProject = Join-Path $WowViewerRoot "tools\converter\WowViewer.Tool.Converter\WowViewer.Tool.Converter.csproj"
$script:ConverterDll = Join-Path $WowViewerRoot "tools\converter\WowViewer.Tool.Converter\bin\Debug\net10.0\WowViewer.Tool.Converter.dll"
$script:HarvestProject = Join-Path $WowViewerRoot "tools\harvest\WowViewer.Tool.Harvest\WowViewer.Tool.Harvest.csproj"
$script:HarvestDll = Join-Path $WowViewerRoot "tools\harvest\WowViewer.Tool.Harvest\bin\Debug\net10.0\WowViewer.Tool.Harvest.dll"
$DataHarvesterRoot = Join-Path $WowViewerRoot "data-harvester"

Write-Step "Building converter and harvest tools once"
& dotnet build $script:ConverterProject -c Debug
if ($LASTEXITCODE -ne 0) { throw "Converter build failed: $script:ConverterProject" }
& dotnet build $script:HarvestProject -c Debug
if ($LASTEXITCODE -ne 0) { throw "Harvest build failed: $script:HarvestProject" }
if (-not (Test-Path $script:ConverterDll)) { throw "Converter DLL not found: $script:ConverterDll" }
if (-not (Test-Path $script:HarvestDll)) { throw "Harvest DLL not found: $script:HarvestDll" }

if (-not $OutputDir) {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputDir = Join-Path $WowViewerRoot "output\datasets\full_shard_batch_$timestamp"
}

$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
Write-Ok "Output root: $OutputDir"

$MapListDir = Join-Path $OutputDir "map_lists"
$ShardRoot = Join-Path $OutputDir "shards"
$ManifestDir = Join-Path $OutputDir "manifests"
$ValidationSampleDir = Join-Path $OutputDir "validation_samples"
$VisualizationDir = Join-Path $OutputDir "visualizations"

$builds = Get-ClientBuilds
foreach ($build in $builds) {
    if (-not (Test-Path $build.Root)) {
        throw "Missing staged client root for $($build.Build): $($build.Root)"
    }
}

Write-Step "Discovering maps from Map.dbc"
$runManifest = [System.Collections.Generic.List[object]]::new()
foreach ($build in $builds) {
    $mapListPath = Join-Path $MapListDir "$($build.Build).json"
    if (-not ($Resume -and (Test-Path $mapListPath))) {
        Invoke-Converter @("dataset-list-maps", "--client-root", $build.Root, "--output", $mapListPath)
    }

    $maps = Get-Content $mapListPath | ConvertFrom-Json
    Write-Ok "$($build.Build): $($maps.Count) discovered maps"

    foreach ($map in $maps) {
        $mapOutputDir = Join-Path $ShardRoot (Join-Path $build.Build $map.Directory)
        $existingCount = 0
        if (Test-Path $mapOutputDir) {
            $existingCount = @(Get-ChildItem $mapOutputDir -Filter "*.npz" -File -ErrorAction SilentlyContinue).Count
        }
        if ($Resume -and $existingCount -gt 0) {
            Write-Host "  Reusing $($build.Build)/$($map.Directory) ($existingCount shards)"
        }
        else {
            Write-Host "  Harvesting $($build.Build)/$($map.Directory)"
            New-Item -ItemType Directory -Force -Path $mapOutputDir | Out-Null
            Invoke-Harvest @(
                "harvest-map-mpq",
                "--client-root", $build.Root,
                "--map", $map.Directory,
                "--output-dir", $mapOutputDir
            )
            $existingCount = @(Get-ChildItem $mapOutputDir -Filter "*.npz" -File -ErrorAction SilentlyContinue).Count
        }

        $runManifest.Add([PSCustomObject]@{
            build = $build.Build
            client_root = $build.Root
            map_id = $map.Id
            map_directory = $map.Directory
            map_name = $map.Name
            shard_dir = $mapOutputDir
            shard_count = $existingCount
        })
    }
}

$HarvestManifestPath = Join-Path $ManifestDir "harvest_manifest.json"
Write-Json $HarvestManifestPath $runManifest
Write-Ok "Wrote harvest manifest: $HarvestManifestPath"

if (-not $SkipValidation) {
    Write-Step "Selecting validation sample from harvested NPZ shards"
    $ValidationJson = Join-Path $ManifestDir "validation_selection.json"
    Push-Location $DataHarvesterRoot
    try {
        uv run python scripts/select_validation_tiles.py $ShardRoot --output-json $ValidationJson --copy-dir $ValidationSampleDir --per-bucket $ValidationPerBucket
    }
    finally {
        Pop-Location
    }

    if (-not $SkipVisualization) {
        Write-Step "Rendering validation NPZ visualizations"
        $sampleDirs = Get-ChildItem $ValidationSampleDir -Directory -Recurse | Where-Object { @(Get-ChildItem $_.FullName -Filter "*.npz" -File -ErrorAction SilentlyContinue).Count -gt 0 }
        foreach ($dir in $sampleDirs) {
            $relative = $dir.FullName.Substring($ValidationSampleDir.Length).TrimStart('\')
            $outDir = Join-Path $VisualizationDir $relative
            Push-Location $DataHarvesterRoot
            try {
                uv run python scripts/visualize_npz.py $dir.FullName --output-dir $outDir
            }
            finally {
                Pop-Location
            }
        }
    }
}

Write-Ok "Harvest-first shard batch complete: $OutputDir"

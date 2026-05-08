#!/usr/bin/env pwsh

param(
    [string]$OutputDir = "",
    [int]$ValidationPerBucket = 5,
    [switch]$Resume,
    [switch]$SkipFullCache,
    [switch]$SkipVisualization
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step([string]$Message) { Write-Host "`n==> $Message" -ForegroundColor Cyan }
function Write-Ok([string]$Message) { Write-Host "[OK]  $Message" -ForegroundColor Green }

function Invoke-Converter([string[]]$CommandArgs) {
    & dotnet run --project $script:ConverterProject -c Debug -- @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Converter command failed: dotnet run --project $script:ConverterProject -c Debug -- $($CommandArgs -join ' ')"
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

function Get-SanitizedDatasetKeySegment([string]$Value) {
    $chars = foreach ($c in $Value.ToCharArray()) {
        if ([char]::IsLetterOrDigit($c)) { [char]::ToLowerInvariant($c) } else { '_' }
    }
    $result = (-join $chars).Trim('_')
    if ([string]::IsNullOrWhiteSpace($result)) { return "dataset" }
    return $result
}

function Get-DatasetKey($Entry) {
    return "$(Get-SanitizedDatasetKeySegment $Entry.BuildLabel)__$(Get-SanitizedDatasetKeySegment $Entry.MapName)"
}

function Get-ComplexityScore($Entry) {
    $metrics = $Entry.Metrics
    $signals = $Entry.Signals

    $heightScore = [Math]::Min([double]$metrics.HeightRange / 64.0, 4.0)
    $minimapScore = 0.0
    if ($signals.HasMinimap) {
        $minimapScore = [Math]::Min([double]$metrics.MinimapGradient / 0.02, 3.0) + [Math]::Min([double]$metrics.MinimapVariance / 0.01, 3.0)
    }
    $wdlPenalty = 0.0
    if ($signals.HasWdl) {
        $wdlPenalty = [Math]::Min([double]$metrics.MeanWdlDelta / 128.0, 2.0) + [Math]::Min([double]$metrics.MaxAbsWdlDelta / 512.0, 2.0)
    }
    $holePenalty = [Math]::Min([double]$metrics.HoleCoverage * 2.0, 1.0)
    return $heightScore + $minimapScore - $wdlPenalty - $holePenalty
}

function Write-Json($Path, $Object) {
    $dir = Split-Path -Parent $Path
    if ($dir) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    $Object | ConvertTo-Json -Depth 16 | Set-Content -Path $Path -Encoding UTF8
}

$WowViewerRoot = Split-Path -Parent $PSScriptRoot
$ParpToolsRoot = Split-Path -Parent $WowViewerRoot
$script:ConverterProject = Join-Path $WowViewerRoot "tools\converter\WowViewer.Tool.Converter\WowViewer.Tool.Converter.csproj"
$DataHarvesterRoot = Join-Path $WowViewerRoot "data-harvester"

if (-not $OutputDir) {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputDir = Join-Path $ParpToolsRoot "output\tmp\full_shard_batch_$timestamp"
}

$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$MapListDir = Join-Path $OutputDir "map_lists"
$ScanDir = Join-Path $OutputDir "scans"
$ManifestDir = Join-Path $OutputDir "manifests"
$FullCacheDir = Join-Path $OutputDir "full_cache"
$ValidationCacheDir = Join-Path $OutputDir "validation_cache"
$VisualizationDir = Join-Path $OutputDir "visualizations"

$builds = Get-ClientBuilds
foreach ($build in $builds) {
    if (-not (Test-Path $build.Root)) {
        throw "Missing client root for $($build.Build): $($build.Root)"
    }
}

Write-Step "Discovering maps from Map.dbc"
$scanPaths = [System.Collections.Generic.List[string]]::new()
foreach ($build in $builds) {
    $mapListPath = Join-Path $MapListDir "$($build.Build).json"
    if (-not ($Resume -and (Test-Path $mapListPath))) {
        Invoke-Converter @("dataset-list-maps", "--client-root", $build.Root, "--output", $mapListPath)
    }

    $maps = Get-Content $mapListPath | ConvertFrom-Json
    Write-Ok "$($build.Build): $($maps.Count) maps"

    foreach ($map in $maps) {
        $scanPath = Join-Path $ScanDir ("scan_{0}_{1}.json" -f $build.Build, $map.Directory)
        if ($Resume -and (Test-Path $scanPath)) {
            $scanPaths.Add($scanPath)
            continue
        }

        Write-Host "  Scanning $($build.Build)/$($map.Directory)"
        try {
            Invoke-Converter @(
                "dataset-scan",
                "--client-root", $build.Root,
                "--map", $map.Directory,
                "--build", $build.Build,
                "--output", $scanPath
            )
            $scanPaths.Add($scanPath)
        }
        catch {
            Write-Warning "Skipping $($build.Build)/$($map.Directory): $($_.Exception.Message)"
        }
    }
}

if ($scanPaths.Count -eq 0) {
    throw "No scan manifests were produced."
}

$MergedPath = Join-Path $ManifestDir "merged_scan.json"
$AuditPath = Join-Path $ManifestDir "audit.json"
$CuratedAllPath = Join-Path $ManifestDir "curated_all.json"
$CurateReportPath = Join-Path $ManifestDir "curation_report.json"
$ValidationManifestPath = Join-Path $ManifestDir "validation_sample.json"
$ValidationSummaryPath = Join-Path $ManifestDir "validation_summary.json"

Write-Step "Merging scan manifests"
$mergeArgs = @("dataset-merge", "--output", $MergedPath)
foreach ($scanPath in $scanPaths) {
    $mergeArgs += @("--input", $scanPath)
}
Invoke-Converter $mergeArgs

Write-Step "Auditing merged dataset"
Invoke-Converter @("dataset-audit", "--input", $MergedPath, "--output", $AuditPath)

Write-Step "Curating accepted pool"
Invoke-Converter @(
    "dataset-curate",
    "--input", $AuditPath,
    "--output", $CuratedAllPath,
    "--report", $CurateReportPath,
    "--max-per-group", "1000000"
)

if (-not $SkipFullCache) {
    Write-Step "Building full shard cache"
    Invoke-Converter @("dataset-build-cache", "--input", $AuditPath, "--output-dir", $FullCacheDir, "--overwrite")
}

Write-Step "Selecting validation sample by complexity bucket"
$curatedManifest = Get-Content $CuratedAllPath | ConvertFrom-Json
$entriesByBuild = $curatedManifest.Entries | Group-Object BuildLabel
$selected = [System.Collections.Generic.List[object]]::new()
$validationSummary = [System.Collections.Generic.List[object]]::new()

foreach ($group in $entriesByBuild) {
    $entries = @($group.Group)
    if ($entries.Count -eq 0) { continue }

    $ranked = $entries | ForEach-Object {
        [PSCustomObject]@{
            Entry = $_
            Score = Get-ComplexityScore $_
        }
    } | Sort-Object Score

    $count = $ranked.Count
    $lowCut = [Math]::Floor($count / 3)
    $midCut = [Math]::Floor(($count * 2) / 3)

    $buckets = @{
        low = @($ranked | Select-Object -First $lowCut)
        medium = @($ranked | Select-Object -Skip $lowCut -First ($midCut - $lowCut))
        high = @($ranked | Select-Object -Skip $midCut)
    }

    foreach ($bucketName in $buckets.Keys) {
        $bucketEntries = @($buckets[$bucketName])
        if ($bucketEntries.Count -eq 0) { continue }
        $sampled = $bucketEntries | Get-Random -Count ([Math]::Min($ValidationPerBucket, $bucketEntries.Count))
        foreach ($item in $sampled) {
            $selected.Add($item.Entry)
        }

        $validationSummary.Add([PSCustomObject]@{
            build = $group.Name
            bucket = $bucketName
            available = $bucketEntries.Count
            selected = @($sampled).Count
            min_score = ($bucketEntries | Measure-Object -Property Score -Minimum).Minimum
            max_score = ($bucketEntries | Measure-Object -Property Score -Maximum).Maximum
        })
    }
}

$validationManifest = [PSCustomObject]@{
    SchemaVersion = $curatedManifest.SchemaVersion
    CreatedAtUtc = [DateTimeOffset]::UtcNow.ToString("o")
    SourceManifestKind = "curate"
    Entries = @($selected)
}
Write-Json $ValidationManifestPath $validationManifest
Write-Json $ValidationSummaryPath $validationSummary
Write-Ok "Validation sample: $($selected.Count) entries"

Write-Step "Building validation shard cache"
Invoke-Converter @("dataset-build-cache", "--input", $ValidationManifestPath, "--output-dir", $ValidationCacheDir, "--overwrite")

if (-not $SkipVisualization) {
    Write-Step "Rendering validation NPZ visualizations"
    $shardRoot = Join-Path $ValidationCacheDir "shards"
    $datasetDirs = Get-ChildItem $shardRoot -Directory | Sort-Object Name
    foreach ($dir in $datasetDirs) {
        $outDir = Join-Path $VisualizationDir $dir.Name
        Push-Location $DataHarvesterRoot
        try {
            uv run python scripts/visualize_npz.py $dir.FullName --output-dir $outDir --quilt
        }
        finally {
            Pop-Location
        }
    }
}

Write-Ok "Full shard batch complete: $OutputDir"

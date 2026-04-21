#!/usr/bin/env pwsh

param(
    [ValidateSet("audit", "train")]
    [string]$Mode = "audit",

    [string[]]$IncludeBuilds = @(
        "0_5_3_3368",
        "0_5_5_3494",
        "0_7_0_3694",
        "3_0_1_8303",
        "3_3_5_12340",
        "4_0_0_11927"
    ),

    [int]$PerMapScanLimit = 0,
    [int]$AuditLimit = 0,
    [int]$CurateLimit = 0,
    [int]$CacheLimit = 0,
    [int]$TrainerLimit = 0,

    [string]$OutputDir = "",
    [string]$PythonExe = "",
    [string]$ConverterProject = "",
    [string]$TrainerScript = "",
    [string[]]$TrainerArgs = @(),

    [switch]$AllowMissingRoots,
    [switch]$WowArchiveOnly,
    [switch]$NoRequireMinimap,
    [switch]$NoRequireWdl,
    [switch]$DryRun,
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step([string]$Message) { Write-Host "`n==> $Message" -ForegroundColor Cyan }
function Write-Ok([string]$Message) { Write-Host "[OK]  $Message" -ForegroundColor Green }
function Write-WarnLine([string]$Message) { Write-Warning $Message }
function Write-Err([string]$Message) { Write-Host "[ERR] $Message" -ForegroundColor Red }

function ConvertTo-SerializableObject($Value) {
    if ($null -eq $Value) {
        return $null
    }

    if ($Value -is [System.Array]) {
        return @($Value | ForEach-Object { ConvertTo-SerializableObject $_ })
    }

    if ($Value -is [System.Collections.IDictionary]) {
        $result = [ordered]@{}
        foreach ($key in $Value.Keys) {
            $result[[string]$key] = ConvertTo-SerializableObject $Value[$key]
        }

        return $result
    }

    if ($Value -is [System.Management.Automation.PSCustomObject] -or $Value -is [psobject]) {
        $result = [ordered]@{}
        foreach ($property in $Value.PSObject.Properties) {
            $result[$property.Name] = ConvertTo-SerializableObject $property.Value
        }

        return $result
    }

    return $Value
}

function Get-OptionalTrainerArgValue([string[]]$Args, [string]$FlagName) {
    if (!$Args) {
        return $null
    }

    for ($index = 0; $index -lt $Args.Count; $index++) {
        if ($Args[$index] -ne $FlagName) {
            continue
        }

        if ($index + 1 -ge $Args.Count) {
            return $true
        }

        $candidate = $Args[$index + 1]
        if ($candidate.StartsWith("--")) {
            return $true
        }

        return $candidate
    }

    return $null
}

function Write-PipelineMetadata {
    param(
        [string]$Path,
        [string]$Mode,
        [string]$OutputDir,
        [string]$PythonExe,
        [string]$ConverterProject,
        [string]$TrainerScript,
        [string[]]$TrainerArgs,
        [string]$CacheManifest,
        [string]$TrainOutputDir,
        [object[]]$ResolvedBuilds,
        [string[]]$ScanPaths,
        [string]$MergedScanPath,
        [string]$AuditPath,
        [string]$CuratePath,
        [string]$CurateReportPath,
        [string]$CacheDir,
        [bool]$NoRequireMinimap,
        [bool]$NoRequireWdl,
        [bool]$WowArchiveOnly
    )

    $metadata = [ordered]@{
        schema_version = "wow-viewer-direct-v9-pipeline.v1"
        created_at_utc = [DateTime]::UtcNow.ToString("o")
        mode = $Mode
        output_dir = $OutputDir
        python_exe = $PythonExe
        converter_project = $ConverterProject
        trainer_script = $TrainerScript
        trainer_args = @($TrainerArgs)
        cache_manifest = $CacheManifest
        train_output_dir = $TrainOutputDir
        dev_eval_cache_manifest = Get-OptionalTrainerArgValue -Args $TrainerArgs -FlagName "--dev-eval-cache-manifest"
        selection_metric = Get-OptionalTrainerArgValue -Args $TrainerArgs -FlagName "--selection-metric"
        no_require_minimap = $NoRequireMinimap
        no_require_wdl = $NoRequireWdl
        wowarchive_only = $WowArchiveOnly
        scan_manifests = @($ScanPaths)
        merged_scan_manifest = $MergedScanPath
        audit_manifest = $AuditPath
        curated_manifest = $CuratePath
        curation_report = $CurateReportPath
        cache_dir = $CacheDir
        resolved_builds = @($ResolvedBuilds | ForEach-Object {
            [ordered]@{
                build_label = $_.BuildLabel
                client_root = $_.ClientRoot
                maps = @($_.Maps)
            }
        })
    }

    $json = ConvertTo-SerializableObject $metadata | ConvertTo-Json -Depth 8
    Set-Content -Path $Path -Value $json -Encoding UTF8
}

function Invoke-Step {
    param(
        [string]$Exe,
        [string[]]$CommandArgs,
        [string]$Cwd = ""
    )

    $display = "$Exe $($CommandArgs -join ' ')"
    Write-Host "  $ $display" -ForegroundColor DarkGray
    if ($DryRun) {
        return
    }

    if ($Cwd) {
        Push-Location $Cwd
    }

    try {
        & $Exe @CommandArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed (exit $LASTEXITCODE): $display"
        }
    }
    finally {
        if ($Cwd) {
            Pop-Location
        }
    }
}

function Find-Python {
    foreach ($candidate in @(
        ".venv\Scripts\python.exe",
        ".venv\Scripts\python",
        ".venv-train\Scripts\python.exe",
        ".venv-train\Scripts\python",
        "python",
        "python3"
    )) {
        $resolved = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($resolved) {
            return $resolved.Source
        }
    }

    return $null
}

function Get-BuildMapDefaults([string]$BuildLabel) {
    switch ($BuildLabel) {
        "0_5_3_3368" { return @("Azeroth", "Kalimdor") }
        "0_5_5_3494" { return @("Azeroth", "Kalimdor", "EmeraldDream") }
        "0_7_0_3694" { return @("Azeroth", "Kalimdor", "EmeraldDream") }
        "3_0_1_8303" { return @("Northrend") }
        "3_3_5_12340" { return @("Azeroth", "Kalimdor", "EmeraldDream", "Northrend", "PVPZone01", "PVPZone02", "PVPZone03", "PVPZone04") }
        "4_0_0_11927" { return @("Azeroth", "Kalimdor", "EmeraldDream", "Deepholm", "LostIsles", "LostIslesPhase1", "LostIslesPhase2") }
        default { return @("Azeroth") }
    }
}

function Resolve-ExistingPath([string[]]$Candidates) {
    foreach ($candidate in $Candidates) {
        if ([string]::IsNullOrWhiteSpace($candidate)) {
            continue
        }

        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }

    return $null
}

function Resolve-ClientRoot([string]$BuildLabel, [string]$ParpToolsRoot, [bool]$WowArchiveOnly) {
    switch ($BuildLabel) {
        "0_5_3_3368" {
            return Resolve-ExistingPath @(
                (Join-Path $ParpToolsRoot "output\tmp\wowarchive-clients\0_5_3_3368\World of Warcraft"),
                (Join-Path $ParpToolsRoot "output\tmp\wowarchive-clients\0_5_3_3368")
            )
        }
        "0_5_5_3494" {
            return Resolve-ExistingPath @(
                (Join-Path $ParpToolsRoot "output\tmp\wowarchive-clients\0_5_5_3494\World of Warcraft"),
                (Join-Path $ParpToolsRoot "output\tmp\wowarchive-clients\0_5_5_3494")
            )
        }
        "0_7_0_3694" {
            $candidates = @(
                (Join-Path $ParpToolsRoot "output\tmp\wowarchive-clients\0_7_0_3694\World of Warcraft"),
                (Join-Path $ParpToolsRoot "output\tmp\wowarchive-clients\0_7_0_3694")
            )
            if (-not $WowArchiveOnly) {
                $candidates += "H:\CLIENTS\0.X_Pre-Release_Windows_enUS_0.7.0.3694\World of Warcraft"
            }

            return Resolve-ExistingPath $candidates
        }
        "3_0_1_8303" {
            $candidates = @(
                (Join-Path $ParpToolsRoot "output\tmp\wowarchive-clients\3_0_1_8303\World of Warcraft"),
                (Join-Path $ParpToolsRoot "output\tmp\wowarchive-clients\3_0_1_8303")
            )
            if (-not $WowArchiveOnly) {
                $candidates += "H:\CLIENTS\3.X_Pre-Release_Windows_enUS_3.0.1.8303\World of Warcraft"
            }

            return Resolve-ExistingPath $candidates
        }
        "3_3_5_12340" {
            $candidates = @(
                (Join-Path $ParpToolsRoot "output\tmp\wowarchive-clients\3_3_5_12340\World of Warcraft"),
                (Join-Path $ParpToolsRoot "output\tmp\wowarchive-clients\3_3_5_12340")
            )
            if (-not $WowArchiveOnly) {
                $candidates += "H:\CLIENTS\WoW335\3.X_Retail_Windows_enUS_3.3.5.12340\World of Warcraft"
            }

            return Resolve-ExistingPath $candidates
        }
        "4_0_0_11927" {
            $candidates = @(
                (Join-Path $ParpToolsRoot "output\tmp\wowarchive-clients\4_0_0_11927\World of Warcraft"),
                (Join-Path $ParpToolsRoot "output\tmp\wowarchive-clients\4_0_0_11927")
            )
            if (-not $WowArchiveOnly) {
                $candidates += "H:\CLIENTS\World of Warcraft Cata beta 11927"
            }

            return Resolve-ExistingPath $candidates
        }
        default {
            return $null
        }
    }
}

$ScriptSelf = $PSScriptRoot
$WowViewerRoot = (Get-Item $ScriptSelf).Parent.FullName
$ParpToolsRoot = (Get-Item $WowViewerRoot).Parent.FullName

if (!$ConverterProject) {
    $ConverterProject = Join-Path $WowViewerRoot "tools\converter\WowViewer.Tool.Converter\WowViewer.Tool.Converter.csproj"
}

if (!$TrainerScript) {
    $TrainerScript = Join-Path $ParpToolsRoot "gillijimproject_refactor\src\WoWMapConverter\scripts\train_v9.py"
}

if (!$PythonExe) {
    Push-Location $ParpToolsRoot
    try {
        $PythonExe = Find-Python
    }
    finally {
        Pop-Location
    }
}

if (!$PythonExe) {
    Write-Err "Python not found. Activate .venv or pass -PythonExe."
    exit 1
}

if (!$OutputDir) {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputDir = Join-Path $ParpToolsRoot "output\tmp\v9_direct_pipeline_$timestamp"
}

$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

if ($Mode -eq "train" -and !(Get-Command "nvidia-smi" -ErrorAction SilentlyContinue)) {
    Write-WarnLine "nvidia-smi not found. Training may fall back to CPU depending on the Python environment."
}

Write-Step "Resolving client roots"
$resolvedBuilds = @()
foreach ($buildLabel in $IncludeBuilds) {
    $clientRoot = Resolve-ClientRoot -BuildLabel $buildLabel -ParpToolsRoot $ParpToolsRoot -WowArchiveOnly:$WowArchiveOnly
    if (!$clientRoot) {
        $message = if ($WowArchiveOnly) {
            "Client root for $buildLabel was not found under output/tmp/wowarchive-clients while -WowArchiveOnly is enabled."
        }
        else {
            "Client root for $buildLabel was not found. Expected fixed local root or staged copy under output/tmp/wowarchive-clients."
        }
        if ($AllowMissingRoots) {
            Write-WarnLine $message
            continue
        }

        Write-Err $message
        exit 1
    }

    $resolvedBuilds += [PSCustomObject]@{
        BuildLabel = $buildLabel
        ClientRoot = $clientRoot
        Maps = @(Get-BuildMapDefaults -BuildLabel $buildLabel)
    }
    Write-Ok "$buildLabel => $clientRoot"
}

if ($resolvedBuilds.Count -eq 0) {
    Write-Err "No client roots were resolved."
    exit 1
}

$scanPaths = @()
Write-Step "Scanning direct dataset manifests"
foreach ($build in $resolvedBuilds) {
    foreach ($mapName in $build.Maps) {
        $scanPath = Join-Path $OutputDir ("scan_{0}_{1}.json" -f $build.BuildLabel, $mapName)
        $scanArgs = @(
            "run",
            "--project", $ConverterProject,
            "--",
            "dataset-scan",
            "--client-root", $build.ClientRoot,
            "--map", $mapName,
            "--build", $build.BuildLabel,
            "--output", $scanPath
        )
        if ($PerMapScanLimit -gt 0) {
            $scanArgs += @("--limit", $PerMapScanLimit)
        }

        try {
            Invoke-Step -Exe "dotnet" -CommandArgs $scanArgs -Cwd $ParpToolsRoot
            $scanPaths += $scanPath
        }
        catch {
            Write-WarnLine "Skipping $($build.BuildLabel)/$mapName because dataset-scan failed: $($_.Exception.Message)"
        }
    }
}

if ($scanPaths.Count -eq 0) {
    Write-Err "No scan manifests were produced."
    exit 1
}

$mergedScanPath = Join-Path $OutputDir "merged_scan.json"
$mergeArgs = @(
    "run",
    "--project", $ConverterProject,
    "--",
    "dataset-merge",
    "--output", $mergedScanPath
)
foreach ($scanPath in $scanPaths) {
    $mergeArgs += @("--input", $scanPath)
}

Write-Step "Merging direct scan manifests"
Invoke-Step -Exe "dotnet" -CommandArgs $mergeArgs -Cwd $ParpToolsRoot

$auditPath = Join-Path $OutputDir "audit.json"
$auditArgs = @(
    "run",
    "--project", $ConverterProject,
    "--",
    "dataset-audit",
    "--input", $mergedScanPath,
    "--output", $auditPath
)
if ($AuditLimit -gt 0) {
    $auditArgs += @("--limit", $AuditLimit)
}

Write-Step "Auditing merged direct dataset"
Invoke-Step -Exe "dotnet" -CommandArgs $auditArgs -Cwd $ParpToolsRoot

$curatePath = Join-Path $OutputDir "curated.json"
$curateReportPath = Join-Path $OutputDir "curation_report.json"
$curateArgs = @(
    "run",
    "--project", $ConverterProject,
    "--",
    "dataset-curate",
    "--input", $auditPath,
    "--output", $curatePath,
    "--report", $curateReportPath
)
if ($CurateLimit -gt 0) {
    $curateArgs += @("--limit", $CurateLimit)
}
if ($NoRequireMinimap) {
    $curateArgs += "--no-require-minimap"
}
if ($NoRequireWdl) {
    $curateArgs += "--no-require-wdl"
}

Write-Step "Curating merged direct dataset"
Invoke-Step -Exe "dotnet" -CommandArgs $curateArgs -Cwd $ParpToolsRoot

$cacheDir = Join-Path $OutputDir "cache"
$cacheArgs = @(
    "run",
    "--project", $ConverterProject,
    "--",
    "dataset-build-cache",
    "--input", $curatePath,
    "--output-dir", $cacheDir,
    "--overwrite"
)
if ($CacheLimit -gt 0) {
    $cacheArgs += @("--limit", $CacheLimit)
}

Write-Step "Building direct v9 cache"
Invoke-Step -Exe "dotnet" -CommandArgs $cacheArgs -Cwd $ParpToolsRoot

$cacheManifest = Join-Path $cacheDir "v9_tensor_cache_manifest.json"
if (!(Test-Path $cacheManifest) -and !$DryRun) {
    Write-Err "Expected cache manifest was not written: $cacheManifest"
    exit 1
}

$trainOutputLeaf = "train_audit"
if ($Mode -eq "train") {
    $trainOutputLeaf = "train"
}

$trainOutputDir = Join-Path $OutputDir $trainOutputLeaf
$trainArgs = @(
    $TrainerScript,
    $cacheManifest,
    "--output-dir", $trainOutputDir
)
if ($Mode -eq "audit") {
    $trainArgs += "--audit-only"
}
if ($TrainerLimit -gt 0) {
    $trainArgs += @("--limit", $TrainerLimit)
}
if ($NoRequireMinimap) {
    $trainArgs += "--no-require-minimap"
}
if ($NoRequireWdl) {
    $trainArgs += "--no-require-wdl"
}
if ($TrainerArgs.Count -gt 0) {
    $trainArgs += $TrainerArgs
}

$pipelineMetadataPath = Join-Path $OutputDir "pipeline_run.json"
Write-PipelineMetadata \
    -Path $pipelineMetadataPath \
    -Mode $Mode \
    -OutputDir $OutputDir \
    -PythonExe $PythonExe \
    -ConverterProject $ConverterProject \
    -TrainerScript $TrainerScript \
    -TrainerArgs $TrainerArgs \
    -CacheManifest $cacheManifest \
    -TrainOutputDir $trainOutputDir \
    -ResolvedBuilds $resolvedBuilds \
    -ScanPaths $scanPaths \
    -MergedScanPath $mergedScanPath \
    -AuditPath $auditPath \
    -CuratePath $curatePath \
    -CurateReportPath $curateReportPath \
    -CacheDir $cacheDir \
    -NoRequireMinimap $NoRequireMinimap.IsPresent \
    -NoRequireWdl $NoRequireWdl.IsPresent \
    -WowArchiveOnly $WowArchiveOnly.IsPresent
Write-Ok "Wrote pipeline metadata: $pipelineMetadataPath"

$trainStepLabel = "Auditing trainer contract"
if ($Mode -eq "train") {
    $trainStepLabel = "Training v9 model"
}

Write-Step $trainStepLabel
Invoke-Step -Exe $PythonExe -CommandArgs $trainArgs -Cwd $ParpToolsRoot

Write-Ok "Direct v9 pipeline complete. Output: $OutputDir"
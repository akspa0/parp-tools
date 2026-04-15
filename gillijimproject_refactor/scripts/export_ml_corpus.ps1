param(
    [string]$ConfigPath = (Join-Path $PSScriptRoot 'ml_corpus_fixed_clients.json'),
    [string]$ProjectPath = (Join-Path $PSScriptRoot '..\src\WoWMapConverter\WoWMapConverter.Cli\WoWMapConverter.Cli.csproj'),
    [string]$OutputRoot,
    [string]$ArchiveRoot,
    [string]$ArchiveMountRoot,
    [string]$MountScript,
    [string]$StagingRoot,
    [string]$ListfilePath,
    [string]$Configuration = 'Debug',
    [switch]$SkipHarvest,
    [switch]$PruneStagedClients,
    [switch]$ForceRestage,
    [switch]$Resume,
    [switch]$Force,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$wowArchiveHelperPath = Join-Path $PSScriptRoot 'wowarchive_client_staging.ps1'
if (-not (Test-Path $wowArchiveHelperPath)) {
    throw "WoWArchive helper script not found: $wowArchiveHelperPath"
}

. $wowArchiveHelperPath

$configDirectory = $null
$mlCorpusResumeStateFileName = '.ml-corpus-resume-state.json'

function Resolve-ConfigPathValue {
    param(
        [AllowNull()]
        [string]$Value,
        [AllowNull()]
        [string]$BaseRoot
    )

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return $null
    }

    if ([System.IO.Path]::IsPathRooted($Value)) {
        return [System.IO.Path]::GetFullPath($Value)
    }

    if (-not [string]::IsNullOrWhiteSpace($BaseRoot)) {
        return [System.IO.Path]::GetFullPath((Join-Path $BaseRoot $Value))
    }

    return [System.IO.Path]::GetFullPath((Join-Path $configDirectory $Value))
}

function Get-JsonPropertyValue {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Object,
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) {
        return $null
    }

    return $property.Value
}

function Get-DatasetJsonFiles {
    param(
        [Parameter(Mandatory = $true)]
        [string]$DatasetOutput
    )

    $datasetJsonDir = Join-Path $DatasetOutput 'dataset'
    if (-not (Test-Path $datasetJsonDir)) {
        return @()
    }

    return @(
        Get-ChildItem -Path $datasetJsonDir -Filter '*.json' -File -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -ne 'texture_database.json' }
    )
}

function Test-DatasetManifestCurrent {
    param(
        [Parameter(Mandatory = $true)]
        [string]$DatasetOutput,
        [Parameter(Mandatory = $true)]
        [object[]]$DatasetFiles
    )

    if ($DatasetFiles.Count -eq 0) {
        return $false
    }

    $manifestPath = Join-Path $DatasetOutput 'ml_dataset_manifest.json'
    if (-not (Test-Path $manifestPath)) {
        return $false
    }

    try {
        $manifest = Get-Content -Raw -Path $manifestPath | ConvertFrom-Json
    }
    catch {
        return $false
    }

    $coverage = Get-JsonPropertyValue -Object $manifest -Name 'coverage'
    if ($null -eq $coverage) {
        return $false
    }

    $tilesProcessed = Get-JsonPropertyValue -Object $coverage -Name 'tiles_processed'
    if ($null -eq $tilesProcessed) {
        return $false
    }

    if ([int]$tilesProcessed -ne $DatasetFiles.Count) {
        return $false
    }

    $manifestWriteTimeUtc = (Get-Item -LiteralPath $manifestPath).LastWriteTimeUtc
    $latestDatasetWriteTimeUtc = ($DatasetFiles | Measure-Object -Property LastWriteTimeUtc -Maximum).Maximum
    return $manifestWriteTimeUtc -ge $latestDatasetWriteTimeUtc
}

function Read-ResumeState {
    param(
        [Parameter(Mandatory = $true)]
        [string]$DatasetOutput
    )

    $statePath = Join-Path $DatasetOutput $mlCorpusResumeStateFileName
    if (-not (Test-Path $statePath)) {
        return $null
    }

    try {
        return Get-Content -Raw -Path $statePath | ConvertFrom-Json
    }
    catch {
        return $null
    }
}

function Test-ResumeStateMatches {
    param(
        [AllowNull()]
        [object]$State,
        [Parameter(Mandatory = $true)]
        [string]$ClientLabel,
        [Parameter(Mandatory = $true)]
        [string]$ClientVersion,
        [Parameter(Mandatory = $true)]
        [string]$Map,
        [Parameter(Mandatory = $true)]
        [bool]$GenerateDepth,
        [AllowNull()]
        [int]$TileLimit,
        [Parameter(Mandatory = $true)]
        [bool]$InterestingOnly,
        [Parameter(Mandatory = $true)]
        [int]$InterestingMinScore,
        [Parameter(Mandatory = $true)]
        [bool]$SkipDerivedAssets,
        [AllowNull()]
        [string]$MinimapRoot
    )

    if ($null -eq $State) {
        return $false
    }

    $stateSchemaVersion = [string](Get-JsonPropertyValue -Object $State -Name 'schema_version')
    if ($stateSchemaVersion -ne 'ml-corpus-resume-state.v1') {
        return $false
    }

    $stateTileLimitValue = Get-JsonPropertyValue -Object $State -Name 'tile_limit'
    $stateTileLimit = if ($null -eq $stateTileLimitValue) { $null } else { [int]$stateTileLimitValue }

    $stateMinimapRootValue = Get-JsonPropertyValue -Object $State -Name 'minimap_root'
    $stateMinimapRoot = if ([string]::IsNullOrWhiteSpace([string]$stateMinimapRootValue)) { $null } else { [string]$stateMinimapRootValue }

    return [string]::Equals([string](Get-JsonPropertyValue -Object $State -Name 'client_label'), $ClientLabel, [System.StringComparison]::OrdinalIgnoreCase) -and
        [string]::Equals([string](Get-JsonPropertyValue -Object $State -Name 'client_version'), $ClientVersion, [System.StringComparison]::OrdinalIgnoreCase) -and
        [string]::Equals([string](Get-JsonPropertyValue -Object $State -Name 'map_name'), $Map, [System.StringComparison]::OrdinalIgnoreCase) -and
        ([bool](Get-JsonPropertyValue -Object $State -Name 'generate_depth') -eq $GenerateDepth) -and
        (($null -eq $stateTileLimit -and $null -eq $TileLimit) -or ($null -ne $stateTileLimit -and $null -ne $TileLimit -and $stateTileLimit -eq $TileLimit)) -and
        ([bool](Get-JsonPropertyValue -Object $State -Name 'interesting_only') -eq $InterestingOnly) -and
        ([int](Get-JsonPropertyValue -Object $State -Name 'interesting_min_score') -eq $InterestingMinScore) -and
        ([bool](Get-JsonPropertyValue -Object $State -Name 'skip_derived_assets') -eq $SkipDerivedAssets) -and
        [string]::Equals($stateMinimapRoot, $MinimapRoot, [System.StringComparison]::OrdinalIgnoreCase)
}

function Write-ResumeState {
    param(
        [Parameter(Mandatory = $true)]
        [string]$DatasetOutput,
        [Parameter(Mandatory = $true)]
        [string]$ClientLabel,
        [Parameter(Mandatory = $true)]
        [string]$ClientVersion,
        [Parameter(Mandatory = $true)]
        [string]$Map,
        [Parameter(Mandatory = $true)]
        [bool]$HarvestRequested,
        [Parameter(Mandatory = $true)]
        [bool]$ExportCompleted,
        [Parameter(Mandatory = $true)]
        [bool]$HarvestCompleted,
        [Parameter(Mandatory = $true)]
        [bool]$GenerateDepth,
        [AllowNull()]
        [int]$TileLimit,
        [Parameter(Mandatory = $true)]
        [bool]$InterestingOnly,
        [Parameter(Mandatory = $true)]
        [int]$InterestingMinScore,
        [Parameter(Mandatory = $true)]
        [bool]$SkipDerivedAssets,
        [AllowNull()]
        [string]$MinimapRoot,
        [Parameter(Mandatory = $true)]
        [int]$TileJsonCount
    )

    if ($DryRun) {
        return
    }

    $statePath = Join-Path $DatasetOutput $mlCorpusResumeStateFileName
    $state = [ordered]@{
        schema_version = 'ml-corpus-resume-state.v1'
        client_label = $ClientLabel
        client_version = $ClientVersion
        map_name = $Map
        harvest_requested = $HarvestRequested
        export_completed = $ExportCompleted
        harvest_completed = $HarvestCompleted
        generate_depth = $GenerateDepth
        tile_limit = $TileLimit
        interesting_only = $InterestingOnly
        interesting_min_score = $InterestingMinScore
        skip_derived_assets = $SkipDerivedAssets
        minimap_root = $MinimapRoot
        tile_json_count = $TileJsonCount
        updated_at_utc = [DateTime]::UtcNow.ToString('o')
    }

    $state | ConvertTo-Json -Depth 4 | Set-Content -Path $statePath -Encoding UTF8
}

function Get-ResumeDecision {
    param(
        [Parameter(Mandatory = $true)]
        [string]$DatasetOutput,
        [Parameter(Mandatory = $true)]
        [string]$ClientLabel,
        [Parameter(Mandatory = $true)]
        [string]$ClientVersion,
        [Parameter(Mandatory = $true)]
        [string]$Map,
        [Parameter(Mandatory = $true)]
        [bool]$HarvestRequested,
        [Parameter(Mandatory = $true)]
        [bool]$GenerateDepth,
        [AllowNull()]
        [int]$TileLimit,
        [Parameter(Mandatory = $true)]
        [bool]$InterestingOnly,
        [Parameter(Mandatory = $true)]
        [int]$InterestingMinScore,
        [Parameter(Mandatory = $true)]
        [bool]$SkipDerivedAssets,
        [AllowNull()]
        [string]$MinimapRoot
    )

    $datasetFiles = @(Get-DatasetJsonFiles -DatasetOutput $DatasetOutput)
    $datasetTileCount = $datasetFiles.Count
    $manifestCurrent = Test-DatasetManifestCurrent -DatasetOutput $DatasetOutput -DatasetFiles $datasetFiles
    $state = Read-ResumeState -DatasetOutput $DatasetOutput
    $stateMatches = Test-ResumeStateMatches -State $state -ClientLabel $ClientLabel -ClientVersion $ClientVersion -Map $Map -GenerateDepth $GenerateDepth -TileLimit $TileLimit -InterestingOnly $InterestingOnly -InterestingMinScore $InterestingMinScore -SkipDerivedAssets $SkipDerivedAssets -MinimapRoot $MinimapRoot
    $stateExportCompleted = $stateMatches -and [bool](Get-JsonPropertyValue -Object $state -Name 'export_completed')
    $stateHarvestCompleted = $stateMatches -and [bool](Get-JsonPropertyValue -Object $state -Name 'harvest_completed')

    if ($stateExportCompleted) {
        if (-not $HarvestRequested) {
            return [pscustomobject]@{ Kind = 'skip-all'; Reason = 'resume state already marks export complete for the same job settings'; DatasetTileCount = $datasetTileCount }
        }

        if ($stateHarvestCompleted -and $manifestCurrent) {
            return [pscustomobject]@{ Kind = 'skip-all'; Reason = 'resume state and manifest already mark this map complete'; DatasetTileCount = $datasetTileCount }
        }

        return [pscustomobject]@{ Kind = 'run-harvest-only'; Reason = 'resume state marks export complete but harvest metadata is missing or stale'; DatasetTileCount = $datasetTileCount }
    }

    if ($manifestCurrent) {
        return [pscustomobject]@{ Kind = 'skip-all'; Reason = 'existing manifest is current for this dataset root'; DatasetTileCount = $datasetTileCount }
    }

    return [pscustomobject]@{ Kind = 'run-export'; Reason = 'no matching completion state was found'; DatasetTileCount = $datasetTileCount }
}

function Get-JobPlanAction {
    param(
        [Parameter(Mandatory = $true)]
        [bool]$ResumeEnabled,
        [Parameter(Mandatory = $true)]
        [string]$DatasetOutput,
        [Parameter(Mandatory = $true)]
        [string]$ClientLabel,
        [Parameter(Mandatory = $true)]
        [string]$ClientVersion,
        [Parameter(Mandatory = $true)]
        [string]$Map,
        [Parameter(Mandatory = $true)]
        [bool]$HarvestRequested,
        [Parameter(Mandatory = $true)]
        [bool]$GenerateDepth,
        [AllowNull()]
        [int]$TileLimit,
        [Parameter(Mandatory = $true)]
        [bool]$InterestingOnly,
        [Parameter(Mandatory = $true)]
        [int]$InterestingMinScore,
        [Parameter(Mandatory = $true)]
        [bool]$SkipDerivedAssets,
        [AllowNull()]
        [string]$MinimapRoot
    )

    if (-not $ResumeEnabled) {
        return [pscustomobject]@{ Kind = 'run-export'; Reason = 'force requested; resume bypassed' }
    }

    return Get-ResumeDecision -DatasetOutput $DatasetOutput -ClientLabel $ClientLabel -ClientVersion $ClientVersion -Map $Map -HarvestRequested $HarvestRequested -GenerateDepth $GenerateDepth -TileLimit $TileLimit -InterestingOnly $InterestingOnly -InterestingMinScore $InterestingMinScore -SkipDerivedAssets $SkipDerivedAssets -MinimapRoot $MinimapRoot
}

function Write-JobPlanSummary {
    param(
        [Parameter(Mandatory = $true)]
        [object[]]$JobPlans,
        [Parameter(Mandatory = $true)]
        [bool]$ResumeEnabled,
        [Parameter(Mandatory = $true)]
        [bool]$HarvestRequested,
        [Parameter(Mandatory = $true)]
        [bool]$ForceRequested
    )

    $skipCount = @($JobPlans | Where-Object { $_.Action -eq 'skip-all' }).Count
    $harvestOnlyCount = @($JobPlans | Where-Object { $_.Action -eq 'run-harvest-only' }).Count
    $fullExportCount = @($JobPlans | Where-Object { $_.Action -eq 'run-export' }).Count

    Write-Host "ML corpus preflight summary:" -ForegroundColor Yellow
    Write-Host ("  total map jobs : {0}" -f $JobPlans.Count) -ForegroundColor DarkYellow
    Write-Host ("  skip complete  : {0}" -f $skipCount) -ForegroundColor DarkYellow
    Write-Host ("  harvest-only   : {0}" -f $harvestOnlyCount) -ForegroundColor DarkYellow
    Write-Host ("  full export    : {0}" -f $fullExportCount) -ForegroundColor DarkYellow
    Write-Host ("  resume enabled : {0}" -f $ResumeEnabled) -ForegroundColor DarkYellow
    Write-Host ("  harvest after  : {0}" -f $HarvestRequested) -ForegroundColor DarkYellow
    Write-Host ("  force requested: {0}" -f $ForceRequested) -ForegroundColor DarkYellow
}

function Resolve-ConfiguredWorkingRoot {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [AllowNull()]
        [string]$DirectPath,
        [AllowNull()]
        [string]$DirectBaseRoot,
        [AllowNull()]
        [string]$LocalPath,
        [AllowNull()]
        [string]$ArchivePath,
        [AllowNull()]
        [string]$ArchiveBaseRoot,
        [AllowNull()]
        [string]$MountRoot,
        [AllowNull()]
        [string]$MountScript,
        [AllowNull()]
        [string]$StagingRoot,
        [switch]$ForceRestage,
        [switch]$DryRun
    )

    $resolvedDirectPath = if ([string]::IsNullOrWhiteSpace($DirectPath)) { $null } else { Resolve-ConfigPathValue -Value $DirectPath -BaseRoot $DirectBaseRoot }
    $resolvedLocalPath = if ([string]::IsNullOrWhiteSpace($LocalPath)) { $null } else { Resolve-ConfigPathValue -Value $LocalPath -BaseRoot $null }
    $resolvedArchivePath = if ([string]::IsNullOrWhiteSpace($ArchivePath)) { $null } else { Resolve-ConfigPathValue -Value $ArchivePath -BaseRoot $ArchiveBaseRoot }

    return Resolve-WoWClientWorkingRoot -Label $Label -DirectPath $resolvedDirectPath -PreferredLocalPath $resolvedLocalPath -ArchiveSourcePath $resolvedArchivePath -MountRoot $MountRoot -MountScript $MountScript -StagingRoot $StagingRoot -ForceRestage:$ForceRestage -DryRun:$DryRun
}

function Invoke-LoggedCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $commandText = 'dotnet ' + (($Arguments | ForEach-Object {
        if ($_ -match '\s') { '"' + $_ + '"' } else { $_ }
    }) -join ' ')

    Write-Host $commandText -ForegroundColor Cyan
    if ($DryRun) {
        return
    }

    & dotnet @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $commandText"
    }
}

function Resolve-PythonExecutable {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ProjectRoot
    )

    $candidateRoots = @(
        $ProjectRoot,
        [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot '..'))
    ) | Select-Object -Unique

    foreach ($candidateRoot in $candidateRoots) {
        $venvPython = Join-Path $candidateRoot '.venv\Scripts\python.exe'
        if (Test-Path $venvPython) {
            return $venvPython
        }
    }

    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $pythonCommand) {
        return $pythonCommand.Source
    }

    throw "Python interpreter not found. Expected a workspace .venv or a python executable on PATH."
}

function Invoke-MinimalManifestCuration {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ProjectRoot,
        [Parameter(Mandatory = $true)]
        [string]$DatasetsRoot,
        [AllowNull()]
        [string]$ManifestOutput,
        [AllowNull()]
        [string]$PlanOutput
    )

    $pythonExecutable = Resolve-PythonExecutable -ProjectRoot $ProjectRoot
    $scriptPath = Join-Path $ProjectRoot 'scripts\build_minimal_ml_manifest.py'
    if (-not (Test-Path $scriptPath)) {
        throw "Minimal manifest curation script not found: $scriptPath"
    }

    $arguments = @($scriptPath, '--datasets-root', $DatasetsRoot)
    if (-not [string]::IsNullOrWhiteSpace($ManifestOutput)) {
        $arguments += @('--output', $ManifestOutput)
    }

    if (-not [string]::IsNullOrWhiteSpace($PlanOutput)) {
        $arguments += @('--plan-output', $PlanOutput)
    }

    $commandText = '"' + $pythonExecutable + '" ' + (($arguments | ForEach-Object {
        if ($_ -match '\s') { '"' + $_ + '"' } else { $_ }
    }) -join ' ')

    Write-Host $commandText -ForegroundColor Cyan
    if ($DryRun) {
        return
    }

    & $pythonExecutable @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Minimal manifest curation failed with exit code ${LASTEXITCODE}: $commandText"
    }
}

function Resolve-ClientMapList {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ClientPath,
        [Parameter(Mandatory = $true)]
        [string]$ProjectPath,
        [Parameter(Mandatory = $true)]
        [string]$Configuration,
        [string]$ListfilePath
    )

    $tmpJson = [System.IO.Path]::GetTempFileName()
    try {
        $args = @(
            'run',
            '--project', $ProjectPath,
            '--configuration', $Configuration,
            '--',
            'ml-list-maps',
            '--client', $ClientPath,
            '--output-json', $tmpJson
        )

        if (-not [string]::IsNullOrWhiteSpace($ListfilePath)) {
            $args += @('--listfile', $ListfilePath)
        }

        $commandText = 'dotnet ' + (($args | ForEach-Object {
            if ($_ -match '\s') { '"' + $_ + '"' } else { $_ }
        }) -join ' ')

        Write-Host $commandText -ForegroundColor Cyan
        if ($DryRun) {
            return ,@()
        }

        & dotnet @args
        if ($LASTEXITCODE -ne 0) {
            throw "Map discovery command failed with exit code ${LASTEXITCODE}: $commandText"
        }

        if (-not (Test-Path $tmpJson)) {
            throw "Map discovery output not found: $tmpJson"
        }

        $json = Get-Content -Raw -Path $tmpJson
        if ([string]::IsNullOrWhiteSpace($json)) {
            return ,@()
        }

        $parsed = $json | ConvertFrom-Json
        if ($parsed -is [System.Array]) {
            return ,@($parsed | ForEach-Object { [string]$_ } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
        }

        if ($null -eq $parsed) {
            return ,@()
        }

        return ,@([string]$parsed)
    }
    finally {
        if (Test-Path $tmpJson) {
            Remove-Item -Path $tmpJson -Force -ErrorAction SilentlyContinue
        }
    }
}

if (-not (Test-Path $ConfigPath)) {
    throw "Config file not found: $ConfigPath"
}

$resolvedConfigPath = (Resolve-Path $ConfigPath).Path
$configDirectory = Split-Path -Parent $resolvedConfigPath

$config = Get-Content -Raw -Path $ConfigPath | ConvertFrom-Json
$configArchiveRoot = Get-JsonPropertyValue -Object $config -Name 'archive_root'
$configMountRoot = Get-JsonPropertyValue -Object $config -Name 'mount_root'
$configMountScript = Get-JsonPropertyValue -Object $config -Name 'mount_script'
$configStagingRoot = Get-JsonPropertyValue -Object $config -Name 'staging_root'
$configDefaultOutputRoot = Get-JsonPropertyValue -Object $config -Name 'default_output_root'
$configLegacyOutputRoot = Get-JsonPropertyValue -Object $config -Name 'output_root'
$configListfilePath = Get-JsonPropertyValue -Object $config -Name 'listfile_path'
$configHarvestAfterExport = Get-JsonPropertyValue -Object $config -Name 'harvest_after_export'
$configPruneStagedClients = Get-JsonPropertyValue -Object $config -Name 'prune_staged_clients'
$configRunMinimalCurationAfterExport = Get-JsonPropertyValue -Object $config -Name 'run_minimal_curation_after_export'
$configMinimalCurationOutput = Get-JsonPropertyValue -Object $config -Name 'minimal_curation_output'
$configMinimalCurationPlanOutput = Get-JsonPropertyValue -Object $config -Name 'minimal_curation_plan_output'

$resolvedArchiveRoot = if ($ArchiveRoot) { Resolve-ConfigPathValue -Value $ArchiveRoot -BaseRoot $null } elseif ($configArchiveRoot) { Resolve-ConfigPathValue -Value ([string]$configArchiveRoot) -BaseRoot $null } else { $null }
$resolvedMountRoot = if ($ArchiveMountRoot) { Resolve-ConfigPathValue -Value $ArchiveMountRoot -BaseRoot $null } elseif ($configMountRoot) { Resolve-ConfigPathValue -Value ([string]$configMountRoot) -BaseRoot $null } else { $null }
$resolvedMountScript = if ($MountScript) { Resolve-ConfigPathValue -Value $MountScript -BaseRoot $null } elseif ($configMountScript) { Resolve-ConfigPathValue -Value ([string]$configMountScript) -BaseRoot $null } else { $null }
$resolvedStagingRoot = if ($StagingRoot) { Resolve-ConfigPathValue -Value $StagingRoot -BaseRoot $null } elseif ($configStagingRoot) { Resolve-ConfigPathValue -Value ([string]$configStagingRoot) -BaseRoot $null } else { [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\output\tmp\wowarchive-clients')) }
$resolvedLegacyPathBase = if (-not [string]::IsNullOrWhiteSpace($resolvedArchiveRoot)) { $resolvedArchiveRoot } else { $resolvedMountRoot }
$resolvedArchiveSourceBase = if (-not [string]::IsNullOrWhiteSpace($resolvedMountRoot)) { $resolvedMountRoot } else { $resolvedArchiveRoot }
$resolvedOutputRoot = if ($OutputRoot) { Resolve-ConfigPathValue -Value $OutputRoot -BaseRoot $null } elseif ($configDefaultOutputRoot) { Resolve-ConfigPathValue -Value ([string]$configDefaultOutputRoot) -BaseRoot $null } elseif ($configLegacyOutputRoot) { Resolve-ConfigPathValue -Value ([string]$configLegacyOutputRoot) -BaseRoot $null } else { [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\datasets')) }
$resolvedListfilePath = if ($ListfilePath) { Resolve-ConfigPathValue -Value $ListfilePath -BaseRoot $null } elseif ($configListfilePath) { Resolve-ConfigPathValue -Value ([string]$configListfilePath) -BaseRoot $null } else { [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\test_data\community-listfile-withcapitals.csv')) }
$harvestConfigured = if ($null -ne $configHarvestAfterExport) { [bool]$configHarvestAfterExport } else { $true }
$harvestAfterExport = -not $SkipHarvest -and $harvestConfigured
$pruneConfigured = if ($null -ne $configPruneStagedClients) { [bool]$configPruneStagedClients } else { $false }
$shouldPruneStagedClients = $PruneStagedClients -or $pruneConfigured
$resumeEnabled = -not $Force
$runMinimalCurationAfterExport = if ($null -ne $configRunMinimalCurationAfterExport) { [bool]$configRunMinimalCurationAfterExport } else { $false }
$resolvedMinimalCurationOutput = if ($configMinimalCurationOutput) { Resolve-ConfigPathValue -Value ([string]$configMinimalCurationOutput) -BaseRoot $null } else { $null }
$resolvedMinimalCurationPlanOutput = if ($configMinimalCurationPlanOutput) { Resolve-ConfigPathValue -Value ([string]$configMinimalCurationPlanOutput) -BaseRoot $null } else { $null }
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))

if (-not $DryRun) {
    New-Item -ItemType Directory -Path $resolvedOutputRoot -Force | Out-Null
}

$jobFailures = 0
$jobPlans = @()
$stagedLabelsToKeep = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)

foreach ($client in $config.clients) {
    $clientLabelValue = Get-JsonPropertyValue -Object $client -Name 'label'
    $clientOutputRootValue = Get-JsonPropertyValue -Object $client -Name 'output_root'
    $clientGenerateDepth = Get-JsonPropertyValue -Object $client -Name 'generate_depth'
    $clientSkipDerivedAssets = Get-JsonPropertyValue -Object $client -Name 'skip_derived_assets'
    $clientTileLimitValue = Get-JsonPropertyValue -Object $client -Name 'tile_limit'
    $clientInterestingOnlyValue = Get-JsonPropertyValue -Object $client -Name 'interesting_only'
    $clientInterestingMinScoreValue = Get-JsonPropertyValue -Object $client -Name 'interesting_min_score'
    $clientMinimapRootValue = Get-JsonPropertyValue -Object $client -Name 'minimap_root'
    $clientLocalPathValue = Get-JsonPropertyValue -Object $client -Name 'local_client_path'
    $clientArchivePathValue = Get-JsonPropertyValue -Object $client -Name 'archive_client_path'
    $clientLocalMinimapRootValue = Get-JsonPropertyValue -Object $client -Name 'local_minimap_root'
    $clientArchiveMinimapRootValue = Get-JsonPropertyValue -Object $client -Name 'archive_minimap_root'
    $clientAllMapsValue = Get-JsonPropertyValue -Object $client -Name 'all_maps'

    $clientLabel = if ($clientLabelValue) { [string]$clientLabelValue } else { ([string]$client.version).Replace('.', '_') }
    $clientOutputRoot = if ($clientOutputRootValue) { Resolve-ConfigPathValue -Value ([string]$clientOutputRootValue) -BaseRoot $null } else { Join-Path $resolvedOutputRoot $clientLabel }

    $clientRootInfo = Resolve-ConfiguredWorkingRoot -Label $clientLabel -DirectPath ([string](Get-JsonPropertyValue -Object $client -Name 'client_path')) -DirectBaseRoot $resolvedLegacyPathBase -LocalPath ([string]$clientLocalPathValue) -ArchivePath ([string]$clientArchivePathValue) -ArchiveBaseRoot $resolvedArchiveSourceBase -MountRoot $resolvedMountRoot -MountScript $resolvedMountScript -StagingRoot $resolvedStagingRoot -ForceRestage:$ForceRestage -DryRun:$DryRun
    $resolvedClientPath = $clientRootInfo.WorkingPath

    $minimapRootInfo = $null
    $resolvedMinimapRoot = $null
    if ($clientMinimapRootValue -or $clientLocalMinimapRootValue -or $clientArchiveMinimapRootValue) {
        $minimapRootInfo = Resolve-ConfiguredWorkingRoot -Label ("{0}-minimap" -f $clientLabel) -DirectPath ([string]$clientMinimapRootValue) -DirectBaseRoot $resolvedLegacyPathBase -LocalPath ([string]$clientLocalMinimapRootValue) -ArchivePath ([string]$clientArchiveMinimapRootValue) -ArchiveBaseRoot $resolvedArchiveSourceBase -MountRoot $resolvedMountRoot -MountScript $resolvedMountScript -StagingRoot $resolvedStagingRoot -ForceRestage:$ForceRestage -DryRun:$DryRun
        $resolvedMinimapRoot = $minimapRootInfo.WorkingPath
    }

    if ($clientRootInfo.Staged) {
        [void]$stagedLabelsToKeep.Add($clientLabel)
    }

    if ($minimapRootInfo -and $minimapRootInfo.Staged) {
        [void]$stagedLabelsToKeep.Add(("{0}-minimap" -f $clientLabel))
    }

    if (-not $DryRun) {
        New-Item -ItemType Directory -Path $clientOutputRoot -Force | Out-Null
    }

    Write-Host ("Client {0} -> {1} [{2}]" -f $clientLabel, $resolvedClientPath, $clientRootInfo.SourceType) -ForegroundColor Yellow
    if ($clientRootInfo.Staged) {
        Write-Host ("  staged from: {0}" -f $clientRootInfo.SourcePath) -ForegroundColor DarkYellow
    }
    if ($resolvedMinimapRoot) {
        $minimapMode = if ($minimapRootInfo) { $minimapRootInfo.SourceType } else { 'direct' }
        Write-Host ("  minimap -> {0} [{1}]" -f $resolvedMinimapRoot, $minimapMode) -ForegroundColor DarkYellow
    }

    $mapsToProcess = @()
    $clientAllMaps = $false
    if ($null -ne $clientAllMapsValue) {
        $clientAllMaps = [bool]$clientAllMapsValue
    }

    $mapDiscoveryClientPath = if ($DryRun -and $clientRootInfo.Staged -and -not (Test-Path $resolvedClientPath)) { $clientRootInfo.SourcePath } else { $resolvedClientPath }

    if ($clientAllMaps) {
        $mapsToProcess = Resolve-ClientMapList -ClientPath $mapDiscoveryClientPath -ProjectPath $ProjectPath -Configuration $Configuration -ListfilePath $resolvedListfilePath
        if ($mapsToProcess.Count -eq 0) {
            if ($DryRun) {
                Write-Warning "Dry-run map discovery returned no maps for client '$clientLabel'."
            }
            else {
                Write-Warning "No maps discovered for client '$clientLabel' at '$resolvedClientPath'."
                $jobFailures++
            }
            continue
        }

        Write-Host "Discovered $($mapsToProcess.Count) maps for $clientLabel" -ForegroundColor DarkCyan
    }
    else {
        $mapsToProcess = @($client.maps)
    }

    foreach ($map in $mapsToProcess) {
        $datasetOutput = Join-Path $clientOutputRoot $map
        $clientGenerateDepthEnabled = $null -ne $clientGenerateDepth -and [bool]$clientGenerateDepth
        $clientTileLimit = if ($null -ne $clientTileLimitValue) { [int]$clientTileLimitValue } else { $null }
        $clientInterestingOnly = $null -ne $clientInterestingOnlyValue -and [bool]$clientInterestingOnlyValue
        $clientInterestingMinScore = if ($null -ne $clientInterestingMinScoreValue) { [int]$clientInterestingMinScoreValue } else { 1 }
        $clientSkipDerivedAssetsEnabled = $null -ne $clientSkipDerivedAssets -and [bool]$clientSkipDerivedAssets

        $jobDecision = Get-JobPlanAction -ResumeEnabled $resumeEnabled -DatasetOutput $datasetOutput -ClientLabel $clientLabel -ClientVersion ([string]$client.version) -Map ([string]$map) -HarvestRequested $harvestAfterExport -GenerateDepth $clientGenerateDepthEnabled -TileLimit $clientTileLimit -InterestingOnly $clientInterestingOnly -InterestingMinScore $clientInterestingMinScore -SkipDerivedAssets $clientSkipDerivedAssetsEnabled -MinimapRoot $resolvedMinimapRoot

        $jobPlans += [pscustomobject]@{
            ClientLabel = $clientLabel
            ClientVersion = [string]$client.version
            ResolvedClientPath = $resolvedClientPath
            ClientRootInfo = $clientRootInfo
            ResolvedMinimapRoot = $resolvedMinimapRoot
            MinimapRootInfo = $minimapRootInfo
            DatasetOutput = $datasetOutput
            DatasetJsonDir = (Join-Path $datasetOutput 'dataset')
            Map = [string]$map
            GenerateDepth = $clientGenerateDepthEnabled
            TileLimit = $clientTileLimit
            InterestingOnly = $clientInterestingOnly
            InterestingMinScore = $clientInterestingMinScore
            SkipDerivedAssets = $clientSkipDerivedAssetsEnabled
            Action = [string]$jobDecision.Kind
            ActionReason = [string]$jobDecision.Reason
        }
    }
}

Write-JobPlanSummary -JobPlans $jobPlans -ResumeEnabled $resumeEnabled -HarvestRequested $harvestAfterExport -ForceRequested $Force

foreach ($job in $jobPlans) {
    if ($job.Action -eq 'skip-all') {
        Write-Host ("Resume skip for {0}/{1}: {2}" -f $job.ClientLabel, $job.Map, $job.ActionReason) -ForegroundColor DarkGreen
        continue
    }

    if ($Force -and (Test-Path $job.DatasetOutput)) {
        Write-Host "Clearing existing export output: $($job.DatasetOutput)" -ForegroundColor DarkYellow
        if (-not $DryRun) {
            Remove-Item -Path $job.DatasetOutput -Recurse -Force
        }
    }

    if ($job.Action -eq 'run-harvest-only') {
        Write-Host ("Resume export skip for {0}/{1}: {2}" -f $job.ClientLabel, $job.Map, $job.ActionReason) -ForegroundColor DarkGreen
    }

    if ($job.Action -eq 'run-export') {
        $exportArgs = @(
            'run',
            '--project', $ProjectPath,
            '--configuration', $Configuration,
            '--',
            'ml-export',
            '--client', $job.ResolvedClientPath,
            '--map', $job.Map,
            '--out', $job.DatasetOutput,
            '--listfile', $resolvedListfilePath
        )

        if (-not [string]::IsNullOrWhiteSpace($job.ResolvedMinimapRoot)) {
            $exportArgs += @('--minimap-root', $job.ResolvedMinimapRoot)
        }

        if ($job.GenerateDepth) {
            $exportArgs += '--depth'
        }

        if ($null -ne $job.TileLimit -and $job.TileLimit -gt 0) {
            $exportArgs += @('--limit', [string]$job.TileLimit)
        }

        if ($job.InterestingOnly) {
            $exportArgs += '--interesting-only'
        }

        $exportArgs += @('--interesting-min-score', [string]$job.InterestingMinScore)

        if ($job.SkipDerivedAssets) {
            $exportArgs += '--skip-derived-assets'
        }

        try {
            Invoke-LoggedCommand -Arguments $exportArgs
            $datasetTileCountAfterExport = (Get-DatasetJsonFiles -DatasetOutput $job.DatasetOutput).Count
            Write-ResumeState -DatasetOutput $job.DatasetOutput -ClientLabel $job.ClientLabel -ClientVersion $job.ClientVersion -Map $job.Map -HarvestRequested $harvestAfterExport -ExportCompleted $true -HarvestCompleted $false -GenerateDepth $job.GenerateDepth -TileLimit $job.TileLimit -InterestingOnly $job.InterestingOnly -InterestingMinScore $job.InterestingMinScore -SkipDerivedAssets $job.SkipDerivedAssets -MinimapRoot $job.ResolvedMinimapRoot -TileJsonCount $datasetTileCountAfterExport
        }
        catch {
            Write-Warning ("Export failed for {0}/{1}: {2}" -f $job.ClientLabel, $job.Map, $_.Exception.Message)
            $jobFailures++
            continue
        }
    }

    if ($harvestAfterExport) {
        $datasetFiles = @(Get-DatasetJsonFiles -DatasetOutput $job.DatasetOutput)

        if ($datasetFiles.Count -eq 0) {
            Write-Warning "Skipping harvest for $($job.DatasetOutput) because no tile JSON files were found under $($job.DatasetJsonDir)."
            continue
        }

        $manifestPath = Join-Path $job.DatasetOutput 'ml_dataset_manifest.json'
        $harvestArgs = @(
            'run',
            '--project', $ProjectPath,
            '--configuration', $Configuration,
            '--',
            'ml-harvest',
            '--dataset', $job.DatasetOutput,
            '--output', $manifestPath
        )

        try {
            Invoke-LoggedCommand -Arguments $harvestArgs
            Write-ResumeState -DatasetOutput $job.DatasetOutput -ClientLabel $job.ClientLabel -ClientVersion $job.ClientVersion -Map $job.Map -HarvestRequested $harvestAfterExport -ExportCompleted $true -HarvestCompleted $true -GenerateDepth $job.GenerateDepth -TileLimit $job.TileLimit -InterestingOnly $job.InterestingOnly -InterestingMinScore $job.InterestingMinScore -SkipDerivedAssets $job.SkipDerivedAssets -MinimapRoot $job.ResolvedMinimapRoot -TileJsonCount $datasetFiles.Count
        }
        catch {
            Write-Warning ("Harvest failed for {0}/{1}: {2}" -f $job.ClientLabel, $job.Map, $_.Exception.Message)
            $jobFailures++
            continue
        }
    }
    elseif ($job.Action -eq 'run-export') {
        $datasetTileCount = (Get-DatasetJsonFiles -DatasetOutput $job.DatasetOutput).Count
        Write-ResumeState -DatasetOutput $job.DatasetOutput -ClientLabel $job.ClientLabel -ClientVersion $job.ClientVersion -Map $job.Map -HarvestRequested $harvestAfterExport -ExportCompleted $true -HarvestCompleted $false -GenerateDepth $job.GenerateDepth -TileLimit $job.TileLimit -InterestingOnly $job.InterestingOnly -InterestingMinScore $job.InterestingMinScore -SkipDerivedAssets $job.SkipDerivedAssets -MinimapRoot $job.ResolvedMinimapRoot -TileJsonCount $datasetTileCount
    }
}

if ($shouldPruneStagedClients) {
    $removedStages = @(Remove-StaleWoWArchiveStages -StagingRoot $resolvedStagingRoot -KeepLabels @($stagedLabelsToKeep) -DryRun:$DryRun)
    if ($removedStages.Count -gt 0) {
        Write-Host ("Pruned {0} stale staged client(s)." -f $removedStages.Count) -ForegroundColor DarkGreen
    }
}

if ($jobFailures -gt 0) {
    throw "ML corpus export workflow completed with $jobFailures failed map jobs."
}

if ($runMinimalCurationAfterExport) {
    Invoke-MinimalManifestCuration -ProjectRoot $projectRoot -DatasetsRoot $resolvedOutputRoot -ManifestOutput $resolvedMinimalCurationOutput -PlanOutput $resolvedMinimalCurationPlanOutput
}

Write-Host "ML corpus export workflow complete." -ForegroundColor Green
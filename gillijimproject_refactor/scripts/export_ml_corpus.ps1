param(
    [string]$ConfigPath = (Join-Path $PSScriptRoot 'ml_corpus_fixed_clients.json'),
    [string]$ProjectPath = (Join-Path $PSScriptRoot '..\src\WoWMapConverter\WoWMapConverter.Cli\WoWMapConverter.Cli.csproj'),
    [string]$OutputRoot,
    [string]$ListfilePath,
    [string]$Configuration = 'Debug',
    [switch]$SkipHarvest,
    [switch]$Force,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$configDirectory = $null

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
            return @()
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
            return @()
        }

        $parsed = $json | ConvertFrom-Json
        if ($parsed -is [System.Array]) {
            return @($parsed | ForEach-Object { [string]$_ } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
        }

        if ($null -eq $parsed) {
            return @()
        }

        return @([string]$parsed)
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
$configDefaultOutputRoot = Get-JsonPropertyValue -Object $config -Name 'default_output_root'
$configLegacyOutputRoot = Get-JsonPropertyValue -Object $config -Name 'output_root'
$configListfilePath = Get-JsonPropertyValue -Object $config -Name 'listfile_path'
$configHarvestAfterExport = Get-JsonPropertyValue -Object $config -Name 'harvest_after_export'

$resolvedArchiveRoot = if ($configArchiveRoot) { Resolve-ConfigPathValue -Value ([string]$configArchiveRoot) -BaseRoot $null } else { $null }
$resolvedOutputRoot = if ($OutputRoot) { Resolve-ConfigPathValue -Value $OutputRoot -BaseRoot $null } elseif ($configDefaultOutputRoot) { Resolve-ConfigPathValue -Value ([string]$configDefaultOutputRoot) -BaseRoot $null } elseif ($configLegacyOutputRoot) { Resolve-ConfigPathValue -Value ([string]$configLegacyOutputRoot) -BaseRoot $null } else { [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\datasets')) }
$resolvedListfilePath = if ($ListfilePath) { Resolve-ConfigPathValue -Value $ListfilePath -BaseRoot $null } elseif ($configListfilePath) { Resolve-ConfigPathValue -Value ([string]$configListfilePath) -BaseRoot $null } else { [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\test_data\community-listfile-withcapitals.csv')) }
$harvestConfigured = if ($null -ne $configHarvestAfterExport) { [bool]$configHarvestAfterExport } else { $true }
$harvestAfterExport = -not $SkipHarvest -and $harvestConfigured

if (-not $DryRun) {
    New-Item -ItemType Directory -Path $resolvedOutputRoot -Force | Out-Null
}

$jobFailures = 0

foreach ($client in $config.clients) {
    $clientLabelValue = Get-JsonPropertyValue -Object $client -Name 'label'
    $clientOutputRootValue = Get-JsonPropertyValue -Object $client -Name 'output_root'
    $clientGenerateDepth = Get-JsonPropertyValue -Object $client -Name 'generate_depth'
    $clientMinimapRootValue = Get-JsonPropertyValue -Object $client -Name 'minimap_root'
    $clientAllMapsValue = Get-JsonPropertyValue -Object $client -Name 'all_maps'

    $clientLabel = if ($clientLabelValue) { [string]$clientLabelValue } else { ([string]$client.version).Replace('.', '_') }
    $clientOutputRoot = if ($clientOutputRootValue) { Resolve-ConfigPathValue -Value ([string]$clientOutputRootValue) -BaseRoot $null } else { Join-Path $resolvedOutputRoot $clientLabel }
    $resolvedClientPath = Resolve-ConfigPathValue -Value ([string]$client.client_path) -BaseRoot $resolvedArchiveRoot
    $resolvedMinimapRoot = if ($clientMinimapRootValue) { Resolve-ConfigPathValue -Value ([string]$clientMinimapRootValue) -BaseRoot $resolvedArchiveRoot } else { $null }

    if (-not $DryRun) {
        New-Item -ItemType Directory -Path $clientOutputRoot -Force | Out-Null
    }

    Write-Host "Client $clientLabel -> $resolvedClientPath" -ForegroundColor Yellow

    $mapsToProcess = @()
    $clientAllMaps = $false
    if ($null -ne $clientAllMapsValue) {
        $clientAllMaps = [bool]$clientAllMapsValue
    }

    if ($clientAllMaps) {
        $mapsToProcess = Resolve-ClientMapList -ClientPath $resolvedClientPath -ProjectPath $ProjectPath -Configuration $Configuration -ListfilePath $resolvedListfilePath
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
        $datasetJsonDir = Join-Path $datasetOutput 'dataset'

        if ((-not $Force) -and (Test-Path $datasetJsonDir)) {
            Write-Host "Skipping existing export: $datasetOutput" -ForegroundColor DarkYellow
        }
        else {
            $exportArgs = @(
                'run',
                '--project', $ProjectPath,
                '--configuration', $Configuration,
                '--',
                'ml-export',
                '--client', $resolvedClientPath,
                '--map', [string]$map,
                '--out', $datasetOutput,
                '--listfile', $resolvedListfilePath
            )

            if (-not [string]::IsNullOrWhiteSpace($resolvedMinimapRoot)) {
                $exportArgs += @('--minimap-root', $resolvedMinimapRoot)
            }

            if ($null -ne $clientGenerateDepth -and [bool]$clientGenerateDepth) {
                $exportArgs += '--depth'
            }

            try {
                Invoke-LoggedCommand -Arguments $exportArgs
            }
            catch {
                Write-Warning ("Export failed for {0}/{1}: {2}" -f $clientLabel, $map, $_.Exception.Message)
                $jobFailures++
                continue
            }
        }

        if ($harvestAfterExport) {
            $datasetFiles = @()
            if (Test-Path $datasetJsonDir) {
                $datasetFiles = @(Get-ChildItem -Path $datasetJsonDir -Filter '*.json' -File -ErrorAction SilentlyContinue)
            }

            if ($datasetFiles.Count -eq 0) {
                Write-Warning "Skipping harvest for $datasetOutput because no tile JSON files were found under $datasetJsonDir."
                continue
            }

            $manifestPath = Join-Path $datasetOutput 'ml_dataset_manifest.json'
            $harvestArgs = @(
                'run',
                '--project', $ProjectPath,
                '--configuration', $Configuration,
                '--',
                'ml-harvest',
                '--dataset', $datasetOutput,
                '--output', $manifestPath
            )

            try {
                Invoke-LoggedCommand -Arguments $harvestArgs
            }
            catch {
                Write-Warning ("Harvest failed for {0}/{1}: {2}" -f $clientLabel, $map, $_.Exception.Message)
                $jobFailures++
                continue
            }
        }
    }
}

if ($jobFailures -gt 0) {
    throw "ML corpus export workflow completed with $jobFailures failed map jobs."
}

Write-Host "ML corpus export workflow complete." -ForegroundColor Green
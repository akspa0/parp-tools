Set-StrictMode -Version Latest

$script:WowArchiveStageMetadataFileName = '.wowarchive-stage.json'

function Resolve-WowArchiveAbsolutePath {
    param(
        [AllowNull()]
        [string]$Path,
        [AllowNull()]
        [string]$BaseRoot
    )

    if ([string]::IsNullOrWhiteSpace($Path)) {
        return $null
    }

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }

    if (-not [string]::IsNullOrWhiteSpace($BaseRoot)) {
        return [System.IO.Path]::GetFullPath((Join-Path $BaseRoot $Path))
    }

    return [System.IO.Path]::GetFullPath($Path)
}

function Test-PathWithinRoot {
    param(
        [AllowNull()]
        [string]$Path,
        [AllowNull()]
        [string]$Root
    )

    if ([string]::IsNullOrWhiteSpace($Path) -or [string]::IsNullOrWhiteSpace($Root)) {
        return $false
    }

    $fullPath = [System.IO.Path]::GetFullPath($Path).TrimEnd('\', '/')
    $fullRoot = [System.IO.Path]::GetFullPath($Root).TrimEnd('\', '/')

    if ($fullPath.Length -lt $fullRoot.Length) {
        return $false
    }

    if ($fullPath.Equals($fullRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $true
    }

    return $fullPath.StartsWith($fullRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)
}

function Ensure-WoWArchiveMounted {
    param(
        [Parameter(Mandatory = $true)]
        [string]$MountRoot,
        [AllowNull()]
        [string]$MountScript,
        [switch]$DryRun
    )

    if (Test-Path $MountRoot) {
        return $true
    }

    if ([string]::IsNullOrWhiteSpace($MountScript)) {
        throw "WoWArchive mount root '$MountRoot' is not available and no mount script was provided."
    }

    if (-not (Test-Path $MountScript)) {
        throw "WoWArchive mount script not found: $MountScript"
    }

    Write-Host "Mounting WoWArchive via $MountScript" -ForegroundColor DarkCyan
    if ($DryRun) {
        return $false
    }

    & $MountScript
    if ($LASTEXITCODE -ne 0) {
        throw "WoWArchive mount script failed with exit code ${LASTEXITCODE}: $MountScript"
    }

    if (-not (Test-Path $MountRoot)) {
        throw "WoWArchive mount root still not available after running mount script: $MountRoot"
    }

    return $true
}

function Get-WoWArchiveStageMetadataPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$StagePath
    )

    return Join-Path $StagePath $script:WowArchiveStageMetadataFileName
}

function Read-WoWArchiveStageMetadata {
    param(
        [Parameter(Mandatory = $true)]
        [string]$StagePath
    )

    $metadataPath = Get-WoWArchiveStageMetadataPath -StagePath $StagePath
    if (-not (Test-Path $metadataPath)) {
        return $null
    }

    $raw = Get-Content -Raw -Path $metadataPath -ErrorAction Stop
    if ([string]::IsNullOrWhiteSpace($raw)) {
        return $null
    }

    return $raw | ConvertFrom-Json
}

function Write-WoWArchiveStageMetadata {
    param(
        [Parameter(Mandatory = $true)]
        [string]$StagePath,
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [string]$SourcePath,
        [Parameter(Mandatory = $true)]
        [string]$SourceType
    )

    if (-not (Test-Path $StagePath)) {
        New-Item -ItemType Directory -Path $StagePath -Force | Out-Null
    }

    $metadata = [ordered]@{
        label = $Label
        source_path = [System.IO.Path]::GetFullPath($SourcePath)
        source_type = $SourceType
        updated_at_utc = [DateTime]::UtcNow.ToString('o')
    }

    $metadataPath = Get-WoWArchiveStageMetadataPath -StagePath $StagePath
    $metadata | ConvertTo-Json | Set-Content -Path $metadataPath -Encoding UTF8
}

function Remove-WoWArchiveStageDirectory {
    param(
        [Parameter(Mandatory = $true)]
        [string]$StagePath,
        [Parameter(Mandatory = $true)]
        [string]$StagingRoot,
        [switch]$DryRun
    )

    if (-not (Test-PathWithinRoot -Path $StagePath -Root $StagingRoot)) {
        throw "Refusing to remove stage path outside staging root: $StagePath"
    }

    if (-not (Test-Path $StagePath)) {
        return
    }

    Write-Host "Removing staged client: $StagePath" -ForegroundColor DarkYellow
    if ($DryRun) {
        return
    }

    Remove-Item -Path $StagePath -Recurse -Force
}

function Copy-WoWArchiveClientToStage {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SourcePath,
        [Parameter(Mandatory = $true)]
        [string]$DestinationPath,
        [switch]$DryRun
    )

    if (-not (Test-Path $SourcePath)) {
        throw "WoWArchive source path not found: $SourcePath"
    }

    if (-not (Test-Path $DestinationPath)) {
        New-Item -ItemType Directory -Path $DestinationPath -Force | Out-Null
    }

    $robocopyArgs = @(
        $SourcePath,
        $DestinationPath,
        '/E',
        '/COPY:DAT',
        '/DCOPY:DAT',
        '/R:1',
        '/W:1',
        '/NFL',
        '/NDL',
        '/NJH',
        '/NJS',
        '/NP'
    )

    Write-Host ("Staging client with robocopy: {0} -> {1}" -f $SourcePath, $DestinationPath) -ForegroundColor DarkCyan
    if ($DryRun) {
        return
    }

    & robocopy @robocopyArgs | Out-Null
    $robocopyExitCode = $LASTEXITCODE
    if ($robocopyExitCode -gt 7) {
        throw "robocopy failed with exit code ${robocopyExitCode} while staging '$SourcePath' to '$DestinationPath'"
    }
}

function Resolve-WoWArchiveWorkingRoot {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [string]$ArchiveSourcePath,
        [AllowNull()]
        [string]$MountRoot,
        [AllowNull()]
        [string]$MountScript,
        [AllowNull()]
        [string]$StagingRoot,
        [switch]$ForceRestage,
        [switch]$DryRun
    )

    $resolvedSourcePath = [System.IO.Path]::GetFullPath($ArchiveSourcePath)
    $isMountedArchivePath = Test-PathWithinRoot -Path $resolvedSourcePath -Root $MountRoot

    if ($isMountedArchivePath -and -not (Test-Path $resolvedSourcePath)) {
        Ensure-WoWArchiveMounted -MountRoot $MountRoot -MountScript $MountScript -DryRun:$DryRun | Out-Null
    }

    if (-not (Test-Path $resolvedSourcePath) -and -not $DryRun) {
        throw "Archive-backed client root not found: $resolvedSourcePath"
    }

    if ([string]::IsNullOrWhiteSpace($StagingRoot)) {
        return [pscustomobject]@{
            WorkingPath = $resolvedSourcePath
            SourcePath = $resolvedSourcePath
            SourceType = 'archive-mounted'
            Staged = $false
            StagePath = $null
        }
    }

    $resolvedStagingRoot = [System.IO.Path]::GetFullPath($StagingRoot)
    if (-not (Test-Path $resolvedStagingRoot) -and -not $DryRun) {
        New-Item -ItemType Directory -Path $resolvedStagingRoot -Force | Out-Null
    }

    $stagePath = Join-Path $resolvedStagingRoot $Label
    $existingMetadata = if (Test-Path $stagePath) { Read-WoWArchiveStageMetadata -StagePath $stagePath } else { $null }
    $sourceMatchesMetadata = $false
    if ($null -ne $existingMetadata) {
        $sourceMatchesMetadata = [System.IO.Path]::GetFullPath([string]$existingMetadata.source_path).Equals($resolvedSourcePath, [System.StringComparison]::OrdinalIgnoreCase)
    }

    $shouldRestage = $ForceRestage -or (-not (Test-Path $stagePath)) -or (-not $sourceMatchesMetadata)
    if ($shouldRestage) {
        if (Test-Path $stagePath) {
            Remove-WoWArchiveStageDirectory -StagePath $stagePath -StagingRoot $resolvedStagingRoot -DryRun:$DryRun
        }

        if (-not $DryRun) {
            New-Item -ItemType Directory -Path $stagePath -Force | Out-Null
        }

        Copy-WoWArchiveClientToStage -SourcePath $resolvedSourcePath -DestinationPath $stagePath -DryRun:$DryRun
    }
    else {
        Write-Host "Reusing staged client: $stagePath" -ForegroundColor DarkGreen
    }

    if (-not $DryRun) {
        Write-WoWArchiveStageMetadata -StagePath $stagePath -Label $Label -SourcePath $resolvedSourcePath -SourceType 'archive-staged'
    }

    return [pscustomobject]@{
        WorkingPath = $stagePath
        SourcePath = $resolvedSourcePath
        SourceType = 'archive-staged'
        Staged = $true
        StagePath = $stagePath
    }
}

function Resolve-WoWClientWorkingRoot {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [AllowNull()]
        [string]$DirectPath,
        [AllowNull()]
        [string]$PreferredLocalPath,
        [AllowNull()]
        [string]$ArchiveSourcePath,
        [AllowNull()]
        [string]$MountRoot,
        [AllowNull()]
        [string]$MountScript,
        [AllowNull()]
        [string]$StagingRoot,
        [switch]$ForceRestage,
        [switch]$DryRun
    )

    $resolvedDirectPath = if ([string]::IsNullOrWhiteSpace($DirectPath)) { $null } else { [System.IO.Path]::GetFullPath($DirectPath) }
    $resolvedLocalPath = if ([string]::IsNullOrWhiteSpace($PreferredLocalPath)) { $null } else { [System.IO.Path]::GetFullPath($PreferredLocalPath) }
    $resolvedArchiveSourcePath = if ([string]::IsNullOrWhiteSpace($ArchiveSourcePath)) { $null } else { [System.IO.Path]::GetFullPath($ArchiveSourcePath) }

    if (-not [string]::IsNullOrWhiteSpace($resolvedLocalPath) -and (Test-Path $resolvedLocalPath)) {
        return [pscustomobject]@{
            WorkingPath = $resolvedLocalPath
            SourcePath = $resolvedLocalPath
            SourceType = 'local-fixed'
            Staged = $false
            StagePath = $null
        }
    }

    if ([string]::IsNullOrWhiteSpace($resolvedArchiveSourcePath) -and -not [string]::IsNullOrWhiteSpace($resolvedDirectPath) -and (Test-PathWithinRoot -Path $resolvedDirectPath -Root $MountRoot)) {
        $resolvedArchiveSourcePath = $resolvedDirectPath
    }

    if (-not [string]::IsNullOrWhiteSpace($resolvedArchiveSourcePath)) {
        return Resolve-WoWArchiveWorkingRoot -Label $Label -ArchiveSourcePath $resolvedArchiveSourcePath -MountRoot $MountRoot -MountScript $MountScript -StagingRoot $StagingRoot -ForceRestage:$ForceRestage -DryRun:$DryRun
    }

    if (-not [string]::IsNullOrWhiteSpace($resolvedDirectPath) -and (Test-Path $resolvedDirectPath)) {
        return [pscustomobject]@{
            WorkingPath = $resolvedDirectPath
            SourcePath = $resolvedDirectPath
            SourceType = 'direct'
            Staged = $false
            StagePath = $null
        }
    }

    $checkedPaths = @($resolvedLocalPath, $resolvedArchiveSourcePath, $resolvedDirectPath) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    $checkedDisplay = if ($checkedPaths.Count -gt 0) { $checkedPaths -join '; ' } else { '<none>' }
    throw "Could not resolve a usable client root for '$Label'. Checked: $checkedDisplay"
}

function Remove-StaleWoWArchiveStages {
    param(
        [AllowNull()]
        [string]$StagingRoot,
        [string[]]$KeepLabels = @(),
        [switch]$DryRun
    )

    if ([string]::IsNullOrWhiteSpace($StagingRoot)) {
        return @()
    }

    $resolvedStagingRoot = [System.IO.Path]::GetFullPath($StagingRoot)
    if (-not (Test-Path $resolvedStagingRoot)) {
        return @()
    }

    $keepSet = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($label in $KeepLabels) {
        if (-not [string]::IsNullOrWhiteSpace($label)) {
            [void]$keepSet.Add($label)
        }
    }

    $removed = New-Object System.Collections.Generic.List[string]
    foreach ($entry in Get-ChildItem -Path $resolvedStagingRoot -Directory -ErrorAction SilentlyContinue) {
        if ($keepSet.Contains($entry.Name)) {
            continue
        }

        $metadataPath = Get-WoWArchiveStageMetadataPath -StagePath $entry.FullName
        if (-not (Test-Path $metadataPath)) {
            continue
        }

        Remove-WoWArchiveStageDirectory -StagePath $entry.FullName -StagingRoot $resolvedStagingRoot -DryRun:$DryRun
        [void]$removed.Add($entry.FullName)
    }

    return $removed
}
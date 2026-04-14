param(
    [string]$ClientPath,
    [string]$Label,
    [string]$ArchiveRoot = 'G:\WoW\WoWArchive-0.X-3.X\Mount',
    [string]$StagingRoot = 'i:\parp\parp-tools\output\tmp\wowarchive-clients',
    [string]$MountScript = 'G:\WoW\WoWArchive-0.X-3.X\MountAll.bat',
    [string[]]$KeepLabel = @(),
    [switch]$Prune,
    [switch]$ForceRestage,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$helperPath = Join-Path $PSScriptRoot 'wowarchive_client_staging.ps1'
. $helperPath

if ([string]::IsNullOrWhiteSpace($ClientPath) -and -not $Prune) {
    throw 'Provide -ClientPath <path> to stage a client, or use -Prune to remove stale staged clients.'
}

$keepLabels = New-Object System.Collections.Generic.List[string]
foreach ($keep in $KeepLabel) {
    if (-not [string]::IsNullOrWhiteSpace($keep)) {
        [void]$keepLabels.Add($keep)
    }
}

$result = $null
if (-not [string]::IsNullOrWhiteSpace($ClientPath)) {
    if ([string]::IsNullOrWhiteSpace($Label)) {
        throw 'Provide -Label <name> when staging a client.'
    }

    $resolvedClientPath = Resolve-WowArchiveAbsolutePath -Path $ClientPath -BaseRoot $ArchiveRoot
    $result = Resolve-WoWArchiveWorkingRoot -Label $Label -ArchiveSourcePath $resolvedClientPath -MountRoot $ArchiveRoot -MountScript $MountScript -StagingRoot $StagingRoot -ForceRestage:$ForceRestage -DryRun:$DryRun
    [void]$keepLabels.Add($Label)

    Write-Host ("Resolved working root [{0}] -> {1}" -f $result.SourceType, $result.WorkingPath) -ForegroundColor Green
    Write-Output $result.WorkingPath
}

if ($Prune) {
    $removed = @(Remove-StaleWoWArchiveStages -StagingRoot $StagingRoot -KeepLabels @($keepLabels) -DryRun:$DryRun)
    if ($removed.Count -eq 0) {
        Write-Host 'No stale staged clients removed.' -ForegroundColor DarkGreen
    }
    else {
        Write-Host ("Removed {0} stale staged client(s)." -f $removed.Count) -ForegroundColor DarkGreen
        foreach ($path in $removed) {
            Write-Host ("  pruned: {0}" -f $path) -ForegroundColor DarkYellow
        }
    }
}
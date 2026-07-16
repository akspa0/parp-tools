[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $WorkspaceRoot,

    [switch] $Apply,

    [string] $Confirmation = "",

    [switch] $MeasureBytes,

    [string] $ReportPath = ""
)

$ErrorActionPreference = "Stop"
$requiredConfirmation = "DELETE-LEGACY-OUTPUTS"

function Resolve-ExistingDirectory([string] $Path) {
    $item = Get-Item -LiteralPath $Path -Force
    if (-not $item.PSIsContainer) {
        throw "Expected a directory: $Path"
    }
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Refusing reparse-point root: $($item.FullName)"
    }
    return $item.FullName.TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)
}

function Get-TreeMeasurement([System.IO.FileSystemInfo] $Item) {
    if (-not $Item.PSIsContainer) {
        return [pscustomobject]@{
            bytes = [int64]$Item.Length
            errors = @()
        }
    }

    $sum = [int64]0
    $scanErrors = @()
    Get-ChildItem -LiteralPath $Item.FullName -File -Force -Recurse `
        -ErrorAction SilentlyContinue -ErrorVariable +scanErrors | ForEach-Object {
        $sum += [int64]$_.Length
    }
    return [pscustomobject]@{
        bytes = $sum
        errors = @($scanErrors | ForEach-Object { $_.Exception.Message })
    }
}

$workspace = Resolve-ExistingDirectory $WorkspaceRoot
$expectedRoots = @(
    [IO.Path]::GetFullPath((Join-Path $workspace "output")),
    [IO.Path]::GetFullPath((Join-Path $workspace "wow-viewer\output"))
)

$trackedExisting = @(
    git -C $workspace ls-files -- output wow-viewer/output |
        Where-Object { Test-Path -LiteralPath (Join-Path $workspace $_) }
)
if ($LASTEXITCODE -ne 0) {
    throw "git ls-files failed; cleanup safety cannot be established"
}
if ($trackedExisting.Count -gt 0) {
    throw "Refusing cleanup while tracked files still exist under output roots: $($trackedExisting -join ', ')"
}

$records = [System.Collections.Generic.List[object]]::new()
foreach ($expectedRoot in $expectedRoots) {
    if (-not (Test-Path -LiteralPath $expectedRoot)) {
        $records.Add([pscustomobject]@{
            root = $expectedRoot
            path = $expectedRoot
            kind = "absent-root"
            bytes = [int64]0
        })
        continue
    }

    $root = Resolve-ExistingDirectory $expectedRoot
    if (-not [string]::Equals($root, $expectedRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Resolved output root escaped its expected path: $root"
    }

    foreach ($child in Get-ChildItem -LiteralPath $root -Force) {
        if (($child.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Refusing reparse-point cleanup target: $($child.FullName)"
        }
        $measurement = if ($MeasureBytes) {
            Get-TreeMeasurement $child
        } else {
            [pscustomobject]@{ bytes = $null; errors = @() }
        }
        $records.Add([pscustomobject]@{
            root = $root
            path = $child.FullName
            kind = if ($child.PSIsContainer) { "directory" } else { "file" }
            bytes = $measurement.bytes
            measurement_complete = ($measurement.errors.Count -eq 0)
            measurement_errors = @($measurement.errors)
        })
    }
}

$measurementErrorCount = @($records | Where-Object { -not $_.measurement_complete }).Count

$report = [ordered]@{
    schema = "v50-legacy-output-cleanup-v1"
    workspace = $workspace
    mode = if ($Apply) { "apply" } else { "dry-run" }
    measured_bytes = [bool]$MeasureBytes
    expected_recovered_bytes = if ($MeasureBytes) {
        [int64](($records | Measure-Object -Property bytes -Sum).Sum)
    } else {
        $null
    }
    measurement_complete = ($measurementErrorCount -eq 0)
    measurement_error_count = $measurementErrorCount
    targets = @($records)
}

if (-not [string]::IsNullOrWhiteSpace($ReportPath)) {
    $reportParent = Split-Path -Parent $ReportPath
    if (-not [string]::IsNullOrWhiteSpace($reportParent)) {
        New-Item -ItemType Directory -Path $reportParent -Force | Out-Null
    }
    $report | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $ReportPath -Encoding utf8
}

if (-not $Apply) {
    $report | ConvertTo-Json -Depth 6
    Write-Host "Dry run only. Re-run with -Apply -Confirmation $requiredConfirmation after review."
    exit 0
}

if ($Confirmation -cne $requiredConfirmation) {
    throw "Apply requires -Confirmation $requiredConfirmation"
}

$deleteFailures = [System.Collections.Generic.List[object]]::new()
foreach ($record in $records) {
    if ($record.kind -eq "absent-root") {
        continue
    }
    $target = [IO.Path]::GetFullPath([string]$record.path)
    $parent = [IO.Path]::GetFullPath((Split-Path -Parent $target))
    if ($expectedRoots -notcontains $parent) {
        throw "Cleanup target escaped an approved output root: $target"
    }
    try {
        Remove-Item -LiteralPath $target -Recurse -Force
    } catch {
        $deleteFailures.Add([pscustomobject]@{
            path = $target
            error = $_.Exception.Message
        })
    }
}

if ($deleteFailures.Count -gt 0) {
    $deleteFailures | Format-Table -AutoSize | Out-Host
    throw "Cleanup was incomplete: $($deleteFailures.Count) top-level target(s) could not be removed. Repair their ACLs, then rerun the same apply command."
}

Write-Host "Legacy output cleanup complete. The two output roots were retained and emptied."

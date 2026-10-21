#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Build check for the UI control registry and the generated UI reference (Spec 280 T117/T118).

.DESCRIPTION
    Enforces ui-manager.md §7: "A control that is not in the registry is not allowed to ship."

    The check runs three linked steps and exits non-zero if any of them fails:

      1. Regenerate the control inventory from the viewer source
         (eng/ui/Get-UiControlInventory.ps1).
      2. Regenerate the UI control registry from that inventory into a scratch copy
         (eng/ui/New-UiControlRegistry.ps1) and compare it with the committed registry
         (eng/ui/ui-controls.json).
         - A visible control that is missing from the committed registry is a FAILURE.
         - A registry entry that no longer matches a control in the source is a WARNING
           (hand-authored help text stays authoritative; the entry is simply stale).
      3. Verify that the committed generated document (docs/WoWViewer/UI-REFERENCE.md) still
         matches the committed registry (eng/ui/Update-UiReference.ps1, compare mode).

    The script never writes to the committed registry or the committed document. The scratch
    copy lives in eng/ui/.verify-scratch and is removed on exit.

    All paths are resolved from this script's own location, so the repository can be checked
    out at any path. The script only needs PowerShell 7; it runs no build and no client.

.PARAMETER RepositoryRoot
    Repository root. Defaults to the parent of the parent of the parent of this script.

.PARAMETER ResultDirectory
    Optional directory (relative to the current directory) that receives the machine-readable
    result of the run: "ui-reference-check.started", "ui-reference-check.pass" or
    "ui-reference-check.fail", plus the "ui-reference-check.log" transcript. The build target
    uses these files so that a machine without PowerShell 7 can skip the check instead of
    failing the build.

.EXAMPLE
    pwsh -NoProfile -File wow-viewer/eng/ui/Verify-UiReference.ps1
    Standalone drift check. Exits 0 when the registry and the generated reference match the
    viewer source, 1 otherwise.

.EXAMPLE
    pwsh -NoProfile -File wow-viewer/eng/ui/Verify-UiReference.ps1 -RepositoryRoot I:\parp\parp-tools
#>
[CmdletBinding()]
param(
    [string] $RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path,
    [string] $ResultDirectory = ''
)

$ErrorActionPreference = 'Stop'

$inventoryScript = Join-Path $PSScriptRoot 'Get-UiControlInventory.ps1'
$registrySeeder = Join-Path $PSScriptRoot 'New-UiControlRegistry.ps1'
$referenceScript = Join-Path $PSScriptRoot 'Update-UiReference.ps1'

$registryPath = Join-Path $RepositoryRoot 'wow-viewer/eng/ui/ui-controls.json'
$documentPath = Join-Path $RepositoryRoot 'wow-viewer/docs/WoWViewer/UI-REFERENCE.md'
# The scratch copy is per process: a solution build can run several viewer projects (and
# several shells) at the same time, and they must not delete each other's working copy.
$scratchRoot = Join-Path $RepositoryRoot ('wow-viewer/eng/ui/.verify-scratch-' + $PID)

$failures = [System.Collections.Generic.List[string]]::new()
$warnings = [System.Collections.Generic.List[string]]::new()
$transcript = [System.Collections.Generic.List[string]]::new()

$resultDirectoryFull = ''
if (-not [string]::IsNullOrWhiteSpace($ResultDirectory))
{
    $resultDirectoryFull = if ([System.IO.Path]::IsPathRooted($ResultDirectory))
    {
        $ResultDirectory
    }
    else
    {
        Join-Path (Get-Location).Path $ResultDirectory
    }
}

function Write-Transcript
{
    param([Parameter(Mandatory = $true)][string] $Text)
    $transcript.Add($Text)
    Write-Output $Text
}

function Write-Failure
{
    param([Parameter(Mandatory = $true)][string] $Text)
    $transcript.Add($Text)
    [Console]::Error.WriteLine($Text)
}

function Write-ResultFile
{
    param(
        [Parameter(Mandatory = $true)][string] $Name,
        [string] $Content = ''
    )

    if ([string]::IsNullOrWhiteSpace($resultDirectoryFull)) { return }

    try
    {
        if (-not (Test-Path -LiteralPath $resultDirectoryFull))
        {
            New-Item -ItemType Directory -Force -Path $resultDirectoryFull | Out-Null
        }

        Set-Content -LiteralPath (Join-Path $resultDirectoryFull $Name) -Value $Content -Encoding utf8 -NoNewline
    }
    catch
    {
        # A missing result file must never change the check outcome.
    }
}

# The current host executable is used for child script invocations so the check works under
# PowerShell 7 even when "pwsh" is not on PATH for the child process.
$hostExecutable = 'pwsh'
try
{
    $current = (Get-Process -Id $PID).Path
    if (-not [string]::IsNullOrWhiteSpace($current)) { $hostExecutable = $current }
}
catch { $hostExecutable = 'pwsh' }

function Invoke-Script
{
    param(
        [Parameter(Mandatory = $true)][string] $ScriptPath,
        [string[]] $Arguments = @()
    )

    $output = & $hostExecutable -NoProfile -File $ScriptPath @Arguments 2>&1
    $exitCode = $LASTEXITCODE

    $lines = @()
    foreach ($line in @($output))
    {
        $text = $line.ToString()
        if ($text -match '^\s*Write-Error:') { $text = $text -replace '^\s*Write-Error:\s*', '' }
        if (-not [string]::IsNullOrWhiteSpace($text)) { $lines += $text }
    }

    return [pscustomobject]@{
        ExitCode = $exitCode
        Output = ($lines -join [Environment]::NewLine)
    }
}

function Get-ControlMap
{
    param([Parameter(Mandatory = $true)] $Registry)

    $map = @{}
    foreach ($control in @($Registry.controls))
    {
        $map[[string]$control.label] = $control
    }

    return $map
}

Write-ResultFile -Name 'ui-reference-check.started' -Content 'started'
Write-ResultFile -Name 'ui-reference-check.log' -Content ''

try
{
    foreach ($required in @($inventoryScript, $registrySeeder, $referenceScript, $registryPath))
    {
        if (-not (Test-Path -LiteralPath $required))
        {
            Write-Failure ("UI_REFERENCE_CHECK ERROR: cannot run, missing: {0}" -f $required)
            exit 1
        }
    }

    # ------------------------------------------------------------ step 1: source inventory
    $inventoryRun = Invoke-Script -ScriptPath $inventoryScript -Arguments @('-RepositoryRoot', $RepositoryRoot)
    if ($inventoryRun.ExitCode -ne 0)
    {
        Write-Failure 'UI_REFERENCE_CHECK ERROR: the control inventory could not be regenerated.'
        Write-Failure $inventoryRun.Output
        exit 1
    }

    $inventoryRows = @($inventoryRun.Output -split "`r?`n" | Where-Object { $_.Trim() })
    if ($inventoryRows.Count -eq 0)
    {
        Write-Failure ("UI_REFERENCE_CHECK ERROR: the control inventory is empty. The viewer source was not found under {0}." -f (Join-Path $RepositoryRoot 'wow-viewer/src/viewer/WoWViewer'))
        exit 1
    }

    $inventoryLabels = @{}
    foreach ($row in $inventoryRows)
    {
        $parts = $row -split '\|'
        if ($parts.Count -lt 4) { continue }
        if (-not $inventoryLabels.ContainsKey($parts[0]))
        {
            $inventoryLabels[$parts[0]] = ('{0}:{1}' -f $parts[1], $parts[2])
        }
    }

    # ------------------------------------------ step 2: regenerate the registry and compare it
    try
    {
        if (Test-Path -LiteralPath $scratchRoot) { Remove-Item -LiteralPath $scratchRoot -Recurse -Force }

        $scratchInventory = Join-Path $scratchRoot 'wow-viewer/specs/280-ui-audit-refactor/evidence/ui-control-inventory.txt'
        $scratchRegistryDirectory = Join-Path $scratchRoot 'wow-viewer/eng/ui'
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $scratchInventory) | Out-Null
        New-Item -ItemType Directory -Force -Path $scratchRegistryDirectory | Out-Null

        $inventoryRows | Set-Content -LiteralPath $scratchInventory -Encoding utf8
        Copy-Item -LiteralPath $registryPath -Destination (Join-Path $scratchRegistryDirectory 'ui-controls.json') -Force

        $seedRun = Invoke-Script -ScriptPath $registrySeeder -Arguments @('-RepositoryRoot', $scratchRoot)
        if ($seedRun.ExitCode -ne 0)
        {
            Write-Failure 'UI_REFERENCE_CHECK ERROR: the UI control registry could not be regenerated from the current source inventory.'
            Write-Failure $seedRun.Output
            exit 1
        }

        $candidate = Get-Content -LiteralPath (Join-Path $scratchRegistryDirectory 'ui-controls.json') -Raw | ConvertFrom-Json
        $committed = Get-Content -LiteralPath $registryPath -Raw | ConvertFrom-Json

        $candidateMap = Get-ControlMap -Registry $candidate
        $committedMap = Get-ControlMap -Registry $committed

        foreach ($label in ($candidateMap.Keys | Sort-Object))
        {
            if (-not $committedMap.ContainsKey($label))
            {
                $location = if ($inventoryLabels.ContainsKey($label)) { $inventoryLabels[$label] } else { 'unknown source' }
                $failures.Add(("visible control '{0}' has no UI control registry entry (source: {1}). Add it to wow-viewer/eng/ui/ui-controls.json and regenerate the reference." -f $label, $location))
                continue
            }

            $committedControl = $committedMap[$label]
            $candidateControl = $candidateMap[$label]
            foreach ($field in @('region', 'help'))
            {
                if ([string]$committedControl.$field -cne [string]$candidateControl.$field)
                {
                    $failures.Add(("registry entry '{0}' has a stale {1}: committed '{2}', regenerated '{3}'." -f $label, $field, $committedControl.$field, $candidateControl.$field))
                }
            }

            $committedShortcut = if ($null -eq $committedControl.shortcut) { '' } else { [string]$committedControl.shortcut }
            $candidateShortcut = if ($null -eq $candidateControl.shortcut) { '' } else { [string]$candidateControl.shortcut }
            if ($committedShortcut -cne $candidateShortcut)
            {
                $failures.Add(("registry entry '{0}' has a stale shortcut: committed '{1}', regenerated '{2}'." -f $label, $committedShortcut, $candidateShortcut))
            }
        }

        foreach ($label in ($committedMap.Keys | Sort-Object))
        {
            if (-not $candidateMap.ContainsKey($label))
            {
                $warnings.Add(("registry entry '{0}' no longer matches a visible control in the manager-owned sources. Remove it from wow-viewer/eng/ui/ui-controls.json and regenerate the reference." -f $label))
            }
        }
    }
    finally
    {
        if (Test-Path -LiteralPath $scratchRoot) { Remove-Item -LiteralPath $scratchRoot -Recurse -Force }
    }

    # ------------------------------- step 3: the generated reference must match the registry
    $documentRun = Invoke-Script -ScriptPath $referenceScript -Arguments @('-RepositoryRoot', $RepositoryRoot)
    if ($documentRun.ExitCode -ne 0)
    {
        $failures.Add('the generated UI reference is out of date: docs/WoWViewer/UI-REFERENCE.md no longer matches the committed UI control registry. Run: pwsh -NoProfile -File wow-viewer/eng/ui/Update-UiReference.ps1 -Write')
    }

    # ---------------------------------------------------------- summary and exit status
    $committedCount = @((Get-Content -LiteralPath $registryPath -Raw | ConvertFrom-Json).controls).Count
    Write-Transcript ("UI_REFERENCE_CHECK inventory_rows={0} distinct_labels={1} registry_entries={2}" -f $inventoryRows.Count, $inventoryLabels.Count, $committedCount)

    foreach ($warning in $warnings) { Write-Transcript ("UI_REFERENCE_CHECK WARNING: {0}" -f $warning) }

    if ($failures.Count -gt 0)
    {
        foreach ($failure in $failures) { Write-Failure ("UI_REFERENCE_CHECK ERROR: {0}" -f $failure) }
        Write-Transcript ("UI_REFERENCE_CHECK FAILED errors={0} warnings={1}" -f $failures.Count, $warnings.Count)
        Write-ResultFile -Name 'ui-reference-check.fail' -Content ("errors={0}" -f $failures.Count)
        exit 1
    }

    Write-Transcript ("UI_REFERENCE_CHECK PASSED errors=0 warnings={0}" -f $warnings.Count)
    Write-ResultFile -Name 'ui-reference-check.pass' -Content 'pass'
    exit 0
}
catch
{
    Write-Failure ("UI_REFERENCE_CHECK ERROR: the check failed with an unexpected error: {0}" -f $_.Exception.Message)
    Write-ResultFile -Name 'ui-reference-check.fail' -Content 'unexpected error'
    exit 1
}
finally
{
    Write-ResultFile -Name 'ui-reference-check.log' -Content ($transcript -join [Environment]::NewLine)
}

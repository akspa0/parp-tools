#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Lists every visible interactive control in the manager-owned viewer surfaces.

.DESCRIPTION
    The UI reference (Spec 280) is generated from a registry, and the registry is
    checked against the source so a visible control cannot ship undocumented. This
    script extracts the visible control labels from the shell surfaces the UI manager
    owns: the top menu, the bottom bar and options, the minimap, the chat dock, the
    left navigator, and the right inspector.

    Output is deterministic: one line per control, "label|file|line|kind".

.EXAMPLE
    pwsh -NoProfile -File wow-viewer/eng/ui/Get-UiControlInventory.ps1
#>
[CmdletBinding()]
param(
    [string] $RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
)

$ErrorActionPreference = 'Stop'

# Forward slashes keep the script usable on non-Windows hosts (PowerShell 7 on Linux/macOS).
$viewerRoot = Join-Path $RepositoryRoot 'wow-viewer/src/viewer/WoWViewer'

# Surfaces owned by the UI manager (Spec 280 ui-manager.md §2).
$coveredRoots = @(
    'Workbench/Services/UiManager',
    'Workbench/Services/MainMenu',
    'Workbench/Services/Chrome',
    'Workbench/Services/Navigator',
    'Workbench/Services/WorkbenchPanels'
)

# The visible label is the first argument. It may be a plain literal or a conditional
# whose branches are literals, so every non-interpolated literal in the first argument
# counts. This mirrors UiControlScanner in src/viewer/WoWViewer/UI/UiControlScanner.cs.
$callPattern = 'ImGui\.(Button|SmallButton|Checkbox|RadioButton|MenuItem|Selectable|TreeNode|CollapsingHeader|Combo|SliderFloat|SliderInt)\(([^;]*)\)'
$actionBarIconPattern = 'DrawActionBarIcon\(([^;]*)\)'
$widgetPattern = 'SharedUiWidgets\.(ActionButton|DrawActionGroup|SectionHeader|DrawSectionHeader)\(([^;]*)\)'
$literalPattern = '(?<!\$)"((?:[^"\\]|\\.)*)"'

function Get-FirstArgumentLabels {
    param([string] $Arguments)

    $depth = 0
    $end = $Arguments.Length
    for ($index = 0; $index -lt $Arguments.Length; $index++)
    {
        $char = $Arguments[$index]
        if ($char -eq '(') { $depth++ }
        elseif ($char -eq ')') { $depth-- }
        elseif ($char -eq ',' -and $depth -eq 0) { $end = $index; break }
    }

    $first = $Arguments.Substring(0, [Math]::Max(0, $end))
    $labels = [System.Collections.Generic.List[string]]::new()
    foreach ($match in [regex]::Matches($first, $literalPattern))
    {
        $label = ($match.Groups[1].Value -replace '##.*$', '').Trim()
        if ([string]::IsNullOrWhiteSpace($label)) { continue }
        $labels.Add($label)
    }
    return $labels
}

$rows = [System.Collections.Generic.List[string]]::new()

foreach ($relative in $coveredRoots)
{
    $directory = Join-Path $viewerRoot $relative
    if (-not (Test-Path -LiteralPath $directory)) { continue }

    foreach ($file in Get-ChildItem -LiteralPath $directory -Filter '*.cs' -File | Sort-Object FullName)
    {
        $lines = Get-Content -LiteralPath $file.FullName
        for ($index = 0; $index -lt $lines.Count; $index++)
        {
            $line = $lines[$index]
            foreach ($pattern in @($callPattern, $widgetPattern))
            {
                foreach ($match in [regex]::Matches($line, $pattern))
                {
                    $kind = $match.Groups[1].Value
                    $relativePath = $file.FullName.Substring($viewerRoot.Length).TrimStart([char]'\', [char]'/')
                    foreach ($label in (Get-FirstArgumentLabels -Arguments $match.Groups[2].Value))
                    {
                        $rows.Add(('{0}|{1}|{2}|{3}' -f $label, $relativePath.Replace('\', '/'), ($index + 1), $kind))
                    }
                }
            }
            foreach ($match in [regex]::Matches($line, $actionBarIconPattern))
            {
                $relativePath = $file.FullName.Substring($viewerRoot.Length).TrimStart([char]'\', [char]'/')
                foreach ($label in (Get-FirstArgumentLabels -Arguments $match.Groups[1].Value))
                {
                    $rows.Add(('{0}|{1}|{2}|ActionBarIcon' -f $label, $relativePath.Replace('\', '/'), ($index + 1)))
                }
            }
        }
    }
}

$rows | Sort-Object -Unique

#!/usr/bin/env pwsh
<#
.SYNOPSIS
    One-off seeder that writes the UI control registry (wow-viewer/eng/ui/ui-controls.json).

.DESCRIPTION
    The registry is the source of truth for the generated UI reference. This script
    seeds it from the audited control inventory plus an authored help table, then
    it is not needed again: edit the JSON directly for later changes.
#>
[CmdletBinding()]
param(
    [string] $RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
)

$ErrorActionPreference = 'Stop'

$help = @{
    'UI Reference' = @('Top menu > Help', $null, 'Open the searchable list of every documented control.')
    'Fog' = @('Left sidebar > Workspace', $null, 'Fog start and end distance for the scene.')
    'Expand chat' = @('Chat', $null, 'Expand the collapsed chat dock.')
    'Base Layer (L0)' = @('Options > Layers', $null, 'Show the base terrain layer (L0).')
    'Model WF' = @('Options > Objects', $null, 'Wireframe overlay for the standalone model.')
    'WMO WF' = @('Options > Objects', $null, 'Wireframe overlay for the standalone WMO.')
    'Collapse' = @('Minimap', 'M', 'Collapse the minimap to its small header button.')
    'Collapse left sidebar' = @('Left sidebar', '[', 'Collapse the left sidebar to its rail.')
    'Collapse right sidebar' = @('Right sidebar', ']', 'Collapse the right sidebar to its rail.')
    'Expand left sidebar' = @('Left sidebar', '[', 'Expand the left sidebar.')
    'Expand right sidebar' = @('Right sidebar', ']', 'Expand the right sidebar.')
    '3D Cursor' = @('Options > Objects', $null, 'Toggle the in-scene 3D cursor used for placement aiming.')
    'Reload cursor model' = @('Options > Objects', $null, 'Re-resolve the client cursor.mdx/cursor.m2 model from the active data source.')
    'Show minimap' = @('Options > Display > Minimap', $null, 'Show or hide the minimap panel.')
    'Collapse minimap' = @('Options > Display > Minimap', $null, 'Collapse the minimap panel to a one-line bar.')
    'Dock minimap in right sidebar' = @('Options > Display > Minimap', $null, 'Dock the minimap at the top of the right sidebar instead of floating it.')
    'Map Spawns' = @('Chat > Tools', $null, 'Open the current map spawn list and population controls.')
    'Minimap' = @('Chat > Tools', 'M', 'Show or hide the minimap panel.')
    'Inspector' = @('Chat > Tools', 'I', 'Open the Inspector in the right sidebar.')
    'PM4' = @('Chat > Tools', 'P', 'Open the PM4 workbench in the right sidebar.')
    'Clear chat' = @('Chat > Tools > Barkers', $null, 'Clear the chat log.')
    'About' = @('Top menu > Help', $null, 'Show viewer version, credits and project links.')
    'ADT Detail Tiles' = @('Viewer > World', $null, 'How many terrain tiles stay resident around the camera.')
    'ADT Retain Radius' = @('Viewer > World', $null, 'Radius in tiles kept resident for terrain streaming.')
    'All Layers' = @('Options > Layers', $null, 'Show every terrain layer at once.')
    'Alpha' = @('Options > Overlays', $null, 'Show the terrain alpha (opacity) map overlay.')
    'Anim' = @('Options > Objects', $null, 'Advance animations on world doodad MDX/M2 models.')
    'Archaeology Workspace' = @('Left sidebar > Archaeology', $null, 'Current map, analysis scope and playback state.')
    'Assets' = @('Left sidebar > Viewer', $null, 'Asset catalog for the loaded data source.')
    'Attach Loose Map Folder...' = @('Top menu > File', $null, 'Attach a loose map folder to the active MPQ archive.')
    'Audio' = @('Options > Audio', $null, 'Audio output and per-zone audio settings.')
    'Auto' = @('Viewer > World', $null, 'Use automatic detail-tile and retention defaults.')
    'Automatic 24-minute cycle' = @('Viewer > World', $null, 'Advance the world time of day on the in-game day cycle.')
    'Base Only' = @('Options > Layers', $null, 'Show only the base terrain layer.')
    'Bounding Boxes' = @('Options > Overlays', $null, 'Draw world object bounding boxes.')
    'Camera Speed' = @('Viewer > World', $null, 'Camera movement speed for keyboard and drag navigation.')
    'Cancel' = @('Popup', 'Esc', 'Close the dialog without applying anything.')
    'Cells' = @('Options > Grids', $null, 'Show the terrain chunk cell grid.')
    'Chunks' = @('Options > Grids', $null, 'Show the terrain chunk grid.')
    'Close Options' = @('Options', 'Esc', 'Close the options sub-menu.')
    'Close' = @('Popup', 'Esc', 'Close the dialog.')
    'Color NPC/GO BBs' = @('Viewer > World', $null, 'Colour bounding boxes by NPC and GameObject spawn type.')
    'Color Spawns' = @('Options > Overlays', $null, 'Colour spawn markers by type.')
    'Contours' = @('Options > Overlays', $null, 'Show terrain height contours.')
    'Copy Path' = @('Left sidebar > File Browser', $null, 'Copy the selected file path to the clipboard.')
    'DAT Conversion' = @('Viewer > World', $null, 'Export the loaded DAT map or nearby tiles as DAT v26.')
    'Detailed graph attribution (costs frame time)' = @('Viewer > World', $null, 'Attribute frame time to detailed scene-graph nodes.')
    'Diagnostics' = @('Left sidebar', $null, 'Log, performance counters and runtime statistics.')
    'Display' = @('Options', $null, 'Render quality, lighting and theme settings.')
    'Dynamic Hover Range' = @('Viewer > World', $null, 'Scale the hover/pick range with the camera distance.')
    'Editor Workspace' = @('Left sidebar > Editor', $null, 'Editor session, undo state and output folder.')
    'Export Loaded DAT Map...' = @('Viewer > World', $null, 'Export the loaded DAT map with the existing exporter.')
    'Export Nearby Tiles as DAT v26 (experimental)' = @('Viewer > World', $null, 'Export nearby tiles as DAT v26 (experimental format path).')
    'File Browser' = @('Left sidebar', $null, 'Browse the active data source and open assets.')
    'Fog Objects' = @('Viewer > World', $null, 'Apply distance fog to world objects.')
    'FOV' = @('Viewer > World', $null, 'Camera field of view in degrees.')
    'Grids' = @('Options', $null, 'Terrain tile, chunk and cell grid overlays.')
    'Group Names' = @('Options > Objects', $null, 'Show WMO group names in the overlay.')
    'Hide All' = @('Options > Layers', $null, 'Hide every terrain layer.')
    'Hide UI Chrome' = @('Top menu > View', 'Tab', 'Hide or show all viewer chrome.')
    'Holes' = @('Options > Layers', $null, 'Mask terrain holes (missing chunks) in the render.')
    'Hover Tooltips' = @('Viewer > World', $null, 'Show hover tooltips for the asset under the pointer.')
    'Hover/Pick Range' = @('Viewer > World', $null, 'Maximum hover and click-pick distance in yards.')
    'Keyboard Shortcuts' = @('Top menu > Help', 'F1', 'List the registered keyboard shortcuts.')
    'Launch Map Converter' = @('Editor > Converters', $null, 'Open the map converter dialog.')
    'Launch WMO Converter' = @('Editor > Converters', $null, 'Open the WMO converter dialog.')
    'Layer Projects' = @('Left sidebar > Archaeology', $null, 'Save and load layer projects.')
    'Layers' = @('Options', $null, 'Terrain layer visibility and WDL overlay.')
    'Left Sidebar' = @('Top menu > View', '[', 'Show or hide the left sidebar.')
    'Lighting' = @('Options > Display', $null, 'Scene lighting diagnostics and settings.')
    'Limit Hover/Pick Range' = @('Viewer > World', $null, 'Restrict hover and pick to a bounded distance.')
    'Load Area POIs' = @('Viewer > World', $null, 'Load area points of interest for the current map.')
    'Load AreaTriggers' = @('Viewer > World', $null, 'Load area trigger volumes for the current map.')
    'Log' = @('Left sidebar > Diagnostics', $null, 'Open the viewer log window.')
    'M2/WMO WF' = @('Options > Objects', $null, 'Wireframe overlay for M2 doodads and WMO buildings.')
    'Map Converter' = @('Editor > Converters', $null, 'Convert maps with the existing converter service.')
    'MCCV' = @('Options > Overlays', $null, 'Show the terrain vertex colour (MCCV) overlay.')
    'Objects' = @('Options', $null, 'Object transparency, wireframes, animation and 3D cursor.')
    'Open Alpha WDT (loose map)...' = @('Top menu > File', $null, 'Open a loose alpha-era WDT map file.')
    'Open Cartography' = @('Left sidebar > Archaeology', $null, 'Open the archaeology cartography page.')
    'Open CASC Install (local + CDN fill)...' = @('Top menu > File', $null, 'Open a local CASC install and fill missing files from the CDN.')
    'Open CASC Install (local)...' = @('Top menu > File', $null, 'Open a local CASC install.')
    'Open DAT Terrain Folder...' = @('Top menu > File', $null, 'Open a folder of DAT terrain files.')
    'Open File...' = @('Top menu > File', $null, 'Open a single file through the shared path picker.')
    'Open Game Folder (MPQ)...' = @('Top menu > File', $null, 'Open a game folder as an MPQ data source.')
    'Open Game Folder...' = @('Left sidebar > Workspace', $null, 'Open a game folder as an MPQ data source.')
    'Open Loose PM4 / PD4 File...' = @('Top menu > File', $null, 'Open a loose PM4 or PD4 file.')
    'Open Selected' = @('Left sidebar > File Browser', $null, 'Open the selected file from the data source.')
    'Options' = @('Bottom bar', $null, 'Open the options sub-menu next to the chat.')
    'Overlays' = @('Options', $null, 'Surface overlays: alpha, shadows, MCCV, contours, liquid, WL, BBs, PM4.')
    'Performance' = @('Left sidebar > Diagnostics', $null, 'Open the performance window.')
    'PM4 Overlay' = @('Options > Overlays', $null, 'Show the PM4 surface overlay in the scene.')
    'Project Output' = @('Left sidebar > Editor', $null, 'Current editor project folder and new-folder action.')
    'Quit' = @('Top menu > File', $null, 'Close the viewer.')
    'Remove' = @('Popup', $null, 'Remove the saved known-good base.')
    'Render Quality' = @('Options > Display', $null, 'Render quality settings for the scene.')
    'Reset Camera' = @('Top menu > View', $null, 'Reset the camera to the default view.')
    'Reset load phases' = @('Viewer > World', $null, 'Reset the staged load phases to the start.')
    'Reset Layout Only' = @('Top menu > View', $null, 'Restore the default panel layout and close tool windows. Text size and the loaded client/map are unchanged.')
    'Reset Text Size Only' = @('Top menu > View', $null, 'Return the user text size to 100%. Layout and the loaded client/map are unchanged; high-resolution displays keep their automatic sizing.')
    'Reset Layout and Text Size' = @('Top menu > View', $null, 'Restore the default panel layout and return the text size to 100%. The loaded client/map is unchanged.')
    'Text Scale' = @('Top menu > View > Text Size', $null, 'Interface text size from 0.85x to 2.20x, shared with Settings. The 100% to 175% presets sit below it; the current size is checked.')
    'Visual Options...' = @('Top menu > View', $null, 'Open Options: render quality, lighting, theme, minimap, overlays, layers, grids, objects, audio and simulation.')
    'Camera and Fog...' = @('Top menu > View', $null, 'Open Workspace: camera, fog and client source controls.')
    'Return To Last World' = @('Left sidebar > Workspace', $null, 'Return to the last loaded world.')
    'Right Sidebar' = @('Top menu > View', ']', 'Show or hide the right sidebar.')
    'Save Current Game Folder As Known-Good Base' = @('Top menu > File', $null, 'Remember the current game folder as a known-good base.')
    'Scene Tools' = @('Left sidebar > Viewer', $null, 'Taxi routes, capture automation and camera paths.')
    'Selected Object' = @('Viewer > Inspector', $null, 'Details of the currently selected object.')
    'Settings...' = @('Top menu > File', $null, 'Open the viewer settings.')
    'Shadows' = @('Options > Overlays', $null, 'Show terrain shadow overlays.')
    'Show Selected Object Bounds' = @('Viewer > World', $null, 'Draw the bounds of the selected object.')
    'Show WDL Terrain' = @('Viewer > World', $null, 'Show the WDL terrain layer.')
    'Sidebar width' = @('Left sidebar', $null, 'Width of the left sidebar in pixels.')
    'Simulation' = @('Options', $null, 'Population streaming and spawn display settings.')
    'Spawn BBs Only' = @('Options > Simulation', $null, 'Show bounding boxes for spawns only.')
    'Terrain WF' = @('Options > Objects', $null, 'Wireframe overlay for terrain.')
    'Theme' = @('Options > Display', $null, 'Viewer colour theme.')
    'Tiles' = @('Options > Grids', $null, 'Show the terrain tile grid.')
    'Time of Day' = @('Viewer > World', $null, 'World time of day in hours.')
    'Use For Taxi Override' = @('Left sidebar > File Browser', $null, 'Use the selected path as the taxi actor model override.')
    'WDL' = @('Options > Layers', $null, 'Show the WDL terrain layer in the viewport.')
    'WMO Converter' = @('Editor > Converters', $null, 'Convert WMO files with the existing converter service.')
    'WMO Group BBs' = @('Options > Objects', $null, 'Draw WMO group bounding boxes.')
    'Workspace' = @('Left sidebar', $null, 'Open data and return to the last world.')
    'World Bounding Boxes' = @('Viewer > World', $null, 'Draw world object bounding boxes.')
    'World Overview' = @('Left sidebar', $null, 'Map list and map export.')
}

$inventoryPath = Join-Path $RepositoryRoot 'wow-viewer/specs/280-ui-audit-refactor/evidence/ui-control-inventory.txt'
if (-not (Test-Path -LiteralPath $inventoryPath)) { throw "Missing control inventory: $inventoryPath" }

$rows = Get-Content -LiteralPath $inventoryPath | Where-Object { $_.Trim() } | ForEach-Object {
    $parts = $_ -split '\|'
    [pscustomobject]@{ Label = $parts[0]; File = $parts[1]; Line = [int]$parts[2]; Kind = $parts[3] }
}

# Existing registry entries win: the JSON file is the source of truth for help text.
$existing = @{}
$registryPath = Join-Path $RepositoryRoot 'wow-viewer/eng/ui/ui-controls.json'
if (Test-Path -LiteralPath $registryPath)
{
    $previous = Get-Content -LiteralPath $registryPath -Raw | ConvertFrom-Json
    foreach ($control in @($previous.controls)) { $existing[$control.label] = $control }
}

$missing = [System.Collections.Generic.List[string]]::new()
$entries = [System.Collections.Generic.List[object]]::new()
$seen = @{}
foreach ($row in $rows)
{
    $key = $row.Label
    if ($seen.ContainsKey($key)) { continue }
    $seen[$key] = $true
    if ($existing.ContainsKey($key)) { $meta = $null; $previous = $existing[$key] }
    else
    {
        $meta = $help[$key]
        if ($null -eq $meta)
        {
            $missing.Add($key)
            continue
        }
    }
    if ($null -ne $meta)
    {
        $id = ($key -replace '[^A-Za-z0-9]+', '-').ToLowerInvariant().Trim('-')
        if ([string]::IsNullOrWhiteSpace($id)) { $id = 'control-' + ($entries.Count + 1) }
        $entries.Add([ordered]@{
            id = $id
            label = $key
            region = $meta[0]
            shortcut = $meta[1]
            help = $meta[2]
            visible = $true
        })
    }
    else
    {
        $entries.Add([ordered]@{
            id = $previous.id
            label = $key
            region = $previous.region
            shortcut = $previous.shortcut
            help = $previous.help
            visible = $previous.visible
        })
    }
}

if ($missing.Count -gt 0)
{
    Write-Error ("Controls without a registry entry: " + ($missing -join ', '))
    exit 1
}

$registry = [ordered]@{
    version = 1
    generatedFrom = 'wow-viewer/src/viewer/WoWViewer/Workbench/Services (manager-owned surfaces)'
    controls = $entries
}

$outputPath = Join-Path $RepositoryRoot 'wow-viewer/eng/ui/ui-controls.json'
$registry | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $outputPath -Encoding utf8
Write-Output ("Wrote {0} entries to {1}" -f $entries.Count, $outputPath)

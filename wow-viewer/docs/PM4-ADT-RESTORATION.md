# PM4 → ADT Restoration Guide

How to extract PM4 object data and write it into viewable ADT files.

## What This Does

PM4 files contain collision/pathfinding surfaces for M2 and WMO objects. This pipeline:

1. Reads PM4 segments (CK24 grouping, TypeFlags classification)
2. Reads _obj0.adt placements (MDDF + MODF entries from the game client)
3. Matches PM4 segments to placements (shape, footprint, TypeFlags scoring)
4. Writes a new LK ADT file with matched placements
5. The viewer can open the written ADT to see restored objects

## Prerequisites

- .NET 10 SDK
- A staged game client under `output/tmp/wowarchive-clients/` (e.g. `3_3_5_12340`)
- PM4 files (loose or extracted from game data)
- Corresponding `_obj0.adt` files with M2/WMO placement data

## Quick Start: Single Tile

```powershell
cd I:\parp\parp-tools

# Build the inspect tool
dotnet build wow-viewer/tools/inspect/WowViewer.Tool.Inspect -c Debug

# Write one tile's placements to ADT
dotnet run --project wow-viewer/tools/inspect/WowViewer.Tool.Inspect -c Debug -- pm4 write-adt `
  --input "path\to\development_00_00.pm4" `
  --archive-root "output\tmp\wowarchive-clients\3_3_5_12340" `
  --output "output\pm4restored_0_0.adt"
```

The tool auto-derives tile coordinates from the PM4 filename and finds the matching `_obj0.adt` in the same directory.

## Quick Start: All Tiles in a Directory

```powershell
$pm4Dir = "path\to\pm4\directory"
$archiveRoot = "output\tmp\wowarchive-clients\3_3_5_12340"
$outputDir = "output\pm4restored"

New-Item -ItemType Directory -Path $outputDir -Force

Get-ChildItem -Path $pm4Dir -Filter "*.pm4" | ForEach-Object {
    $baseName = $_.BaseName
    if ($baseName -match '_(\d+)_(\d+)$') {
        $tileX = $matches[1]
        $tileY = $matches[2]
        $output = Join-Path $outputDir "pm4restored_${tileX}_${tileY}.adt"
        
        Write-Host "Processing $baseName -> $output"
        dotnet run --project wow-viewer/tools/inspect/WowViewer.Tool.Inspect -c Debug -- pm4 write-adt `
            --input $_.FullName `
            --archive-root $archiveRoot `
            --output $output
    }
}
```

## Viewing Results

Open the written ADT files in the viewer:

```powershell
dotnet run --project wow-viewer/src/viewer/WoWViewer -c Debug -- <client-root> <map-name>
```

Or use the `map inspect` command to verify the ADT structure:

```powershell
dotnet run --project wow-viewer/tools/inspect/WowViewer.Tool.Inspect -c Debug -- map inspect --input "output\pm4restored_0_0.adt"
```

Expected output shows MDDF (M2 placements) and MODF (WMO placements) chunks with the correct counts.

## CLI Reference

### `pm4 write-adt`

```
pm4 write-adt --input <file.pm4> --archive-root <client-dir> [options]

Options:
  --input, -i        PM4 input file (required)
  --archive-root, -r Staged game client root (required)
  --placements, -p   _obj0.adt file (auto-derived if omitted)
  --output, -o       Output ADT path (auto-generated if omitted)
  --map-name, -m     Map name in ADT header (default: "pm4restored")
```

### `pm4 match` (diagnostic)

Shows the raw match data without writing an ADT:

```
pm4 match --input <file.pm4> --archive-root <client-dir> [--placements <obj0.adt>] [--output <report.json>]
```

### `pm4 export-segments` (diagnostic)

Exports PM4 segments to JSON for inspection:

```
pm4 export-segments --input <file.pm4|directory> [--output <report.json>]
```

## TypeFlags Classification

PM4 surfaces are classified by `MSLK.TypeFlags`:

| TypeFlags | Label | Asset Kind |
|-----------|-------|------------|
| 0x03 | M2 top surfaces | M2 (doodads) |
| 0x10 | Interior WMO floors | WMO |
| 0x12 | Exterior WMO solid | WMO |

CK24 types determine the asset kind:

| CK24 Type | Asset Kind |
|-----------|------------|
| 0x40, 0x41, 0xC0-0xC3 | M2 |
| 0x42, 0x43 | WMO |

## Troubleshooting

### "Error: --archive-root is required"

The tool needs the staged game client to resolve M2/WMO model paths from _obj0.adt. Point `--archive-root` to the staged client directory.

### "Error: placement source does not exist"

The tool auto-derives the _obj0.adt path from the PM4 filename. If the naming convention doesn't match (`<map>_<x>_<y>.pm4`), provide `--placements` explicitly.

### 0 placements written

If M2/WMO assets can't be resolved from the archive root, the match falls back to ±2 unit default bounds. The placements still get written, but they may not render at correct positions. Check that the archive root contains the expected game data.

### File naming conventions

PM4 files: `<map>_<tileX>_<tileY>.pm4` (e.g. `development_00_00.pm4`)
_obj0.adt files: `<map>_<tileX>_<tileY>_obj0.adt` (e.g. `development_0_0_obj0.adt`)

Note: PM4 uses zero-padded coordinates, _obj0.adt uses unpadded. The tool handles this automatically.

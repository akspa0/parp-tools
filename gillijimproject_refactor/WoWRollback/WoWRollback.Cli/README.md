# WoWRollback.Cli

## Overview
Command-line interface for the WoWRollback toolkit. Provides:
- analyze-map-adts (analyze loose ADTs, generate viewer)
- alpha-to-lk (Alpha WDT → LK ADTs with burial, holes, MCSH, AreaIDs)
- lk-to-alpha (patch existing LK ADTs with same terrain logic)
- serve-viewer (Kestrel-based static server)
- probe-archive / probe-minimap (diagnostics)

## Prerequisites
- .NET SDK 9.0
- Optional: LK client root for AreaTable mapping (`--lk-client-path`)
- Optional: Crosswalk CSVs (`--crosswalk-dir`/`--crosswalk-file`)

## Build
```powershell
dotnet build ..\WoWRollback.sln -c Release
```

## Usage
### Alpha → LK
```powershell
dotnet run --project . -- \
  alpha-to-lk \
  --input ..\test_data\0.5.3\tree\World\Maps\Azeroth\Azeroth.wdt \
  --max-uniqueid 43000 \
  --fix-holes --disable-mcsh \
  --out wrb_out \
  --lk-out wrb_out\lk_adts\World\Maps\Azeroth \
  --lk-client-path "J:\\wowDev\\modernwow" \
  --default-unmapped 0
```

### LK → Alpha Patcher
```powershell
dotnet run --project . -- \
  lk-to-alpha \
  --lk-adts-dir .\wrb_out\lk_adts\World\Maps\Azeroth \
  --map Azeroth \
  --max-uniqueid 43000 \
  --fix-holes --disable-mcsh \
  --out .\patched_lk_az
```

### Analyze Loose ADTs
```powershell
dotnet run --project . -- analyze-map-adts \
  --map <name> \
  --map-dir <dir> \
  --out analysis_output
```

### Serve Viewer
```powershell
dotnet run --project . -- serve-viewer --viewer-dir analysis_output\viewer --port 8080
```

## Notes
- Neighbor-aware MCNK holes clearing uses pre-scan of buried placement refs and clears holes in self+8-neighbors.
## Development Map Repair Pipeline

Automated pipeline for repairing the `development` map (PM4 -> WMO matching -> ADT patching).

```bash
dotnet run --project WoWRollback/WoWRollback.Cli -- development-repair \
  --pm4-dir <path/to/pm4faces_output> \
  --source-adt <path/to/source_adts> \
  --client-path <path/to/3.3.5_client> \
  --out <path/to/output_dir> \
  [--map <map_name>]
```

### Arguments
- `--pm4-dir`: Root directory of PM4FacesTool output (containing `ck_instances.csv` files).
- `--source-adt`: Directory containing source ADTs to patch (e.g., WDL-generated or existing split ADTs).
- `--client-path`: Path to 3.3.5 WoW client (root folder containing `Data/`).
- `--out`: Output directory for repaired ADTs and intermediate files.
- `--map`: Map name (default: `development`).

### What it does
1.  **Extracts WMOs**: Scans client MPQs for Stormwind/Ogrimmar WMOs and extracts collision geometry.
2.  **Matches Objects**: Matches PM4 geometry against extracted WMOs to reconstruct `MODF` placements.
3.  **Fixes Coordinates**: Applies `ServerToAdtPosition` transform to fix PM4 coordinate system issues.
4.  **Patches ADTs**: Injects new placements into source ADTs, creating valid 3.3.5 files ready for Noggit.

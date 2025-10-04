# WoWRollback

Visual comparison toolkit for World of Warcraft Alpha map evolution.

## 🚀 Quick Start (5 Minutes)

### 1. Organize Your Data
```
test_data/
├── 0.5.3/
│   └── tree/
│       └── World/
│           ├── Maps/
│           │   ├── Azeroth/Azeroth.wdt
│           │   ├── Kalimdor/Kalimdor.wdt
│           │   └── ...
│           └── Minimaps/
│               ├── Azeroth/
│               └── Kalimdor/
└── 0.5.5/
    └── (same structure)
```

### 2. Generate Viewer (One Command!)
```powershell
cd WoWRollback
.\rebuild-and-regenerate.ps1 -AlphaRoot ..\test_data\ -Serve
```

That's it! The script will:
- ✅ **Auto-discover** all maps from test_data (no manual list needed!)
- ✅ **Extract terrain data** - MCNK flags, liquids, holes, AreaIDs, multi-layer terrain
- ✅ **Convert to LK format** - Using raw coordinates (matches original client)
- ✅ **Generate overlays** - Terrain properties, area boundaries, object placements
- ✅ **Start web server** at http://localhost:8080

### 3. Explore in Browser
Open http://localhost:8080 and:
- **Toggle terrain overlays** - Impassible areas (red), liquids (blue/green/orange), multi-layer terrain (blue), vertex colors (green)
- **View area boundaries** - Real area names from AreaTable.dbc with adjustable opacity
- **Compare versions** - Switch between 0.5.3, 0.5.5, etc. and see object additions/removals
- **Filter overlays** - Sub-options for each overlay type (rivers, oceans, magma, slime)
- **Pan and zoom** - Explore the entire continent with Leaflet map controls

## Common Workflows

### Generate Specific Maps
```powershell
.\rebuild-and-regenerate.ps1 `
  -Maps @("Azeroth","Kalimdor") `
  -Versions @("0.5.3.3368") `
  -AlphaRoot ..\test_data\
```

### Compare Two Versions
```powershell
.\rebuild-and-regenerate.ps1 `
  -Versions @("0.5.3.3368","0.5.5.3494") `
  -AlphaRoot ..\test_data\ `
  -Serve
```

### Refresh Cached Data (Force Re-Extract)
```powershell
.\rebuild-and-regenerate.ps1 `
  -RefreshCache `
  -AlphaRoot ..\test_data\
```

### Process Instance Maps
```powershell
.\rebuild-and-regenerate.ps1 `
  -Maps @("DeadminesInstance","Shadowfang","StormwindJail") `
  -Versions @("0.5.3.3368") `
  -AlphaRoot ..\test_data\ `
  -Serve
```

## Overview
WoWRollback is a digital archaeology toolkit for exploring the evolution of World of Warcraft map content. It ingests Alpha-era WDT/ADT exports alongside converted Wrath (LK) ADTs, catalogs placement `UniqueID` ranges, compares versions, and optionally produces rollback-ready assets and a portable viewer for visual analysis.

## Prerequisites
- .NET SDK 9.0 (64-bit)
- Real WoW Alpha/LK data exports produced by `AlphaWDTAnalysisTool` (or equivalent) using the same directory layout described below
- Optional: CASC or filesystem access to minimap BLP/PNG tiles for richer viewer imagery

## Project Layout
```
WoWRollback/
├── WoWRollback.Core/       # Core services and models
├── WoWRollback.Cli/        # Command-line entry point
├── ViewerAssets/           # Static viewer (HTML/CSS/JS)
├── docs/                   # Design notes & plans
│   └── architecture/       # System architecture documentation
├── memory-bank/            # Persistent project context
└── rollback_outputs/       # Default output root (timestamped sessions & comparisons)
```

## Architecture Documentation

See `docs/architecture/` for detailed design documents:
- **`overlay-system-architecture.md`** - Complete overlay pipeline design (ADT → CSV → JSON → Viewer)
- **`mcnk-flags-overlay.md`** - MCNK terrain flags implementation (impassible areas, holes)
- **`IMPLEMENTATION_ROADMAP.md`** - Step-by-step implementation guide for new overlays

## Building
```
dotnet build WoWRollback/WoWRollback.sln
```
All commands below assume execution from the repository root via `dotnet run --project WoWRollback/WoWRollback.Cli`.

## Command Reference

### analyze-alpha-wdt
Extract Alpha placement ranges from a single WDT.
```
dotnet run --project WoWRollback/WoWRollback.Cli -- \
  analyze-alpha-wdt \
  --wdt-file path/to/AlphaMap.wdt \
  --out rollback_outputs
```
Outputs land in `rollback_outputs/session_*/<map>/alpha_<map>_ranges.csv` plus supporting ledgers/timelines when available.

### analyze-lk-adt
Analyze converted LK ADTs for a specific map.
```
dotnet run --project WoWRollback/WoWRollback.Cli -- \
  analyze-lk-adt \
  --map Arathi \
  --input-dir path/to/lk/adts \
  --out rollback_outputs
```
Generates `lk_<map>_ranges.csv` and related summaries under the session directory.

### compare-versions
Compare multiple version roots (e.g., Alpha vs LK) and emit CSV/YAML summaries, optional viewer artifacts.
```
dotnet run --project WoWRollback/WoWRollback.Cli -- \
  compare-versions \
  --versions alpha_053,lk_335 \
  --root rollback_inputs \
  --maps Arathi,DunMorogh \
  --yaml-report \
  --viewer-report \
  --default-version 0.5.3 \
  --diff alpha_053,lk_335
```
Key flags:
- `--versions`: Comma-separated list of version folder names under `--root`
- `--maps`: Optional comma-separated map filter
- `--yaml-report`: Emit per-tile YAML summaries (`.../comparisons/<key>/yaml/`)
- `--viewer-report`: Produce minimaps, overlays, diffs, and static viewer config under `.../comparisons/<key>/viewer/`
- `--default-version`: Preferred default layer in the viewer (falls back to earliest version)
- `--diff`: Explicit baseline/comparison pair for diff JSON (defaults to earliest→latest)

Viewer output structure (`.../viewer/`):
- `minimap/<Version>/<Map>/<Map>_<Col>_<Row>.png`
- `overlays/<Version>/<Map>/<Variant>/tile_r<Row>_c<Col>.json`
- `diffs/<Map>/tile_r<Row>_c<Col>.json`
- `index.json`, `config.json`
- Static viewer bundle (HTML/CSS/JS) copied from `ViewerAssets/`

Overlay variants:
- `combined` – all placements for the selected version
- `m2` – MDX/M2 doodads only
- `wmo` – WMO placements only

The viewer UI now exposes:
- Version, map, and overlay dropdowns in `index.html`
- Overlay selector in `tile.html`
- Per-variant marker colors/radii

Regeneration tip: `rebuild-and-regenerate.ps1` writes the new directory layout. Ensure no files under `rollback_outputs/comparisons/<comparison-key>/` are open before running, otherwise CSV locks will abort the CLI.

### dry-run
Simulate rollback filtering using keep/drop configs without writing ADTs.
```
dotnet run --project WoWRollback/WoWRollback.Cli -- \
  dry-run \
  --map Arathi \
  --input-dir path/to/lk/adts \
  --config configs/arathi_keep_ranges.json \
  --out rollback_outputs
```
Reports counts of placements that would be removed per tile and overall.

## Typical Workflow
1. Run `analyze-alpha-wdt` for each Alpha map of interest.
2. Run `analyze-lk-adt` for the converted LK outputs of the same maps.
3. Populate `rollback_inputs/<version>/<map>/...` with the session CSVs or preprocessed data.
4. Call `compare-versions --viewer-report` to generate comparison CSVs, YAML (optional), and viewer assets.
5. Open the viewer bundle (once static assets are in place) to inspect sediment layers, diffs, and annotations.
6. Iterate with `dry-run` (and future rollback commands) to plan selective removals.

## Outputs & Layout
- Sessions: `rollback_outputs/session_YYYYMMDD_HHmmss/`
- Comparisons: `rollback_outputs/comparisons/<comparisonKey>/`
  - `csv/`: Core range, timeline, design kit reports
  - `yaml/`: Optional per-tile YAML when `--yaml-report` is used
  - `viewer/`: Viewer JSON/PNG assets when `--viewer-report` is used

## Troubleshooting

### Build Issues
- **ImageSharp vulnerability warnings**: The build references `SixLabors.ImageSharp 2.1.9` via `Warcraft.NET`. These are known issues. The warnings can be safely ignored in controlled environments, or upgrade when upstream packages allow.
- **"Project file does not exist"**: Make sure you're in the `WoWRollback/` directory when running commands.

### Data Issues
- **Missing minimaps**: Ensure source data contains BLP/PNG tiles at `test_data/{version}/tree/World/Minimaps/{map}/`. The viewer will show placeholder crosshairs if minimaps are missing.
- **"Unable to locate {Map}.wdt"**: Check that WDT files exist at `test_data/{version}/tree/World/Maps/{Map}/{Map}.wdt`.
- **Empty viewer output**: If `AssetTimelineDetailed` is empty, confirm AlphaRoot contains valid WDT/ADT files.

### Terrain Overlay Issues
- **"Terrain CSV not found"**: The script will log `[warn] ✗ Terrain CSV NOT created`. Check AlphaWdtAnalyzer output for errors.
- **404 errors for terrain_complete files**: This means terrain CSVs weren't extracted. Look for the green checkmark: `[debug] ✓ Terrain CSV created`.
- **No colored chunks in viewer**: Enable "Terrain Properties" overlay in the sidebar. If still nothing, check browser console for 404 errors.

### Area Boundary Issues
- **"Unknown Area 1234" instead of names**: AreaTable CSVs must be in `rollback_outputs/{version}/`. Run DBCTool.V2 to extract AreaTable.dbc first.
- **Area boundaries don't disappear**: Fixed in latest version. Make sure you have the updated `areaIdLayer.js`.

### Performance
- **Slow generation**: Large maps (Azeroth, Kalimdor) can take 5-10 minutes. Use smaller maps like DeadminesInstance for testing.
- **Cached maps reused**: Delete `cached_maps/` directory or use `-RefreshCache` flag to force re-extraction.

### Server Issues
- **Port 8080 already in use**: Another process is using the port. Stop it or change the port in the Python server command.
- **Browser shows blank page**: Make sure the script finished completely. Check for `Serving viewer at http://localhost:8080` message.

## Features

### Terrain Overlays
- **Terrain Properties**: Impassible areas, vertex-colored chunks, multi-layer terrain
- **Liquids**: Rivers, oceans, magma, slime with distinct colors
- **Holes**: Terrain holes (caves, tunnels)
- **Area Boundaries**: Zone/subzone boundaries with real names from AreaTable.dbc

### Object Overlays
- **Combined**: All M2 and WMO placements
- **M2 Only**: Just M2 model placements
- **WMO Only**: Just WMO object placements

### Comparison Features
- **Version switching**: Compare multiple Alpha versions side-by-side
- **Diff visualization**: See object additions/removals between versions
- **UniqueID tracking**: Track object ID ranges across versions

### Data Export
- **CSV exports**: All comparison data exported to CSV for analysis
- **YAML reports**: Optional per-tile YAML summaries
- **LK ADT conversion**: Convert Alpha ADTs to Wrath format

## Default Behavior

### Coordinate System
- **Raw coordinates** are the default (matches original WoW client)
- No transformations applied to placement data
- Easier debugging and comparison with wow.tools

### Auto-Discovery
- Script automatically finds all maps in `test_data/`
- Searches version-specific paths: `{version}/tree/World/Maps/`
- No need to manually list maps unless you want specific ones

## Future Enhancements
- UniqueID timeline selector for per-tile filtering
- Patched ADT export (write modified ADTs with selected object ranges)
- Automated minimap sourcing via CASC/file lookup
- Rollback APPLY command that rewrites ADTs using keep/drop configurations
- Integration tests on real data fixtures

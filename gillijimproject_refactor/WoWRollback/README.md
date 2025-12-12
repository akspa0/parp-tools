# WoWRollback - World of Warcraft Map Analysis & Rollback Toolkit

**Digital archaeology + conversion toolkit** focused on:
## Condensed Overview (2025-11-08)

- Main issue: LK ADT positions → Alpha WDT writeout (compute MAIN offsets, embed ADT payloads, validate).
- Implemented: MPQ overlay precedence (FS > root-letter > locale-letter > root-numeric > locale-numeric > base), DBC locale‑first for `DBFilesClient/*`, plain patch support, WDT tile presence fallback, tee logging (`--log-dir`/`--log-file`).

### Quick Start (CLI)

Alpha → LK (minimal):
```powershell
dotnet run --project WoWRollback.Cli -- alpha-to-lk \
  --input ..\test_data\0.5.3\tree\World\Maps\Azeroth\Azeroth.wdt \
  --max-uniqueid 75000 --fix-holes --disable-mcsh \
  --out wrb_out \
  --lk-out wrb_out\lk_adts\World\Maps\Azeroth \
  --lk-client-path "J:\\wowDev\\modernwow"
```

Pack monolithic Alpha WDT from MPQ client (diagnostics):
```powershell
dotnet run --project WoWRollback.Cli -- pack-monolithic-alpha-wdt \
  --client-path "H:\\WoWDev\\modernwow" \
  --map "CataclysmCTF" \
  --out out\CataclysmCTF.wdt \
  --verbose --verbose-logging --log-dir .\logs\
```

---

## PM4 → ADT Restoration Pipeline (NEW!)

Reconstruct lost WMO placements from PM4 pathfinding mesh data. This pipeline:
1. Parses PM4 objects (CK24-grouped surfaces from .pm4 files)
2. Matches them to retail WMO geometry via principal component analysis
3. Generates MODF entries and patches into ADT files
4. Produces a Noggit-ready project

### Usage

```powershell
dotnet run --project WoWRollback.PM4Module -- patch-pipeline \
  --game "C:\WoW335\World of Warcraft" \
  --listfile ".\test_data\World of Warcraft 3x.txt" \
  --pm4 ".\test_data\development\World\Maps\development\" \
  --split-adt ".\test_data\development\World\Maps\development\" \
  --museum-adt ".\test_data\WoWMuseum\335-dev\World\Maps\development\" \
  --out "PM4_to_ADT"
```

### Required Inputs

| Argument | Description |
|----------|-------------|
| `--game` | Path to WoW 3.3.5 client (for WMO extraction from MPQs) |
| `--listfile` | Listfile with WMO paths (e.g., `World of Warcraft 3x.txt`) |
| `--pm4` | Directory containing .pm4 files (e.g., `development_29_30.pm4`) |
| `--split-adt` | Directory with split ADT data (for WDL file) |
| `--museum-adt` | WoWMuseum LK ADTs to patch (preserves terrain data) |
| `--out` | Output directory for Noggit project |
| `--wmo-filter` | *Optional*: Filter WMOs by path prefix (e.g., `Northrend` or `Kalimdor`) |
| `--use-full-mesh` | *Optional*: Use full WMO mesh for matching (not just walkable surfaces) |

### Output Structure

```
PM4_to_ADT/
├── development.noggitproj    # Open this in Noggit Red
├── Final_Assembly/           # Patched ADTs ready to use
├── Patched_Museum_ADTs/      # Museum ADTs + MODF placements
├── WDL_Painted/              # WDL-generated ADTs + MODF
├── modf_csv/
│   ├── modf_entries.csv      # All matched placements
│   ├── mwmo_names.csv        # WMO path string table
│   └── match_candidates.csv  # Full match analysis
└── wmo_library_cache.json    # Cached WMO fingerprints
```

### Pipeline Stages

1. **Stage 1 & 1.5**: In-memory WMO processing (extracts walkable surfaces)
2. **Stage 2**: PM4 → WMO geometry matching (PCA-based fingerprinting)
3. **Stage 3**: WDL → ADT generation (terrain from low-res heightmap)
4. **Stage 4**: Museum ADT patching (adds MWMO/MWID/MODF chunks)
5. **Stage 4b**: WDL ADT patching (same for WDL-generated tiles)
6. **Stage 5**: Noggit project assembly

### Technical Details

- **PM4 Parsing**: MSVT vertices stored as (Y,X,Z), reordered to (X,Y,Z)
- **Tile Assignment**: Derived from PM4 filename (e.g., `development_29_30.pm4` → tile 29,30)
- **MODF Format**: Coordinates written as XZY (X, Height, Y) per WoW spec
- **Path Format**: UPPERCASE backslashes (`WORLD\WMO\...`)
- **Match Threshold**: 88% confidence minimum for auto-matching

### Links
- memory-bank/activeContext.md — Current focus and TODOs
- memory-bank/progress.md — Snapshot of progress
- memory-bank/systemPatterns.md — Overlay precedence, WDT fallback
- memory-bank/techContext.md — Runtime/env and modules

---
## Requirements
- .NET SDK 9.0 (x64)
- Alpha data (extracted WDT/ADT/DBC) as needed
- LK 3.3.5 client or DBCs when required for AreaTable/Map guards
- Optional: MPQ client install for minimap extraction

## License
See LICENSE in repository root.

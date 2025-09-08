# AlphaWDTAnalysisTool

AlphaWDTAnalysisTool analyzes World of Warcraft Alpha WDT/ADT data and can export converted WotLK-compatible ADTs per tile. It also performs AreaID analysis and (optionally) patches MCNK AreaIDs in the generated ADTs using DBC/DBD-provided mapping.

This folder contains the tool and its CLI.

---

## Features

- Parses Alpha WDTs to find existing ADTs and model name tables (MDNM/MONM).
- Scans ADTs to index referenced assets (WMO/M2/BLP) and per-tile placements.
- Exports WotLK ADTs using a faithful ported writer:
  - ADTs are written to `World/Maps/<MapName>/<MapName>_<x>_<y>.adt`.
  - A WDT file is written once per map as `World/Maps/<MapName>/<MapName>.wdt`.
- Area ID analysis (Alpha):
  - Emits `csv/maps/<MapName>/areaid_mapping.csv` with per-MCNK rows.
  - Patches MCNK `AreaId` fields in-place in the exported ADTs using `AreaTable.dbc` mappings (name-based, with fallback).
- Asset fixups using a community listfile and an optional LK listfile path.
- Optional web UI assets to view analysis output.

---

## Requirements

- .NET SDK 9.0 (or newer compatible with `net9.0`).

---

## Dependencies & Setup

This tool expects a couple of external resources to be available locally.

- Community listfile (CSV) — required
  - Download the latest “with capitals” CSV:
    - https://github.com/wowdev/wow-listfile/releases/download/202509072101/community-listfile-withcapitals.csv
  - Save it somewhere accessible (examples below use `test_data/community-listfile-withcapitals.csv`).

- LK 3.x listfile (smaller, optional but recommended)
  - Download the ZIP from: http://www.zezula.net/download/listfiles.zip
  - Extract the archive and locate the 3.x-specific listfile (commonly named like `listfile - 3.x.txt`).
  - Optionally rename it to `World of Warcraft 3x.txt` to match example commands.

- Libraries to clone into the repository (as source dependencies)
  - Clone these into the repository `lib/` folder:
    - wow.tools.local: https://github.com/Marlamin/wow.tools.local/
    - WoWDBDefs: https://github.com/wowdev/WoWDBDefs

  Example (PowerShell from repo root):

  ```powershell
  # Ensure lib folder exists
  mkdir lib -Force | Out-Null

  # Clone wow.tools.local with submodules
  git clone --recurse-submodules https://github.com/Marlamin/wow.tools.local.git lib/wow.tools.local

  # Clone WoWDBDefs
  git clone https://github.com/wowdev/WoWDBDefs.git lib/WoWDBDefs
  ```

  Notes:
  - If you already have these folders, ensure they are up to date and run `git submodule update --init --recursive` in `lib/wow.tools.local` if needed.
  - The solution is wired to consume these as source dependencies under `lib/`.

---

## Build

```powershell
# From repository root
> dotnet build
```

The CLI assembly is at `AlphaWDTAnalysisTool/AlphaWdtAnalyzer.Cli/bin/Debug/net9.0/AlphaWdtAnalyzer.Cli.dll` after a successful build.

---

## CLI Usage

CLI entry point: `AlphaWDTAnalysisTool/AlphaWdtAnalyzer.Cli/Program.cs`.

```
AlphaWdtAnalyzer
Usage:
  Single map: AlphaWdtAnalyzer --input <path/to/map.wdt> --listfile <community_listfile.csv> [--lk-listfile <3x.txt>] --out <output_dir> [--cluster-threshold N] [--cluster-gap N] [--dbc-dir <dir>] [--area-alpha <AreaTable.dbc>] [--area-lk <AreaTable.dbc>] [--web] [--export-adt --export-dir <dir> [--fallback-tileset <blp>] [--fallback-wmo <wmo>] [--fallback-m2 <m2>] [--fallback-blp <blp>] [--no-mh2o] [--asset-fuzzy on|off]]
  Batch maps:  AlphaWdtAnalyzer --input-dir <root_of_wdts> --listfile <community_listfile.csv> [--lk-listfile <3x.txt>] --out <output_dir> [--cluster-threshold N] [--cluster-gap N] [--dbc-dir <dir>] [--web] [--export-adt --export-dir <dir> [--fallback-tileset <blp>] [--fallback-wmo <wmo>] [--fallback-m2 <m2>] [--fallback-blp <blp>] [--no-mh2o] [--asset-fuzzy on|off]]
```

Key options:

- `--input <map.wdt>`: Single-map mode (can also pass a directory; treated as `--input-dir`).
- `--input-dir <dir>`: Batch mode; recurses for `*.wdt`.
- `--listfile <community.csv>`: Community listfile used for path normalization and asset fixups.
- `--lk-listfile <3x.txt>`: Optional LK listfile for better asset resolution.
- `--out <out_dir>`: Analysis output root.
- `--dbc-dir <dir>`: Optional directory for DBC reading (when doing deeper analysis).
- `--area-alpha <AreaTable_alpha.dbc>`: Alpha AreaTable for AreaID mapping (recommended for patching).
- `--area-lk <AreaTable_lk.dbc>`: LK AreaTable to resolve LK names and IDs (recommended for patching).
- `--web`/`--no-web`: Emit web assets for the analysis summary (single-map mode).
- Export flags:
  - `--export-adt`: Enable ADT export.
  - `--export-dir <dir>`: ADT export root (separate from `--out`).
  - `--fallback-tileset <blp>`: Fallback BLP path for tileset textures.
  - `--fallback-wmo <wmo>`: Fallback WMO when references are missing.
  - `--fallback-m2 <m2>`: Fallback M2/MDX when references are missing.
  - `--fallback-blp <blp>`: Fallback non-tileset BLP.
  - `--no-mh2o`: Disable MH2O.
  - `--asset-fuzzy on|off`: Toggle fuzzy matching for asset fixups (default: on).

---

## Examples

Examples are run from the repository root using `--project` to target the CLI:

### Single map (analyze + export)

```powershell
> dotnet run --project .\AlphaWDTAnalysisTool\AlphaWdtAnalyzer.Cli \
  -- --input "..\..\test_data\0.5.5\tree\World\Maps\Azeroth\Azeroth.wdt" \
     --listfile "..\..\test_data\community-listfile-withcapitals.csv" \
     --lk-listfile "..\..\test_data\World of Warcraft 3x.txt" \
     --out "..\..\out\0.5.5_decode2" \
     --export-adt --export-dir "..\..\out\0.5.5_decode2" \
     --area-alpha "..\..\test_data\0.5.5\tree\DBFilesClient\AreaTable.dbc" \
     --dbc-dir "..\..\lib\WoWDBDefs\definitions" \
     --asset-fuzzy on
```

### Batch (analyze multiple maps + export)

```powershell
> dotnet run --project .\AlphaWDTAnalysisTool\AlphaWdtAnalyzer.Cli \
  -- --input-dir "..\..\test_data\0.5.5\tree" \
     --listfile "..\..\test_data\community-listfile-withcapitals.csv" \
     --lk-listfile "..\..\test_data\World of Warcraft 3x.txt" \
     --out "..\..\out\0.5.5_decode2" \
     --export-adt --export-dir "..\..\out\0.5.5_decode2" \
     --area-alpha "..\..\test_data\0.5.5\tree\DBFilesClient\AreaTable.dbc" \
     --dbc-dir "..\..\lib\WoWDBDefs\definitions" \
     --asset-fuzzy on
```

---

## Output Layout

When `--export-adt` is enabled, files are written under the export root (`--export-dir`) as follows:

```
<export-dir>/
  World/
    Maps/
      <MapName>/
        <MapName>.wdt
        <MapName>_<x>_<y>.adt            # one per tile that exists in WDT (offset > 0)
  csv/
    maps/
      <MapName>/
        areaid_mapping.csv                # per-MCNK AreaID analysis
```

Notes:

- Tiles are selected as the union of tiles with placements and tiles present in the WDT’s `MAIN` table (non-zero offsets). Tiles with zero offsets are not written.
- WDT export happens once per map per run. The tool converts the Alpha WDT and writes it to `World/Maps/<MapName>/<MapName>.wdt`.

---

## AreaID Mapping & Patching

- Provide `--area-alpha` (Alpha `AreaTable.dbc`) to decode Alpha AreaID names.
- Provide `--area-lk` (LK `AreaTable.dbc`) to resolve LK names and IDs.
- The tool writes a CSV with one row per present MCNK:
  - `tile_x,tile_y,mcnk_index,alpha_area_id,alpha_name,lk_area_id,lk_name,mapped,mapping_reason`
- After writing an ADT, the tool patches the LK `MCNK.AreaId` field in-place when a mapping is found. Name-based matching is preferred; a minimal fallback may apply (e.g., "On Map Dungeon").

---

## Asset Fixups

- The tool normalizes and fixes asset paths via the community listfile and optional LK listfile.
- Before building each LK ADT, it applies fixups to the MDNM/MONM tables while preserving indices.
- Fallbacks:
  - `--fallback-tileset`, `--fallback-wmo`, `--fallback-m2`, `--fallback-blp` can be used when assets are missing or unmapped.
  - `--asset-fuzzy on|off` toggles fuzzy matching during fixups.

---

## Quick Start

1) Clone required libraries under `lib/`:

```powershell
mkdir lib -Force | Out-Null
git clone --recurse-submodules https://github.com/Marlamin/wow.tools.local.git lib/wow.tools.local
git clone https://github.com/wowdev/WoWDBDefs.git lib/WoWDBDefs
```

2) Download listfiles:

- Community CSV: https://github.com/wowdev/wow-listfile/releases/download/202509072101/community-listfile-withcapitals.csv
- LK 3.x listfile: http://www.zezula.net/download/listfiles.zip (extract and locate the 3.x file; optionally rename to `World of Warcraft 3x.txt`)

3) Build:

```powershell
dotnet build
```

4) Run (single map example):

```powershell
dotnet run --project .\AlphaWDTAnalysisTool\AlphaWdtAnalyzer.Cli -- \
  --input "<path-to>\World\Maps\<MapName>\<MapName>.wdt" \
  --listfile "<path-to>\community-listfile-withcapitals.csv" \
  --lk-listfile "<path-to>\World of Warcraft 3x.txt" \
  --out "<output-root>" \
  --export-adt --export-dir "<export-root>" \
  --area-alpha "<path-to>\AreaTable.dbc" \
  --dbc-dir ".\lib\WoWDBDefs\definitions"
```

---

## Troubleshooting

- __Listfile not found__: Ensure `--listfile` and `--lk-listfile` paths are valid. The CLI verifies and will exit with a message if not found.
- __No ADTs written__: Check if the WDT has non-zero tile offsets. The tool only writes tiles with `MAIN` offsets > 0. See the exported folder for the WDT to confirm tiles.
- __AreaID CSV empty__: Provide `--area-alpha` to decode Alpha Area IDs. `--area-lk` improves LK mapping quality but is optional.
- __Unexpected asset paths__: Turn `--asset-fuzzy off` to disable fuzzy fixups. Verify listfile contents.
- __Web UI assets__: Use `--web` (single-map mode) to emit web assets to `<out>/web/`.

---

## Notes

- The ADT export uses the ported LK writer and maintains parity with the original logic wherever possible.
- Changes avoid modifying the core library under `src/gillijimproject-csharp/WowFiles/`.

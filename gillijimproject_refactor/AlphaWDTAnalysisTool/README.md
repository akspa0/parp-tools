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
- Area ID patching (strict, CSV-only):
  - Consumes DBCTool.V2 crosswalks from `compare/v2/` and applies per-source-map numeric mappings only.
  - No name-based heuristics and no zone-base fallback. If there is no explicit per-map CSV row, we write `0`.
  - A per-tile verify CSV is emitted in verbose mode to audit what was written.
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
  Single map: AlphaWdtAnalyzer --input <path/to/map.wdt> --listfile <community_listfile.csv> --out <output_dir> \
              [--lk-listfile <3x.txt>] [--viz-html] [--no-fixups] \
              --export-adt --export-dir <dir> --dbctool-patch-dir <compare/v2/dir> [--dbctool-lk-dir <LK DBC dir>]
  Batch maps:  AlphaWdtAnalyzer --input-dir <root_of_wdts> --listfile <community_listfile.csv> --out <output_dir> \
              [--lk-listfile <3x.txt>] [--viz-html] [--no-fixups] \
              --export-adt --export-dir <dir> --dbctool-patch-dir <compare/v2/dir> [--dbctool-lk-dir <LK DBC dir>]
```

Key options (mapping/patching):

- `--dbctool-patch-dir <dir>`: Path to DBCTool.V2 session `compare/v2/` directory. Consumed for strict per-map numeric mapping.
- `--dbctool-lk-dir <dir>`: Optional LK DBC folder (for map guard/name legend only). Mapping itself is CSV-driven only.

Other options:

- `--input` / `--input-dir`, `--listfile`, `--lk-listfile`, `--out`, `--viz-html`, `--no-fixups`, `--export-adt`, `--export-dir`.

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
        areaid_verify_<x>_<y>.csv        # per-MCNK AreaID verify (verbose runs)
```

Notes:

- Tiles are selected as the union of tiles with placements and tiles present in the WDT’s `MAIN` table (non-zero offsets). Tiles with zero offsets are not written.
- WDT export happens once per map per run. The tool converts the Alpha WDT and writes it to `World/Maps/<MapName>/<MapName>.wdt`.

---

## AreaID Mapping & Patching

- Strict CSV-only mapping: no heuristics, no zone-base fallback.
- The tool consumes DBCTool.V2 crosswalks and patches `MCNK.AreaId` in-place when a per-map numeric CSV row exists for the given `src_areaNumber (zone<<16|sub)`.
- If no per-map CSV mapping exists, `0` is written. Special rows that explicitly map to 0 (e.g., "On Map Dungeon") remain 0.
- In verbose mode, a per-tile verify CSV is written to audit the alpha value, chosen LK area, and reason.

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
  --dbctool-patch-dir ".\DBCTool.V2\dbctool_outputs\<session>\compare\v2" \
  --dbctool-lk-dir ".\test_data\3.3.5\tree\DBFilesClient"
```

---

## Troubleshooting

- **Listfile not found**: Ensure `--listfile` and `--lk-listfile` paths are valid.
- **No ADTs written**: Check if the WDT has non-zero tile offsets. Only tiles with `MAIN` offsets > 0 are written.
- **Area IDs all zero**: Confirm `--dbctool-patch-dir` points at the correct DBCTool.V2 `compare/v2/` folder and that your map-specific CSV has non-zero `tgt_areaID` rows for the `src_areaNumber` values you expect.
- **Unexpected asset paths**: Turn `--no-fixups` on or disable fuzzy in your profile to preserve original paths.
- **Web UI assets**: Use `--viz-html` to emit HTML summary when available.

---

## Notes

- The ADT export uses the ported LK writer and maintains parity with the original logic wherever possible.
- Changes avoid modifying the core library under `src/gillijimproject-csharp/WowFiles/`.

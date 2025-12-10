# Parp Tools — Gillijim Project (C# refactor)

This repository contains tools and libraries to restore and analyze AreaTable mappings across early World of Warcraft builds, with a preservation-first philosophy. The current focus is strict, numeric, auditable mapping from Alpha-era data (0.5.x) to LK 3.3.5 — without editing DBC files.

Key docs:

- AreaID Restoration Approach: `DBCTool.V2/docs/areaid-restoration-approach.md`
- Input Data Preparation (test_data layout and Alpha WDT extraction): `DBCTool.V2/docs/input-data-prep.md`

---

## Repository structure (what each folder is for)

- `AlphaWDTAnalysisTool/`
  - `AlphaWdtAnalyzer.Cli/` — CLI to remap LK ADTs using numeric crosswalks and Alpha-captured area values; writes verify CSVs and (optionally) visualizations.
  - `AlphaWdtAnalyzer.Core/` — core exporter/patcher, CSV schemas, visualization helpers.
  - Uses per-map numeric crosswalks from `DBCTool.V2` and enforces strict, map-locked, non-heuristic behavior.

- `DBCTool.V2/`
  - CLI that generates crosswalks (`Area_patch_crosswalk_via060_map*_*.csv`), dumps, and audits under `DBCTool.V2/dbctool_outputs/session_*/compare/v2/`.
  - Rules are strict and map-locked; only via060 rows with non-zero targets are used for patching.
  - See `DBCTool.V2/README.md` and the docs linked above.

- `DBCTool.V2.Core/`
  - Shared types and helpers used by `DBCTool.V2`.

- `DBCTool/` (legacy — unmaintained; use `DBCTool.V2`)
  - Earlier implementation with issues we could not resolve; kept for historical reference. Please use `DBCTool.V2`.

- `lib/`
  - Vendored reference libraries and tooling used by this project and experiments:
    - `wow.tools.local/` (e.g., `WoWFormatLib`, `DBCD`, `TACTSharp`) — CASC/TACT access, format parsers, and DBC helpers.
    - `Warcraft.NET/` — WoW format domain models/utilities.
    - `WoWDBDefs/` — DBDefs parsing and utilities.
    - `MapUpconverter/` — reference materials for map/terrain conversions.
  - Not all subprojects are required at runtime; treat as a toolbox/reference.

- `test_data/`
  - Standard input tree: `test_data/<version>/tree/...`. Examples:
    - `test_data/0.5.3/tree/World/Maps/Azeroth/Azeroth.wdt`
    - `test_data/3.3.5/tree/DBFilesClient/AreaTable.dbc`
  - See `DBCTool.V2/docs/input-data-prep.md` for full layout and Alpha WDT extraction notes (MPQEditor from Zezula.net).

- `tools/`
  - Utility scripts like `tools/agg-area-verify.ps1` to summarize verify CSVs (per-map `alpha_areas_used_*.csv` and `unknown_area_numbers_*.csv`).

- `reference_data/`
  - Local copies of WoWDev wiki pages used during development. Treat as reference only; can be out of date and is not authoritative.

- `053-chain/`, `053-viz/`, `053-viz-36/`
  - Example output trees generated during development from prior commands; not required by tools (which choose sensible output locations by default).

- `next/`, `refactor/`, `src/`
  - Experimental or in-progress tools and refactors outside the current preservation workflow.

- `out/`, `docs/`, `memory-bank/`
  - `out/` — ad hoc outputs (varies by experiment).
  - `docs/` — planning or top-level documentation.
  - `memory-bank/` — working context notes.

---

## Quick workflows (high level)

1) Generate crosswalks and audits (DBCTool.V2)

```bash
dotnet run --project DBCTool.V2/DBCTool.V2.csproj -- --s53
```

- Inputs come from `test_data/<version>/tree` (see docs). Outputs land under `DBCTool.V2/dbctool_outputs/session_*/compare/v2/`.

2) Remap ADTs and verify (AlphaWDTAnalysisTool)

- Use a patch directory that contains the via060 per-map crosswalks (and optional overrides), and provide the LK 3.3.5 DBC directory for names in verify.
- After export, run `tools/agg-area-verify.ps1` on the export root to see which `alpha_raw` values were present and whether they mapped.

---

## Alpha / Legacy asset gating (prevent old client crashes)

Older clients (e.g., 0.5.3) crash if MDNM/MONM reference assets that don’t exist in that build. Use the workflow below to build a version-accurate listfile and strictly gate assets when packing.

### 1) Build a version listfile (scan MPQs + loose files)

```bash
dotnet run --project WoWRollback.Cli/WoWRollback.Cli.csproj snapshot-listfile \
  --client-path "C:/Path/To/ClientRoot" \
  --alias "alpha-0.5.3" \
  --community-listfile "../test_data/community-listfile-withcapitals.csv" \
  --out ./alpha_053.json \
  --csv-out ./alpha_053.csv \
  --csv-missing-fdid 0
```

What it does:

- Scans all MPQs (including locale subfolders) and loose files under the client.
- Reads embedded `(listfile)` when present and derives asset names from wrappers (e.g., `Azeroth.wdt.MPQ` → `Azeroth.wdt`, `*.m2.MPQ` → `*.m2` and `.mdx`).
- Optionally enriches with a community listfile to set FDIDs (exact path, `.m2`↔`.mdx` alias, and filename-only heuristic when unique).
- Writes:
  - `alpha_053.json` (snapshot with optional `fdid` fields)
  - `alpha_053.csv` (community-style CSV; unknown FDIDs use the provided token, default `0`)
  - `snapshot_missing_fdid.csv` and `snapshot_ambiguous_fdid.csv` for diagnostics

Useful flags:

- `--community-listfile <file>`: CSV/JSON/plain community listfile for FDID enrichment
- `--csv-out <file>`: emit a community-style CSV (preferred for downstream gating)
- `--csv-missing-fdid <token>`: placeholder for unknown FDIDs (e.g., `0`, `XXXXXXXX`, or `none` for path-only)

### 2) Pack with strict asset gating

```bash
dotnet run --project WoWRollback.Cli/WoWRollback.Cli.csproj pack-monolithic-alpha-wdt \
  --client-path "C:/Path/To/ClientRoot" \
  --map "expansion01" \
  --out ./build/expansion01.wdt \
  --target-listfile ./alpha_053.csv \
  --strict-target-assets true \
  --verbose
```

Notes:

- Gating is applied for both M2 (MDNM) and WMO (MONM) names; unknowns are dropped.
- Placements referencing dropped names are skipped (no invalid MCRF refs).
- A `dropped_assets.csv` report is written next to the output WDT.

---

## Development Map Repair Pipeline

Automated pipeline to reconstruct the "development" map (The Vice) by extracting WMO geometry, matching against PM4 pathfinding objects, and patching clean 3.3.5 ADTs.

### Usage

```bash
dotnet run --project WoWRollback/WoWRollback.Cli -- development-repair \
  --pm4-dir "path/to/development/pm4" \
  --source-adt "path/to/clean_adts" \
  --client-path "path/to/3.3.5/client" \
  --out "output_dir" \
  --cache-dir "work/cache" \
  --listfile "listfile.txt" \
  --dump-objs
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--pm4-dir` | Path to extracted PM4 files (containing `.obj` geometry). |
| `--source-adt` | Path to clean 3.3.5 ADTs (must have valid textures!). |
| `--client-path` | Path to 3.3.5 client (for WMO extraction). |
| `--out` | Output directory for patched ADTs and diagnostics. |
| `--cache-dir` | **Highly Recommended**. Directory to store heavy cache files (`wmo_library.json`, `pm4_library.json`). Reusing this speeds up runs from 20m to 10s. |
| `--listfile` | Listfile to filter WMO candidates (optional). |
| `--dump-objs` | If set, dumps extracted `.obj` files to output for inspection. |

### Output Structure

- `adt_335/`: The patched ADT files ready for Noggit.
- `modf_reconstruction/`: CSVs detailing the matched placements.
- `chunk_dump/`: **(Debug)** Raw binary dumps of `MTEX`, `MODF`, `MWMO` chunks for verification.

### Troubleshooting

- **Empty Textures / Missing Objects**: Check `chunk_dump/` in the output. If `MTEX_....bin` or `MODF_....bin` are 0 bytes, the patching failed.
- **"Domino Effect"**: If the input `--source-adt` files are corrupt (missing chunk sizes), the output will also be corrupt. Ensure source ADTs are generated with the latest `wdl-to-adt` and `adt-merge` tools.

---

## Policies

- Strict numeric mapping only; map-locked; no heuristics.
- Overrides contain exact numeric rows and never override explicit crosswalk mappings.
- We do not edit DBCs in this preservation track.

---

## Housekeeping

- Legacy: `DBCTool/` is unmaintained and kept for historical reference. Please use `DBCTool.V2` for all current workflows.
- License: undecided — pending review of third-party library licenses in `lib/`.


# Parp Tools — Gillijim Project (C# refactor)

WoW format preservation, analysis, and visualization tooling. The **primary project** is [MdxViewer](src/MdxViewer/) — a high-performance .NET 9 / OpenGL 3.3 world viewer supporting **Alpha 0.5.3**, **0.6.0**, and **LK 3.3.5** game data.

## Quick Start — MdxViewer

```bash
cd src/MdxViewer
dotnet build
dotnet run -- path/to/game/directory
```

The viewer auto-detects the WoW build version from the game path and loads terrain, WMOs, MDX models, liquids, and DBC overlays.

See [`src/MdxViewer/README.md`](src/MdxViewer/README.md) for full documentation.

---

## Repository Structure

### Primary Project

- **`src/MdxViewer/`** — 3D world viewer (active development)
  - Multi-version terrain: Alpha monolithic WDT, 0.6.0 split ADTs, 3.3.5 split ADTs
  - WMO v14/v16/v17 rendering with 4-pass transparency, doodads, liquids
  - MDX/M2 model rendering with blend modes, alpha cutout, DBC texture resolution
  - AOI streaming with persistent cache, directional lookahead, MPQ throttling
  - DBC overlays: AreaPOI, TaxiPaths, area names, zone-based lighting
  - ImGui UI with minimap, file browser, object list, terrain controls

### Supporting Projects

- `src/MDX-L_Tool/` — MDX file format parser for Alpha 0.5.3 model archaeology
- `src/WoWMapConverter/` — Map conversion tools, VLM, ADT format library

### Data & Conversion Tools

- `AlphaWDTAnalysisTool/` — CLI to remap LK ADTs using numeric crosswalks
- `DBCTool.V2/` — DBC crosswalk generation, dumps, and audits
- `WoWRollback/` — ADT merger, PM4 tools, MCCV painting, asset gating
- `BlpResizer/` — BLP texture conversion

### Libraries & References

- `lib/` — Vendored libraries: `wow.tools.local` (DBCD), `Warcraft.NET`, `WoWDBDefs`, `MapUpconverter`
- `reference_data/` — Local WoWDev wiki copies (reference only)
- `test_data/` — Input data tree (`test_data/<version>/tree/...`)
- `memory-bank/` — Working context notes

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

## Development Map PM4 Pipeline (C# refactor)

End-to-end pipeline to reconstruct the "development" map using PM4 pathfinding output and WMO collision geometry, then inject the resulting WMO placements into clean 3.3.5 ADTs.

### Canonical data locations

These paths are fixed in this repo (see `.windsurf/rules/data-paths.md`):

| Data | Path | Notes |
|------|------|-------|
| Split Cata ADTs + PM4 | `test_data/development/World/Maps/development` | 466 root ADTs, 616 PM4 files + PM4Faces output |
| Minimap PNGs | `test_data/minimaps/development` | For MCCV painting |
| WoWMuseum LK ADTs | `test_data/WoWMuseum/335-dev/World/Maps/development` | Clean 3.3.5 baseline |
| WMO collision (per-group/flag) | `pm4-adt-test13/wmo_flags/` | One folder per WMO, one OBJ per group/flag |
| MODF reconstruction (current) | `pm4-adt-test13/modf_reconstruction/` | `modf_entries.csv`, `mwmo_names.csv`, `placement_verification.json` |

### Step 1 — Build base LK ADTs (terrain/textures/MCCV)

Use the existing ADT tooling to produce a "clean" LK ADT set that has terrain, textures, and MCCV applied where needed:

- `merge-split` — merge split LK ADTs (root + `_obj0` + `_tex0`) into monolithic form.
- `merge-minimap` — MCCV-paint tiles that have no textures at all using minimap PNGs.
- `merge-textures` — pull texture chunks from older monolithic ADTs when `_tex0` is missing.

The result is a directory such as:

- `PM4ADTs/clean_v3/` or `test_output/merged_minimap/`

This directory is the **`--in`** argument for the MODF injector.

### Step 2 — Reconstruct MODF from PM4 + WMO collision library

Run the PM4 reconstruction CLI from the repo root:

```bash
dotnet run --project WoWRollback/WoWRollback.PM4Module/WoWRollback.PM4Module.csproj -- pm4-reconstruct-modf \
  --pm4 "test_data/development/World/Maps/development" \
  --wmo "pm4-adt-test13/wmo_flags" \
  --out "pm4-adt-test13/modf_reconstruction" \
  --min-confidence 0.7
```

This will:

- Load PM4 objects from all `ck_instances.csv` files under `--pm4`.
- Load WMO collision geometry from the per-group/per-flag OBJ files under `--wmo`.
- Match PM4 objects to WMOs, compute placement transforms, and apply the PM4→ADT world coordinate transform.
- Write:
  - `modf_entries.csv` — world-space MODF placements (post-transform).
  - `mwmo_names.csv` — MWMO string table.
  - `placement_verification.json` — per-tile/per-WMO summary for audits.

### Step 3 — Inject MODF into base LK ADTs

With the reconstruction CSVs in place, inject MODF into your base ADT set:

```bash
dotnet run --project WoWRollback/WoWRollback.PM4Module/WoWRollback.PM4Module.csproj -- inject-modf \
  --modf "pm4-adt-test13/modf_reconstruction/modf_entries.csv" \
  --mwmo "pm4-adt-test13/modf_reconstruction/mwmo_names.csv" \
  --in  "PM4ADTs/clean_v3" \
  --out "PM4ADTs/museum_patched_test13" \
  --map development
```

Notes:

- `inject-modf` uses `MuseumAdtPatcher` / `AdtPatcher` to preserve all existing chunks and only modify `MWMO`, `MWID`, and `MODF`.
- If a tile has no reconstructed placements in `modf_entries.csv`, its ADT is copied unchanged.
- You can inspect on-disk MODF via:

  ```bash
  dotnet run --project WoWRollback/WoWRollback.PM4Module/WoWRollback.PM4Module.csproj -- dump-modf \
    --in PM4ADTs/museum_patched_test13 --map development --tile 22_18
  ```

  and compare against `dump-modf-csv` for the same tile.

The `PM4ADTs/museum_patched_test13/` directory is suitable for Noggit / 3.3.5 client testing once paired with an appropriate WDT.

---

## Policies

- Strict numeric mapping only; map-locked; no heuristics.
- Overrides contain exact numeric rows and never override explicit crosswalk mappings.
- We do not edit DBCs in this preservation track.

---

## Housekeeping

- Legacy: `DBCTool/` is unmaintained and kept for historical reference. Please use `DBCTool.V2` for all current workflows.
- License: undecided — pending review of third-party library licenses in `lib/`.


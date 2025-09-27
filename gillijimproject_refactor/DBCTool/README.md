# DBCTool (LEGACY — Unmaintained)

> Legacy notice: This tool is no longer maintained and is kept for historical context.
> Please use `DBCTool.V2` instead:
> - `DBCTool.V2/README.md`
> - `DBCTool.V2/docs/areaid-restoration-approach.md`
> - `DBCTool.V2/docs/input-data-prep.md`

---

Filesystem-only DBC → CSV exporter and Area remapping helper built on DBCD (wow.tools.local).

## Setup

- .NET 9.0 (x64)
- DBCD referenced as a project:
  - `..\lib\wow.tools.local\DBCD\DBCD\DBCD.csproj`
- Default WoWDBDefs path:
  - `lib/WoWDBDefs/definitions`

## Quick Start: Export Tables

Filesystem (extracted DBCs):
```powershell
# Using build alias in --input
dotnet run --project .\DBCTool\DBCTool.csproj -- \
  --dbd-dir .\lib\WoWDBDefs\definitions \
  --out out \
  --locale enUS \
  --table Map --table AreaTable \
  --input 3.3.5=H:\extract\DBFilesClient

# Or bare directory: alias inferred from path tokens (0.5.3|0.5.5|3.3.5)
dotnet run --project .\DBCTool\DBCTool.csproj -- \
  --dbd-dir .\lib\WoWDBDefs\definitions \
  --out out \
  --locale enUS \
  --table AreaTable \
  --input ..\test_data\0.5.3\tree\DBFilesClient
```

Outputs are written to version folders (no timestamp):
- `out/0.5.3/AreaTable.csv`
- `out/0.5.5/AreaTable.csv`
- `out/3.3.5/AreaTable.csv`

Supported build aliases map to canonical builds for DBCD:
- `0.5.3` → `0.5.3.3368`
- `0.5.5` → `0.5.5.3494`
- `3.3.5` → `3.3.5.12340`

## Single-Tool Remap Workflow (discover → export → apply)

DBCTool discovers Area mappings from an early source (0.5.3 or 0.5.5) to 3.3.5, exports your decisions into a JSON remap file, and can re-run deterministically by applying that remap.

CLI flags of interest:
- `--compare-area` run the mapping workflow
- `--src-alias` choose source alias: `0.5.3` or `0.5.5` (optional; inferred from inputs)
- `--src-build` canonical source build (default from alias)
- `--tgt-build` canonical target build (default `3.3.5.12340`)
- `--export-remap <path>` write a JSON with aliases/explicit maps/options
- `--apply-remap <path>` load a JSON and apply aliases/explicit maps before matching
- `--allow-do-not-use` allow 3.3.5 targets with names containing “DO NOT USE” (default is to exclude)

### 0.5.3 → 3.3.5 (discover + export)
```powershell
dotnet run --project .\DBCTool\DBCTool.csproj -- \
  --dbd-dir .\lib\WoWDBDefs\definitions \
  --out out \
  --locale enUS \
  --compare-area \
  --input 0.5.3=..\test_data\0.5.3\tree\DBFilesClient \
  --input 3.3.5=..\test_data\3.3.5\tree\DBFilesClient \
  --export-remap .\defs\053_to_335.remap.json
```

### 0.5.3 → 3.3.5 (apply remap; deterministic)
```powershell
dotnet run --project .\DBCTool\DBCTool.csproj -- \
  --dbd-dir .\lib\WoWDBDefs\definitions \
  --out out \
  --locale enUS \
  --compare-area \
  --input 0.5.3=..\test_data\0.5.3\tree\DBFilesClient \
  --input 3.3.5=..\test_data\3.3.5\tree\DBFilesClient \
  --apply-remap .\defs\053_to_335.remap.json
```

### 0.5.5 → 3.3.5 (discover + export)
```powershell
dotnet run --project .\DBCTool\DBCTool.csproj -- \
  --dbd-dir .\lib\WoWDBDefs\definitions \
  --out out \
  --locale enUS \
  --compare-area --src-alias 0.5.5 --src-build 0.5.5.3494 --tgt-build 3.3.5.12340 \
  --input 0.5.5=..\test_data\0.5.5\tree\DBFilesClient \
  --input 3.3.5=..\test_data\3.3.5\tree\DBFilesClient \
  --export-remap .\defs\055_to_335.remap.json
```

### 0.5.5 → 3.3.5 (apply remap; deterministic)
```powershell
dotnet run --project .\DBCTool\DBCTool.csproj -- \
  --dbd-dir .\lib\WoWDBDefs\definitions \
  --out out \
  --locale enUS \
  --compare-area --src-alias 0.5.5 --src-build 0.5.5.3494 --tgt-build 3.3.5.12340 \
  --input 0.5.5=..\test_data\0.5.5\tree\DBFilesClient \
  --input 3.3.5=..\test_data\3.3.5\tree\DBFilesClient \
  --apply-remap .\defs\055_to_335.remap.json
```

## Outputs

Comparison outputs (written to `out/compare/`):
- `Map_crosswalk_{053|055}_to_335.csv`
- `MapId_to_Name_{0.5.3|0.5.5}.csv`, `MapId_to_Name_3.3.5.csv`
- `AreaTable_mapping_{053|055}_to_335.csv`
- `AreaTable_unmatched_{053|055}_to_335.csv`
- `AreaTable_ambiguous_{053|055}_to_335.csv`
- `AreaTable_rename_suggestions_{053|055}_to_335.csv`
- `Area_patch_crosswalk_{053|055}_to_335.csv`

Remap definition:
- `defs/{053|055}_to_335.remap.json`
  - `meta`: `{ src_alias, src_build, tgt_build, generated_at }`
  - `aliases`: name → name variants to try first (e.g., `{"demonic stronghold":["dreadmaul hold"]}`)
  - `explicit_map`: per-source `src_areaNumber` → `tgt_areaID` with optional `note`
  - `ignore_area_numbers`: excluded source area numbers (e.g., dev placeholders)
  - `options`: `{ disallow_do_not_use_targets: true|false }`

## API Documentation

For developers who wish to consume the `.remap.json` files in other tools, detailed documentation of the file format is available.

- **[DBCTool Remap File API](./docs/api.md)**

### Column reference (patch CSV)
`Area_patch_crosswalk_{053|055}_to_335.csv`:
- `src_mapId`, `src_mapName`
- `src_areaNumber`, `src_parentNumber`, `src_name`
- `tgt_mapId_xwalk`, `tgt_mapName_xwalk`
- `tgt_areaID`, `tgt_parentID`, `tgt_name`

## Using the Patch CSV to fix ADT AreaIDs

When converting legacy (0.5.x) ADTs to 3.3.5 ADT format:
- Read the legacy area field (32-bit) as the 0.5.x `AreaNumber`.
- Join to `Area_patch_crosswalk_{053|055}_to_335.csv` on `src_areaNumber`.
- Write `tgt_areaID` into the 3.3.5 ADT.
- For unmatched rows, either leave as-is or map to `0` per your policy (dev placeholders are excluded from stats by default).

## Matching Strategy (high level)

- Crosswalk maps first by `Directory`, with name fallback; then bias area matching by that map.
- Exact name matches with variants (toggle leading "The ", targeted aliases).
- Fuzzy: Levenshtein within the map, then global fallback.
- Exclude 3.3.5 targets containing “DO NOT USE” by default (use `--allow-do-not-use` to override).
- Skip dev placeholders from unmatched stats (e.g., `***On Map Dungeon***`, Programmer Isle, Plains of Snow, Jeff Quadrant).

## Troubleshooting

- Ensure `lib/WoWDBDefs/definitions` path is correct (or pass `--dbd-dir`).
- If `--export-remap` targets a new directory, the tool creates it automatically.
- If unmatched rows remain, consult `AreaTable_rename_suggestions_{053|055}_to_335.csv` and add aliases or explicit maps to the remap JSON, then re-run with `--apply-remap`.

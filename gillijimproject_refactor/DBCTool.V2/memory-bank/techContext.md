# Tech Context — DBCTool.V2

## Data sources
- Alpha/Classic DBCs via DBCD + WoWDBDefs (AreaTable, Map, etc.)
- LK 3.3.5 DBCs via DBCD + WoWDBDefs
- Optional 0.6.0 pivot set to bridge renames between 0.5.x and 3.3.5

## Decode rules (Alpha → chains)
- AreaNumber is packed `zone<<16 | sub` in Alpha ADTs (see ADT v18 docs, L536).
- ZoneBase = `zone<<16 | 0`.
- Validate `ParentAreaNum(zoneBase)` for sub rows.
- Build chain:
  - If `lo16 == 0`: `[zone]`
  - If `lo16 > 0` and SubIndex hit within same continent: `[zone, sub]`
  - Else: `[zone]` (no sub)

## Matching rules (to LK)
- Map-lock by source row’s continent using Map crosswalk (no cross-map in primary pass).
- Try exact chain inside locked map first.
- Optional pivot through 0.6.0 then crosswalk to LK.
- Rename fallbacks (global) — labeled and unique-only:
  - `rename_global` for zones, `rename_global_child` for children
  - Fuzzy variants: `rename_fuzzy`, `rename_fuzzy_child` (Levenshtein ≤ 2), still unique-only
- Special-case `On Map Dungeon` → 0 in fallback streams.

## Outputs
- `compare/v2/AreaTable_mapping_{src}_to_335.csv`
- `compare/v2/AreaTable_unmatched_{src}_to_335.csv`
- `compare/v2/Area_patch_crosswalk_{src}_to_335.csv`
- `compare/v2/Area_patch_crosswalk_map{mapId}_{src}_to_335.csv`
- `compare/v2/Area_patch_with_fallback0_{src}_to_335.csv`
- `compare/v2/Area_patch_with_fallback0_map{mapId}_{src}_to_335.csv`
- Dumps/diagnostics: `AreaTable_dump_{src}.csv`, `AreaTable_dump_3.3.5.csv`, `zone_missing_in_target_map_*.csv`

## Consumption contract (other tools)
- Use per-map `Area_patch_crosswalk_map{mapId}_{src}_to_335.csv` for strict mapping.
- Numeric mapping should be keyed by `(src_mapName, src_areaNumber)` only.
- Do not invent targets — if not present, treat as unmatched/0.

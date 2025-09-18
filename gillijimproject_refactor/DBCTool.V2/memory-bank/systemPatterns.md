# System Patterns — DBCTool.V2

## Pipeline overview
1) Decode Alpha/Classic AreaTable + Map using DBCD + WoWDBDefs.
2) Build per-continent ZoneIndex/SubIndex from Alpha AreaTable.
3) Construct deterministic chains from Alpha AreaNumber:
   - `lo16 == 0` → `[zone]`
   - `lo16 > 0` and sub exists in same continent → `[zone, sub]`
   - else → `[zone]`
4) Map-lock to target map via Map crosswalk (no cross-map during primary matching).
5) Try exact chain match in locked map; else apply rename/fuzzy fallbacks (unique-only, labeled).
6) Emit per-map patch CSVs plus unmatched/fallback variants and dumps.

## Matching contract
- Primary pass never emits cross-map matches.
- Fallbacks:
  - `rename_global`, `rename_global_child` (explicitly labeled)
  - `rename_fuzzy`, `rename_fuzzy_child` (Levenshtein ≤ 2), still unique-only
- `On Map Dungeon` is allowed to map to `0` in fallback streams.

## Outputs (compare/v2)
- `AreaTable_mapping_{src}_to_335.csv`
- `AreaTable_unmatched_{src}_to_335.csv`
- `Area_patch_crosswalk_{src}_to_335.csv`
- `Area_patch_crosswalk_map{mapId}_{src}_to_335.csv` (strict per-map patch source)
- `Area_patch_with_fallback0_{src}_to_335.csv`
- `Area_patch_with_fallback0_map{mapId}_{src}_to_335.csv`
- Dumps: `AreaTable_dump_{src}.csv`, `AreaTable_dump_3.3.5.csv`, `zone_missing_in_target_map_*.csv`

## CSV schema stability
- Headers remain stable across sessions.
- `match_method` communicates how a row was resolved.
- Per-map patch CSVs are the authoritative inputs for consumers (e.g., AlphaWDTAnalysisTool).

## Notes on compounded LK names
- LK may prefix child names with their parent (e.g., `western plaguelands: hearthglen`).
- Prefer exact matches; rely on rename/fuzzy only when unique and safe.
- When in doubt, prefer explicit per-map rows to avoid ambiguity.

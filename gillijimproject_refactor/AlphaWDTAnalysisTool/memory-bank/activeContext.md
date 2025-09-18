# Active Context — AlphaWDTAnalysisTool

## Current Focus
- Apply strict CSV-only AreaID patching using DBCTool.V2 outputs (`compare/v2/`).
- Keep mapping scoped to `(src_mapName, src_areaNumber)`; no name/zone fallback.
- Emit per-tile verify CSVs in verbose runs to audit writes.

## Recent Changes
- Enforced per-map numeric mapping only; removed all heuristics and LK-dump fallbacks.
- README updated with `--dbctool-patch-dir` and `--dbctool-lk-dir` usage.
- Memory bank updated to reflect strict behavior and binary write offsets.

## Outstanding Items
- DeadminesInstance still 0 for some `src_areaNumber` due to missing per-map rows in CSVs. Action: update DBCTool.V2 crosswalk for that map.
- Compound LK names (e.g., `western plaguelands: hearthglen`) can cause non-matches. Resolution expected via DBCTool.V2 aliases/fuzzy or explicit per-map rows.

## Next Steps
- Optional: add a verbose diagnostic toggle to trace which CSV file produced each `patch_csv_num` hit.
- Coordinate with DBCTool.V2 to add explicit rows or aliases for compounded names.

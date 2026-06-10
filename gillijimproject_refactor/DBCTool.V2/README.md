# DBCTool.V2 — AreaTable mapping (Alpha → 3.3.5)
## Minimal end-to-end run

1. Prepare inputs as described in [Input Data Preparation](docs/input-data-prep.md).
2. Generate crosswalks and audits (example: 0.5.3 source):
   ```bash
   dotnet run --project DBCTool.V2/DBCTool.V2.csproj -- --s53
   ```
   After running, files are written under `dbctool_outputs/session_*/compare/` and `compare/v2/`.
3. Use the generated per-map crosswalks in downstream tools (e.g., AlphaWDTAnalysisTool) and, after export, run the verify aggregator (see that tool's docs).


DBCTool.V2 generates per-map CSVs and patches to remap Alpha-era AreaTable area numbers (zone<<16|sub) to Wrath of the Lich King (3.3.5) AreaTable IDs with strong safety rules.

This README explains how it works, how to run it, and how to reuse it from other tools.

## Documentation

- AreaID Restoration Approach: [docs/areaid-restoration-approach.md](docs/areaid-restoration-approach.md)
- Input Data Preparation: [docs/input-data-prep.md](docs/input-data-prep.md)

## Highlights
- Zone-only matching with strict map lock from the source row’s ContinentID
- Optional pivot via 0.6.0 to bridge Alpha name changes
- Global rename and fuzzy-rename fallbacks (unique-only) across LK
- Parent-agnostic patching: `tgt_parentID = tgt_areaID`
- Explicit per-map outputs plus a separate “fallback0” variant for unmatched rows
- Audit CSVs, raw dumps, and missing-zone diagnostics

## Mapping rules
- **Map-lock**: Determine target map from the source row’s `ContinentID` using a Map.dbc crosswalk (Alpha → LK). No cross-map matches during primary matching.
- **Zone-only chain**: Build `[zoneName]` chain. For subzones, resolve the zone name from the zoneBase row on the same continent as the current row.
- **Exact match first**: Try an exact chain match (zone-only) inside the locked map.
- **Pivot 0.6.0 (optional)**: When provided, map the Alpha chain into 0.6.0 first, then crosswalk into LK and match there.
- **Rename fallback (global)**:
  - `rename_global`: If the zone name is unique across LK top-level zones, use it (cross-map allowed).
  - `rename_global_child`: Same for child names.
- **Fuzzy rename (unique-only)**:
  - `rename_fuzzy` / `rename_fuzzy_child`: Levenshtein ≤ 2 and unique best candidate.
- **On Map Dungeon**: Any “On Map Dungeon” rows map to 0 in the fallback streams.
- **Parent-agnostic patch**: For matched rows, set `tgt_parentID = tgt_areaID`.

## What gets written after running
After you run DBCTool.V2, outputs are written under `dbctool_outputs/session_*/compare/` and `compare/v2/`:
- `compare/v2/AreaTable_mapping_{src}_to_335.csv`
- `compare/v2/AreaTable_unmatched_{src}_to_335.csv`
- `compare/v2/Area_patch_crosswalk_{src}_to_335.csv`
- `compare/v2/Area_patch_crosswalk_map{mapId}_{src}_to_335.csv`
- `compare/v2/Area_patch_with_fallback0_{src}_to_335.csv`
- `compare/v2/Area_patch_with_fallback0_map{mapId}_{src}_to_335.csv`
- Audit/diagnostics:
  - `compare/alpha_areaid_anomalies.csv` (0/1 churn filtered)
  - `compare/v2/AreaTable_dump_{src}.csv`, `compare/v2/AreaTable_dump_3.3.5.csv`
  - `compare/v2/zone_missing_in_target_map_{src}_to_335.csv`

## Usage (CLI)

Examples:

- Compare 0.5.3 → 3.3.5 and write CSVs:
```
dotnet run --project DBCTool.V2/DBCTool.V2.csproj -- --s53
```
- Other sources (0.5.5/0.6.0) are auto-detected from input dirs. 3.3.5 is required.

## Normalization and aliases
- Lowercase, drop punctuation/spaces, drop leading article `the `
- Aliases (examples):
  - `shadowfang` → `shadowfang keep`
  - `south seas` → `south seas unused`

### Compound zone strings (parent + child in LK names)

Some LK 3.3.5 AreaTable names include the parent zone as a prefix (e.g., `western plaguelands: hearthglen`). Alpha-side children may exist only as child rows without the parent prefix. This can lead to non-matches in strict map-locked exact name matching.

- Strategy:
  - Keep primary matching strict and map-locked.
  - Rely on rename/fuzzy fallbacks (unique-only) to bridge these differences where safe.
  - Prefer explicit per-map CSV corrections when available to avoid ambiguity.

If you encounter systematic non-matches due to compounded names, consider adding explicit rows in the per-map patch CSVs or expanding aliases to normalize prefixes.

## Notes & guarantees
- Per-map outputs are map-locked where appropriate. Cross-map renames are tagged via `match_method = rename_*`.
- Fallback0 CSVs contain only truly unmatched rows (or special-cased `onmapdungeon`).
- Anomalies CSV suppresses 0↔1 churn which is expected for early content duplication.

Additional guarantees:
- Primary map-locked pass never emits cross-map targets.
- Rename/fuzzy steps are gated by uniqueness to avoid accidental cross-map leakage.

## Reuse as a library
See `docs/DBCTool.V2-API.md` for the `AreaIdMapperV2` programmatic API.

## Troubleshooting
- If a known area doesn’t map, check the dumps:
  - `compare/v2/AreaTable_dump_{src}.csv` and `AreaTable_dump_3.3.5.csv`
- If a target exists but is a renamed variant, add an alias or rely on rename/fuzzy steps.
- If you want to keep outputs map-locked only, you can post-filter on `match_method`.

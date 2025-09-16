# 001 – Alpha→LK AreaID Mapping (DBCTool-first)

## Objective
Produce a verified remap.json from DBCD truth, plus two diagnostics CSVs:
- alpha_areaid_decode.csv: proof that Alpha Unknown3 is (zone<<16 | sub) with parent validation
- alpha_to_335_suggestions.csv: advisory LK zone/sub suggestions via name + continent bias

AlphaWDTAnalysisTool remains remap-only for AreaID writes and will later consume remap.json. No heuristics for writes.

## Alpha AreaID On ADTs
Alpha MCNK “AreaID” is NOT a single AreaID:
- Unknown3 = (zone<<16 | sub)
- zone = AreaTable.AreaNumber of the zone row
- sub = AreaTable.AreaNumber of the subzone row (0 means “none”)
- Parent validation: when sub != 0, ParentAreaNum(subRow) MUST equal (zone<<16)

Continent is derived from the zone row (ContinentID of the zone row).

Reference: This behavior is documented in the retail MCNK structure comment: `/*0x034*/  uint32_t areaid; // in alpha: both zone id and sub zone id, as uint16s.` See `reference_data/wowdev.wiki/ADT_v18.md` line 536. We apply this strictly: hi16 = zone, lo16 = sub.

## Decoding Algorithm
Given raw = Unknown3:
- z = raw >> 16; s = raw & 0xFFFF
- zoneBase = (z << 16)
- zone_name_alpha = name(AreaNumber == zoneBase)
- sub_name_alpha = name(AreaNumber == raw) when s != 0
- parent_ok = (s == 0) or (ParentAreaNum(subRow) == zoneBase)
- alpha_continent = ContinentID(zone-row)

No swap heuristics. Strict hi16→zone, lo16→sub.

## CSV: alpha_areaid_decode.csv
Columns:
- alpha_raw, alpha_raw_hex, zone_num, zone_name_alpha, sub_num, sub_name_alpha, parent_ok, alpha_continent

Row selection:
- Unique per map by alpha_raw (raw Unknown3). Avoid duplicates.

## CSV: alpha_to_335_suggestions.csv
Columns:
- alpha_raw, alpha_raw_hex, zone_num, zone_name_alpha, sub_num, sub_name_alpha, alpha_continent, lk_zone_id_suggested, lk_zone_name, lk_sub_id_suggested, lk_sub_name, method

Method:
- Build LK indices: normalized-name → [IDs], and map-bias index (MapID → normalized-name → [IDs])
- Suggest LK zone by normalized zone name:
  - Prefer entries with LK.MapID == alpha_continent → method "map_biased"
  - Else fall back to global name → method "global"
- If sub != 0, suggest LK sub by normalized sub name filtered to LKParent == suggested-zone
  - If no sub match, append ":fallback_to_zone" to method

Suggestions are advisory only (never used for writes).

## remap.json (explicit only)
Schema (example):
{
  "meta": {
    "src_alias": "0.5.3",
    "src_build": "0.5.3.3368",
    "tgt_build": "3.3.5.12340",
    "generated_at": "2025-09-13T00:00:00Z"
  },
  "explicit_map": [
    { "src_areaNumber": 1441792, "lk_areaID": 20, "note": "Example" }
  ],
  "ignore": [
    { "src_areaNumber": 655360, "reason": "no_equivalent_or_do_not_use" }
  ],
  "options": { "disallow_do_not_use_targets": true }
}

- src_areaNumber is the full 32-bit alpha Unknown3 (zone<<16 | sub), matching ADT raw.
- Writes in AlphaWDTAnalysisTool will be driven by explicit_map only.

Outputs written by `--compare-area`:
- `out/compare/alpha_areaid_decode.csv`
- `out/compare/alpha_to_335_suggestions.csv`
- `out/compare/Map_crosswalk_<src>_to_335.csv`
- Mapping diagnostics CSVs under `out/compare/`

## DBCTool Implementation Plan
- Extend Program.CompareAreas() to emit:
  - out/compare/alpha_areaid_decode.csv
  - out/compare/alpha_to_335_suggestions.csv
- If `--export-remap <file>` is provided, write remap.json with explicit_map only
- CLI examples:
  dotnet run -- --dbd-dir lib/WoWDBDefs/definitions --out out --compare-area --input 0.5.3=.../DBFilesClient --input 3.3.5=.../DBFilesClient --export-remap out/alpha053_to_335_remap.json --allow-do-not-use

## AlphaWDTAnalysisTool Backport (later)
- Keep remap-only writes:
  - For each alpha MCNK raw Unknown3, look up in explicit_map
  - If matched: write LK AreaID
  - If ignored or missing: no write
- Verification CSVs:
  - remap_dump.csv
  - remap_explicit_used.csv
  - Optional: per-tile sample CSV with readback only when --remap is present
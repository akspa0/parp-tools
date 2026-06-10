# Alpha Area Decode V2 (DBCTool.V2)

- Output root: dbctool_outputs/<session>/...
- Audit CSVs:
  - dbctool_outputs/<session>/compare/alpha_areaid_decode_v2.csv
  - dbctool_outputs/<session>/compare/alpha_areaid_anomalies.csv
- No confidence tags are surfaced in CSVs.

Integration steps:
1) Decode hi16/lo16 halves from Alpha AreaNumber and validate ParentAreaNum == zoneBase.
2) Build per-continent ZoneIndex/SubIndex, resolve cross-continent ownership per zoneBase.
3) Build deterministic chains:
   - lo16 == 0 -> [zone]
   - lo16 > 0 with sub hit -> [zone, sub]
   - lo16 > 0 without sub hit -> [zone]
4) Map-locked exact matching against LK indices (no cross-map results).

---

## Source packing (ADT v18)
- Alpha ADTs pack the area number into the `MCNK` header `Unknown3` 32-bit field (`mcnk+8+0x38`) as `zone<<16 | sub`.
- Reference: ADT v18 documentation (L536) confirming hi16/lo16 halves for zone/sub fields.

## Validation
- For any subzone (`lo16 > 0`), the parent zone is determined from the `zoneBase = zone<<16` row.
- Validate `ParentAreaNum(sub) == zoneBase`. Any mismatches are surfaced in `alpha_areaid_anomalies.csv` (0/1 churn filtered).

## Indices
- Build `ZoneIndex` and `SubIndex` per continent for Alpha. When a `zoneBase` appears in multiple continents, resolve canonical ownership based on majority and ParentAreaNum references.

## Chain construction
- Deterministic chain per area number, used as the lookup key:
  - `lo16 == 0` → `[zone]`
  - `lo16 > 0` and SubIndex hit on the same continent → `[zone, sub]`
  - `lo16 > 0` but no sub hit → `[zone]` (sub dropped)

## Matching into LK (3.3.5)
- Map-lock to a candidate target map based on the Alpha row’s continent via a Map crosswalk. Primary matching never emits cross-map results.
- Try exact chain match inside the locked map.
- Optionally pivot via 0.6.0 first to bridge early renames, then re-lock and match in LK.
- If still unmatched, emit into `AreaTable_unmatched_*` and `Area_patch_with_fallback0_*` streams (with special-case `onmapdungeon -> 0`).

## Outputs
- `compare/alpha_areaid_decode_v2.csv` — source decode with `zone`, `sub`, `zoneBase`, and validation fields.
- `compare/v2/AreaTable_mapping_{src}_to_335.csv` — resolved mappings (map-locked primary pass annotated with `match_method`).
- `compare/v2/AreaTable_unmatched_{src}_to_335.csv` — areas without a primary match.
- `compare/v2/Area_patch_crosswalk_{src}_to_335.csv` and `Area_patch_crosswalk_map{mapId}_{src}_to_335.csv` — patch-ready numeric crosswalks for consumers.

## Consumption guidance (AlphaWDTAnalysisTool)
- Use only the per-map crosswalks keyed by `(src_mapName, src_areaNumber)` for strict patching.
- Do not invent targets. If no row exists or `tgt_areaID == 0`, write `0` and audit.

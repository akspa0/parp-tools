# Active Context

- Current Focus:
  - **Documentation and Usability**: The core area remapping functionality is complete and stable. The current focus is on creating clear, comprehensive documentation so that the tool's output (specifically the `remap.json` files) can be reliably consumed by other downstream tools and developers.

- Core Logic Summary:
  - The tool provides a deterministic workflow for mapping AreaIDs between different client builds.
  - It uses a combination of map cross-walking, exact name matching (with aliases), and fuzzy matching to find the best candidates.
  - The entire process can be saved to a `.remap.json` file and re-applied later for consistent results.

- Next Steps:
  1.  **Create API Documentation**: Write a clear `api.md` file explaining the structure of the `remap.json` output.
  2.  **Update README**: Add a link in the main `README.md` pointing to the new API documentation.
  3.  **Align Memory Bank**: Ensure all memory bank files reflect the tool's current, functional state and its new focus on developer experience.

## Alpha AreaID Decoding & Mapping (DBCTool-first)
- Strict rule: Alpha MCNK area field packs zone/sub as uint16 halves → hi16 = zone, lo16 = sub.
  - Reference: `reference_data/wowdev.wiki/ADT_v18.md` line 536: `/*0x034*/ uint32_t areaid; // in alpha: both zone id and sub zone id, as uint16s.`
- Outputs produced under `out/compare/` when using `--compare-area`:
  - `alpha_areaid_decode.csv` (proof: zone/sub decode with parent validation)
  - `alpha_to_335_suggestions.csv` (advisory LK zone/sub by name + map bias)
  - Crosswalk/mapping diagnostics unchanged
- AlphaWDTAnalysisTool remains remap-only for writes and will consume remap.json later (explicit_map only).

### Stepwise Plan (small patches)
- Part A: Restrict remap export to explicit-only (gate `exportExplicit.Add` by `method == "explicit"`).
- Part B: Add `ChooseSubWithinZone(subName, zoneId)` helper.
- Part C: Add `alpha_areaid_decode.csv` generation (strict hi16/lo16 + parent validation).
- Part D: Add `alpha_to_335_suggestions.csv` generation (zone: name+map bias; sub: within chosen zone; annotated methods).
- Part E: Write the two CSVs and console logs.

Status: A–E queued; docs updated; implementing one part per commit to keep changes small.

## Updates — 2025-09-15
- Zone suggestions are now strict: LK zone must be on the same map (map-locked) and top-level only (ParentAreaID==ID). No global fallback for zones.
- Sub suggestions are constrained to the chosen zone and same map:
  - Try NameVariants (article flip + aliases) for exact matches first, then fuzzy among children of the zone.
  - Method annotations: "name" / "name_alias" / "fuzzy"; if no match, sub=-1 and ":fallback_to_zone" is appended.
- Alpha raw 0 policy:
  - `alpha_raw==0` is emitted with method "alpha_zero" and no LK IDs (-1). 3.3.5 has no catch-all 0 AreaID; engine shows "Unknown Zone".
- AlphaWDTAnalysisTool alignment:
  - Writer remains remap-only (explicit mappings only). Mapping stats are map-aware and count only `remap_explicit`.
  - Optional verification CSV per tile (`csv/maps/<Map>/areaid_verify_<x>_<y>.csv`) confirms on-disk LK AreaID after patch.

## Collision-resistant AreaTable mapping (2025-09-15)

- Problem
  - Cross-continent collisions from name-only mapping caused Azshara (map 1) to appear as parent for map 0 areas in mapping CSVs.

- Decisions
  - Zones are map-locked and top-level-only:
    - Use `ChooseTargetByName(srcName, mapIdX, requireMap: true, topLevelOnly: true)`.
  - Sub-areas are anchored to the chosen zone and same map:
    - Resolve parent zone first (same call as above), then `ChooseSubWithinZone(subName, zoneId)` constrained to zone’s children and same map.
    - If no sub match, do NOT coerce to the zone; mark unmatched with `zone_only_no_sub`.
  - Cross-map guard:
    - After any selection, if `mapIdX >= 0` and `Extract335(chosen).mapId != mapIdX`, treat as unmatched with `cross_map_violation`.
  - Outputs (QA-oriented):
    - In addition to global CSVs, write per-map CSVs (map 0 and map 1) for mapping, unmatched, and patch crosswalk to make validation explicit.
    - Route rows by `src.mapIdX` when available; unknown `mapIdX` remain only in global CSVs.
  - Remap export remains explicit-only; no heuristics.

- Method annotations
  - Zones: `name+map` (exact), `fuzzy+map` (fuzzy), `unmatched`.
  - Subs: `name` / `name_alias` / `fuzzy` within `:sub(...)`; when sub misses, append `:zone_only_no_sub`.
  - Explicit: `explicit` (and `explicit_cross_map` flagged only for diagnostics if a mismatch is detected).

- Impact
  - Eliminates cross-map parent/sub leakage, aligns mapping strictness with suggestions, and simplifies QA via per-map outputs.

- AlphaWDTAnalysisTool alignment
  - Writer remains remap-only (explicit mappings only).
  - Add validation before writes: reject cross-map entries (relative to ADT’s map context) and entries missing DBCD targets; log/report skipped items.
